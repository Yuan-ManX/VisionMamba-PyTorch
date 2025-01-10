import math
from dataclasses import dataclass
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pscan import pscan


"""
- 一个 Mamba 模型由多个层组成，这些层是 ResidualBlock。

- 一个 ResidualBlock 由一个 MambaBlock、一个归一化层和一个残差连接组成：ResidualBlock(x) = mamba(norm(x)) + x

- 这就引出了 MambaBlock：其输入 x 是 (B, L, D)，输出 y 也是 (B, L, D)（B=批次大小，L=序列长度，D=模型维度）。

首先，将 x 扩展为 (B, L, 2*ED)（其中 E 通常为 2），并将其拆分为 x 和 z，每个都是 (B, L, ED)。
然后，对 x 应用短 1D 卷积，然后是激活函数（silu），然后是 SSM。
然后将其乘以 silu(z)。

"""


@dataclass
class MambaConfig:
    """
    Mamba 模型配置类，用于存储模型的各种超参数配置。

    属性:
        d_model (int): 模型维度 D。
        n_layers (int): 模型的层数。
        dt_rank (Union[int, str], optional): 离散时间矩阵的秩。如果设置为 'auto'，则自动计算。默认为 'auto'。
        d_state (int, optional): 状态维度 N，在论文和注释中为 N。默认为 16。
        expand_factor (int, optional): 扩展因子 E，在论文和注释中为 E。默认为 2。
        d_conv (int, optional): 卷积层的维度。默认为 4。

        dt_min (float, optional): 离散时间步长的最小值。默认为 0.001。
        dt_max (float, optional): 离散时间步长的最大值。默认为 0.1。
        dt_init (str, optional): 离散时间步长的初始化方法，可以是 'random' 或 'constant'。默认为 'random'。
        dt_scale (float, optional): 离散时间步长的缩放因子。默认为 1.0。
        dt_init_floor (float, optional): 离散时间步长初始化的下限。默认为 1e-4。

        rms_norm_eps (float, optional): RMS 归一化中的 epsilon 参数。默认为 1e-5。
        base_std (float, optional): 基础标准差，用于初始化参数。默认为 0.02。

        bias (bool, optional): 是否使用偏置。默认为 False。
        conv_bias (bool, optional): 卷积层是否使用偏置。默认为 True。
        inner_layernorms (bool, optional): 是否对内部激活应用层归一化。默认为 False。

        mup (bool, optional): 是否使用 muP（模型并行化）。默认为 False。
        mup_base_width (float, optional): muP 的基础宽度，默认为 128。

        pscan (bool, optional): 训练时是否使用并行扫描模式。如果为 False，则使用顺序模式。默认为 True。
        use_cuda (bool, optional): 训练时是否使用官方的 CUDA 实现（与 (b)float16 不兼容）。默认为 False。
    """
    d_model: int # D
    n_layers: int
    dt_rank: Union[int, str] = 'auto'
    d_state: int = 16 # N in paper/comments
    expand_factor: int = 2 # E in paper/comments
    d_conv: int = 4

    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init: str = "random" # "random" or "constant"
    dt_scale: float = 1.0
    dt_init_floor = 1e-4

    rms_norm_eps: float = 1e-5
    base_std: float = 0.02

    bias: bool = False
    conv_bias: bool = True
    inner_layernorms: bool = False # apply layernorms to internal activations

    mup: bool = False
    mup_base_width: float = 128 # width=d_model

    pscan: bool = True # use parallel scan mode or sequential mode when training
    use_cuda: bool = False # use official CUDA implementation when training (not compatible with (b)float16)

    def __post_init__(self):
        # 计算内部维度 D_inner = E * D
        self.d_inner = self.expand_factor * self.d_model # E*D = ED in comments

        # 如果 dt_rank 设置为 'auto'，则自动计算 dt_rank
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)

        # muP 设置
        if self.mup:
            self.mup_width_mult = self.d_model / self.mup_base_width


class Mamba(nn.Module):
    """
    Mamba 模型类，实现了 Mamba 架构。

    该模型由多个 ResidualBlock 组成，每个 ResidualBlock 包含一个 MambaBlock、一个归一化层和一个残差连接。

    参数:
        config (MambaConfig): Mamba 模型的配置，包含各种超参数。
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        # 保存配置
        self.config = config

        # 构建模型层，每个层是一个 ResidualBlock
        self.layers = nn.ModuleList([ResidualBlock(config) for _ in range(config.n_layers)])

    def forward(self, x):
        """
        前向传播方法。

        对输入张量 x 进行多层处理，每一层都是 ResidualBlock。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, D)。

        返回:
            Tensor: 输出张量，形状为 (B, L, D)。
        """
        # x 的形状为 (B, L, D)

        # 对每一层进行处理
        for layer in self.layers:
            x = layer(x)

        return x
    
    def step(self, x, caches):
        """
        单步前向传播方法，用于推理或训练时逐步处理输入。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, D)。
            caches (List[dict]): 每一层的缓存列表，每个缓存是一个字典，包含历史状态。

        返回:
            Tuple[Tensor, List[dict]]: 处理后的输出张量和更新后的缓存列表。
        """
        # x 的形状为 (B, L, D)
        # caches 是一个包含每个层的缓存的列表，缓存的形状为 (h, inputs)

        # 对每一层进行处理，并更新缓存
        for i, layer in enumerate(self.layers):
            x, caches[i] = layer.step(x, caches[i])

        return x, caches


class ResidualBlock(nn.Module):
    """
    残差块（ResidualBlock）类，用于构建 Mamba 模型。

    该模块由一个 MambaBlock、一个归一化层和一个残差连接组成。
    具体来说，ResidualBlock 的输出是 MambaBlock 的输出与输入的加和：
        ResidualBlock(x) = MambaBlock(norm(x)) + x

    参数:
        config (MambaConfig): Mamba 模型的配置，包含各种超参数。
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        # 初始化 MambaBlock 作为混合器（mixer）
        self.mixer = MambaBlock(config)
        # 初始化 RMS 归一化层，归一化的维度为 d_model，epsilon 为 config.rms_norm_eps，mup 为 config.mup
        self.norm = RMSNorm(config.d_model, config.rms_norm_eps, config.mup)

    def forward(self, x):
        """
        前向传播方法。

        对输入张量 x 进行归一化处理，通过 MambaBlock 处理后，与原始输入 x 相加，实现残差连接。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, D)。

        返回:
            Tensor: 输出张量，形状为 (B, L, D)。
        """
        # 对输入张量 x 进行归一化处理
        # x 的形状为 (B, L, D)

        # 通过 MambaBlock 处理归一化后的张量
        # 输出形状为 (B, L, D)
        output = self.mixer(self.norm(x)) + x
        return output
    
    def step(self, x, cache):
        """
        单步前向传播方法，用于推理或训练时逐步处理输入。

        参数:
            x (Tensor): 输入张量，形状为 (B, D)。
            cache (dict): 当前层的缓存，包含历史状态。

        返回:
            Tuple[Tensor, dict]: 处理后的输出张量和更新后的缓存。
        """
        # 对输入张量 x 进行归一化处理
        # x 的形状为 (B, D)
        # cache 的形状为 (h, inputs)
        # h 的形状为 (B, ED, N)
        # inputs 的形状为 (B, ED, d_conv-1)

        # 通过 MambaBlock 的单步前向传播方法处理归一化后的张量，并更新缓存
        # 输出形状为 (B, D)
        # cache 的形状保持不变
        output, cache = self.mixer.step(self.norm(x), cache)

        # 将输出与原始输入 x 相加，实现残差连接
        output = output + x
        return output, cache


class MambaBlock(nn.Module):
    """
    MambaBlock 类实现了 Mamba 模型的基本构建块。

    MambaBlock 是 Mamba 模型的核心组件，负责处理输入特征并进行状态空间建模。
    它通过线性投影、1D 卷积、状态空间模型（SSM）和残差连接来实现复杂的特征变换。

    参数:
        config (MambaConfig): Mamba 模型的配置，包含各种超参数。
    """
    def __init__(self, config: MambaConfig):
        super().__init__()

        self.config = config

        # projects block input from D to 2*ED (two branches)
        # 将输入从 D 维度投影到 2*ED（两个分支）
        # ED 是扩展维度，通常为 D 的倍数，用于扩展特征表示
        self.in_proj = nn.Linear(config.d_model, 2 * config.d_inner, bias=config.bias)

        # 定义 1D 卷积层，用于对输入进行卷积操作
        self.conv1d = nn.Conv1d(in_channels=config.d_inner, out_channels=config.d_inner, 
                              kernel_size=config.d_conv, bias=config.conv_bias, 
                              groups=config.d_inner,
                              padding=config.d_conv - 1)
        
        # projects x to input-dependent delta, B, C
        # 将输入张量投影到依赖于输入的 delta, B, C
        # delta: 增量，用于状态空间模型的更新
        # B, C: 状态空间模型的参数
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + 2 * config.d_state, bias=False)

        # projects delta from dt_rank to d_inner
        # 将 delta 从 dt_rank 投影到 d_inner
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        # dt initialization
        # dt weights
        # dt 初始化
        # 计算 dt 的初始化标准差
        dt_init_std = config.dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            # 使用常数初始化 dt_proj 的权重
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            # 使用均匀分布随机初始化 dt_proj 的权重
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            # 如果初始化方法未实现，则抛出异常
            raise NotImplementedError
        
        # delta bias
        # 初始化 dt 偏置
        # 计算 dt 的初始值，使用指数函数和随机数生成器
        dt = torch.exp(
            torch.rand(config.d_inner) * (math.log(config.dt_max) - math.log(config.dt_min)) + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # 计算 softplus 的逆，用于初始化 dt 的偏置
        inv_dt = dt + torch.log(-torch.expm1(-dt)) 
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # S4D 真实初始化
        # 生成状态空间模型的参数 A
        A = torch.arange(1, config.d_state + 1, dtype=torch.float32).repeat(config.d_inner, 1)
        # 将 A 存储在对数中以保持 A < 0，确保数值稳定性
        self.A_log = nn.Parameter(torch.log(A)) # why store A in log ? to keep A < 0 (cf -torch.exp(...)) ? for gradient stability ?
        # 不对 A_log 进行权重衰减
        self.A_log._no_weight_decay = True

        # 初始化状态空间模型的参数 D
        # 初始化 D 为全 1
        self.D = nn.Parameter(torch.ones(config.d_inner))
        # 不对 D 进行权重衰减
        self.D._no_weight_decay = True

        # projects block output from ED back to D
        # 将块输出从 ED 投影回 D
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

        # used in jamba
        # 在 jamba 中使用
        if self.config.inner_layernorms:
            # 如果启用内部层归一化，则定义 RMS 归一化层
            self.dt_layernorm = RMSNorm(self.config.dt_rank, config.rms_norm_eps, config.mup)
            self.B_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
            self.C_layernorm = RMSNorm(self.config.d_state, config.rms_norm_eps, config.mup)
        else:
            # 否则，将归一化层设置为 None
            self.dt_layernorm = None
            self.B_layernorm = None
            self.C_layernorm = None

        if self.config.use_cuda:
            # 如果使用 CUDA，则尝试导入 CUDA 实现的选择性扫描函数
            try:
                from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
                self.selective_scan_cuda = selective_scan_fn
            except ImportError:
                # 如果导入失败，则打印错误信息并将 use_cuda 设置为 False
                print("Failed to import mamba_ssm. Falling back to mamba.py.")
                self.config.use_cuda = False

    def _apply_layernorms(self, dt, B, C):
        """
        应用层归一化到 dt, B, C。

        参数:
            dt (Tensor): 增量张量。
            B (Tensor): B 张量。
            C (Tensor): C 张量。

        返回:
            Tuple[Tensor, Tensor, Tensor]: 归一化后的 dt, B, C。
        """
        if self.dt_layernorm is not None:
            dt = self.dt_layernorm(dt)
        if self.B_layernorm is not None:
            B = self.B_layernorm(B)
        if self.C_layernorm is not None:
            C = self.C_layernorm(C)
        return dt, B, C

    def forward(self, x):
        """
        前向传播方法。

        对输入张量 x 进行处理，包括线性投影、1D 卷积、SiLU 激活、状态空间模型（SSM）和输出投影。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, D)。

        返回:
            Tensor: 输出张量，形状为 (B, L, D)。
        """
        # 获取输入张量的批次大小 B、长度维度 L 和其他维度
        _, L, _ = x.shape

        # 将输入张量投影到 2*ED（两个分支）
        xz = self.in_proj(x) # (B, L, 2*ED)

        # 将投影后的张量拆分为 x 和 z 两个分支
        x, z = xz.chunk(2, dim=-1) # (B, L, ED), (B, L, ED)

        # x branch
        # **x 分支处理**：
        # 对 x 进行转置，从 (B, L, ED) 变为 (B, ED, L)
        x = x.transpose(1, 2) # (B, ED, L)

        # 对 x 进行 1D 卷积操作，卷积核大小为 `config.d_conv`，并进行填充以保持长度不变
        # 然后截取前 L 个元素，以保持长度不变
        x = self.conv1d(x)[:, :, :L] # depthwise convolution over time, with a short filter

        # 对 x 进行转置，从 (B, ED, L) 变为 (B, L, ED)
        x = x.transpose(1, 2) # (B, L, ED)

        # 对 x 进行 SiLU 激活
        x = F.silu(x)

        # 通过状态空间模型（SSM）处理 x 和 z，得到输出 y
        y = self.ssm(x, z)

        # 使用 CUDA 的情况
        if self.config.use_cuda:
            # 通过输出投影层将 y 映射到最终输出
            output = self.out_proj(y) # (B, L, D)
            # 剩余的操作在 ssm 函数中完成（与 CUDA 融合）
            return output # the rest of the operations are done in the ssm function (fused with the CUDA pscan)

        # z branch
        # z 分支处理
        # 对 z 进行 SiLU 激活
        z = F.silu(z) # (B, L, D)

        # 将 y 和 z 相乘，得到输出
        output = y * z # (B, L, ED)

        # 通过输出投影层将输出映射到最终输出
        output = self.out_proj(output) # (B, L, D)

        return output
    
    def ssm(self, x, z):
        """
        状态空间模型（SSM）方法。

        对输入特征 x 和辅助向量 z 进行状态空间建模。

        参数:
            x (Tensor): 输入特征张量，形状为 (B, L, ED)。
            z (Tensor): 辅助向量张量，形状为 (B, L, ED)。

        返回:
            Tensor: SSM 的输出，形状为 (B, L, ED)。
        """
        # 计算 A 和 D 参数
        A = -torch.exp(self.A_log.float()) # (ED, N)
        D = self.D.float()

        # 将输入 x 投影到 delta, B, C
        deltaBC = self.x_proj(x) # (B, L, dt_rank+2*N)

        # 将 deltaBC 拆分为 delta, B, C
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # (B, L, dt_rank), (B, L, N), (B, L, N)
        
        # 应用层归一化到 delta, B, C
        delta, B, C = self._apply_layernorms(delta, B, C)

        # 将 delta 投影到 d_inner 维度
        delta = self.dt_proj.weight @ delta.transpose(1, 2) # (ED, dt_rank) @ (B, L, dt_rank) -> (B, ED, L)

        # 选择使用哪种选择性扫描函数，根据配置决定
        if self.config.use_cuda:
            # 如果使用 CUDA，则对张量进行转置，以适应 CUDA 实现
            x = x.transpose(1, 2)
            B = B.transpose(1, 2)
            C = C.transpose(1, 2)
            z = z.transpose(1, 2)

            # "softplus" + "bias" + "y * silu(z)" operations are fused
            # 将 "softplus" + "bias" + "y * silu(z)" 操作融合在一起
            y = self.selective_scan_cuda(x, delta, A, B, C, D, z=z, delta_softplus=True, delta_bias=self.dt_proj.bias.float())
            # 将输出转置回 (B, L, ED)
            y = y.transpose(1, 2) # (B, L, ED)
        
        else:
            # 如果不使用 CUDA，则对 delta 进行转置，并应用 softplus 和偏置
            delta = delta.transpose(1, 2)
            delta = F.softplus(delta + self.dt_proj.bias) # (B, L, ED)

            # 根据配置选择使用并行扫描还是顺序扫描
            if self.config.pscan:
                y = self.selective_scan(x, delta, A, B, C, D) # (B, L, ED)
            else:
                y = self.selective_scan_seq(x, delta, A, B, C, D) # (B, L, ED)

        return y
    
    def selective_scan(self, x, delta, A, B, C, D):
        """
        选择性扫描方法（并行版本）。

        对输入张量 x 进行选择性扫描操作，结合增量 delta、参数 A, B, C, D 生成输出。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, ED)。
            delta (Tensor): 增量张量，形状为 (B, L, ED)。
            A (Tensor): 参数张量，形状为 (ED, N)。
            B (Tensor): 参数张量，形状为 (B, L, N)。
            C (Tensor): 参数张量，形状为 (B, L, N)。
            D (Tensor): 参数张量，形状为 (ED,)。

        返回:
            Tensor: 选择性扫描的输出，形状为 (B, L, ED)。
        """
        # 计算 deltaA，形状为 (B, L, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        # 计算 deltaB，形状为 (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        # 计算 BX，形状为 (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)
        
        # 调用并行扫描函数 pscan，得到 hs，形状为 (B, L, ED, N)
        hs = pscan(deltaA, BX)

        # 计算输出 y，形状为 (B, L, ED, 1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # 将 D 与 x 相乘并加到 y 上，得到最终输出，形状为 (B, L, ED)
        y = y + D * x

        return y
    
    def selective_scan_seq(self, x, delta, A, B, C, D):
        """
        选择性扫描方法（顺序版本）。

        对输入张量 x 进行选择性扫描操作，结合增量 delta、参数 A, B, C, D 生成输出。

        参数:
            x (Tensor): 输入张量，形状为 (B, L, ED)。
            delta (Tensor): 增量张量，形状为 (B, L, ED)。
            A (Tensor): 参数张量，形状为 (ED, N)。
            B (Tensor): 参数张量，形状为 (B, L, N)。
            C (Tensor): 参数张量，形状为 (B, L, N)。
            D (Tensor): 参数张量，形状为 (ED,)。

        返回:
            Tensor: 选择性扫描的输出，形状为 (B, L, ED)。
        """
        # 获取输入张量的批次大小 B 和长度维度 L
        _, L, _ = x.shape

        # 计算 deltaA，形状为 (B, L, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # (B, L, ED, N)
        # 计算 deltaB，形状为 (B, L, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(2) # (B, L, ED, N)

        # 计算 BX，形状为 (B, L, ED, N)
        BX = deltaB * (x.unsqueeze(-1)) # (B, L, ED, N)

        # 初始化 h，形状为 (B, ED, N)
        h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # (B, ED, N)
        # 初始化 hs 列表，用于存储中间结果
        hs = []

        # 顺序扫描过程
        for t in range(0, L):
            # 更新 h，形状为 (B, ED, N)
            h = deltaA[:, t] * h + BX[:, t]
            # 将 h 添加到 hs 列表中
            hs.append(h)
        
        # 将 hs 列表堆叠为张量，形状为 (B, L, ED, N)
        hs = torch.stack(hs, dim=1) # (B, L, ED, N)

        # 计算输出 y，形状为 (B, L, ED, 1)
        y = (hs @ C.unsqueeze(-1)).squeeze(3) # (B, L, ED, N) @ (B, L, N, 1) -> (B, L, ED, 1)

        # 将 D 与 x 相乘并加到 y 上，得到最终输出，形状为 (B, L, ED)
        y = y + D * x

        return y
    
    # -------------------------- inference -------------------------- #
    """
    关于自回归推理

    使用 Mamba 的优点在于推理时间与序列长度无关。
    我们只需要为每个层在缓存中保留两个东西：
    - 隐藏状态 h（形状为 (B, ED, N)），就像在 RNN 中进行推理时一样。
    - 层的最后 d_conv-1 个输入，以便能够计算 1D 卷积，这是对时间维度的卷积。
      （d_conv 是固定的，因此随着序列生成的进行，缓存不会增长）
      （并且 d_conv 通常非常小，比如 4，所以我们只需要“记住”最后 3 个输入）

    具体来说，这两个量被放入一个缓存元组中，分别命名为 h 和 inputs。
    h 的形状为 (B, ED, N)，inputs 的形状为 (B, ED, d_conv-1)。
    MambaBlock.step() 方法接收这个缓存，并且除了输出输出外，还输出下一个调用的更新缓存。

    缓存对象初始化如下：(None, torch.zeros())。
    当 h 为 None 时，选择性扫描函数会检测到它并从 h=0 开始。
    torch.zeros() 并不是问题（它与仅输入输入相同，因为 conv1d 是填充的）

    由于我们需要每个层一个这样的缓存变量，我们存储一个缓存对象，它只是一个缓存对象的列表。
    """
    
    def step(self, x, cache):
        """
        单步前向传播方法，用于推理或训练时逐步处理输入。

        参数:
            x (Tensor): 输入张量，形状为 (B, D)。
            cache (Tuple[Optional[Tensor], Tensor]): 当前层的缓存，包含历史隐藏状态和输入缓存。

        返回:
            Tuple[Tensor, Tuple[Optional[Tensor], Tensor]]: 处理后的输出张量和更新后的缓存。
        """
        # 获取缓存中的隐藏状态 h 和输入缓存 inputs
        h, inputs = cache  # h: (B, ED, N), inputs: (B, ED, d_conv-1)
        
        # 将输入 x 投影到 2*ED（两个分支）
        xz = self.in_proj(x) # (B, 2*ED)

        # 将投影后的张量拆分为 x 和 z 两个分支
        x, z = xz.chunk(2, dim=1) # (B, ED), (B, ED)

        # x branch
        # x 分支处理
        # 对 x 进行扩展，以便与输入缓存 inputs 进行拼接
        x_cache = x.unsqueeze(2)  # (B, ED, 1)

        # 将输入缓存 inputs 和 x_cache 在时间维度上拼接
        # 然后通过 1D 卷积层进行处理
        # 最后，截取最后一个时间步的结果，以保持长度不变
        x = self.conv1d(torch.cat([inputs, x_cache], dim=2))[:, :, self.config.d_conv-1] # (B, ED)

        # 对 x 进行 SiLU 激活
        x = F.silu(x)  # (B, ED)

        # 通过状态空间模型的单步前向传播方法处理 x 和 h，得到输出 y 和新的隐藏状态 h
        y, h = self.ssm_step(x, h)  # y: (B, ED), h: (B, ED, N)

        # z branch
        # z 分支处理
        # 对 z 进行 SiLU 激活
        z = F.silu(z)  # (B, ED)

        # 将 y 和 z 相乘，得到输出
        output = y * z  # (B, ED)

        # 通过输出投影层将输出映射到最终输出
        output = self.out_proj(output) # (B, D)

        # prepare cache for next call
        # 准备缓存以供下一次调用
        # 将输入缓存 inputs 向左移动一位，并拼接新的输入 x_cache
        inputs = torch.cat([inputs[:, :, 1:], x_cache], dim=2) # (B, ED, d_conv-1)

        # 将新的隐藏状态 h 和输入缓存 inputs 组合成新的缓存
        cache = (h, inputs)
        
        # 返回输出和缓存
        return output, cache

    def ssm_step(self, x, h):
        """
        状态空间模型（SSM）的单步前向传播方法。

        对输入特征 x 和隐藏状态 h 进行状态空间建模，生成输出 y 和新的隐藏状态 h。

        参数:
            x (Tensor): 输入特征张量，形状为 (B, ED)。
            h (Optional[Tensor]): 隐藏状态张量，形状为 (B, ED, N)。如果为 None，则初始化为 0。

        返回:
            Tuple[Tensor, Tensor]: 输出张量 y 和新的隐藏状态 h，形状分别为 (B, ED) 和 (B, ED, N)。
        """
        # 计算 A 和 D 参数
        A = -torch.exp(self.A_log.float()) # A: (ED, N)
        D = self.D.float() # D: (ED,)

        # 将输入 x 投影到 delta, B, C
        deltaBC = self.x_proj(x) # deltaBC: (B, dt_rank + 2 * N)

        # 将 deltaBC 拆分为 delta, B, C
        delta, B, C = torch.split(deltaBC, [self.config.dt_rank, self.config.d_state, self.config.d_state], dim=-1) # delta: (B, dt_rank), B: (B, N), C: (B, N)

        # 应用层归一化到 delta, B, C
        delta, B, C = self._apply_layernorms(delta, B, C) # delta: (B, dt_rank), B: (B, N), C: (B, N)

        # 将 delta 从 dt_rank 投影到 d_inner
        delta = F.softplus(self.dt_proj(delta)) # delta: (B, ED)

        # 计算 deltaA，形状为 (B, ED, N)
        deltaA = torch.exp(delta.unsqueeze(-1) * A) # deltaA: (B, ED, N)
        # 计算 deltaB，形状为 (B, ED, N)
        deltaB = delta.unsqueeze(-1) * B.unsqueeze(1) # deltaB: (B, ED, N)

        # 计算 BX，形状为 (B, ED, N)
        BX = deltaB * (x.unsqueeze(-1)) # BX: (B, ED, N)

        # 如果隐藏状态 h 为 None，则初始化为全零张量
        if h is None:
            h = torch.zeros(x.size(0), self.config.d_inner, self.config.d_state, device=deltaA.device) # h: (B, ED, N)

        # 更新隐藏状态 h，形状为 (B, ED, N)
        h = deltaA * h + BX  # h: (B, ED, N)

        # 计算输出 y，形状为 (B, ED, 1)
        y = (h @ C.unsqueeze(-1)).squeeze(2) # (B, ED, N) @ (B, N, 1) -> (B, ED, 1)

        # 将 D 与 x 相乘并加到 y 上，得到最终输出 y，形状为 (B, ED)
        y = y + D * x  # y: (B, ED)

        return y, h


class RMSNorm(nn.Module):
    """
    RMS 归一化（Root Mean Square Normalization）类。

    RMSNorm 通过计算输入张量的均方根（RMS）来进行归一化处理，以稳定训练过程并加速收敛。

    参数:
        d_model (int): 输入特征的维度。
        eps (float, optional): 防止除零的小常数。默认为 1e-5。
        use_mup (bool, optional): 是否使用 muP（模型并行化）。默认为 False。
    """
    def __init__(self, d_model: int, eps: float = 1e-5, use_mup: bool = False):
        super().__init__()

        # 是否使用 muP
        self.use_mup = use_mup
        # 防止除零的小常数
        self.eps = eps

        # 如果不使用 muP，则定义权重参数，用于缩放归一化后的输出
        if not use_mup:
            # 定义可学习的权重参数
            self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        前向传播方法。

        对输入张量进行 RMS 归一化处理，并根据是否使用 muP 进行缩放。

        参数:
            x (Tensor): 输入张量。

        返回:
            Tensor: 归一化后的输出张量。
        """
        # 计算 RMS 归一化后的输出
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # 如果不使用 muP，则对输出进行缩放
        if not self.use_mup:
            # 应用权重参数进行缩放
            return output * self.weight
        else:
            # 直接返回归一化后的输出
            return output
    
