import math

import torch
import torch.nn.functional as F


def npo2(len):
    """
    返回大于或等于给定长度的下一个 2 的幂次。

    例如：
        - 如果 length 是 5，则返回 8（2^3）。
        - 如果 length 是 8，则返回 8（2^3）。
        - 如果 length 是 9，则返回 16（2^4）。

    参数:
        length (int): 输入的长度。

    返回:
        int: 下一个 2 的幂次。
    """
    # 使用 math.ceil 对 log2(length) 进行向上取整，然后计算 2 的幂次
    return 2 ** math.ceil(math.log2(len))


def pad_npo2(X):
    """
    将输入张量在长度维度上填充到下一个 2 的幂次。

    该函数用于确保张量的长度是 2 的幂次，这在某些并行算法（如并行扫描）中是必要的。

    参数:
        X (Tensor): 输入张量，形状为 (B, L, D, N)，其中：
            - B: 批次大小（Batch size）
            - L: 长度维度（Length dimension）
            - D: 数据维度（Data dimension）
            - N: 其他维度（Other dimension）

    返回:
        Y (Tensor): 填充后的张量，形状为 (B, npo2(L), D, N)，其中：
            - npo2(L): 大于或等于 L 的下一个 2 的幂次。
            - 其他维度保持不变。
    """
    # 计算长度维度 L 的下一个 2 的幂次
    len_npo2 = npo2(X.size(1))  # X.size(1) 获取长度维度 L 的值

    # 定义填充元组，格式为 (左填充, 右填充, 上填充, 下填充, 前填充, 后填充)
    # 在这里，我们只对长度维度进行填充，因此其他维度填充为 0
    pad_tuple = (0, 0, 0, 0, 0, len_npo2 - X.size(1))  # (左, 右, 上, 下, 前, 后)

    # 使用 F.pad 对张量进行填充
    # 参数说明：
    # - X: 要填充的输入张量
    # - pad_tuple: 填充元组
    # - "constant": 使用常数填充
    # - 0: 填充值为 0
    return F.pad(X, pad_tuple, "constant", 0)


class PScan(torch.autograd.Function):
    """
    PScan 类实现了并行扫描操作，包括前向和反向传播。
    该类继承自 torch.autograd.Function，用于自定义自动求导过程。
    """
    @staticmethod
    def pscan(A, X):
        """
        前向传播的并行扫描操作。

        该方法对输入张量 X 进行原地修改，执行并行扫描操作。
        更正式地，X 将被填充为以下值：
            H[t] = A[t] * H[t-1] + X[t]  其中 H[0] = 0
        这些值是并行计算的（理想情况下需要 2*log2(T) 个顺序步骤，而不是 T 个顺序步骤）。

        注意：
            该方法仅支持长度为 2 的幂次的输入长度 L（主要是为了代码更清晰）。

        参数:
            A (Tensor): 输入张量，形状为 (B, D, L, N)。
            X (Tensor): 输入张量，形状为 (B, D, L, N)。

        返回:
            Tensor: 并行扫描后的输出张量，形状与 X 相同。
        """
        # 获取输入张量的维度
        B, D, L, _ = A.size()
        # 计算 log2(L)，即扫描的步骤数
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        # 上行扫描（Up-Sweep）阶段
        # 初始化 Aa 和 Xa 为输入张量 A 和 X
        Aa = A
        Xa = X

        # 进行 (num_steps - 2) 次迭代，逐步减少并行处理的元素数量
        for _ in range(num_steps-2):
            # 当前处理的元素数量
            T = Xa.size(2)
            # 重塑 Aa 和 Xa 以便并行处理
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            # 并行计算 Xa 的第 1 个元素
            # Xa[:, :, :, 1] += Aa[:, :, :, 1] * Xa[:, :, :, 0]
            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            # 并行计算 Aa 的第 1 个元素
            # Aa[:, :, :, 1] *= Aa[:, :, :, 0]
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            # 更新 Aa 和 Xa 为当前处理的子张量
            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # we have only 4, 2 or 1 nodes left
        # 处理剩余的 4, 2 或 1 个节点
        if Xa.size(2) == 4:
            # 处理第 1 个元素
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            Aa[:, :, 1].mul_(Aa[:, :, 0])

            # 处理第 3 个元素
            Xa[:, :, 3].add_(Aa[:, :, 3].mul(Xa[:, :, 2] + Aa[:, :, 2].mul(Xa[:, :, 1])))
        elif Xa.size(2) == 2:
            # 处理第 1 个元素
            Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        # 下行扫描（Down-Sweep）阶段
        # 重新初始化 Aa 和 Xa 为输入张量的特定部分
        Aa = A[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        Xa = X[:, :, 2**(num_steps-2)-1:L:2**(num_steps-2)]
        # 并行计算 Xa 的第 2 个元素
        Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 1]))
        Aa[:, :, 2].mul_(Aa[:, :, 1])

        # 进行 (num_steps - 3) 次迭代，逐步增加并行处理的元素数量
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 2**k-1:L:2**k]
            Xa = X[:, :, 2**k-1:L:2**k]

            T = Xa.size(2)
            # 重塑 Aa 和 Xa 以便并行处理
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # 并行计算 Xa 的第 1 个元素
            Xa[:, :, 1:, 0].add_(Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1]))
            # 并行计算 Aa 的第 1 个元素
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def pscan_rev(A, X):
        """
        反向传播的并行扫描操作。

        该方法与上述方法相同，但方向相反。
        （如果你翻转输入，调用 pscan，然后翻转输出，你将得到此函数输出的结果）
        它用于反向传播过程中。

        注意：
            该方法仅支持长度为 2 的幂次的输入长度 L（主要是为了代码更清晰）。

        参数:
            A (Tensor): 输入张量，形状为 (B, D, L, N)。
            X (Tensor): 输入张量，形状为 (B, D, L, N)。

        返回:
            Tensor: 反向并行扫描后的输出张量，形状与 X 相同。
        """
        # 获取输入张量的维度
        B, D, L, _ = A.size()
        # 计算 log2(L)，即扫描的步骤数
        num_steps = int(math.log2(L))

        # up sweep (last 2 steps unfolded)
        # 上行扫描（Up-Sweep）阶段
        # 初始化 Aa 和 Xa 为输入张量 A 和 X
        Aa = A
        Xa = X

        # 进行 (num_steps - 2) 次迭代，逐步减少并行处理的元素数量
        for _ in range(num_steps-2):
            # 当前处理的元素数量
            T = Xa.size(2)
            # 重塑 Aa 和 Xa 以便并行处理
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)
            
            # 并行计算 Xa 的第 0 个元素
            # Xa[:, :, :, 0] += Aa[:, :, :, 0] * Xa[:, :, :, 1]
            Xa[:, :, :, 0].add_(Aa[:, :, :, 0].mul(Xa[:, :, :, 1]))
            # 并行计算 Aa 的第 0 个元素
            # Aa[:, :, :, 0] *= Aa[:, :, :, 1]
            Aa[:, :, :, 0].mul_(Aa[:, :, :, 1])

            # 更新 Aa 和 Xa 为当前处理的子张量
            Aa = Aa[:, :, :, 0]
            Xa = Xa[:, :, :, 0]

        # we have only 4, 2 or 1 nodes left
        # 处理剩余的 4, 2 或 1 个节点
        if Xa.size(2) == 4:
            # 处理第 2 个元素
            Xa[:, :, 2].add_(Aa[:, :, 2].mul(Xa[:, :, 3]))
            Aa[:, :, 2].mul_(Aa[:, :, 3])

            # 处理第 0 个元素
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1].add(Aa[:, :, 1].mul(Xa[:, :, 2]))))
        elif Xa.size(2) == 2:
            # 处理第 0 个元素
            Xa[:, :, 0].add_(Aa[:, :, 0].mul(Xa[:, :, 1]))
            return
        else:
            return

        # down sweep (first 2 steps unfolded)
        # 下行扫描（Down-Sweep）阶段
        # 重新初始化 Aa 和 Xa 为输入张量的特定部分
        Aa = A[:, :, 0:L:2**(num_steps-2)]
        Xa = X[:, :, 0:L:2**(num_steps-2)]
        # 并行计算 Xa 的第 1 个元素
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 2]))
        Aa[:, :, 1].mul_(Aa[:, :, 2])

        # 进行 (num_steps - 3) 次迭代，逐步增加并行处理的元素数量
        for k in range(num_steps-3, -1, -1):
            Aa = A[:, :, 0:L:2**k]
            Xa = X[:, :, 0:L:2**k]

            T = Xa.size(2)
            # 重塑 Aa 和 Xa 以便并行处理
            Aa = Aa.view(B, D, T//2, 2, -1)
            Xa = Xa.view(B, D, T//2, 2, -1)

            # 并行计算 Xa 的第 1 个元素
            Xa[:, :, :-1, 1].add_(Aa[:, :, :-1, 1].mul(Xa[:, :, 1:, 0]))
            # 并行计算 Aa 的第 1 个元素
            Aa[:, :, :-1, 1].mul_(Aa[:, :, 1:, 0])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        应用并行扫描操作，如上所述。返回一个新的张量。
        如果可能，请优先使用长度为 2 的幂次的序列长度。

        参数:
            A_in (Tensor): 输入张量，形状为 (B, L, D, N)。
            X_in (Tensor): 输入张量，形状为 (B, L, D, N)。

        返回:
            H (Tensor): 输出张量，形状为 (B, L, D, N)。
        """
        # 获取长度维度 L
        L = X_in.size(1)

        # cloning is requiered because of the in-place ops
        # 因为存在原地操作，所以需要克隆张量
        if L == npo2(L):
            A = A_in.clone()
            X = X_in.clone()
        else:
            # pad tensors (and clone btw)
            # 对张量进行填充（同时进行克隆）
            A = pad_npo2(A_in) # (B, npo2(L), D, N)
            X = pad_npo2(X_in) # (B, npo2(L), D, N)
        
        # prepare tensors
        # 准备张量
        A = A.transpose(2, 1) # (B, D, npo2(L), N)
        X = X.transpose(2, 1) # (B, D, npo2(L), N)

        # parallel scan (modifies X in-place)
        # 执行并行扫描（原地修改 X）
        PScan.pscan(A, X)

        # 保存输入张量以供反向传播使用
        ctx.save_for_backward(A_in, X)
        
        # slice [:, :L] (cut if there was padding)
        # 切片 [:, :L]（如果有填充，则进行裁剪）
        return X.transpose(2, 1)[:, :L]
    
    @staticmethod
    def backward(ctx, grad_output_in):
        """
        将梯度从输出传递到输入。返回两个新的张量。

        参数:
            ctx: A_in (Tensor): (B, L, D, N), X (Tensor): (B, D, L, N)
            grad_output_in (Tensor): (B, L, D, N)

        返回:
            gradA (Tensor): (B, L, D, N), gradX (Tensor): (B, L, D, N)
        """
        # 获取保存的输入张量
        A_in, X = ctx.saved_tensors

        # 获取长度维度 L
        L = grad_output_in.size(1)

        # cloning is requiered because of the in-place ops
        # 因为存在原地操作，所以需要克隆张量
        if L == npo2(L):
            grad_output = grad_output_in.clone()
            # the next padding will clone A_in
            # 下一个填充操作将克隆 A_in
        else:
            grad_output = pad_npo2(grad_output_in) # (B, npo2(L), D, N)
            A_in = pad_npo2(A_in) # (B, npo2(L), D, N)

        # prepare tensors
        # 准备张量
        grad_output = grad_output.transpose(2, 1) # (B, D, npo2(L), N)
        A_in = A_in.transpose(2, 1) # (B, D, npo2(L), N)
        A = torch.nn.functional.pad(A_in[:, :, 1:], (0, 0, 0, 1)) # (B, D, npo2(L), N) shift 1 to the left (see hand derivation)

        # reverse parallel scan (modifies grad_output in-place)
        # 反向并行扫描（原地修改 grad_output）
        PScan.pscan_rev(A, grad_output)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output[:, :, 1:])

        return Q.transpose(2, 1)[:, :L], grad_output.transpose(2, 1)[:, :L]
    
pscan = PScan.apply
