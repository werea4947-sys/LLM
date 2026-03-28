import torch
import torch.nn as nn
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim:int,end:int,theta:float=100000.0) -> torch.Tensor:#end是预计算的最大序列长度
    # torch.arange(0, dim, 2)[: (dim // 2)].float()生成了一个从0开始，步长为2的序列，长度为dim的一半
    # 然后每个元素除以dim，再取theta的倒数，得到频率
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # torch.arange(end)生成一个从0到end-1的序列，乘以频率，得到每个位置的频率
    t=torch.arange(end,device=freqs.device)
    #计算外积 得到一个二维矩阵，每一行都是t的元素乘以freqs的元素
    freqs = torch.outer(t, freqs).float()
    # 将频率矩阵转换为复数形式，实部为cos(freqs)，虚部为sin(freqs)，得到一个形状为(end, dim//2)的复数矩阵
    freqs_cos = freqs.cos()
    freqs_sin = freqs.sin()
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis:torch.Tensor, x:torch.Tensor) -> torch.Tensor:
    #获取x的维度信息
    ndim=x.ndim
    assert 0<=1<ndim

     # 断言，确保freqs_cis的形状与x的第二维和最后一维相同，即seqlen和head_dim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    # 构造一个新的形状，除了第二维和最后一维，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    
    # 将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

def apply_rotary_emb(xq:torch.Tensor, xk:torch.Tensor, freqs_cos:torch.Tensor, freqs_sin:torch.Tensor) -> torch.Tensor:
    #将查询和键张量转为浮点数，以便进行旋转嵌入的计算
    xq_r,xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1) #将xq沿最后一个维度分成两部分，实部和虚部
    xk_r,xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1) #将xk沿最后一个维度分成两部分，实部和虚部
    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)