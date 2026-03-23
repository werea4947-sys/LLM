import torch

##开方求倒数
t=torch.rsqrt(torch.tensor(4.0))
print(t)

# 创建一个全1张量
t2=torch.ones(3,4)
print(t2)


class RMSNorm(nn.Module):
#__init__方法中定义了模型的结构和参数，forward方法中定义了模型的前向传播过程。
    def __init__(self,dim:int,eps:float=1e-5):
        super().init__()
        self.dim=dim
        self.eps=eps
        self.weight=nn.Parameter(torch.ones(dim))
#_norm
    def _norm(self,x):
        return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)#rmsnorm的计算公式
    #其中x是输入张量，pow(2)表示对每个元素进行平方，mean(-1,keepdim=True)表示在最后一个维度上求平均值，并保持维度不变，eps是一个小常数，用于防止除以零的情况。最后将输入张量x乘以权重参数weight，并返回结果。
#forword方法中输入x，经过层归一化、线性变换、激活函数、dropout等操作，最后返回输出结果。
    def forward(self,x):
        output=self._norm(x)*self.weight.type_as(x)
        return output