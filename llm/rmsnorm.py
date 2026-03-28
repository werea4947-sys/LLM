import torch
import torch.nn as nn
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self,x): #在dim维度求均值，保持维度不变
        return x*torch.rsqrt(x*pow(2).mean(-1,keepdim=True)+self.eps)
    
    def forward(self,x):
        output = self._norm(x.float()).type_as(x)
        return output*self.weight