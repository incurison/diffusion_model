import torch
import torch.nn as nn
import math
from einops import rearrange
class pos_embed(nn.Module):
    def __init__(self,dim):#根据正弦公式PE(t,2i)=sin(t/10000^(2i/d))，这里的d就是dim
        super().__init__()
        self.dim=dim
    def forward(self,t):
        half_dim=self.dim/2
        emb_base=math.log(10000)/half_dim
        freqs=torch.exp(torch.arange(half_dim)*emb_base)
        freqs=freqs.to(t.device)
        # t 的形状[batch_size]变 [batch_size, 1]，freqs 变 [1, half_dim]，相乘得到 [batch_size, half_dim]
        args=t[:,None]*freqs[None,:]
        embeddings=torch.cat((torch.sin(args),torch.cos(args)),dim=-1)#对于单个图片单个t，这里的话简单拼接为(1,512)(sin,sin...,cos,cos..)并没有按照严格的公式那样交叉排列，这是因为我们想给模型留有学习空间，通过最后的对时间向量的全连接层，自己学习排序
        return embeddings

