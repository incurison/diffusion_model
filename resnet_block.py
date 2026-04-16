import torch
import torch.nn as nn
from einops import reduce,rearrange
import torch.nn.functional as F
# partial把函数的一部分参数先固定住，返回一个新函数
from functools import partial
from tools import exists

# 卷积核权重标准化
class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self,x):
        self.eps=1e-5 if x.dtype==torch.float32 else 1e-3
        weight=self.weight
        mean=reduce(weight,'o ...->o 1 1 1','mean')#reduce魔法方法，把[o,i,k,k]转换成[o,1,1,1]计算每个输出通道的对应均值
        var=reduce(weight,'o ...->o 1 1 1',partial(torch.var,unbiased=False))#不需要使用无偏估计
        normalized_weight=(weight-mean)/torch.sqrt(var+self.eps)
        return F.conv2d(x,#等到把卷积层的权重给标准化之后我们才进行卷积层的输出
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups)

class Block(nn.Module):
    def __init__(self,dim,dim_out,groups=8):
        super().__init__()
        self.proj=WeightStandardizedConv2d(in_channels=dim,out_channels=dim_out,padding=1,kernel_size=3)
        self.norm=nn.GroupNorm(groups,dim_out)#groupnorm归一化
        self.act=nn.SiLU()
    def forward(self,x,scale_shift=None):#这里通过film将时间向量融入到x输入中，使得时间可以控制x的方差和均值，进而影响他的噪声的预测
        x=self.proj(x)
        x=self.norm(x)
        if exists(scale_shift):
            scale,shift=scale_shift
            x=x*(scale+1)+shift
        x=self.act(x)
        return x


class ResNet(nn.Module):
    def __init__(self,dim,dim_out,time_emp_dim=None,groups=8):#time_emp_dim是传入的时间向量的特征维度，后续我们在unet拼接中会把时间向量维度升维四倍，然后分割给不同的resnet块，之后还会有一个全局的MLP
        super().__init__()
        self.block1=Block(dim,dim_out,groups)
        self.block2=Block(dim_out,dim_out,groups)
        self.mlp=nn.Sequential(nn.SiLU(),nn.Linear(time_emp_dim,dim_out*2)) if exists(time_emp_dim) else None
        self.res_conv=nn.Conv2d(in_channels=dim,out_channels=dim_out,kernel_size=1) if dim!=dim_out else nn.Identity()#这里的话意思就是如果我们的残差连接中x输入的维度和卷积神经网络的维度不同的话，我们就调整x输入的维度让他和卷积输出的维度保持一致，这样才能相加
    def forward(self,x,time_emb=None):
        scale_shift=None
        if exists(self.mlp):
            time_emb=self.mlp(time_emb)
            time_emb=rearrange(time_emb,"b c -> b c 1 1")
            scale_shift=time_emb.chunk(2,dim=1)
        h=self.block1(x,scale_shift)
        # 第二层残差块就不用在融入时间向量了直接那前一层的输出当成输入就行
        h=self.block2(h)
        return h+self.res_conv(x)
            