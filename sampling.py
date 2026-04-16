import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
def default(val,d):#默认值函数，在我们没有输入采样想要输出的维度的时候就按输入维度输出不变
    if val:
        return val
    return d() if d is callable() else d


def upsampling(dim,dim_out):
    return nn.Sequential(
        nn.Upsample(scale_factor=2,mode='nearest'),#scale_factor是空间放大倍数，mode='nearst'是最近邻插值，最简单的方法
        nn.Conv2d(dim,default(dim_out,dim),kernel_size=3,padding=1)
    )
    
def downsampling(dim,dim_out):
     # No More Strided Convolutions or Pooling
    # 再也不用步长卷积和池化啦！
    return nn.Sequential(
        Rearrange("b c (h p1) (w p2)->b (c p1 p2) h w",p1=2,p2=2),
        nn.Conv2d(dim*4,default(dim_out,dim),kernel_size=1,padding=0)#kernal_size=1就不用加padding了
    )
