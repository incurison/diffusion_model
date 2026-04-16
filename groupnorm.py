"""groupNorm模块，用来我们搭建unet时的组归一化"""
import torch 
import torch.nn as nn

class preNorm(nn.Module):
    def __init__(self,dim,fn):
        super().__init__()
        self.fn=fn
        self.norm=nn.GroupNorm(num_groups=1,num_channels=dim)#bum_groups是分组数量，这里当成一个大组去处理，num_channels是我们输入的数据的通道数
    def forward(self,x):
        x=self.norm(x)
        return self.fn(x)