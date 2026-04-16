import torch
import torch.nn as nn
from einops import rearrange
class Attention(nn.Module):
    def __init__(self,dim,head=4,head_dim=32):#这里的dim是我们的输入的通道数比如512，hiddendim是我们多头注意力矩阵合并之后算出来的通道数可能是128，实现降维，从而减少在计算点积的时候的计算量节省显存，当然我们最后还需要在加一层卷积层将其膨胀回512，方便后续的上采样
        super().__init__()
        self.heads=head
        hidden_dim=head*head_dim#总的通道数
        self.scale=head_dim**(-0.5)#√d，softmax用来控制尺度
        self.to_qkv=nn.Conv2d(in_channels=dim,out_channels=hidden_dim*3,kernel_size=1,stride=1,bias=False)#这里的hidden_dim*3方便后续分割,一般在卷积层后面有group norm或者其他归一化的时候就不加bias，因为回减均值抵消，没用。因为我们算出来的q，k，v公式是Q=Wx不需要有偏置
        self.to_out=nn.Conv2d(in_channels=hidden_dim,out_channels=dim,kernel_size=1)
    def forward(self,x):
        b,c,h,w=x.shape
        qkv=self.to_qkv(x).chunk(3,dim=1)#按照特征维度通道数c进行切分
        q,k,v=map(
            lambda t:rearrange(t,"b (h c) x y -> b h c (x y)",h=self.heads),qkv#把q,k,v变成(batch,head,dim_head,N(x*y))
        )
        q=self.scale*q
        sim=torch.einsum('b h d i,b h d j -> b h i j',q,k)#爱因斯坦积，将对应元素进行点积也可以写成sim = torch.matmul(q.transpose(-2, -1), k)
        sim=sim-sim.amax(keepdim=True,dim=-1).detach()#截断梯度不让他通过这一部分
        scores=sim.softmax(dim=-1)#d对每个像素做softmax
        out=torch.einsum('b h i j,b h d j -> b h d i', scores,v)
        out=rearrange(out,'b h d (x y) -> b (h d) x y',x=h,y=w)
        return self.to_out(out)#最后一步给他还原回原来的通道数


class LinearAttention(nn.Module):
    def __init__(self,dim,heads=4,head_dim=32):
        super().__init__()
        self.heads=heads
        self.hidden_dim=heads*head_dim
        self.scale=head_dim**(-0.5)
        self.to_qkv=nn.Conv2d(in_channels=dim,out_channels=self.hidden_dim*3,kernel_size=1,stride=1,bias=False)
        self.to_out=nn.Conv2d(in_channels=self.hidden_dim,out_channels=dim,kernel_size=1)
        
    def forward(self,x):
        b,c,h,w = x.shape
        qkv=self.to_qkv(x).chunk(3,dim=1)
        q,k,v=map(
            lambda t: rearrange(t,"b (h c) x y -> b h c (x y)",h=self.heads),qkv
        )
        q=q.softmax(dim=-1)#我们对q和k分别做一个线性映射函数softmax
        k=k.softmax(dim=-2)
        
        q=self.scale*q
        context=torch.einsum('b h e n,b h d n -> b h e d',k,v)#我们改变原来的矩阵乘法顺序，先让K和V做点积，对V先进行一个提取上下文,点积的话即使维度不一样也要用不同的字母求表示：e,d不然回强制对齐变成矩阵乘法而不是点积
        out=torch.einsum('b h e d,b h e n -> b h d n',context,q)#这里在让上下文信息去乘q查询矩阵
        out=rearrange(out,'b h d (x y) -> b (h d) x y',h=self.heads,x=h,y=w)
        return self.to_out(out)
        
        