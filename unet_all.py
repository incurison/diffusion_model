import torch
import torch.nn as nn
from attention import LinearAttention,Attention
from resnet_block import ResNet
from Position_embedding import pos_embed
from tools import default,Residual
from functools import partial
from groupnorm import preNorm
from sampling import downsampling,upsampling
class Unet(nn.Module):
    def __init__(self,dim,init_dim=None,out_dim=None,dim_mults=(1,2,4,8),channels=3,self_condition=False,resnet_block_groups=4):
        super().__init__()
        self.self_condition=self_condition
        input_channels=channels*(2 if self_condition else 1)
        init_dim=default(init_dim,dim)
        self.init_conv=nn.Conv2d(input_channels,init_dim,1,padding=0)
        dims=[init_dim,*map(lambda m: dim*m,dim_mults)]
        in_out=list(zip(dims[:-1],dims[1:]))#[(32,64),(64,128),...,(256,512)]这样就可以得到每一次的输入维度和输出维度
        block_klass=partial(ResNet,groups=resnet_block_groups)#固定组归一化的参数，下次使用不用再传惨进去了
        # 时间向量化
        time_dim=dim*4#dim是我们设置的维度基数，用来根据维度放缩倍数决定后续上采样和下采样的维度数,(这里指的是通道数而不是分辨率)
        self.time_mlp=nn.Sequential(
            pos_embed(dim),
            nn.Linear(dim,time_dim),
            nn.GELU(),
            nn.Linear(time_dim,time_dim)#通过激活函数和全连接层让时间向量的动态选择sin和cos
        )
        
        # 采样层
        self.downs=nn.ModuleList([])
        self.ups=nn.ModuleList([])
        num_resolutions=len(in_out)
        
        
        
        # 下采样层
        for index,(dim_in,dim_out) in enumerate(in_out):
            is_last=index>=(num_resolutions-1)#用来判断是否维度达到目标预期还要继续叠加
            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in,dim_in,time_emp_dim=time_dim),
                    block_klass(dim_in,dim_in,time_emp_dim=time_dim),
                    Residual(preNorm(dim_in,LinearAttention(dim_in))),#残差连接层，之后把x输进去，他会自动去计算线性自注意力矩阵并且进行group_norm组归一化，并加上x残差跳跃
                    downsampling(dim_in,dim_out)
                    if not  is_last#最后一层直接卷积输出
                    else nn.Conv2d(dim_in,dim_out,kernel_size=3,padding=1)
                    
                ])
            )
        
        # 瓶颈层
        #下采样完了之后的通道数
        mid_dim=dims[-1]
        self.mid_block1=block_klass(mid_dim,mid_dim,time_emp_dim=time_dim)
        self.mid_attn=Residual(preNorm(mid_dim,Attention(mid_dim)))
        self.mid_block2=block_klass(mid_dim,mid_dim,time_emp_dim=time_dim)
        
        # 上采样
        for index,(dim_in,dim_out) in enumerate(reversed(in_out)):#因为上采样的话我们就要逐步减少通道特征数了所以反过来遍历
            is_last=index==(len(in_out)-1)
            
            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out+dim_in,dim_out,time_emp_dim=time_dim),#这里上采样要拼接前面下采样的特征图，来学习上采样学习的局部特征，最后再把他压缩回去,因为我们前面规定了下采样最后一层不降维，所以这里直接可以那上采样的第一层和下采样倒数第二层进行拼接就行
                    block_klass(dim_out+dim_in,dim_out,time_emp_dim=time_dim),
                    Residual(preNorm(dim_out,LinearAttention(dim_out))),
                    upsampling(dim_out,dim_in)
                    if not is_last
                    else nn.Conv2d(in_channels=dim_out,out_channels=dim_in,kernel_size=3,padding=1)#这里的话(dim_in=256,dim_out=512)因为是反向的降维，所以是dim_out是输入维度，dim_in是输出维度
                ])
            )
        
        # 最终输出
        self.out_dim=default(out_dim,channels)
        self.final_res_block=block_klass(init_dim*2,dim,time_emp_dim=time_dim)#后面的话还需要再进行一次残差跳跃加上初始化卷积的结果所以在拼接变成了init_dim*2，因为我们最终预测的是噪声，所以要加上原始的底片r，让他最终反向传播输出的结果是纯噪音
        self.final_conv=nn.Conv2d(dim,self.out_dim,1)#最后恢复到原始的dim
    
    def forward(self,x,time,x_self_cond=None):
        if self.self_condition:
            x_self_cond=default(x_self_cond,lambda:torch.zeros_like(x))#如果我们启用自条件了，那就用自条件矩阵，不然的话就用零矩阵，等价于没有自条件
            x=torch.cat((x_self_cond,x),dim=1)
            
        # 先初始化卷积
        x=self.init_conv(x)
        
        # 克隆一份给后续使用
        r=x.clone()
        
        t=self.time_mlp(time)
        
        h=[]#备份文档
        # 下采样
        for block1,block2,attn,downsample in self.downs:
            x=block1(x,t)
            h.append(x)
            
            x=block2(x,t)
            x=attn(x)
            h.append(x)
            
            x=downsample(x)
        
        # 瓶颈层全局注意力
        x=self.mid_block1(x,t)
        x=self.mid_attn(x)
        x=self.mid_block2(x,t)
        
        
        # 上采样
        for block1,block2,attn,upsample in self.ups:
            x=torch.cat((x,h.pop()),dim=1)
            x=block1(x,t)
            x=torch.cat((x,h.pop()),dim=1)
            x=block2(x,t)
            x=attn(x)
            
            x=upsample(x)
        x=torch.cat((x,r),dim=1)
        
        x=self.final_res_block(x,t)
        final_res=self.final_conv(x)
        return final_res
            
            
            
        
        
        