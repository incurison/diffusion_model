import torch
import torch.nn.functional as F
from PIL import Image
import requests
import numpy as np
import torch
from torchvision.transforms import Compose,ToTensor,Lambda,ToPILImage,CenterCrop,Resize
def cosine_beta_schedule(timesteps,s=0.008):
    step=timesteps+1
    x=torch.linspace(0,timesteps,step)#生成0-timesteps的等间隔向量
    alpha_cumprod=torch.cos((x/timesteps+s)/(1+s)*torch.pi*0.5)#连乘向量
    alpha_cumprod=alpha_cumprod/alpha_cumprod[0]#因为第一个元素不一定是1所以除第一个元素归一化
    betas=1-(alpha_cumprod[1:]/alpha_cumprod[:-1])#beta_t=1-alpha_prod_t/alpha_prod_t-1
    return torch.clip(betas,0.0001,0.9999)#截断操作小于0.0001的改成0.0001大于0.9999的改成0.9999

# 线性方差表
def linear_beta_schedule(timesteps):
    beta_start=0.001
    beta_end=0.02
    return torch.linspace(beta_start,beta_end,timesteps) 
    
def quadratic_beta_schedule(timesteps):
    beta_start=0.0001
    beta_end=0.02
    return torch.linspace(beta_start**0.5,beta_end**0.5,timesteps)**2

def sigmoid_beta_schedule(timesteps):
    beta_start=0.0001
    beta_end=0.02
    betas=torch.linspace(-6,6,timesteps)
    return torch.sigmoid(betas)*(beta_end-beta_start)+beta_start




"""分界线 这里是我们的加噪过程，这里我们以线性方差表为例先对一个图片测试"""
timesteps=300
betas=linear_beta_schedule(timesteps=timesteps)
# alpha计算
alphas=1.0-betas

# alpha连乘计算
alphas_cumprod=torch.cumprod(alphas,axis=0)
alphas_cumprod_prev=F.pad(alphas_cumprod[:-1],(1,0),value=1.0)#删除第一个元素替换成1，左边补一个元素，右边补0个元素

# 计算$1 / \sqrt{\alpha_t}$ 这是全局放缩系数
sqrt_recip_alphas=torch.sqrt(1.0/alphas)

# 根号在alpha在t时刻的连乘
sqrt_alphas_cumprod=torch.sqrt(alphas_cumprod)

# 1-alpha的连乘
sqrt_one_minus_alphas_cumprod=torch.sqrt(1.0-alphas_cumprod)


# 计算后验概率分布的方差，即q(xt|x_t-1,x_0)的高斯分布的方差$$\tilde{\beta}_t = \beta_t \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t}$$
posterior_variance=betas*(1.0-alphas_cumprod_prev)/(1.-alphas_cumprod)


# 这个是一个维度匹配的函数，因为我们输入的图像特征图矩阵是四维的，而beta系数这些是一维的，我们要对其进行reshape，这样才能进行矩阵之间的广播运算
def extract(a,t,x_shape):
    if isinstance(t,int):
        t=torch.tensor([t],dtype=torch.long,device=a.device)
    elif t.dim()==0:
        t=t[None]#将标量转换为(1,)等价于t.unsqueeze(0)在当前位置插入一个新维度
    batch_size=t.shape[0]
    out=a.gather(-1,t.cpu())#把a最后一个维度中t索引的值拿出来
    return out.reshape(batch_size,*((1,)*(len(x_shape)-1))).to(t.device)


url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image=Image.open(requests.get(url=url,stream=True).raw)


# 将PIL对象转换为tensor对象并进行裁剪
image_size=128
transform=Compose([
    Resize(image_size),#resize回先把短边拉到image_size,然后再从中间裁出128,128
    CenterCrop(image_size),
    ToTensor(),#turn into torch Tensor of shape CHW, divide by 255，Totensor回把图像的像素值转换为0-1之间，针对图像的unit8数据
    Lambda(lambda t:(t*2)-1)#0-1映射到-1,1
])

x_start=transform(image).unsqueeze(0)#在前面插入1变成(1,C,H,W)
x_start.shape
np.array(x_start)

# reverse transform，将tensor对象转换为pillow对象
reversed_transform=Compose([
    Lambda(lambda t:(t+1)/2),
    Lambda(lambda t:t.permute(1,2,0)),#chw转换为HWC
    Lambda(lambda t:t*255.0),
    Lambda(lambda t : t.numpy().astype(np.uint8)),
    ToPILImage()
    
])

reversed_transform(x_start.squeeze())#squeeze默认清除所有维度为1的维度

# forward diffusion 前向传播
# 前向加噪
def q_sample(x_start,t,noise=None):
    if noise is None:
        noise=torch.randn_like(x_start)
    sqrt_alphas_cumprod_t=extract(sqrt_alphas_cumprod,t,x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod, t, x_start.shape
    )
    return sqrt_alphas_cumprod_t*x_start+sqrt_one_minus_alphas_cumprod_t*noise

#噪声加到图片上并返回
def get_noisy_image(x_start,t):
    x_noisy=q_sample(x_start,t=t)
    noisy_image=reversed_transform(x_noisy.squeeze())
    return noisy_image


# 测试一下t=40时的加噪效果
t=torch.tensor(40)
get_noisy_image(x_start,40)



# add_noise visualization
import matplotlib.pyplot as plt
torch.manual_seed(42)

def plot(imgs,with_orig=False,row_title=None,**imshow_kwargs):
    #with_org是否在每一行最前面加上一个原图
    # 这里我们把图片变成二维列表的形式
    if not isinstance(imgs[0],list):
        imgs=[imgs]
    
    num_rows=len(imgs)
    num_cols=len(imgs[0])+with_orig#如果要加原图的话那就多加一列
    fix,axs=plt.subplots(figsize=(200,200),nrows=num_rows,ncols=num_cols,squeeze=False)
    for row_id,row in enumerate(imgs):
        row=[image]+row if with_orig else row#[image]是原图
        for col_id,img in enumerate(row):
            ax=axs[row_id,col_id]
            ax.imshow(np.asarray(img),**imshow_kwargs)#np.asarray()转换为数组尽量不拷贝，相较于array节省内存
            ax.set(xticklabels=[],yticklabels=[],xticks=[],yticks=[])
    if with_orig:
        axs[0, 0].set(title='Original image')
        axs[0, 0].title.set_size(8)
    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])   
        
    plt.tight_layout()


plot([get_noisy_image(x_start,t) for t in [0,50,100,150,199]],with_orig=True)



# loss funtion 损失计算
# 这里的denoise_model其实就是我们前面定义的unet神经网络
def p_losses(denoise_model,x_start,t,noise=None,loss_type='l1'):
    if noise is None:
        noise=torch.randn_like(x_start)
    x_noisy=q_sample(x_start=x_start,t=t,noise=noise)
    predicted_noise=denoise_model(x_noisy,t)
    if loss_type=='l1':
        loss=F.l1_loss(noise,predicted_noise)
    elif loss_type=='l2':
        loss=F.l2_loss(noise,predicted_noise)
    if loss_type=='huber':
        loss=F.smooth_l1_loss(noise,predicted_noise)
    else:
        raise NotImplementedError#未实现错误
    return loss
        