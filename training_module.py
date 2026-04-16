import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.transforms import Compose
from datasets import load_dataset
from forward_back import cosine_beta_schedule,linear_beta_schedule,sigmoid_beta_schedule,extract,p_losses
import torch.functional as F
from tqdm import tqdm
from unet_all import Unet
"""数据加载"""
dataset=load_dataset('fashion_mnist')
image_size=28#minst数据集都是28*28
channels=1
batch_size=128

transform=Compose([
    transforms.RandomHorizontalFlip(),#以百分之50概率随机反转图形，防止模型过拟合
    transforms.ToTensor(),
    transforms.Lambda(lambda t : t*2-1)#映射到-1,1，输入对称数据能让模型收敛的更快
])

# 把图像转换成灰度图，并且一旦把图像转换完之后命名新列，删除旧列
def apply_transform(examples):
    examples["pixel_values"]=[transform(image.convert('L')) for image in examples['image']]
    del examples['image']
    return examples

transformed_dataset=dataset.with_transform(apply_transform).remove_columns('label')
dataLoader=DataLoader(transformed_dataset['train'],shuffle=True,batch_size=batch_size)

batch=next(iter(dataLoader))
print(batch.keys())


"""图像生成"""
"""
在这里的话我们需要在训练的过程中定义图像降噪的结果并将图片保存，方便我们呢检查模型的图片生成效果
"""


@torch.no_grad()
def p_sample(model,x,t,t_index,T):
    beta=linear_beta_schedule(T)
    alpha=1.0-beta
    alphas_cumprod=torch.cumprod(alpha,axis=0)
    alphas_cumprod_prev=F.pad(alphas_cumprod[:-1],(1,0),value=1.0)#删除第一个元素替换成1，左边补一个元素，右边补0个元素
    one_minus_alpha_cumprod=1.0-torch.cumprod(alpha,axis=0)
    sqrt_one_minus_alpha_cumprod=torch.sqrt(1.0-torch.cumprod(alpha,axis=0))
    betas_t=extract(beta,t,x.shape)
    sqrt_one_minus_alpha_cumprod_t=extract(sqrt_one_minus_alpha_cumprod,t,x.shape)
    
    sqrt_recip_alphas=torch.sqrt(torch.cumprod(alpha,axis=0))   
    sqrt_recip_alphas_t=extract(sqrt_recip_alphas,t,x.shape)
    model_mean= sqrt_recip_alphas_t*(x-betas_t*model(x,t)/sqrt_one_minus_alpha_cumprod_t)
    
    post_variance=(1.0-alphas_cumprod_prev)/(1.0-alphas_cumprod)*beta
    
    if t_index==0:
        return model_mean
    else:
        post_variance_t=extract(post_variance,t,x.shape)
        noise=torch.rand_like(x)
        return model_mean+torch.sqrt(post_variance_t)*noise
    
# trainloop sample
@torch.no_grad()
def p_sample_loop(model,shape,timesteps):
    device=next(iter(model.parameters()).device)
    b=shape[0]
    
    img=torch.randn(shape,device=device)
    imgs=[]
    for i in tqdm(reversed(range(0,timesteps)),desc='sampling loop time step',total=timesteps):
        img=p_sample(model,img,torch.full((b,),i,device=device,dtype=torch.long),i,timesteps)#把当前的的t扩展到(batch_size,1)的维度
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model,image_size,batch_size=16,channels=3):
    return p_sample_loop(model,shape=(batch_size,channels,image_size,image_size),timesteps=300)







# training_section

from torch.optim import Adam
from pathlib import Path
from torchvision.utils import save_image
# 按组划分批次，divisor是我们的设置的最大处理图片数，假如num是10，divisor是4，那么会返回[4,4,2]
def num_to_grpoups(num,divisor):
    groups=num//divisor
    remainder=num%divisor
    arr=[divisor]*groups
    if remainder>0:
        arr.append(remainder)
    return arr

results_folder=Path('./results')
results_folder.mkdir(exist_ok=True)
# 模型训练一千步之后才去跑一次采样生成
save_and_sample_every=200


device='cuda' if torch.cuda.is_available() else 'cpu'

model=Unet(
    dim=image_size,
    channels=channels,
    dim_mults=(1,2,4,)
)
model.to(device)

optimizer=Adam(model.parameters(),lr=1e-3)


# start trianing!


def main(timesteps):
    epochs=6
    checkpoint_path = results_folder / 'latest_checkpoint.pt'
    if checkpoint_path.exists():
        print('发现历史断点')
        checkpoint=torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        start_epoch=checkpoint['epoch']
        print(f'将从第{start_epoch}个断点开始重连')
    else:
        print('未发现断点讲从头开始连')
    for epoch in range(epochs):
        for step,batch in enumerate(tqdm(dataLoader,desc="Training Epoch")):
            optimizer.zero_grad()
            batch_size=batch['pixel_values'].shape[0]
            batch=batch['pixel_values'].to(device)
            t=torch.randint(0,timesteps,(batch_size,),device=device).long()
            loss=p_losses(model,batch,t,loss_type='huber')
            if step%10==0:
                print('loss:',loss.item())
            loss.backward()
            optimizer.step()
            
            # save_images
            if step!=0 and step%save_and_sample_every==0:
                img_nums=step//save_and_sample_every
                batches=num_to_grpoups(4,batch_size)
                all_images_list=list(map(lambda n:sample(model,batch_size=n,channels=channels,image_size=image_size),batches))
                all_images=torch.cat(all_images_list,dim=0)
                all_images=(all_images+1)*0.5#把像素值的取值范围从-1~1变成0-1
                save_image(all_images,str(results_folder/f'sample-{img_nums}.png'),nrow=6)
                checkpoint={
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }
                torch.save(checkpoint,checkpoint_path)
                tqdm.write("[✓] 模型断点及图片已保存！您可以随时 Ctrl+C 终止训练。")


if __name__=='__main__':
    main(300)