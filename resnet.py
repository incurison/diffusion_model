import torch 
import torch.nn as nn
class basic_resnet(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,stride=stride,kernel_size=3,bias=False,padding=1)#这里我们设置bias=False是不用加偏置项了，因为后面的残差连接其实就已经有偏置项了
        self.relu1=nn.ReLU()
        self.conv2=nn.Conv2d(in_channels=out_channels,out_channels=out_channels,stride=1,padding=1,kernel_size=3,bias=False)
        self.relu2=nn.ReLU()
        self.norm=nn.BatchNorm2d(out_channels)
        self.cutout=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.cutout=nn.Sequential([
                nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1,padding=1,stride=stride,bias=False),
                nn.BatchNorm2d(out_channels)
            ])
            
    def forward(self,x):
        identity=self.cutout(x)
        out=self.conv1(x)
        out=self.relu1(out)
        out=self.conv2(out)
        out=self.relu2(out)
        out=self.norm(out)
        out=out+identity
        final_out=nn.ReLU(out)
        return final_out


# unet中的resnet堆叠

