import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class style_mixer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, 3,1,1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )
        
        self.fc = nn.Sequential(
            Norm_Linear(64,64, bias=False),
            nn.LeakyReLU()
            )
        
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

class Norm_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.bias = None

        self.scale = (1 / math.sqrt(in_dim))

    def forward(self, input):
        out = F.linear(input, self.weight*self.scale, bias=self.bias)

        return out

class Feature(nn.Module):
    def __init__(self, in_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.fc1 = nn.Sequential(
            Norm_Linear(64, in_channel),
            nn.LeakyReLU()
            )
        self.fc2 = nn.Sequential(
            Norm_Linear(64, in_channel),
            nn.LeakyReLU()
            )
        
    def forward(self, x, y):
        
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
            
        std_feat = (torch.std(x, dim = 2) + self.eps).view(B,C,1)
        mean_feat = torch.mean(x, dim = 2).view(B,C,1)
        
        mean = self.fc1(y).unsqueeze(2)
        std = self.fc2(y).unsqueeze(2)
        
        new_feature = std * (x - mean_feat)/std_feat + mean
        new_feature = new_feature.view(B,C,H,W)
        
        return new_feature

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU()
        
        self.style1 = Feature(in_channel)
        self.style2 = Feature(out_channel)
        
        self.conv1 = nn.Conv2d(in_channel, in_channel, 3,1,1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 3,1,1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channel, out_channel, 1,1,0, bias=False)
        
    def forward(self, x, z):
        
        out = self.upsample(x)
        res = self.conv3(out)
        
        out = self.relu(out)
        out = self.conv1(out)
        out = self.style1(out, z)
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.style2(out, z)
                
        out = out + res
        
        return out

class RGBBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style = Feature(3)
        self.conv = nn.Conv2d(in_channel, 3, 3,1,1, bias=True)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False)
        
    def forward(self, x, z, prev_rgb=None):
        
        out = self.relu(x)
        out = self.conv(out)
        out = self.style(out, z)       

        if prev_rgb is not None:
            prev_rgb = self.upsample(prev_rgb)
            
            out = out + prev_rgb
        
        return out

class generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.root = nn.Parameter(torch.randn(1, 512, 4, 4))
        
        self.up1 = UpBlock(512,256) # (512,4,4) -> (256,8,8)
        self.up2 = UpBlock(256,128) # (256,8,8) -> (128,16,16)
        self.up3 = UpBlock(128,64) # (128,16,16) -> (64,32,32)
        self.up4 = UpBlock(64,32) # (64,32,32) -> (32,64,64)
        self.up5 = UpBlock(32,16) # (32,64,64) -> (16,128,128)
        self.up6 = UpBlock(16,8) # (16,128,128) -> (8,256,256)
        
        self.rgb1 = RGBBlock(256)
        self.rgb2 = RGBBlock(128)
        self.rgb3 = RGBBlock(64)
        self.rgb4 = RGBBlock(32)
        self.rgb5 = RGBBlock(16)
        self.rgb6 = RGBBlock(8)
        
    def forward(self, mixed_style):
        
        out = self.root.expand((mixed_style.shape[0],512,4,4))
        
        out = self.up1(out, mixed_style)
        rgb = self.rgb1(out, mixed_style)
        
        out = self.up2(out, mixed_style)
        rgb = self.rgb2(out, mixed_style, rgb)
        
        out = self.up3(out, mixed_style)
        rgb = self.rgb3(out, mixed_style, rgb)
        
        out = self.up4(out, mixed_style)
        rgb = self.rgb4(out, mixed_style, rgb)
        
        out = self.up5(out, mixed_style)
        rgb = self.rgb5(out, mixed_style, rgb)
        
        out = self.up6(out, mixed_style)
        rgb = self.rgb6(out, mixed_style, rgb)
        
        return rgb


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        
        self.downsample = nn.AvgPool2d(2)
        self.relu = nn.LeakyReLU()
        
        self.norm1 = nn.InstanceNorm2d(in_channel)
        self.norm2 = nn.InstanceNorm2d(out_channel)
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3,1,1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3,1,1, bias=False)
        
        self.conv3 = nn.Conv2d(in_channel, out_channel, 1,1,0, bias=False)
        
    def forward(self, x):
        
        res = self.conv3(x)
        res = self.downsample(res)
        
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        out = self.downsample(out)
        
        out = out + res
        
        return out

class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        mid_channel = in_channel // 2
        
        self.conv1 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3 = nn.Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv4 = nn.Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=False)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        query = self.conv1(x)
        key = self.conv2(x)
        gate = self.softmax(query * key)
        
        value = self.conv3(x)
        out = gate * value
        
        out = self.conv4(out)
        
        out = out + x
        
        return out

class discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.down1 = DownBlock(6,16) # (3,256,256) -> (16,128,128)
        self.down2 = DownBlock(16,32) # (16,128,128) -> (32,64,64)
        self.down3 = DownBlock(32,64) # (32,64,64) -> (64,32,32)
        self.down4 = DownBlock(64,128) # (64,32,32) -> (128,16,16)
        self.down5 = DownBlock(128,256) # (128,16,16) -> (256,8,8)
        self.down6 = DownBlock(256,512) # (256,8,8) -> (512,4,4)
        
        self.attention1 = nn.Sequential(
            Attention(512,512),
            nn.AvgPool2d(2)
            )
        self.attention2 = nn.Sequential(
            Attention(512,512),
            nn.AvgPool2d(2)
            )
        
        self.fc = nn.Sequential(
            Norm_Linear(512, 256, bias=False),
            nn.LeakyReLU(),
            Norm_Linear(256, 1, bias=False)
            )
        
    def forward(self, x, y):
        out = torch.cat((x,y),dim=1)
        out = self.down1(out)
        out = self.down2(out)
        out = self.down3(out)
        out = self.down4(out)
        out = self.down5(out)
        out = self.down6(out)
        
        out = self.attention1(out)
        out = self.attention2(out)
        
        out = torch.flatten(out, 1)
        out = self.fc(out)
        
        return out

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,3,256,256)).to(device)
    oper1 = style_mixer().to(device)
    b = oper1(a)
    print(b.shape)
    
    parameter1 = list(oper1.parameters())
    cnt = 0
    for i in range(len(parameter1)):
        cnt += parameter1[i].reshape(-1).shape[0]
    
    print(cnt)
    
    oper2 = generator().to(device)
    c = oper2(b)
    print(c.shape)
    
    parameter2 = list(oper2.parameters())
    cnt = 0
    for i in range(len(parameter2)):
        cnt += parameter2[i].reshape(-1).shape[0]
    
    print(cnt)
    
    oper3 = discriminator().to(device)
    d = oper3(c, a)
    print(d.shape)
    
    parameter3 = list(oper3.parameters())
    cnt = 0
    for i in range(len(parameter3)):
        cnt += parameter3[i].reshape(-1).shape[0]
    
    print(cnt)
    
    