import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA_Layer(nn.Module):
    def __init__(self, channels):
        super(ECA_Layer, self).__init__()
        
        kernel = math.ceil((math.log(channels,2)/2 + 0.5))
        if kernel % 2 == 0:
            kernel -= 1
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel, padding=(kernel-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        
        return x*y.expand_as(x)

class DSC2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, padding, bias=False):
        super().__init__()

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=bias)
        
    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)

        return out

class my_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        inter_channel = out_ch//4
        
        self.conv1x1 = nn.Conv2d(in_ch, inter_channel, 1, 1, 0, bias=False)
        self.conv3x3 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        self.conv5x5 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        self.conv7x7 = DSC2d(inter_channel, inter_channel, 3, 1, 1, bias=False)
        
        self.channel_attention = ECA_Layer(out_ch)

    def forward(self, x):
        out1 = self.conv1x1(x)
        out3 = self.conv3x3(out1)
        out5 = self.conv5x5(out3)
        out7 = self.conv7x7(out5)
        
        cat = torch.cat((out1,out3,out5,out7), dim=1)
        cat = self.channel_attention(cat)
        
        return cat

class Equal_Conv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))
        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(input,self.weight * self.scale,bias=self.bias,stride=self.stride,padding=self.padding)
        return out

class Equal_Linear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True):
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
    def __init__(self, out_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.fc1 = nn.Sequential(
            Equal_Linear(64, out_channel),
            nn.LeakyReLU()
            )
        self.fc2 = nn.Sequential(
            Equal_Linear(64, out_channel),
            nn.LeakyReLU()
            )
        
        #self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, y):
        
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        #y = self.pool(y)
        #y = torch.flatten(y,1)
        
        std_feat = (torch.std(x, dim = 2) + self.eps).view(B,C,1)
        mean_feat = torch.mean(x, dim = 2).view(B,C,1)
        
        mean = self.fc1(y).unsqueeze(2)
        std = self.fc2(y).unsqueeze(2)
        
        new_feature = std * (x - mean_feat)/std_feat + mean
        new_feature = new_feature.view(B,C,H,W)
        
        return new_feature



def double_conv_bn(in_ch, out_ch):
    return nn.Sequential(
        my_Conv2d(in_ch, out_ch)
        , nn.LeakyReLU(inplace=True)
        , my_Conv2d(out_ch, out_ch)
        , nn.LeakyReLU(inplace=True)
        )

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv_bn(in_ch, out_ch)
            )
        
    def forward(self, x):
        return self.down(x)

class styler(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.style = nn.Sequential(
            double_conv_bn(3,16),
            Down(16,32),
            Down(32,64),
            Down(64,128),
            Down(128,256),
            Down(256,512)
            )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.vectorize = nn.Sequential(
            nn.Linear(512,128),
            nn.LeakyReLU(inplace=False),
            nn.Linear(128,64)
            )
        
    def forward(self, joint):
        style = self.style(joint)
        style = self.pool(style)
        style = torch.flatten(style,1)
        style = self.vectorize(style)
        return style

class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, size):
        super().__init__()
        
        self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style1 = Feature(in_channel)
        self.style2 = Feature(out_channel)
        
        self.conv1 = my_Conv2d(in_channel, in_channel)
        self.conv2 = my_Conv2d(in_channel, out_channel)
        
    def forward(self, x, z):
        
        out = self.upsample(x)
        
        out = self.relu(out)
        out = self.conv1(out)
        out = self.style1(out, z)
        
        out = self.relu(out)
        out = self.conv2(out)
        out = self.style2(out, z)
        
        return out

class RGBBlock(nn.Module):
    def __init__(self, in_channel, size=None):
        super().__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style = Feature(3)
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            Equal_Conv2d(in_channel, 3, 3,1,0)
            )
        
        if size is not None:
            self.upsample = nn.Upsample(size=size, mode='bilinear', align_corners=False)
        
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
        
        self.root = nn.Parameter(torch.randn(1, 512, 5, 10))
        
        self.up1 = UpBlock(512,256,(11,20))
        self.up2 = UpBlock(256,128,(22,40))
        self.up3 = UpBlock(128,64, (45,80))
        self.up4 = UpBlock(64,32,  (90,160))
        self.up5 = UpBlock(32,16,  (180,320))
        
        self.rgb1 = RGBBlock(256)
        self.rgb2 = RGBBlock(128, (22,40))
        self.rgb3 = RGBBlock(64, (45,80))
        self.rgb4 = RGBBlock(32,  (90,160))
        self.rgb5 = RGBBlock(16,  (180,320))
        
    def forward(self, mix_style):
        out = self.root.expand((mix_style.shape[0],512,5,10))
        
        out = self.up1(out, mix_style)
        rgb = self.rgb1(out, mix_style)
        
        out = self.up2(out, mix_style)
        rgb = self.rgb2(out, mix_style, rgb)
        
        out = self.up3(out, mix_style)
        rgb = self.rgb3(out, mix_style, rgb)
        
        out = self.up4(out, mix_style)
        rgb = self.rgb4(out, mix_style, rgb)
        
        out = self.up5(out, mix_style)
        rgb = self.rgb5(out, mix_style, rgb)
        
        return rgb
'''
class Attention(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        mid_channel = in_channel // 2
        
        self.conv1 = Equal_Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv2 = Equal_Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv3 = Equal_Conv2d(in_channel, mid_channel, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv4 = Equal_Conv2d(mid_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True)
        
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
        
        self.down1 = nn.Sequential(
            DoubleBlock(6, 16, None, 4, 3),
            nn.AvgPool2d(2)
            ) # (3,256,256) -> (16,128,128)
        self.down2 = nn.Sequential(
            DoubleBlock(16, 32, None, 4, 3),
            nn.AvgPool2d(2)
            ) # (16,128,128) -> (32,64,64)
        self.down3 = nn.Sequential(
            DoubleBlock(32, 64, None, 4, 3),
            nn.AvgPool2d(2)
            ) # (32,64,64) -> (64,32,32)
        self.down4 = nn.Sequential(
            DoubleBlock(64, 128, None, 4, 3),
            nn.AvgPool2d(2)
            ) # (64,32,32) -> (128,16,16)
        self.down5 = nn.Sequential(
            DoubleBlock(128, 256, None, 4, 5),
            nn.AvgPool2d(2)
            ) # (128,16,16) -> (256,8,8)
        self.down6 = nn.Sequential(
            DoubleBlock(256, 512, None, 4, 5),
            nn.AvgPool2d(2)
            ) # (256,8,8) -> (512,4,4)
        self.attention1 = nn.Sequential(
            Attention(512,512),
            nn.AvgPool2d(2)
            )
        self.attention2 = nn.Sequential(
            Attention(512,512),
            nn.AvgPool2d(2)
            )
        self.fc = nn.Sequential(
            Equal_Linear(512, 256, bias=True),
            nn.LeakyReLU(),
            Equal_Linear(256, 1, bias=True)
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
'''
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,3,180,320)).to(device)
    
    oper = styler().to(device)
    b = oper(a)
    print(b.shape)
    
    parameter = list(oper.parameters())
    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    
    oper2 = generator().to(device)
    c = oper2(b)
    print(c.shape)
    
    parameter = list(oper2.parameters())
    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)
    

    
    