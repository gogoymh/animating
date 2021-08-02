import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ECA_Layer(nn.Module):
    def __init__(self, channels, kernel=3):
        super(ECA_Layer, self).__init__()

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
    def __init__(self, in_ch, out_ch, kernel, stride, padding):
        super().__init__()

        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        
    def forward(self, x, y): # x is input, y is cumulative concatenation
        out = self.depthwise(x)
        out = self.pointwise(out)

        cat = torch.cat((out, y), dim=1)
        return out, cat

class Our_Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, inter_capacity=None, num_scale=4, kernel=3, stride=1):
        super().__init__()
        if inter_capacity is None:
            inter_capacity = out_ch//num_scale
        
        self.init = nn.Conv2d(in_ch, inter_capacity, 1, 1, 0, bias=False)
        
        self.scale_module = nn.ModuleList()
        for _ in range(num_scale-1):
            self.scale_module.append(DSC2d(inter_capacity, inter_capacity, 3, 1, 1))
        
        self.channel_attention = ECA_Layer(num_scale*inter_capacity, kernel)

    def forward(self, x):
        out = self.init(x)

        cat = out
        for operation in self.scale_module:
            out, cat = operation(out, cat)
        
        cat = self.channel_attention(cat)
        
        return cat

class DoubleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, inter_capacity, num_scale, kernel):
        super().__init__()
        
        self.nn = nn.Sequential(
            Our_Conv2d(in_channel, out_channel, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
            , nn.InstanceNorm2d(out_channel)
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(out_channel, out_channel, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
            , nn.InstanceNorm2d(out_channel)
            , nn.LeakyReLU()
            )
        
    def forward(self, x):
        out = self.nn(x)
        return out

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
    def __init__(self, in_channel, out_channel, eps = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.fc1 = nn.Sequential(
            Equal_Linear(in_channel, out_channel),
            nn.LeakyReLU()
            )
        self.fc2 = nn.Sequential(
            Equal_Linear(in_channel, out_channel),
            nn.LeakyReLU()
            )
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x, y):
        
        B,C,H,W = x.shape
        x = x.view(B,C,-1)
        y = self.pool(y)
        y = torch.flatten(y,1)
        
        std_feat = (torch.std(x, dim = 2) + self.eps).view(B,C,1)
        mean_feat = torch.mean(x, dim = 2).view(B,C,1)
        
        mean = self.fc1(y).unsqueeze(2)
        std = self.fc2(y).unsqueeze(2)
        
        new_feature = std * (x - mean_feat)/std_feat + mean
        new_feature = new_feature.view(B,C,H,W)
        
        return new_feature

class RGBBlock(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        
        self.relu = nn.LeakyReLU(inplace=True)
        
        self.style = Feature(in_channel, 3)
        
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            Equal_Conv2d(in_channel, 3, 3,1,0, bias=True)
            )

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
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16] # 64, 128, 256, 512, 1024
        
        self.nn1 = DoubleBlock(3, filters[0], None, 4, 3)
        self.down1 = nn.MaxPool2d(2)
        
        self.nn2 = DoubleBlock(filters[0], filters[1], None, 4, 3)
        self.down2 = nn.MaxPool2d(2)
        
        self.nn3 = DoubleBlock(filters[1], filters[2], None, 4, 5)
        self.down3 = nn.MaxPool2d(2)
        
        self.nn4 = DoubleBlock(filters[2], filters[3], None, 4, 5)
        self.down4 = nn.MaxPool2d(2)
        
        self.nn5 = DoubleBlock(filters[3], filters[4], None, 4, 5)
        self.rgb5 = RGBBlock(filters[4])
        
        self.up5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DoubleBlock(filters[4], filters[3], None, 4, 5)
            )
        self.nn6 = DoubleBlock(filters[4], filters[3], None, 4, 5)
        self.rgb6 = RGBBlock(filters[3])
        
        self.up6 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DoubleBlock(filters[3], filters[2], None, 4, 5)
            )
        self.nn7 = DoubleBlock(filters[3], filters[2], None, 4, 5)
        self.rgb7 = RGBBlock(filters[2])
        
        self.up7 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DoubleBlock(filters[2], filters[1], None, 2, 3)
            )
        self.nn8 = DoubleBlock(filters[2], filters[1], None, 4, 3)
        self.rgb8 = RGBBlock(filters[1])
        
        self.up8 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            DoubleBlock(filters[1], filters[0], None, 2, 3)
            )
        self.nn9 = DoubleBlock(filters[1], filters[0], None, 4, 3)
        self.rgb9 = RGBBlock(filters[0])
        
        
    def forward(self, joint):
        output1 = self.nn1(joint)
        input2 = self.down1(output1)
        
        output2 = self.nn2(input2)
        input3 = self.down2(output2)
        
        output3 = self.nn3(input3)
        input4 = self.down3(output3)
        
        output4 = self.nn4(input4)
        input5 = self.down4(output4)
        
        output5 = self.nn5(input5)
        input6 = self.up5(output5)
        rgb = self.rgb5(output5, output5)
        
        output6 = self.nn6(torch.cat((input6, output4), dim=1))
        input7 = self.up6(output6)
        rgb = self.rgb6(output6, output4, rgb)
        
        output7 = self.nn7(torch.cat((input7, output3), dim=1))
        input8 = self.up7(output7)
        rgb = self.rgb7(output7, output3, rgb)
        
        output8 = self.nn8(torch.cat((input8, output2), dim=1))
        input9 = self.up8(output8)
        rgb = self.rgb8(output8, output2, rgb)
        
        output9 = self.nn9(torch.cat((input9, output1), dim=1))
        rgb = self.rgb9(output9, output1, rgb)
        
        return rgb

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

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,3,256,256)).to(device)
    
    oper2 = generator().to(device)
    c = oper2(a)
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
    
    