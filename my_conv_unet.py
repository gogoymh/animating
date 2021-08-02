import torch
import torch.nn as nn
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

def double_conv_bn(in_ch, out_ch):
    return nn.Sequential(
        my_Conv2d(in_ch, out_ch)
        , nn.BatchNorm2d(out_ch)
        , nn.LeakyReLU(inplace=True)
        , my_Conv2d(out_ch, out_ch)
        , nn.BatchNorm2d(out_ch)
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

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, size):
        super().__init__()
        
        #self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.up = nn.Sequential(
            nn.Upsample(size=size, mode='nearest'),
            double_conv_bn(in_ch, out_ch)
            )
        
        self.conv = double_conv_bn(in_ch, out_ch)
        
    def forward(self, x, y):
        out = self.up(x)
        out = torch.cat((out, y), dim=1)
        return self.conv(out)
        
class my_unet(nn.Module):
    def __init__(self, in_feat, init_ch, end_class):
        super().__init__()
        
        n1 = init_ch
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.layer1 = double_conv_bn(in_feat, filters[0])
        self.layer2 = Down(filters[0], filters[1])
        self.layer3 = Down(filters[1], filters[2])
        self.layer4 = Down(filters[2], filters[3])
        self.layer5 = Down(filters[3], filters[4])
        self.layer6 = Up(filters[4], filters[3], (22,40))
        self.layer7 = Up(filters[3], filters[2], (45,80))
        self.layer8 = Up(filters[2], filters[1], (90,160))
        self.layer9 = Up(filters[1], filters[0], (180,320))
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5, out4)
        out = self.layer7(out, out3)
        out = self.layer8(out, out2)
        out = self.layer9(out, out1)
        
        return out

class Unet_head(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 32)
            )
        
    def forward(self, x):
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

class Unet_last(nn.Module):
    def __init__(self, num_class):
        super().__init__()
        
        self.conv1x1_out = nn.Conv2d(64, num_class, 1, 1, 0, bias=True)
        #self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        x = self.conv1x1_out(x)
        #x = self.softmax(x)
        return x

class Unet_softmax(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.softmax = nn.Softmax2d()
    
    def forward(self, x):
        x = self.softmax(x)
        return x

class my_unet_full(nn.Module):
    def __init__(self, in_feat, init_ch, end_class):
        super().__init__()
        
        n1 = init_ch
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.layer1 = double_conv_bn(in_feat, filters[0])
        self.layer2 = Down(filters[0], filters[1])
        self.layer3 = Down(filters[1], filters[2])
        self.layer4 = Down(filters[2], filters[3])
        self.layer5 = Down(filters[3], filters[4])
        self.layer6 = Up(filters[4], filters[3], (55,70))
        self.layer7 = Up(filters[3], filters[2], (110,141))
        self.layer8 = Up(filters[2], filters[1], (221,282))
        self.layer9 = Up(filters[1], filters[0], (442,565))
        
        self.conv1x1_out = nn.Conv2d(filters[0], end_class, 1, 1, 0, bias=True)
        
    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out = self.layer6(out5, out4)
        out = self.layer7(out, out3)
        out = self.layer8(out, out2)
        out = self.layer9(out, out1)
        
        out = self.conv1x1_out(out)
        
        return out

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,3,180,320)).to(device)
    oper = my_unet(3,64,20).to(device)
    #b = oper(a)
    for i in range(1000):
        b = oper(a)
    #print(b.shape)
    
    parameter = list(oper.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)























