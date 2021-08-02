import torch
import torch.nn as nn
#from torch.nn import functional as F
#from batchnorm import SynchronizedBatchNorm2d

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
    
def create_upconv_single(in_channels, out_channels, size=None, inter_capacity=None, num_scale=4, kernel=3):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        , Our_Conv2d(in_ch=in_channels, out_ch=out_channels, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        , Our_Conv2d(in_ch=out_channels, out_ch=out_channels, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
        , nn.BatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        )

class Our_Unet_singlegpu_Full(nn.Module):
    def __init__(self, end):
        super().__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.conv_l1 = nn.Sequential(
            Our_Conv2d(in_ch=3, out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[0], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[4], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[4], out_ch=filters[4], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv_single(in_channels=filters[4], out_channels=filters[3], size=(32,32), inter_capacity=None, num_scale=4, kernel=5)

        self.conv_u4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[4], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv_single(in_channels=filters[3], out_channels=filters[2], size=(64,64), inter_capacity=None, num_scale=4, kernel=5)

        self.conv_u3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv_single(in_channels=filters[2], out_channels=filters[1], size=(128,128), inter_capacity=None, num_scale=4, kernel=3)

        self.conv_u2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv_single(in_channels=filters[1], out_channels=filters[0], size=(256,256), inter_capacity=None, num_scale=4, kernel=3)

        self.conv_u1 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.conv1x1_out = nn.Conv2d(filters[0], end, 1, 1, 0, bias=True)
        #self.smout = nn.Softmax2d()
        #self.smout = nn.Sigmoid()
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.conv1x1_out(output9)
        
        return out
        #return self.smout(out)
        
        #return output9

class Our_Unet_singlegpu(nn.Module):
    def __init__(self):
        super().__init__()
        
        n1 = 64
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.conv_l1 = nn.Sequential(
            Our_Conv2d(in_ch=3, out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[0], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[4], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[4], out_ch=filters[4], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv_single(in_channels=filters[4], out_channels=filters[3], size=(32,32), inter_capacity=None, num_scale=4, kernel=5)

        self.conv_u4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[4], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv_single(in_channels=filters[3], out_channels=filters[2], size=(64,64), inter_capacity=None, num_scale=4, kernel=5)

        self.conv_u3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=None, num_scale=4, kernel=5)
            , nn.BatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv_single(in_channels=filters[2], out_channels=filters[1], size=(128,128), inter_capacity=None, num_scale=4, kernel=3)

        self.conv_u2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv_single(in_channels=filters[1], out_channels=filters[0], size=(256,256), inter_capacity=None, num_scale=4, kernel=3)

        self.conv_u1 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=None, num_scale=4, kernel=3)
            , nn.BatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        #self.conv1x1_out = nn.Conv2d(filters[0], 1, 1, 1, 0, bias=True)
        #self.smout = nn.Sigmoid()
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        #out = self.conv1x1_out(output9)
        
        #return self.smout(out)
        
        return output9

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


def create_upconv_multi(in_channels, out_channels, size=None, inter_capacity=None, num_scale=4, kernel=3):
    return nn.Sequential(
        nn.Upsample(size=size, mode='nearest')
        , Our_Conv2d(in_ch=in_channels, out_ch=out_channels, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
        , SynchronizedBatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        , Our_Conv2d(in_ch=out_channels, out_ch=out_channels, inter_capacity=inter_capacity, num_scale=num_scale, kernel=kernel)
        , SynchronizedBatchNorm2d(num_features=out_channels)
        , nn.LeakyReLU(inplace=True)
        )

class Our_Unet_multigpu(nn.Module):
    def __init__(self):
        super().__init__()
        
        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]
        
        self.conv_l1 = nn.Sequential(
            Our_Conv2d(in_ch=1, out_ch=filters[0], inter_capacity=16, num_scale=2, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=16, num_scale=2, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[0], out_ch=filters[1], inter_capacity=16, num_scale=4, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=16, num_scale=4, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[2], inter_capacity=16, num_scale=8, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=16, num_scale=8, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[3], inter_capacity=16, num_scale=16, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=16, num_scale=16, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv_l5 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[4], inter_capacity=16, num_scale=32, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[4], out_ch=filters[4], inter_capacity=16, num_scale=32, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[4])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u4 = create_upconv_multi(in_channels=filters[4], out_channels=filters[3], size=(55,70), inter_capacity=16, num_scale=16, kernel=5)

        self.conv_u4 = nn.Sequential(
            Our_Conv2d(in_ch=filters[4], out_ch=filters[3], inter_capacity=16, num_scale=16, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[3], out_ch=filters[3], inter_capacity=16, num_scale=16, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[3])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u3 = create_upconv_multi(in_channels=filters[3], out_channels=filters[2], size=(110,141), inter_capacity=16, num_scale=8, kernel=5)

        self.conv_u3 = nn.Sequential(
            Our_Conv2d(in_ch=filters[3], out_ch=filters[2], inter_capacity=16, num_scale=8, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[2], out_ch=filters[2], inter_capacity=16, num_scale=8, kernel=5)
            , SynchronizedBatchNorm2d(num_features=filters[2])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u2 = create_upconv_multi(in_channels=filters[2], out_channels=filters[1], size=(221,282), inter_capacity=16, num_scale=4, kernel=3)

        self.conv_u2 = nn.Sequential(
            Our_Conv2d(in_ch=filters[2], out_ch=filters[1], inter_capacity=16, num_scale=4, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[1], out_ch=filters[1], inter_capacity=16, num_scale=4, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[1])
            , nn.LeakyReLU(inplace=True)
            )

        self.deconv_u1 = create_upconv_multi(in_channels=filters[1], out_channels=filters[0], size=(442,565), inter_capacity=16, num_scale=2, kernel=3)

        self.conv_u1 = nn.Sequential(
            Our_Conv2d(in_ch=filters[1], out_ch=filters[0], inter_capacity=16, num_scale=2, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            , Our_Conv2d(in_ch=filters[0], out_ch=filters[0], inter_capacity=16, num_scale=2, kernel=3)
            , SynchronizedBatchNorm2d(num_features=filters[0])
            , nn.LeakyReLU(inplace=True)
            )

        self.conv1x1_out = nn.Conv2d(filters[0], 1, 1, 1, 0, bias=True)
        self.smout = nn.Sigmoid()
        
    def forward(self, x):

        output1 = self.conv_l1(x)
        input2 = self.maxpool1(output1)
        
        output2 = self.conv_l2(input2)
        input3 = self.maxpool2(output2)
        
        output3 = self.conv_l3(input3)
        input4 = self.maxpool3(output3)
        
        output4 = self.conv_l4(input4)
        input5 = self.maxpool4(output4)
        
        output5 = self.conv_l5(input5)
        input6 = self.deconv_u4(output5)
        
        output6 = self.conv_u4(torch.cat((input6, output4), dim=1))
        input7 = self.deconv_u3(output6)
        
        output7 = self.conv_u3(torch.cat((input7, output3), dim=1))
        input8 = self.deconv_u2(output7)
        
        output8 = self.conv_u2(torch.cat((input8, output2), dim=1))
        input9 = self.deconv_u1(output8)
        
        output9 = self.conv_u1(torch.cat((input9, output1), dim=1))
        out = self.conv1x1_out(output9)
        
        return self.smout(out)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    a = torch.randn((1,1,442,565)).to(device)
    oper = Our_Unet_singlegpu().to(device)
    #b = oper(a)
    #print(b.shape)
    
    parameter = list(oper.parameters())

    cnt = 0
    for i in range(len(parameter)):
        cnt += parameter[i].reshape(-1).shape[0]
    
    print(cnt)