import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from Networks.swin_channel import SwinChannel

class ECAModule(nn.Module):
    # Constructs a Dynamic channel attention.
    def __init__(self):
        super(ECAModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=3, padding=(3 - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x
        attn = self.avg_pool(y)
        attn = self.conv(attn.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(attn)

        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, stride=1, bias=True),
        )

    def forward(self, x):
        return self.main(x) + x

class MResModule(nn.Module):
    # Constructs a Residual learning module.
    def __init__(self, channel, num_res=4):
        super(MResModule, self).__init__()

        layers = [ResBlock(channel, channel) for _ in range(num_res)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

#########################################################
# ConvBlock
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True, isuseBN=False):
        super(ConvBlock, self).__init__()

        self.isuseBN = isuseBN
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)
        if self.isuseBN:
            self.bn = nn.BatchNorm2d(output_size)
        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)
        if self.isuseBN:
            out = self.bn(out)
        out = self.act(out)
        return out

class PEM(nn.Module):
    # Constructs a Progressive expansion module.
    def __init__(self, out_plane):#64 128
        super(PEM, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, out_plane//8, kernel_size=3, padding=1, stride=1, bias=True),#3,8
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane//8, out_plane // 4, kernel_size=1, padding=0, stride=1, bias=True),#8,16
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane // 4, out_plane // 4, kernel_size=3, padding=1, stride=1, bias=True),#16,16
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane // 4, out_plane // 2, kernel_size=1, padding=0, stride=1, bias=True),  #16,32
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane // 2, out_plane // 2, kernel_size=3, padding=1, stride=1, bias=True),  #32,32
            nn.ReLU(inplace=True),
            nn.Conv2d(out_plane // 2, out_plane-3, kernel_size=1, padding=0, stride=1, bias=True),  #32,61
            nn.ReLU(inplace=True)
        )

        self.conv = nn.Conv2d(out_plane, out_plane, kernel_size=1, padding=0, stride=1, bias=True)#64,64

    def forward(self, x):
        x = torch.cat([x, self.main(x)], dim=1)
        return self.conv(x)


class FSM(nn.Module):
    # Constructs a Feature supplement module.
    def __init__(self, channel):
        super(FSM, self).__init__()
        self.merge = nn.Conv2d(channel, channel, kernel_size=3, padding=1, stride=1, bias=True)

    def forward(self, x1, x2):
        x = x1 * x2
        out = x1 + self.merge(x)
        return out

class SCAUNet(nn.Module):
    def __init__(self):
        super(SCAUNet, self).__init__()
        base_channel = 32

        self.PEM256 = PEM(base_channel)
        self.PEM128 = PEM(base_channel * 2)
        self.PEM64 = PEM(base_channel * 4)

        self.conv_r1_32 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_r1_64 = nn.Sequential(
            nn.Conv2d(base_channel, base_channel * 2, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True))
        self.conv_r1_128 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel * 4, kernel_size=3, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True))

        self.fsm1 = FSM(base_channel * 2)
        self.fsm2 = FSM(base_channel * 4)

        self.eca1 = ECAModule()
        self.eca2 = ECAModule()
        self.eca3 = ECAModule()

        self.sc = SwinChannel(base_channel)

        #*****************************
        self.convtran_r2_64 = nn.Sequential(
            nn.ConvTranspose2d(base_channel * 4, base_channel * 2, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True))
        self.convtran_r2_32 = nn.Sequential(
            nn.ConvTranspose2d(base_channel * 2, base_channel, kernel_size=4, padding=1, stride=2, bias=True),
            nn.ReLU(inplace=True))

        self.mrm = nn.ModuleList([
            MResModule(base_channel),
            MResModule(base_channel * 2),
            MResModule(base_channel * 4),

            MResModule(base_channel * 4),
            MResModule(base_channel * 2),
            MResModule(base_channel)
        ])

        self.conv_r2_64 = nn.Sequential(
            nn.Conv2d(base_channel * 4, base_channel * 2, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True))
        self.conv_r2_32 = nn.Sequential(
            nn.Conv2d(base_channel * 2, base_channel, kernel_size=3, padding=1, stride=1, bias=True),
            nn.ReLU(inplace=True))

        self.out1 = nn.Conv2d(base_channel * 4, 3, kernel_size=3, padding=1, stride=1, bias=True)
        self.out2 = nn.Conv2d(base_channel * 2, 3, kernel_size=3, padding=1, stride=1, bias=True)
        self.out3 = nn.Conv2d(base_channel, 3, kernel_size=3, padding=1, stride=1, bias=True)


    def forward(self, x):
        x_2 = F.interpolate(x, scale_factor=0.5)
        x_4 = F.interpolate(x_2, scale_factor=0.5)

        # ************1************
        z1 = self.PEM256(x)
        z2 = self.PEM128(x_2)
        z4 = self.PEM64(x_4)

        #256
        z1 = self.conv_r1_32(z1)#32*256*256
        res1 = self.mrm[0](z1)  # 32*256*256
        a1 = self.eca1(res1)#32*256*256

        #128
        z = self.conv_r1_64(a1)#64*128*128
        z = self.fsm1(z, z2)#64*128*128
        res2 = self.mrm[1](z)  # 64*128*128
        a2 = self.eca2(res2)#64*128*128

        #64
        z = self.conv_r1_128(a2)#128*64*64
        z = self.fsm2(z, z4)#128*64*64
        res3 = self.mrm[2](z)  #128*64*64
        a3 = self.eca3(res3)#128*64*64

        sc1, sc2, sc3 = self.sc(a1, a2, a3)

        #************2************
        outputs = list()
        #64
        z = self.mrm[3](sc3)#128*64*64
        z_ = self.out1(z)#3*64*64
        outputs.append(z_+x_4)

        # 128
        z = self.convtran_r2_64(z)#64*128*128
        z = self.conv_r2_64(torch.cat([z, sc2], dim=1))#128->64*128*128

        z = self.mrm[4](z)#64*128*128
        z_ = self.out2(z)#3*128*128
        outputs.append(z_+x_2)

        # 256
        z = self.convtran_r2_32(z)#32*256*256
        z = self.conv_r2_32(torch.cat([z, sc1], dim=1))#64->32*256*256

        z = self.mrm[5](z)#32*256*256
        z = self.out3(z)#3*256*256
        outputs.append(z+x)

        return outputs

