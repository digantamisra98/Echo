import torch.nn as nn
import math

from echoAI.utils import torch_utils


class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False, kernel_size=7):
        super(TripletAttention, self).__init__()
        self.cw = torch_utils.AttentionGate(kernel_size)
        self.hc = torch_utils.AttentionGate(kernel_size)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.hw = torch_utils.AttentionGate(kernel_size)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1 / 3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1 / 2 * (x_out11 + x_out21)
        return x_out


class SpatialGate(nn.Module):
    def __init__(
        self, gate_channel, reduction_ratio=16, dilation_conv_num=2, dilation_val=4
    ):
        super(SpatialGate, self).__init__()
        self.gate_s = nn.Sequential()
        self.gate_s.add_module(
            "gate_s_conv_reduce0",
            nn.Conv2d(gate_channel, gate_channel // reduction_ratio, kernel_size=1),
        )
        self.gate_s.add_module(
            "gate_s_bn_reduce0", nn.BatchNorm2d(gate_channel // reduction_ratio)
        )
        self.gate_s.add_module("gate_s_relu_reduce0", nn.ReLU())
        for i in range(dilation_conv_num):
            self.gate_s.add_module(
                "gate_s_conv_di_%d" % i,
                nn.Conv2d(
                    gate_channel // reduction_ratio,
                    gate_channel // reduction_ratio,
                    kernel_size=3,
                    padding=dilation_val,
                    dilation=dilation_val,
                ),
            )
            self.gate_s.add_module(
                "gate_s_bn_di_%d" % i, nn.BatchNorm2d(gate_channel // reduction_ratio)
            )
            self.gate_s.add_module("gate_s_relu_di_%d" % i, nn.ReLU())
        self.gate_s.add_module(
            "gate_s_conv_final",
            nn.Conv2d(gate_channel // reduction_ratio, 1, kernel_size=1),
        )

    def forward(self, in_tensor):
        return self.gate_s(in_tensor).expand_as(in_tensor)


class CBAM(nn.Module):
    def __init__(
        self,
        gate_channels,
        kernel_size=3,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
        bam=False,
        num_layers=1,
        bn=False,
        dilation_conv_num=2,
        dilation_val=4,
    ):
        super(CBAM, self).__init__()
        self.bam = bam
        self.no_spatial = no_spatial
        if self.bam:
            self.dilatedGate = SpatialGate(
                gate_channels, reduction_ratio, dilation_conv_num, dilation_val
            )
            self.ChannelGate = torch_utils.ChannelGate(
                gate_channels,
                reduction_ratio,
                pool_types,
                bam=self.bam,
                num_layers=num_layers,
                bn=bn,
            )
        else:
            self.ChannelGate = torch_utils.ChannelGate(
                gate_channels, reduction_ratio, pool_types
            )
            if not no_spatial:
                self.SpatialGate = torch_utils.AttentionGate(kernel_size)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.bam:
            if not self.no_spatial:
                x_out = self.SpatialGate(x_out)
            return x_out
        else:
            att = 1 + F.sigmoid(self.ChannelGate(x) * self.dilatedGate(x))
            return att * x


class SE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SE, self).__init__()
        self.ChannelGate = torch_utils.ChannelGate(
            gate_channels, reduction_ratio, ["avg"]
        )

    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out



class ECA(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channels = channels
        self.b = b
        self.gamma = gamma
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size() - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def kernel_size(self):
        k = int(abs((math.log2(self.channels)/self.gamma)+ self.b/self.gamma))
        out = k if k % 2 else k+1
        return out

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)
