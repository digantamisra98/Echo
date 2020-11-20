
import torch.nn as nn
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


class CBAM(nn.Module):
    def __init__(self, gate_channels, kernel_size=3, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, bam=False):
        super(CBAM, self).__init__()
        self.ChannelGate = torch_utils.ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = torch_utils.AttentionGate(kernel_size)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out


class SE(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(SE, self).__init__()
        self.ChannelGate = torch_utils.ChannelGate(gate_channels, reduction_ratio, ['avg'])
    def forward(self, x):
        x_out = self.ChannelGate(x)
        return x_out