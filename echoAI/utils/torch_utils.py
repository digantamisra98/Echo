import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate(nn.Module):
    def __init__(self, kernel_size=7):
        super(AttentionGate, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True),
        )

    def forward(self, x):
        x_compress = torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )
        x_out = self.conv_bn(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        bam=False,
        num_layers=1,
        bn=False,
    ):
        super(ChannelGate, self).__init__()
        self.bam = bam
        self.gate_c = nn.Sequential()
        self.gate_c.add_module("flatten", Flatten())
        gate_channels = [gate_channel]
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        gate_channels += [gate_channel]
        for i in range(len(gate_channels) - 2):
            self.gate_c.add_module(
                "gate_c_fc_%d" % i, nn.Linear(gate_channels[i], gate_channels[i + 1])
            )
            if bn is True:
                self.gate_c.add_module(
                    "gate_c_bn_%d" % (i + 1), nn.BatchNorm1d(gate_channels[i + 1])
                )
            self.gate_c.add_module("gate_c_relu_%d" % (i + 1), nn.ReLU())
        self.gate_c.add_module(
            "gate_c_fc_final", nn.Linear(gate_channels[-2], gate_channels[-1])
        )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == "avg":
                avg_pool = F.avg_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.gate_c(avg_pool)
            elif pool_type == "max":
                max_pool = F.max_pool2d(
                    x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.gate_c(max_pool)
            elif pool_type == "lp":
                lp_pool = F.lp_pool2d(
                    x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))
                )
                channel_att_raw = self.gate_c(lp_pool)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        if self.bam:
            return channel_att_sum.unsqueeze(2).unsqueeze(3).expand_as(x)
        else:
            scale = F.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
            return x * scale
