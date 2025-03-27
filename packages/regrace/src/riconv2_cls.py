"""
Author: Zhiyuan Zhang
Date: Dec 2021
Email: cszyzhang@gmail.com
Website: https://wwww.zhiyuanzhang.net
"""

import torch

from ..utils.riconv2_utils import RIConv2SetAbstraction


class RIConvClassification(torch.nn.Module):

    def __init__(self, n: int, normal_channel: bool = True):
        super().__init__()
        in_channel = 64
        self.normal_channel = normal_channel

        self.sa0 = RIConv2SetAbstraction(
            npoint=512 * n,
            radius=0.0,  # not used
            nsample=8,
            in_channel=0 + in_channel,
            mlp=[32],
            group_all=False)
        self.sa1 = RIConv2SetAbstraction(
            npoint=256 * n,
            radius=0.0,  # not used
            nsample=16,
            in_channel=32 + in_channel,
            mlp=[64],
            group_all=False)
        self.sa2 = RIConv2SetAbstraction(
            npoint=128 * n,
            radius=0.0,  # not used
            nsample=32,
            in_channel=64 + in_channel,
            mlp=[128],
            group_all=True)

    def forward(self, xyz) -> torch.Tensor:
        # get the batch size
        B, _, _ = xyz.shape
        if self.normal_channel:
            # use the normal channel
            norm = xyz[:, :, 3:]
            xyz = xyz[:, :, :3]
        else:
            # compute the LRA and use as normal
            norm = None

        l0_xyz, l0_norm, l0_points = self.sa0(xyz, norm, None)
        l1_xyz, l1_norm, l1_points = self.sa1(l0_xyz, l0_norm, l0_points)
        _, _, l2_points = self.sa2(l1_xyz, l1_norm, l1_points)
        x = l2_points.view(B, 128)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, RIConv2SetAbstraction):
                m.init_weights()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
