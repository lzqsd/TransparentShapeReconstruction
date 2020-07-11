import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from model.pointnet_util import PointNetSetAbstraction,PointNetFeaturePropagation


class PointNetRefinePoint(nn.Module):
    def __init__(self ):
        super(PointNetRefinePoint, self).__init__()
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 64, 6 + 6, [64, 64], [4, 4] )
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 6, [128, 128], [8, 8] )
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32,  128 + 6, [256, 256], [16, 16] )
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 6, [512, 512], [32, 32] )

        self.fp4 = PointNetFeaturePropagation(777, [512, 256], [32, 16] )
        self.fp3 = PointNetFeaturePropagation(393, [256, 128], [16, 8] )
        self.fp2 = PointNetFeaturePropagation(201, [128, 64], [8, 4] )
        self.fp1 = PointNetFeaturePropagation(79, [64], [4] )
        self.conv = nn.Conv1d(in_channels=64, out_channels=6, kernel_size=1 )

    def forward(self, xyz, nxyz, points):
        l1_xyz, l1_nxyz, l1_points = self.sa1(   xyz,    nxyz,    points )
        l2_xyz, l2_nxyz, l2_points = self.sa2(l1_xyz, l1_nxyz, l1_points )
        l3_xyz, l3_nxyz, l3_points = self.sa3(l2_xyz, l2_nxyz, l2_points )
        l4_xyz, l4_nxyz, l4_points = self.sa4(l3_xyz, l3_nxyz, l3_points )

        l3_points_new = self.fp4(l3_xyz, l3_nxyz, l4_xyz, l4_nxyz, l3_points, l4_points)
        l2_points_new = self.fp3(l2_xyz, l2_nxyz, l3_xyz, l3_nxyz, l2_points, l3_points_new )
        l1_points_new = self.fp2(l1_xyz, l1_nxyz, l2_xyz, l2_nxyz, l1_points, l2_points_new )
        l0_points_new = self.fp1(   xyz,    nxyz, l1_xyz, l1_nxyz,    points, l1_points_new )
        output = torch.tanh(self.conv(l0_points_new ) )

        depthDelta, normal = torch.split(output, [3, 3], dim=1 )
        depthDelta = depthDelta * 0.25
        normal = normal / torch.clamp(torch.sqrt(torch.sum(normal * normal, dim=1 ).unsqueeze(1) ), min=1e-10 )

        depthDelta = depthDelta.permute([0, 2, 1])
        normal = normal.permute([0, 2, 1])

        return depthDelta, normal

