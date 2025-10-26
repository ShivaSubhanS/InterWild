# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# 
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# 

import torch
import torch.nn as nn

# PyTorch-only implementation to replace pytorch3d functions
def axis_angle_to_matrix(axis_angle):
    """
    Convert axis-angle rotation to 3x3 rotation matrix using Rodrigues' formula.
    
    Args:
        axis_angle: (B, 3) or (B, N, 3) axis-angle vectors
    Returns:
        (B, 3, 3) or (B, N, 3, 3) rotation matrices
    """
    batch_dims = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)
    
    angle = torch.norm(axis_angle, dim=1, keepdim=True)
    axis = axis_angle / (angle + 1e-8)
    
    # Rodrigues' formula
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)
    
    # Cross product matrix
    K = torch.zeros(axis_angle.shape[0], 3, 3, device=axis_angle.device)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]
    
    # Rotation matrix: R = I + sin(θ)K + (1-cos(θ))K^2
    I = torch.eye(3, device=axis_angle.device).unsqueeze(0).expand(axis_angle.shape[0], -1, -1)
    R = I + sin_angle.unsqueeze(2) * K + (1 - cos_angle).unsqueeze(2) * torch.bmm(K, K)
    
    # Handle small angles (use identity rotation)
    small_angle_mask = (angle.squeeze(1) < 1e-3).unsqueeze(1).unsqueeze(2)
    R = torch.where(small_angle_mask, I, R)
    
    return R.reshape(*batch_dims, 3, 3)

def matrix_to_axis_angle(matrix):
    """
    Convert 3x3 rotation matrices to axis-angle representation.
    
    Args:
        matrix: (B, 3, 3) or (B, N, 3, 3) rotation matrices
    Returns:
        (B, 3) or (B, N, 3) axis-angle vectors
    """
    batch_dims = matrix.shape[:-2]
    matrix = matrix.reshape(-1, 3, 3)
    
    # Compute angle
    trace = matrix[:, 0, 0] + matrix[:, 1, 1] + matrix[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Compute axis
    axis = torch.stack([
        matrix[:, 2, 1] - matrix[:, 1, 2],
        matrix[:, 0, 2] - matrix[:, 2, 0],
        matrix[:, 1, 0] - matrix[:, 0, 1]
    ], dim=1)
    
    # Normalize axis
    axis_norm = torch.norm(axis, dim=1, keepdim=True)
    axis = axis / (axis_norm + 1e-8)
    
    # Axis-angle representation
    axis_angle = angle.unsqueeze(1) * axis
    
    # Handle small angles (use identity rotation)
    small_angle_mask = (angle < 1e-3).unsqueeze(1)
    axis_angle = torch.where(small_angle_mask, torch.zeros_like(axis_angle), axis_angle)
    
    return axis_angle.reshape(*batch_dims, 3)

class CoordLoss(nn.Module):
    def __init__(self):
        super(CoordLoss, self).__init__()

    def forward(self, coord_out, coord_gt, valid, is_3D):
        loss = torch.abs(coord_out - coord_gt) * valid
        loss_z = loss[:,:,2:] * is_3D[:,None,None].float()
        loss = torch.cat((loss[:,:,:2], loss_z),2)
        return loss

class PoseLoss(nn.Module):
    def __init__(self):
        super(PoseLoss, self).__init__()

    def forward(self, pose_out, pose_gt, pose_valid):
        batch_size = pose_out.shape[0]

        pose_out = pose_out.view(batch_size,-1,3)
        pose_gt = pose_gt.view(batch_size,-1,3)

        #pose_out = matrix_to_axis_angle(axis_angle_to_matrix(pose_out))
        #pose_gt = matrix_to_axis_angle(axis_angle_to_matrix(pose_gt))

        #loss = torch.abs(pose_out - pose_gt) * pose_valid[:,:,None]

        pose_out = axis_angle_to_matrix(pose_out)
        pose_gt = axis_angle_to_matrix(pose_gt)

        loss = torch.abs(pose_out - pose_gt) * pose_valid[:,:,None,None]
        return loss


       
