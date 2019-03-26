import torch
from torch import nn
from torch.nn.functional import conv2d
from pytorch_direct_warp.direct_warp import DirectWarper


def dilate(tensor, dilation):
    assert(tensor.dtype == torch.uint8), "must be a bool tensor"
    assert(tensor.ndimension() == 3), "must of size BxHxW"
    if dilation != 0:
        k_size = 2 * abs(dilation) + 1
        kernel = torch.ones(1,1,k_size, k_size).float().to(tensor.device)
        if dilation > 0:
            dilated = conv2d(tensor.unsqueeze(1).float(), kernel, padding=dilation)
            dilated = dilated != 0
        else:
            dilated = conv2d((~tensor.unsqueeze(1)).float(), kernel, padding=-dilation)
            dilated = dilated == 0
        return dilated[:,0]
    else:
        return tensor


class OcclusionMapper(nn.Module):
    def __init__(self, dilation=0):
        super(OcclusionMapper, self).__init__()
        self.warper = DirectWarper()
        self.dilation = dilation

    def forward(self, depth, pose_matrix, intrinsics, intrinsics_inv):

        warped_depth = self.warper(depth, None, pose_matrix, intrinsics, intrinsics_inv)

        inv_rot_matrix = pose_matrix[:,:,:3].transpose(1,2)
        inv_tr = -inv_rot_matrix @ pose_matrix[:,:,-1:]

        inv_pose_matrix = torch.cat([inv_rot_matrix, inv_tr], dim=2)
        warped_depth_back = self.warper(warped_depth, None, inv_pose_matrix, intrinsics, intrinsics_inv)

        invalid_points = warped_depth_back == float('inf')
        return dilate(invalid_points, self.dilation)
