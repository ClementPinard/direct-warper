import torch
import torch.nn as nn

from pytorch_direct_warp.direct_proj import direct_projection


class DirectWarper(nn.Module):
    def __init__(self):
        super(DirectWarper, self).__init__()
        self.index_map = None
        self.index = None

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
        ones = torch.ones(1,h,w).type_as(depth)

        self.index_map = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    def forward(self, depth, img, pose_matrix, intrinsics, intrinsics_inv):
        b, h, w = depth.size()
        if (self.index_map is None) or \
           (self.index_map.size(1) < h) or \
           (self.index_map.size(2) < w):
            self.set_id_grid(depth)
        rot_matrix = intrinsics @ pose_matrix[:,:,:3] @ intrinsics_inv
        tr = intrinsics @ pose_matrix[:,:,-1:]
        point_cloud = (self.index_map[:,:,:h,:w].expand(b,3,h,w)*depth.unsqueeze(1)).view(b, 3, -1)
        point_sizes = point_cloud[:,-1:]

        transformed_points = rot_matrix @ point_cloud + tr

        square_cloud = torch.cat([transformed_points, point_sizes], dim=1)
        if img is not None:
            colors = img.view(b, img.size(1), -1)
        else:
            colors = None

        *result, self.index = direct_projection(square_cloud, colors, h, w)

        return result[0] if len(result) == 1 else result
