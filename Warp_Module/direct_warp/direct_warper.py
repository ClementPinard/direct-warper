import torch
import torch.nn as nn

from pytorch_direct_warp.direct_proj import direct_projection


class DirectWarper(nn.Module):
    def __init__(self, keep_index=False):
        super(DirectWarper, self).__init__()
        self.id_grid = None
        self.keep_index = keep_index

    def set_id_grid(self, depth):
        b, h, w = depth.size()
        i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
        j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
        ones = depth.new_ones(1,h,w)

        self.id_grid = torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]

    def forward(self, depth, img, pose_matrix, intrinsics, dilation=0):
        b, h, w = depth.size()
        if (self.id_grid is None) or \
           (self.id_grid.size(1) < h) or \
           (self.id_grid.size(2) < w):
            self.set_id_grid(depth)

        rot_matrix = intrinsics @ pose_matrix[:,:,:3] @ intrinsics.inverse()
        tr = intrinsics @ pose_matrix[:,:,-1:]
        point_cloud = (self.id_grid[:,:,:h,:w].expand(b,3,h,w)*depth.unsqueeze(1)).view(b, 3, -1)
        point_sizes = point_cloud[:,-1:] * (2*dilation + 1)

        transformed_points = rot_matrix @ point_cloud + tr

        square_cloud = torch.cat([transformed_points, point_sizes], dim=1)
        if img is not None:
            colors = img.view(b, img.size(1), -1)
            w_depth, w_colors, index = direct_projection(square_cloud, colors, h, w)
            if self.keep_index:
                self.index = index
            return w_depth, w_colors
        else:
            w_depth, index = direct_projection(square_cloud, None, h, w)
            if self.keep_index:
                self.index = index
            return w_depth
