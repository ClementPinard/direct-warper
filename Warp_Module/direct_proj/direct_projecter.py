from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import direct_proj_backend as proj


def direct_projection(points, colors, H, W):
    """Apply direct projection from a list of square 3D points, with
    optional corresponding color.

    Args:
        points : List of 3D points with radius R. Size should be BxNx4.
        colors : List of correponsding colors. Size should BxCxN.
        pose : matrix to multiply the points with before projecting Bx3X4
        frame_matrix : K matrix for projection Bx3X3

    Returns:
        depth_map : corresponding depth BxHxW
        color_map : BxCxHxW

    """
    function = DirectProjFunction(H, W)
    if colors is not None:
        return function(points, colors)
    else:
        return function(points)


class DirectProjFunction(Function):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def forward(self, points, colors=None):
        self.with_colors = colors is not None
        self.num_points = points.size(-1)
        assert (points.ndimension() == 3 and points.size(1) == 4), ("points tensor must be Bx4xN, "
                                                                    "got {} instead").format(list(points.size()))
        if self.with_colors:
            assert(colors.ndimension() == 3 and colors.size(-1) == points.size(-1) and colors.size(0) == points.size(0)), \
                ("colors tensor must be BxCxN, and B and N must be same as points, i.e {} and {},"
                 "got {} instead").format(points.size(0), points.size(-1), list(colors.size()))
            depth_map, index, img = proj.forward_img(points, colors, self.H, self.W)
        else:
            depth_map, index = proj.forward_depth(points, self.H, self.W)
        self.save_for_backward(index)
        if self.with_colors:
            return depth_map, img, index
        else:
            return depth_map, index

    @once_differentiable
    def backward(self, depth_grad_output, colors_grad_output=None):
        index = self.saved_variables[0]

        points_grad_input = proj.backward_depth(index, depth_grad_output, self.num_points)
        if self.with_colors:
            colors_grad_input = proj.backward_img(index, colors_grad_output, self.num_points)
        else:
            colors_grad_input = None
        return points_grad_input, colors_grad_input


class DirectProjecter(nn.Module):
    def __init__(self, H, W):
        super(DirectProjecter, self).__init__()
        self.H = H
        self.W = W

    def forward(self, points, colors):
        return direct_projection(points, colors, self.H, self.W)
