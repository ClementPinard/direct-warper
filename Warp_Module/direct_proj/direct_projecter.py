from torch import nn
from torch.autograd import Function
from torch.autograd.function import once_differentiable

import direct_proj_backend as proj


def direct_projection(points, colors, H, W):
    """Apply direct projection from a list of square 3D points, with
    optional corresponding color.

    Args:
        points : List of 3D points with radius R. Size should be BxNx4.
        colors : List of corresponding colors. Size should BxCxN.
        H : Original height of the images in batch.
        W : Original width of the images in batch.

    Returns:
        depth_map : corresponding depth BxHxW.
        color_map : BxCxHxW (optional).
        index :
    """
    if colors is not None:
        return DirectProjFunction.apply(points, colors, (H, W))
    else:
        return DirectProjFunction.apply(points, None, (H, W))


class DirectProjFunction(Function):

    @staticmethod
    def forward(ctx, points, colors=None, dim=None):
        assert type(dim) is tuple
        H, W = dim
        ctx.with_colors = colors is not None
        ctx.num_points = points.size(-1)
        assert (points.ndimension() == 3 and points.size(1) == 4), ("points tensor must be Bx4xN, "
                                                                    "got {} instead").format(list(points.size()))
        if ctx.with_colors:
            assert(colors.ndimension() == 3 and colors.size(-1) == points.size(-1) and colors.size(0) == points.size(0)), \
                ("colors tensor must be BxCxN, and B and N must be same as points, i.e {} and {},"
                 "got {} instead").format(points.size(0), points.size(-1), list(colors.size()))
            depth_map, index, img = proj.forward_img(points, colors, H, W)
        else:
            depth_map, index = proj.forward_depth(points, H, W)
        ctx.save_for_backward(index)
        if ctx.with_colors:
            return depth_map, img, index
        else:
            return depth_map, index

    @staticmethod  # or @once_differentiable if preferred
    def backward(ctx, depth_grad_output, *args):
        index = ctx.saved_variables[0]

        points_grad_input = proj.backward_depth(index, depth_grad_output, ctx.num_points)
        if ctx.with_colors:
            colors_grad_output = args[0]
            colors_grad_input = proj.backward_img(index, colors_grad_output, ctx.num_points)
        else:
            colors_grad_input = None
        return points_grad_input, colors_grad_input, None


class DirectProjecter(nn.Module):
    def __init__(self, H, W):
        super(DirectProjecter, self).__init__()
        self.H = H
        self.W = W

    def forward(self, points, colors):
        return direct_projection(points, colors, self.H, self.W)
