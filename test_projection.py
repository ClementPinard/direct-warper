import torch
import argparse
from pytorch_direct_warp.direct_proj import DirectProjFunction


parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('-w', '--width', type=int, default=10)
parser.add_argument('-v', '--verbose', action='store_true')
parser.add_argument('-i', '--img', action='store_true',
                    help='if selected, will warp an img (composed of 1s) along with depth')
parser.add_argument('-c', '--chanels', help='img chanels', type=int, default=1)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('-d', '--double', action='store_true',
                    help='use float64 instead of float32')


def set_id_grid(depth):
    b, h, w = depth.size()
    i_range = torch.arange(0, h).view(1, h, 1).expand(1,h,w).type_as(depth)  # [1, H, W]
    j_range = torch.arange(0, w).view(1, 1, w).expand(1,h,w).type_as(depth)  # [1, H, W]
    ones = depth.new_ones(1,h,w)

    return torch.stack((j_range, i_range, ones), dim=1)  # [1, 3, H, W]


def pixel2cam(depth):
    global pixel_coords
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    b, h, w = depth.size()
    pixel_coords = set_id_grid(depth)
    cam_coords = pixel_coords[:,:,:h,:w].expand(b,3,h,w)*depth.unsqueeze(1)
    return cam_coords.contiguous()


def main():
    args = parser.parse_args()
    direct_projection = DirectProjFunction(args.height,args.width)
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    dtype = torch.float64 if args.double else torch.float32

    fx, fy = args.width/2, args.height/2

    depth = torch.ones(args.batch_size,
                       args.height,
                       args.width,
                       dtype=dtype,
                       device=device,
                       requires_grad=True)

    intrinsics = depth.new([[fy, 0, fy],
                            [0, fx, fx],
                            [0, 0,   1]]).view(1,3,3)

    points = pixel2cam(depth).view(1,3,-1)
    assert(points.requires_grad)

    translation = depth.new([0,0,0.5]).view(1,3,1)

    projected_translation = intrinsics @ translation

    points += projected_translation

    with_size = torch.cat((points, depth.view(1,1,-1)), dim=1)

    if args.img:

        img = torch.ones(args.batch_size,
                         args.chanels,
                         args.height,
                         args.width,
                         dtype=dtype,
                         device=device,
                         requires_grad=True)

        proj_depth, proj_img, index = direct_projection(with_size, img.view(1,1,-1))
        if args.verbose:
            print("Depth + Img Projection result:")
            print(proj_depth)
            print(proj_img)
        assert(proj_depth.requires_grad)
        assert(proj_img.requires_grad)
    else:
        proj_depth, index = direct_projection(with_size)
        if args.verbose:
            print("Depth Projection result:")
            print(proj_depth)
        assert(proj_depth.requires_grad)
    if args.verbose:
        print("Index map:")
        print(index)

    loss = proj_depth.mean()
    if args.img:
        loss += proj_img.sum(dim=1).mean()

    loss.backward()

    if args.verbose:
        print("Depth grad :")
        print(depth.grad)
    if args.img:
        torch.testing.assert_allclose(img.grad[:,0], depth.grad)


if __name__ == '__main__':
    main()
    print("Done")