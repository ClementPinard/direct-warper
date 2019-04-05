import torch
import argparse
from pytorch_direct_warp.occlusion_mapper import OcclusionMapper

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch-size', type=int, default=1)
parser.add_argument('--height', type=int, default=10)
parser.add_argument('-w', '--width', type=int, default=10)
parser.add_argument('-i', '--img', action='store_true',
                    help='if selected, will warp an img (composed of 1s) along with depth')
parser.add_argument('-c', '--chanels', help='img chanels', type=int, default=1)
parser.add_argument('--cpu', action='store_true')
parser.add_argument('-d', '--double', action='store_true',
                    help='use float64 instead of float32')


def main():
    args = parser.parse_args()
    device = torch.device("cpu") if args.cpu else torch.device("cuda")
    dtype = torch.float64 if args.double else torch.float32
    fx, fy = args.width/2, args.height/2

    mapper = OcclusionMapper(dilation=0, alpha=2)

    depth = torch.ones(args.batch_size,
                       args.height,
                       args.width,
                       dtype=dtype,
                       device=device,
                       requires_grad=True)

    intrinsics = depth.new([[fy, 0, fy],
                            [0, fx, fx],
                            [0, 0,   1]]).view(1,3,3)
    pose_matrix = depth.new([[1,0,0,0],
                             [0,1,0,0],
                             [0,0,1,-0.5]]).unsqueeze(0)

    occlusion = mapper(depth, pose_matrix, intrinsics)
    print("Depth:")
    print(depth)
    print("Occlusion map:")
    print(occlusion)


if __name__ == '__main__':
    main()
    print("Done")