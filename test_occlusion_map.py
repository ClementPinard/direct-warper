import torch
from pytorch_direct_warp.occlusion_mapper import OcclusionMapper
device = torch.device("cuda")

mapper = OcclusionMapper(dilation=0).to(device)
intrinsics = torch.Tensor([[5, 0, 5],
                           [0, 5, 5],
                           [0, 0, 1]]).view(1,3,3).float().to(device)
intrinsics_inv = torch.inverse(intrinsics[0]).view(1,3,3)
print(intrinsics_inv)

depth_map = torch.ones(1,10,10).float().to(device)
img = torch.ones(1,1,10,10).float().view(1,1,-1).to(device)

pose_matrix = torch.Tensor([[1,0,0,0],
                            [0,1,0,1],
                            [0,0,1,0]]).unsqueeze(0).to(device)

occlusion = mapper(depth_map, pose_matrix, intrinsics, intrinsics_inv)

print(occlusion)