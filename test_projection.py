import torch

from notebooks import utils
from pytorch_direct_warp.direct_proj import DirectProjFunction
direct_projection = DirectProjFunction(10,10)
device = torch.device("cuda")

intrinsics = torch.Tensor([[5, 0, 5],
                           [0, 5, 5],
                           [0, 0, 1]]).view(1,3,3).float().to(device)
intrinsics_inv = torch.inverse(intrinsics[0]).view(1,3,3)
print(intrinsics_inv)

depth_map = torch.nn.Parameter(torch.ones(1,10,10, requires_grad=True).float().to(device))
img = torch.ones(1,1,10,10).float().view(1,1,-1).to(device)

points = utils.pixel2cam(depth_map).view(1,3,-1)
print(points.requires_grad)


translation = torch.Tensor([0,0,1]).view(1,3,1).float().to(device)

projected_translation = intrinsics @ translation

print(projected_translation)

points += projected_translation
print(points)
print(points.shape)


with_size = torch.cat((points, points[:,2:]), dim=1)
print(with_size.requires_grad)
warped_depth = direct_projection(with_size)
print(warped_depth.requires_grad)
loss = warped_depth.mean()
loss.backward()
print(depth_map.grad)
print(warped_depth)