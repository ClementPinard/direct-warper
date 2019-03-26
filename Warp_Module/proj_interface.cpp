#include <torch/extension.h>

#include <vector>
#include <iostream>

// declarations

std::vector<at::Tensor> proj_depth_cuda_forward(
    at::Tensor points,
    int H, int W);

std::vector<at::Tensor> proj_depth_cpp_forward(
    at::Tensor points,
    int H, int W);

at::Tensor proj_img_cuda_forward(
    at::Tensor colors,
    at::Tensor index);

at::Tensor proj_img_cpp_forward(
    at::Tensor colors,
    at::Tensor index);

at::Tensor proj_depth_cuda_backward(
    at::Tensor index,
    at::Tensor grad_output,
    int N);

at::Tensor proj_img_cuda_backward(
    at::Tensor index,
    at::Tensor grad_output,
    int N);

at::Tensor proj_depth_cpp_backward(
    at::Tensor index,
    at::Tensor grad_output,
    int N);

at::Tensor proj_img_cpp_backward(
    at::Tensor index,
    at::Tensor grad_output,
    int N);

// C++ interface

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDA tensor")

std::vector<at::Tensor> proj_img_forward(
    at::Tensor points,
    at::Tensor colors,
    int H, int W) {
  std::vector<at::Tensor> result;
  at::Tensor projected_img;
  at::Tensor transposed_points = points.transpose(1,2).contiguous();
  if (transposed_points.type().is_cuda()){
    CHECK_CUDA(colors);
    result = proj_depth_cuda_forward(transposed_points, H, W);
    projected_img = proj_img_cuda_forward(colors, result.at(1));
  }else{
    result = proj_depth_cpp_forward(transposed_points, H, W);
    projected_img = proj_img_cpp_forward(colors, result.at(1));
  }

  result.push_back(projected_img);
  return(result);
}

std::vector<at::Tensor> proj_depth_forward(
    at::Tensor points,
    int H, int W) {
  at::Tensor transposed_points = points.transpose(1,2).contiguous();
  if (transposed_points.type().is_cuda()){
    return proj_depth_cuda_forward(transposed_points, H, W);
  }else{
    return proj_depth_cpp_forward(transposed_points, H, W);
  }
}

at::Tensor proj_depth_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {

  if(gradOutput.type().is_cuda()){
    CHECK_CUDA(index);
    return proj_depth_cuda_backward(index, gradOutput, N);
  }else{
    return proj_depth_cpp_backward(index, gradOutput, N);
  }
}

at::Tensor proj_img_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {

  if(gradOutput.type().is_cuda()){
    CHECK_CUDA(index);
    return proj_img_cuda_backward(index, gradOutput, N);
  }else{
    return proj_img_cpp_backward(index, gradOutput, N);
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_img", &proj_img_forward, "Direct Img Projection Forward");
  m.def("forward_depth", &proj_depth_forward, "Direct Depth Projection Forward");
  m.def("backward_img", &proj_img_backward, "Direct Depth Projection Backward");
  m.def("backward_depth", &proj_depth_backward, "Direct Img Projection Backward");
}
