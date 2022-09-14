#include <ATen/ATen.h>
#include <torch/types.h>
using namespace torch;

#include <cuda.h>
#include <cuda_runtime.h>



#include <vector>
#include <iostream>

#define dTensor3R PackedTensorAccessor<scalar_t, 3, RestrictPtrTraits, size_t>
#define dTensor4R PackedTensorAccessor<scalar_t, 4, RestrictPtrTraits, size_t>
#define dTensor3RLong PackedTensorAccessor<long, 3, RestrictPtrTraits, size_t>
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_NUM_THREADS 512
#define THREADS_PER_BLOCK 64

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  if (val==0.0)
    return __longlong_as_double(old);
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}
#endif

template <typename scalar_t, int dims>
PackedTensorAccessor<scalar_t, dims, RestrictPtrTraits, size_t>
toDeviceTensor(at::Tensor t) {
  return PackedTensorAccessor<scalar_t, dims, RestrictPtrTraits, size_t>
  (t.data<scalar_t>(), (size_t*) t.sizes().data(), (size_t*) t.strides().data());
}

namespace {

__device__ __forceinline__ bool SpecialMin (double z, double* projected_depth) {
  unsigned long long val = (unsigned long long) __double_as_longlong(z);
  unsigned long long old = atomicMin((unsigned long long*) projected_depth, val);
  return(old >= val);
}

__device__ __forceinline__ bool SpecialMin (float z, float* projected_depth) {
  int val = __float_as_int(z);
  int old = atomicMin((int*) projected_depth, val);
  return(old >= val);
}

template <typename scalar_t>
__global__ void proj_cuda_forward_kernel(
    const dTensor3R points,
    dTensor3R projected_depth,
    dTensor3RLong index_map,
    int num_points) {

  const int B = points.size(0);
  const int N = points.size(1);

  const int H = projected_depth.size(1);
  const int W = projected_depth.size(2);

  CUDA_KERNEL_LOOP(i, num_points) {

        int p = i % N;
        int b = (i / N) % B;

        scalar_t X = points[b][p][0];
        scalar_t Y = points[b][p][1];
        scalar_t Z = points[b][p][2];
        scalar_t R = points[b][p][3];

        if (Z < 0 | isinf(X) | isinf(Y))
            return;

        int u0 = floor(0.5 + (X - 0.5*R) / Z);
        int v0 = floor(0.5 + (Y - 0.5*R) / Z);

        int du = ceil(0.5 + (X + 0.5*R) / Z);
        int dv = ceil(0.5 + (Y + 0.5*R) / Z);

        if (u0 >= W | v0 >= H | du < 0 | dv < 0)
            return;
        int v = max(v0, 0);
        do{
            int u = max(u0, 0);
            do{
                unsigned long long* index_addr = (unsigned long long*) &index_map[b][v][u];
                unsigned long long old_index, assumed;
                old_index = *index_addr;
                do{
                    assumed = old_index;
                    if (SpecialMin(Z, &projected_depth[b][v][u])){
                        old_index = atomicCAS(index_addr, assumed, (unsigned long long) p);
                    }
                }while(assumed != old_index);
                u++;
            }while(u < du and u < W);
            v++;
        }while(v < dv and v < H);

    }
}

template <typename scalar_t>
__global__ void proj_img_cuda_forward_kernel(
    const dTensor3RLong indexMap,
    const dTensor3R colors,
    dTensor4R projected_img,
    int num_points) {

  const int B = colors.size(0);
  const int C = colors.size(1);

  const int H = indexMap.size(1);
  const int W = indexMap.size(2);

  CUDA_KERNEL_LOOP(i, num_points) {

    int w = i % W;
    int h = (i / W) % H;
    int c = (i / (H * W)) % C;
    int b = (i / (H * W * C)) % B;

    int n = indexMap[b][h][w];
    if (n >= 0){
      projected_img[b][c][h][w] = colors[b][c][n];
    }

  }
}

template <typename scalar_t>
__global__ void proj_depth_cuda_backward_kernel(
    const dTensor3RLong indexMap,
    const dTensor3R gradOutput,
    dTensor3R gradInput,
    int num_points) {

  const int B = gradInput.size(0);
  const int N = gradInput.size(1);

  const int H = indexMap.size(1);
  const int W = indexMap.size(2);

  CUDA_KERNEL_LOOP(i, num_points) {

    int w = i % W;
    int h = (i / W) % H;
    int b = (i / (H * W)) % B;


    long pIndex = indexMap[b][h][w];
    if (pIndex >=0)
      atomicAdd(&gradInput[b][0][pIndex], (scalar_t) gradOutput[b][h][w]);
  }
}

template <typename scalar_t>
__global__ void proj_img_cuda_backward_kernel(
    const dTensor3RLong indexMap,
    const dTensor4R gradOutput,
    dTensor3R gradInput,
    int num_points) {

  const int B = gradInput.size(0);
  const int C = gradInput.size(1);
  const int N = gradInput.size(2);

  const int H = indexMap.size(1);
  const int W = indexMap.size(2);

  CUDA_KERNEL_LOOP(i, num_points) {

        int w = i % W;
        int h = (i / W) % H;
        int c = (i / (H * W)) % C;
        int b = (i / (H * W * C)) % B;


        long pIndex = indexMap[b][h][w];
        if (pIndex >=0)
          atomicAdd(&gradInput[b][c][pIndex], (scalar_t) gradOutput[b][c][h][w]);
  }
}

}

std::vector<at::Tensor> proj_depth_cuda_forward(
    at::Tensor points,
    int H, int W){

  const int batch_size = points.size(0);
  const int num_points = points.size(1) * points.size(0);
  auto options = points.options();
  auto projected_depth = at::empty({batch_size, H, W}, options);
  auto index = at::empty({batch_size, H, W}, options.dtype(at::kLong));
  index.fill_(-1);

  const int threads = CUDA_NUM_THREADS;
  const int blocks = (num_points + CUDA_NUM_THREADS -1) / CUDA_NUM_THREADS;

  AT_DISPATCH_FLOATING_TYPES(points.type(), "proj_depth_forward_cuda", ([&] {
    projected_depth.fill_(std::numeric_limits<scalar_t>::infinity());
    proj_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<scalar_t,3>(points),
        toDeviceTensor<scalar_t,3>(projected_depth),
        toDeviceTensor<long,3>(index),
        num_points);
  }));

  return {projected_depth, index};
}

at::Tensor proj_img_cuda_forward(
    at::Tensor colors,
    at::Tensor index){

  const int batch_size = colors.size(0);
  const int channel = colors.size(1);
  const int H = index.size(1);
  const int W = index.size(2);
  const int num_points = batch_size * channel * H * W;
  auto projected_img = at::empty({batch_size, channel, H, W}, colors.options());

  const int threads = CUDA_NUM_THREADS;
  const int blocks = (num_points + CUDA_NUM_THREADS -1) / CUDA_NUM_THREADS;

  AT_DISPATCH_FLOATING_TYPES(colors.type(), "proj_img_forward_cuda", ([&] {
    projected_img.fill_(std::numeric_limits<scalar_t>::quiet_NaN());
    proj_img_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<long,3>(index),
        toDeviceTensor<scalar_t,3>(colors),
        toDeviceTensor<scalar_t,4>(projected_img),
        num_points);
  }));

  return projected_img;
}

at::Tensor proj_depth_cuda_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {
  
  const int batch_size = gradOutput.size(0);
  auto depthGradInput = at::zeros({batch_size, 1, N}, gradOutput.options());
  auto posGradInput = at::zeros({batch_size, 2, N}, gradOutput.options());
  auto radiusGradInput = at::zeros({batch_size, 1, N}, gradOutput.options());
  
  const int H = index.size(1);
  const int W = index.size(2);
  const int num_points = batch_size * H * W;

  const int threads = CUDA_NUM_THREADS;
  const int blocks = (num_points + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  
  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "proj_depth_backward_cuda", ([&] {
    proj_depth_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<long,3>(index),
        toDeviceTensor<scalar_t,3>(gradOutput),
        toDeviceTensor<scalar_t,3>(depthGradInput),
        num_points);
  }));

  return at::cat({posGradInput, depthGradInput, radiusGradInput}, 1);
}

at::Tensor proj_img_cuda_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {
  
  const int batch_size = gradOutput.size(0);
  const int C = gradOutput.size(1);
  auto gradInput = at::zeros({batch_size, C, N}, gradOutput.options());
  
  const int H = index.size(1);
  const int W = index.size(2);
  const int num_points = batch_size * C * H * W;

  const int threads = CUDA_NUM_THREADS;
  const int blocks = (num_points + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
  
  AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "proj_img_backward_cuda", ([&] {
    proj_img_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
        toDeviceTensor<long,3>(index),
        toDeviceTensor<scalar_t,4>(gradOutput),
        toDeviceTensor<scalar_t,3>(gradInput),
        num_points);
  }));

  return gradInput;
}
