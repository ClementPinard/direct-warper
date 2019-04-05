#include <torch/extension.h>
#include <ATen/ATen.h>
using namespace at;

#include <vector>

#define WITHIN_BOUNDS(x, y, H, W) (x >= 0 && x < W && y >= 0 && y < H)

template <typename scalar_t>
static void projection(
    TensorAccessor<scalar_t,2> projected_depth,
    TensorAccessor<long,2> index,
    TensorAccessor<scalar_t,1> point,
    int n, int W, int H){

  scalar_t X = point[0];
  scalar_t Y = point[1];
  scalar_t Z = point[2];
  scalar_t R = point[3];

  if ((Z<0) | (std::isinf(X)) | (std::isinf(Y)))
    return;

  int u0 = std::floor(0.5 + (X - 0.5*R) / Z);
  int v0 = std::floor(0.5 + (Y - 0.5*R) / Z);

  int du = std::ceil(0.5 + (X + 0.5*R) / Z);
  int dv = std::ceil(0.5 + (Y + 0.5*R) / Z);
  if ((u0 >= W) | (v0 >= H) | (du < 0) | (dv < 0))
    return;
  int v = std::max(v0, 0);
  do{
      int u = std::max(u0, 0);
      do{
          if (Z < projected_depth[u][v]){
            projected_depth[u][v] = Z;
            index[u][v] = n;
          }
          u++;
      }while(u < du and u < W);
      v++;
  }while(v < dv and v < H);
}

std::vector<at::Tensor> proj_depth_cpp_forward(
    at::Tensor points,
    int H, int W) {
  auto projected_depth = at::empty({points.size(0), H, W}, points.options());
  AT_DISPATCH_FLOATING_TYPES(points.type(), "set_to_inf", ([&] {
    projected_depth.fill_(std::numeric_limits<scalar_t>::infinity());
  }));
  auto index = at::empty({points.size(0), H, W}, points.options().dtype(at::kLong));
  index.fill_(-1);

  const auto batch_size = points.size(0);
  const auto N = points.size(1);
  int b, n;
  #pragma omp parallel for private(b, n)
    for (b = 0; b < batch_size; ++b){
      for (n = 0; n < N; ++n){
        AT_DISPATCH_FLOATING_TYPES(points.type(), "project", ([&] {
          auto points_acc = points.accessor<scalar_t, 3>();
          auto projected_depth_acc = projected_depth.accessor<scalar_t, 3>();
          auto index_acc = index.accessor<long, 3>();

          projection(projected_depth_acc[b], index_acc[b], points_acc[b][n], n, H, W);
      }));
    }
  }
  return {projected_depth, index};
}

at::Tensor proj_img_cpp_forward(
    at::Tensor colors,
    at::Tensor index) {
  const int batch_size = colors.size(0);
  const int C = colors.size(1);
  const int W = index.size(1);
  const int H = index.size(2);
  auto projected_img = at::empty({batch_size, C, H, W}, colors.options());
  AT_DISPATCH_FLOATING_TYPES(colors.type(), "set_to_nan", ([&] {
    projected_img.fill_(std::numeric_limits<scalar_t>::quiet_NaN());
  }));

  int b, c, h, w;
  #pragma omp parallel for private(b, c, h, w) collapse(2)
    for (b = 0; b < batch_size; ++b){
      for (c = 0; c < C; ++c){
        AT_DISPATCH_FLOATING_TYPES(colors.type(), "scatter", ([&] {
          auto colors_acc = colors.accessor<scalar_t, 3>();
          auto projected_img_acc = projected_img.accessor<scalar_t, 4>();
          auto index_acc = index.accessor<long, 3>();

          for (h = 0; h < H; ++h){
            for (w = 0; w < W; ++w){
              auto n = index_acc[b][h][w];
              if (n >= 0)
                projected_img_acc[b][c][h][w] = colors_acc[b][c][index_acc[b][h][w]];
            }
          }
        }));
      }
    }
  return projected_img;
}

at::Tensor proj_img_cpp_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {
  const int batch_size = gradOutput.size(0);
  const int C = gradOutput.size(1);
  const int W = index.size(1);
  const int H = index.size(2);

  auto gradInput = at::zeros({batch_size, C, N}, gradOutput.options());
  int b, c, h, w;
  #pragma omp parallel for private(b, c, h, w) collapse(2)
    for (b = 0; b < batch_size; ++b){
      for (c = 0; c < C; ++c){
        AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "gather", ([&] {
          auto gradOutput_acc = gradOutput.accessor<scalar_t, 4>();
          auto index_acc = index.accessor<long, 3>();
          auto gradInput_acc = gradInput.accessor<scalar_t, 3>();

          for (h = 0; h < H; ++h){
            for (w = 0; w < W; ++w){
              if (index_acc[b][h][w] >= 0)
                gradInput_acc[b][c][index_acc[b][h][w]] += gradOutput_acc[b][c][h][w];
            }
          }
        }));
      }
    }
  return gradInput;
}

at::Tensor proj_depth_cpp_backward(
    at::Tensor index,
    at::Tensor gradOutput,
    int N) {
  const int batch_size = gradOutput.size(0);
  const int W = index.size(1);
  const int H = index.size(2);

  auto gradInput = at::zeros({batch_size, 4, N}, gradOutput.options());
  int b, h, w;
  #pragma omp parallel for private(b, h, w)
    for (b = 0; b < batch_size; ++b){
      AT_DISPATCH_FLOATING_TYPES(gradOutput.type(), "gather_depth", ([&] {
        auto gradOutput_acc = gradOutput.accessor<scalar_t, 3>();
        auto index_acc = index.accessor<long, 3>();
        auto gradInput_acc = gradInput.accessor<scalar_t, 3>();
        for (h = 0; h < H; ++h){
          for (w = 0; w < W; ++w){
            if (index_acc[b][h][w] >= 0)
              gradInput_acc[b][2][index_acc[b][h][w]] += gradOutput_acc[b][h][w];
          }
        }
      }));
    }
  return gradInput;
}