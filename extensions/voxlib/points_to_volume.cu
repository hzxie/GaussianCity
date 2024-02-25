/**
 * @File:   points_to_volume.cu
 * @Author: Haozhe Xie
 * @Date:   2024-02-24 14:09:38
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-02-25 15:06:27
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "voxlib_common.h"

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__device__ int64_t compute_index(int64_t x, int64_t y, int64_t z, int64_t w,
                                 int64_t d) {
  return y * w * d + x * d + z;
}

__global__ void points_to_volume_cuda_cuda_kernel(
    size_t n_pts, int h, int w, int d, const short *__restrict__ points,
    const short *__restrict__ scales, short *__restrict__ volume) {
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int stride = blockDim.x;
  int64_t sz = h * w * d;

  points += blk_idx * 4;
  scales += blk_idx * 3;
  for (int i = thd_idx; i < n_pts; i += stride) {
    short x = points[i * 4];
    short y = points[i * 4 + 1];
    short z = points[i * 4 + 2];
    short c = points[i * 4 + 3];
    short sx = scales[i * 3];
    short sy = scales[i * 3 + 1];
    short sz = scales[i * 3 + 2];
    // if (x > w || y > h || z > d) {
    //   printf("Invalid point: (%d, %d, %d)\n", x, y, z);
    // }
    assert(x < w && y < h && z < d);

    for (int j = x; j < x + sx; ++j) {
      for (int k = y; k < y + sy; ++k) {
        for (int l = z; l < z + sz; ++l) {
          if (j < 0 || j >= w || k < 0 || k >= h || l < 0 || l >= d) {
            continue;
          }
          int64_t idx = compute_index(j, k, l, w, d);
          // if (idx - sz <= 0) {
          //   printf("Invalid index: %ld/%ld. j = %d, k = %d, l = %d, h = %d, w "
          //          "= %d, d = %d\n",
          //          idx, sz, j, k, l, h, w, d);
          // }
          volume[idx] = c;
        }
      }
    }
  }
}

torch::Tensor points_to_volume_cuda(const torch::Tensor &points,
                                    const torch::Tensor &scales, int h, int w,
                                    int d) {
  CHECK_CUDA(points);
  CHECK_CUDA(scales);

  size_t n_pts = points.size(0);
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  torch::Device device = points.device();

  // XYZ, Instance ID
  assert(sizeof(short) == 2);
  assert(points.dtype() == torch::kInt16);
  assert(points.dim() == 2);
  assert(scales.dtype() == torch::kInt16);
  assert(scales.dim() == 2);
  if (n_pts) {
    assert(points.size(1) == 4);
    assert(scales.size(1) == 3);
  }

  torch::Tensor volume = torch::zeros({h, w, d}, torch::kInt16).to(device);
  points_to_volume_cuda_cuda_kernel<<<1, get_n_threads(n_pts), 0, stream>>>(
      n_pts, h, w, d, points.data_ptr<short>(), scales.data_ptr<short>(),
      volume.data_ptr<short>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in points_to_volume_cuda_cuda_kernel: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}