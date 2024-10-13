/**
 * @File:   points_to_volume.cu
 * @Author: Haozhe Xie
 * @Date:   2024-02-24 14:09:38
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-10-13 12:29:46
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "voxlib_common.h"

#define THREADS_PER_BLOCK 256

__global__ void points_to_volume_cuda_cuda_kernel(
    size_t n_pts, int h, int w, int d, const short *__restrict__ points,
    const int *__restrict__ pt_ids, const short *__restrict__ scales,
    int *__restrict__ volume) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx >= n_pts) {
    return;
  }
  int pid = pt_ids[idx];
  int idx3 = idx * 3;
  short x = points[idx3];
  short y = points[idx3 + 1];
  short z = points[idx3 + 2];
  short sx = scales[idx3];
  short sy = scales[idx3 + 1];
  short sz = scales[idx3 + 2];

  if (x >= w || y >= h || z >= d || x < 0 || y < 0 || z < 0) {
    return;
  }
  for (int j = x; j < x + sx && j < w; ++j) {
    for (int k = y; k < y + sy && k < h; ++k) {
      for (int l = z; l < z + sz && l < d; ++l) {
        int64_t idx = static_cast<int64_t>(k) * w * d + j * d + l;
        volume[idx] = pid;
      }
    }
  }
}

torch::Tensor points_to_volume_cuda(const torch::Tensor &points,
                                    const torch::Tensor &pt_ids,
                                    const torch::Tensor &scales, int h, int w,
                                    int d) {
  CHECK_CUDA(points);
  CHECK_CUDA(pt_ids);
  CHECK_CUDA(scales);

  size_t n_pts = points.size(0);
  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  torch::Device device = points.device();

  int n_blocks = (n_pts + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  torch::Tensor volume = torch::zeros(
      {h, w, d}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  points_to_volume_cuda_cuda_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(
      n_pts, h, w, d, points.data_ptr<short>(), pt_ids.data_ptr<int>(),
      scales.data_ptr<short>(), volume.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in points_to_volume_cuda_cuda_kernel: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}
