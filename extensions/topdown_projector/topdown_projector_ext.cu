/**
 * @File:   topdown_projector_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2023-12-13 14:02:33
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-12-13 15:47:37
 * @Email:  root@haozhexie.com
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <torch/extension.h>

#define CUDA_NUM_THREADS 512

// Computer the number of threads needed in GPU
inline int get_n_threads(int n) {
  const int pow_2 = std::log(static_cast<float>(n)) / std::log(2.0);
  return max(min(1 << pow_2, CUDA_NUM_THREADS), 1);
}

__global__ void topdown_projector_ext_cuda_kernel(
    int height, int width, int depth, const int *__restrict__ volumes,
    int *__restrict__ seg_maps, int *__restrict__ height_fields) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;

  volumes += batch_index * height * width * depth;
  seg_maps += batch_index * height * width;
  height_fields += batch_index * height * width;
  for (int i = index; i < height; i += stride) {
    for (int j = 0; j < width; ++j) {
      // From top to bottom
      for (int k = depth - 1; k >= 0; --k) {
        int offset_3d = i * width * depth + j * depth + k;
        if (volumes[offset_3d] != 0) {
          int offset_2d = i * width + j;
          seg_maps[offset_2d] = volumes[offset_3d];
          height_fields[offset_2d] = k;
          break;
        }
      }
    }
  }
}

std::vector<torch::Tensor>
topdown_projector_ext_cuda_forward(torch::Tensor volumes, cudaStream_t stream) {
  int batch_size = volumes.size(0);
  int height = volumes.size(1);
  int width = volumes.size(2);
  int depth = volumes.size(3);
  torch::Tensor seg_maps =
      torch::zeros({batch_size, height, width}, torch::CUDA(torch::kInt32));
  torch::Tensor height_fields =
      torch::zeros({batch_size, height, width}, torch::CUDA(torch::kInt32));

  topdown_projector_ext_cuda_kernel<<<
      batch_size, int(CUDA_NUM_THREADS / CUDA_NUM_THREADS), 0, stream>>>(
      height, width, depth, volumes.data_ptr<int>(), seg_maps.data_ptr<int>(),
      height_fields.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in topdown_projector_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return {seg_maps, height_fields};
}
