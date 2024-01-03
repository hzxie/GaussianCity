/**
 * @File:   extrude_footprint_ext.cu
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:18
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-12-23 11:25:55
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

__global__ void extrude_footprint_ext_cuda_kernel(
    int height, int width, int depth, int l1_height, int roof_height,
    int l1_id_offset, int roof_id_offset, int footprint_id_min,
    int footprint_id_max, const int *__restrict__ height_field,
    const int *__restrict__ seg_map, int *__restrict__ volume) {
  int batch_index = blockIdx.x;
  int index = threadIdx.x;
  int stride = blockDim.x;

  height_field += batch_index * height * width;
  seg_map += batch_index * height * width;
  volume += batch_index * height * width * depth;
  for (int i = index; i < height; i += stride) {
    int offset_2d_r = i * width, offset_3d_r = i * width * depth;
    for (int j = 0; j < width; ++j) {
      int offset_2d_c = offset_2d_r + j, offset_3d_c = offset_3d_r + j * depth;
      int hf = height_field[offset_2d_c];
      int seg = seg_map[offset_2d_c];

      for (int k = 0; k < hf + 1; ++k) {
        volume[offset_3d_c + k] = seg;
      }
      if (seg >= footprint_id_min && seg < footprint_id_max) {
        for (int k = 0; k < l1_height; ++ k) {
          volume[offset_3d_c + k] = seg + l1_id_offset;
        }
        for (int k = hf; k > hf - roof_height; -- k) {
          volume[offset_3d_c + k] = seg + roof_id_offset;
        }
      }
    }
  }
}

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor height_field, torch::Tensor seg_map, int l1_height,
    int roof_height, int l1_id_offset, int roof_id_offset, int footprint_id_min,
    int footprint_id_max, int max_height, cudaStream_t stream) {
  int batch_size = seg_map.size(0);
  int height = seg_map.size(2);
  int width = seg_map.size(3);
  torch::Tensor volume = torch::zeros({batch_size, height, width, max_height},
                                      torch::CUDA(torch::kInt32));

  extrude_footprint_ext_cuda_kernel<<<
      batch_size, int(CUDA_NUM_THREADS / CUDA_NUM_THREADS), 0, stream>>>(
      height, width, max_height, l1_height, roof_height, l1_id_offset,
      roof_id_offset, footprint_id_min, footprint_id_max,
      height_field.data_ptr<int>(), seg_map.data_ptr<int>(),
      volume.data_ptr<int>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in extrude_footprint_ext_cuda_forward: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}
