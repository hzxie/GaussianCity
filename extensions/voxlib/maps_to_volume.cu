/**
 * @File:   maps_to_volume.cu
 * @Author: Haozhe Xie
 * @Date:   2024-10-09 15:42:49
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-10-13 12:26:15
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "voxlib_common.h"

#define TILE_DIM 16
#define BLDG_MAX_HEIGHT 504
#define BLDG_INS_MIN_ID 10
#define BLDG_FACADE_SEM 2
#define BLDG_ROOF_OFFSET 1

__global__ void maps_to_volume_cuda_kernel(int height, int width, int depth,
                                           const int8_t *__restrict__ scales,
                                           const short *__restrict__ inst_map,
                                           const short *__restrict__ td_hf,
                                           const short *__restrict__ bu_hf,
                                           const bool *__restrict__ pts_map,
                                           short *__restrict__ volume) {
  size_t i = blockIdx.x * blockDim.x + threadIdx.x; // width
  size_t j = blockIdx.y * blockDim.y + threadIdx.y; // height

  if (i < width && j < height) {
    bool has_pt = pts_map[j * width + i];
    if (!has_pt) {
      return;
    }

    // Fix: nonzero is not supported for tensors with more than INT_MAX elements
    short hgt_up = td_hf[j * width + i];
    short hgt_lw = bu_hf[j * width + i];
    short inst = inst_map[j * width + i];
    // WARN: The semantic labels for buildings would be merged to facade.
    short sem_cls = inst < BLDG_INS_MIN_ID ? inst : BLDG_FACADE_SEM;
    short scale = scales[sem_cls];

    int64_t vol_offset = static_cast<int64_t>(j) * width * depth + i * depth;
    for (int k = hgt_lw; k <= hgt_up; k += scale) {
      // Make all objects hallow
      bool is_border_1 = (k > hgt_up - scale) || (i < scale) ||
                         (i >= width - scale - 1) || (j < scale) ||
                         (j >= height - scale - 1);
      bool is_border_2 = false;
      bool is_border_3 = false;
      if (!is_border_1) {
        // Check is_border_1 to Prevent OOB
        short nbr_hd_hf[8] = {
            td_hf[(j - scale) * width + (i - scale)],
            td_hf[(j - scale) * width + i],
            td_hf[(j - scale) * width + (i + scale)],
            td_hf[j * width + (i - scale)],
            td_hf[j * width + (i + scale)],
            td_hf[(j + scale) * width + (i - scale)],
            td_hf[(j + scale) * width + i],
            td_hf[(j + scale) * width + (i + scale)],
        };
        for (int ni = 0; ni < 8; ++ni) {
          if (nbr_hd_hf[ni] != hgt_up) {
            is_border_2 = true;
            break;
          }
        }

        short nbr_inst[8] = {
            inst_map[(j - scale) * width + (i - scale)],
            inst_map[(j - scale) * width + i],
            inst_map[(j - scale) * width + (i + scale)],
            inst_map[j * width + (i - scale)],
            inst_map[j * width + (i + scale)],
            inst_map[(j + scale) * width + (i - scale)],
            inst_map[(j + scale) * width + i],
            inst_map[(j + scale) * width + (i + scale)],
        };
        for (int ni = 0; ni < 8; ++ni) {
          if (nbr_inst[ni] != inst) {
            is_border_3 = true;
            break;
          }
        }
      }
      if (!is_border_1 && !is_border_2 && !is_border_3) {
        continue;
      }

      // Building Roof Handler (Recover roof instance ID)
      if (k > hgt_up - scale && sem_cls == BLDG_FACADE_SEM) {
        volume[vol_offset + k] = inst + 1;
      } else {
        volume[vol_offset + k] = inst;
      }
    }
  }
}

torch::Tensor maps_to_volume_cuda(const torch::Tensor &inst_map,
                                  const torch::Tensor &td_hf,
                                  const torch::Tensor &bu_hf,
                                  const torch::Tensor &pts_map,
                                  const torch::Tensor &scales) {
  CHECK_CUDA(inst_map);
  CHECK_CUDA(td_hf);
  CHECK_CUDA(bu_hf);
  CHECK_CUDA(pts_map);
  CHECK_CUDA(scales);

  int curDevice = -1;
  cudaGetDevice(&curDevice);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream(curDevice);
  torch::Device device = inst_map.device();

  int height = inst_map.size(0);
  int width = inst_map.size(1);
  int depth = BLDG_MAX_HEIGHT;

  dim3 blockDim(TILE_DIM, TILE_DIM);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x,
               (height + blockDim.y - 1) / blockDim.y);

  torch::Tensor volume =
      torch::zeros({height, width, depth},
                   torch::TensorOptions().dtype(torch::kInt16).device(device));
  maps_to_volume_cuda_kernel<<<gridDim, blockDim, 0, stream>>>(
      height, width, depth, scales.data_ptr<int8_t>(),
      inst_map.data_ptr<short>(), td_hf.data_ptr<short>(),
      bu_hf.data_ptr<short>(), pts_map.data_ptr<bool>(),
      volume.data_ptr<short>());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    printf("Error in maps_to_volume_cuda_kernel: %s\n",
           cudaGetErrorString(err));
  }
  return volume;
}
