/**
 * @File:   extrude_footprint_ext_cuda.cpp
 * @Author: Haozhe Xie
 * @Date:   2023-03-26 11:06:13
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-12-23 11:17:37
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA footprint")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

torch::Tensor extrude_footprint_ext_cuda_forward(
    torch::Tensor height_field, torch::Tensor seg_map, int l1_height,
    int roof_height, int l1_id_offset, int roof_id_offset, int footprint_id_min,
    int footprint_id_max, int max_height, cudaStream_t stream);

torch::Tensor
extrude_footprint_ext_forward(torch::Tensor height_field, torch::Tensor seg_map,
                              int l1_height, int roof_height, int l1_id_offset,
                              int roof_id_offset, int footprint_id_min,
                              int footprint_id_max, int max_height) {
  CHECK_INPUT(height_field);
  CHECK_INPUT(seg_map);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return extrude_footprint_ext_cuda_forward(
      height_field, seg_map, l1_height, roof_height, l1_id_offset,
      roof_id_offset, footprint_id_min, footprint_id_max, max_height, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &extrude_footprint_ext_forward,
        "Extrude Tensor Ext. Forward (CUDA)");
}
