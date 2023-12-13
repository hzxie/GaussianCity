/**
 * @File:   bindings.cpp
 * @Author: Haozhe Xie
 * @Date:   2023-12-13 13:43:51
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2023-12-13 15:47:53
 * @Email:  root@haozhexie.com
 */

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                                    \
  AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                         \
  CHECK_CUDA(x);                                                               \
  CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor>
topdown_projector_ext_cuda_forward(torch::Tensor volume, cudaStream_t stream);

std::vector<torch::Tensor> topdown_projector_ext_forward(torch::Tensor volume) {
  CHECK_INPUT(volume);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  return topdown_projector_ext_cuda_forward(volume, stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &topdown_projector_ext_forward,
        "TopDown Projector Ext. Forward (CUDA)");
}
