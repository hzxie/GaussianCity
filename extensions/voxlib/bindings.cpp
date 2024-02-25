/**
 * @File:   bindings.cpp
 * @Author: NVIDIA Corporation
 * @Date:   2021-10-13 00:00:00
 * @Last Modified by: Haozhe Xie
 * @Last Modified at: 2024-02-24 14:16:37
 * @Email:  root@haozhexie.com
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <vector>

// Fast voxel traversal along rays
std::vector<torch::Tensor> ray_voxel_intersection_perspective_cuda(
    const torch::Tensor &in_voxel, const torch::Tensor &cam_ori,
    const torch::Tensor &cam_dir, const torch::Tensor &cam_up, float cam_f,
    const std::vector<float> &cam_c, const std::vector<int> &img_dims,
    int max_samples);

torch::Tensor points_to_volume_cuda(const torch::Tensor &points,
                                    const torch::Tensor &scales, int h, int w,
                                    int d);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ray_voxel_intersection_perspective",
        &ray_voxel_intersection_perspective_cuda,
        "Ray-voxel intersections given perspective camera parameters (CUDA)");
  m.def("points_to_volume", &points_to_volume_cuda,
        "Generate 3D volume from points (CUDA)");
}
