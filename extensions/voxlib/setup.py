# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: NVIDIA Corporation
# @Date:   2021-10-13 00:00:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-03-27 10:26:27
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-fopenmp"]
nvcc_args = []

setup(
    name="voxlib_ext",
    version="2.0.1",
    ext_modules=[
        CUDAExtension(
            "voxlib",
            [
                "bindings.cpp",
                "ray_voxel_intersection.cu",
                "points_to_volume.cu",
            ],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
