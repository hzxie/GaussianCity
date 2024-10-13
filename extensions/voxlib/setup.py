# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: NVIDIA Corporation
# @Date:   2021-10-13 00:00:00
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-10-13 03:00:47
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_args = ["-fopenmp"]
nvcc_args = []

setup(
    name="voxlib_ext",
    version="3.0.0",
    ext_modules=[
        CUDAExtension(
            "voxlib",
            [
                "bindings.cpp",
                "ray_voxel_intersection.cu",
                "points_to_volume.cu",
                "maps_to_volume.cu",
            ],
            extra_compile_args={"cxx": cxx_args, "nvcc": nvcc_args},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
