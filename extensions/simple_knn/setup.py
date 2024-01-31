#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

cxx_compiler_flags = ["/wd4624"] if os.name == "nt" else []
nvcc_args = []

setup(
    name="simple_knn",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            name="simple_knn_ext",
            sources=["spatial.cu", "simple_knn.cu", "bindings.cpp"],
            extra_compile_args={"nvcc": nvcc_args, "cxx": cxx_compiler_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
