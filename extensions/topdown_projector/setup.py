# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Haozhe Xie
# @Date:   2023-12-13 13:43:22
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-13 13:43:37
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="topdown_projector",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "topdown_projector_ext",
            [
                "bindings.cpp",
                "topdown_projector_ext.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
