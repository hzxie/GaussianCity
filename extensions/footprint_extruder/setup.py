# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Haozhe Xie
# @Date:   2023-03-24 20:35:43
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2023-12-23 10:41:24
# @Email:  root@haozhexie.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="extrude_footprint",
    version="1.0.0",
    ext_modules=[
        CUDAExtension(
            "extrude_footprint_ext",
            [
                "bindings.cpp",
                "extrude_footprint_ext.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
