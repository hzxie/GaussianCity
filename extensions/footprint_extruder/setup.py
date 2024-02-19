# -*- coding: utf-8 -*-
#
# @File:   setup.py
# @Author: Haozhe Xie
# @Date:   2024-02-12 13:07:13
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2024-02-19 20:55:17
# @Email:  root@haozhexie.com

import numpy

from distutils.core import setup, Extension

# run the setup
setup(
    name="footprint_extruder",
    version="2.1.0",
    ext_modules=[
        Extension(
            "footprint_extruder",
            sources=["footprint_extruder.cpp"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-std=c++11", "-O2"],
        )
    ],
)
