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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

project_root = os.path.dirname(os.path.abspath(__file__))

nvcc_args = [
    "-allow-unsupported-compiler",
    "-I" + os.path.join(project_root, "third_party/glm/"),
]
cxx_args = []

if os.name == "nt":
    nvcc_args.append("-D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
    cxx_args.append("/D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
else:
    nvcc_args.extend(["-O3", "--use_fast_math"])
    cxx_args.append("-O3")

setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={
                "nvcc": nvcc_args,
                "cxx": cxx_args,
            })
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
