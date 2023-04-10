import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='torchcrf_cpp',
    ext_modules=[
        CUDAExtension('torchcrf_cpp', [
            'src/torchcrf.cu'
        ], include_dirs=[os.path.join(os.path.dirname(__file__), "include")])
    ],
    cmdclass={'build_ext': BuildExtension}
)
