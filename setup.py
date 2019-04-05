from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'Warp_Module'
sources = [join(project_root, file) for file in ['proj.cpp',
                                                 'proj_interface.cpp',
                                                 'proj_cuda.cu']]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='direct warp',
    version="0.0.1",
    author="Clément Pinard",
    author_email="clement.pinard@ensta-paristech.fr",
    description="Direct Warp module for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ClementPinard/Pytorch-Direct-Warp-extension",
    install_requires=['torch>=0.4.1','numpy'],
    ext_modules=[
        CUDAExtension('direct_proj_backend',
                      sources,
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':['-gencode=arch=compute_50,code=compute_50',
                                                                       '-gencode=arch=compute_60,code=compute_60']},
                      extra_link_args=['-lgomp'])
    ],
    package_dir={'pytorch_direct_warp': 'Warp_Module'},
    packages=['pytorch_direct_warp.direct_warp','pytorch_direct_warp.direct_proj','pytorch_direct_warp.occlusion_mapper'],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ])
