# SAFIRE


**S**tochastic **A**uxiliary-**F**ields for **I**nte**R**acting **E**lectrons (SAFIRE)
is a flexible, high-performance implementation of the auxiliary-field quantum Monte Carlo (AFQMC) method. 

## Installation

This project uses submodules which must be initialized.
This can be achievied using the following commands:

Next, follow one of the sections below depending on whether you want a CPU-only build,
or a GPU-accelerated build.

Also, see the afqmctools Python package [Readme](utils/README.md#installation) to install the afqmctools Python package.

### Dependencies

For a CPU-only build:

- CMake 3.18+
- a modern C++ compiler with C++20 support. LLVM or GCC are recommended.
- MPI (OpenMPI or other MPI implementation)
- HDF5 (parallel HDF5 recommended)
- Boost 1.61.0+
- BLAS library
- LAPACK library
- Intel oneAPI MKL (for sparse matrices on CPU)

The build system will fetch the following dependencies automatically if they are not installed

- [nda](https://github.com/TRIQS/nda) (tensor branch)
- cxxopts
- spdlog
- cpptrace
- nlohmann_json
- Catch2 (for tests)

For a GPU-build, the following are also required:

- CUDA 12+
- CuTENSOR
- CCCL (fetched automatically)
- A GPU with compute compatability >=8

### CPU Build

```bash
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release
$ make -j10
```

### GPU Build

```bash
$ mkdir build
$ cd build
$ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
$ make -j10
```

## Documentation

The user documentation can be built using sphinx as follows.

First, install the python dependencies of the documentation by installing [afqmctools](utils/README.md#installation) with the `DOCS` optional dependency set.

Next, build the docs by running the following starting from the root of this repository:
```bash
$ cd docs
$ make html
```
some error messages and warnings may occur, but this should generate an html document in `_build/html`.

Finally, to view the documentation locally, navigate in a browser to the documentation using:

```
file:///path/to/SAFIRE/docs/_build/html/index.html
```

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Portions of this software are derived from the QMCPACK
project, which was originally distributed under the [University of Illinois/NCSA Open Source License](LICENSES/NCSA.txt).
The full text of that license can be found in [LICENSES/NCSA.txt](LICENSES/NCSA.txt).
