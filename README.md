# SAFIRE
[![Build Status](https://jenkins-new.flatironinstitute.org/job/CCQ/job/SAFIRE/job/main/badge/icon)](https://jenkins-new.flatironinstitute.org/job/CCQ/job/SAFIRE/job/main/)
[![Docs dev](https://img.shields.io/badge/docs-latest-blue.svg)](https://safire.flatironinstitute.org/docs/dev/)


**S**tochastic **A**uxiliary-**F**ields for **I**nte**R**acting **E**lectrons (SAFIRE)
is a flexible, high-performance implementation of the auxiliary-field quantum Monte Carlo (AFQMC) method. 

## Installation

Next, follow one of the sections below depending on whether you want a CPU-only build, or a GPU-accelerated build.

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
- Intel oneAPI MKL (for optimized sparse matrix operations on CPU)

The build system will fetch the following dependencies automatically if they are not installed

- [nda](https://github.com/TRIQS/nda) (tensor branch)
- cxxopts
- spdlog
- cpptrace
- nlohmann_json
- Catch2 (for tests)

For a GPU build, the following are also required:

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

## License

This project is licensed under the [Apache License, Version 2.0](LICENSE).

Portions of this software are derived from the QMCPACK
project, which was originally distributed under the [University of Illinois/NCSA Open Source License](LICENSES/NCSA.txt).
The full text of that license can be found in [LICENSES/NCSA.txt](LICENSES/NCSA.txt).

## Contributors

The core developers of SAFIRE are listed below. For the full list of contributors, see [CONTRIBUTORS.md](CONTRIBUTORS.md).

<table>
  <tr>
    <td align="center">
      <a href="https://github.com/bkesk">
        <img src="https://github.com/bkesk.png" width="80"/><br/>
        <sub><b>(Brandon) Kyle Eskridge</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ryanlevy">
        <img src="https://github.com/ryanlevy.png" width="80"/><br/>
        <sub><b>Ryan Levy</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/mmorale3">
        <img src="https://github.com/mmorale3.png" width="80"/><br/>
        <sub><b>Miguel Morales</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/prosenberg15">
        <img src="https://github.com/prosenberg15.png" width="80"/><br/>
        <sub><b>Peter Rosenberg</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/lukas-weber">
        <img src="https://github.com/lukas-weber.png" width="80"/><br/>
        <sub><b>Lukas Weber</b></sub>
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/Paul-St-Young">
        <img src="https://github.com/Paul-St-Young.png" width="80"/><br/>
        <sub><b>Paul Yang</b></sub>
      </a>
    </td>
  </tr>
</table>
