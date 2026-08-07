.. _install:

Installation
============

The installation is performed in two essentially independent steps.
The SAFIRE executable must be compiled, 
and the afqmctools python package can optionally be installed.

Dependencies
------------

You must first ensure that you have all of the dependencies
and install any that are missing.

CPU Build
~~~~~~~~~

- CMake 3.18+
- a modern C++ compiler with C++20 support. LLVM or GCC are recommended.
- MPI (OpenMPI or other MPI implementation)
- HDF5 (parallel HDF5 recommended)
- Boost 1.61.0+
- BLAS library
- LAPACK library
- Intel oneAPI MKL (for sparse matrices on CPU)

The build system will fetch the following dependencies automatically if they are not installed

- `nda <https://github.com/TRIQS/nda>`_ (tensor branch)
- cxxopts
- spdlog
- cpptrace
- nlohmann_json
- Catch2 (for tests)

NVIDIA GPU-build
~~~~~~~~~~~~~~~~

All of the above and

- CUDA 12+
- CuTENSOR
- CCCL (fetched automatically)
- A GPU with compute compatability >=8

Compiling SAFIRE
----------------

Assuming that all dependencies are installed and available, use the following steps to compile SAFIRE.

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release
    $ make -j 10

to compile for NVIDIA GPUs use the following.

.. code-block:: bash

    $ mkdir build
    $ cd build
    $ cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
    $ make -j 10

Convenience build scripts
~~~~~~~~~~~~~~~~~~~~~~~~~

For users at Flatiron, we provide compilation scripts for convenience.
We note that for large system sizes, GPU-accelerated builds are highly recommended,
but CPU-only builds are also useful for smaller system sizes.

CPU-only build at CCQ
_____________________

For large systems, building and running with GPU acceleration is highly recommended.
However, for smaller systems, CPU builds are useful.

If you are rusty or sitting at an SCC-managed workstation, a suitable build script is

.. code-block:: bash

    module purge
    module load slurm
    module load cmake
    module load gcc
    module load openmpi
    module load hdf5
    module load boost
    module load intel-oneapi-mkl

    mkdir build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release

    make -j 10


GPU-accelerated build at CCQ
____________________________

Currently the GPU build only works with CUDA 11.
If you are on rusty or using a rusty connected desktop a suitable build script is

.. code-block:: bash

    module purge
    module load slurm
    module load cmake
    module load gcc
    module load openmpi
    module load hdf5
    module load boost
    module load intel-oneapi-mkl
    module load cuda

    mkdir build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_CUDA=ON

    make -j 10


Installing the afqmctools Python package
________________________________________

See :ref:`the afqmctools documentation <afqmctools>` for details on installing afqmctools.

Rusty Installation
__________________

If you are connected to Rusty, you can do the following (tested with `modules-2.3`) 
beginning from the root directory of this repo.

.. code-block:: bash

    $ export VENV_DIR=~/venvs  # replace this with a good directory to install a virtual environment
    $ export AFQMC_ROOT_DIR=$(pwd)
    $ module load openmpi hdf5 python-mpi/3.11
    $ cd $VENV_DIR
    $ python -m venv --system-site-packages afqmctools
    $ source afqmctools/bin/activate
    $ cd $AFQMC_ROOT_DIR/utils
    $ pip install .[AUTOHF] # add -e to make editable
    # optional, install jax with gpu support
    # pip install -U "jax[cuda12]"


afqmctools as a library
_______________________

Additionally, the afqmctools library can be imported and used in Python:

.. code-block:: python

    from afqmctools.hamiltonian.mol import write_hamil_mol

    ... 

See the examples for more.
