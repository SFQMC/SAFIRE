.. _install:

Installation
============

The installation is performed in two essentially independent steps.
The SAFIRE executable must be compiled,
and the safiretools python package can optionally be installed.

Dependencies
------------

Ensure the following dependencies are installed

- CMake ≥3.25
- C++20 compiler
- MPI
- HDF5
- Boost ≥1.70.0
- BLAS, LAPACK
- Intel oneAPI MKL (for sparse matrices on CPU)

The build system will fetch the following dependencies automatically if they are not installed

- `nda <https://github.com/TRIQS/nda>`_
- cxxopts
- spdlog
- cpptrace
- nlohmann_json
- Catch2 (for tests)

For NVIDIA GPU support, you additionally need

- CUDA ≥12, compute capability ≥8.0
- `cuTENSOR <https://developer.nvidia.com/cutensor>`__
- CCCL (fetched automatically)

Getting SAFIRE
--------------

You can download SAFIRE from `GitHub <https://github.com/SFQMC/SAFIRE>`__, either through the latest release tarball or

.. code-block::

    git clone https://github.com/SFQMC/SAFIRE.git

Compiling SAFIRE
----------------

Assuming that all dependencies are installed and available, go to the SAFIRE source directory and run

.. parsed-literal::

    $ cmake -S . -B build \\
            -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install
    $ cmake --build build -j 10
    $ cmake --install build

This installs the ``safire`` executable to ``build/install/bin``.
To compile with NVIDIA GPU support use the following:

.. parsed-literal::

    $ cmake -S . -B build \\
            -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install \\
            **-DENABLE_CUDA=ON**
    $ cmake --build build -j 10
    $ cmake --install build

If CMake does not find `cuTENSOR <https://developer.nvidia.com/cutensor>`__, point it to the unpacked archive with ``-DCutensor_ROOT=/path/to/libcutensor``.

The GPU build can run both CPU and GPU calculations using the ``--compute`` CLI argument.

.. admonition:: For Flatiron Users
    :class: note

    The following snippet will compile SAFIRE in the Rusty environment. Ensure these modules are also loaded whenever you are running SAFIRE.

    .. code-block::

        module purge
        module load slurm cmake gcc hdf5 boost intel-oneapi-mkl
        module load openmpi            # CPU-only build, or
        module load cuda openmpi/cuda  # GPU build

        cmake -S . -B build \
              -DCMAKE_INSTALL_PREFIX=$(pwd)/build/install \
              -DENABLE_CUDA=ON -DCutensor_ROOT=... # GPU build only
        cmake --build build -j 10
        cmake --install build


Installing Companion Software
-----------------------------

It is highly recommended you install :ref:`safiretools`, the companion Python package that simplifies
the generation of input files and the processing of output data.

.. code-block::

    cd utils # TODO update once package is public
    pip install ".[vis]" # [vis]ualization support

For lattice model calculations, the `AutoHF <https://github.com/SFQMC/AutoHF>`__ Hartree-Fock code is very useful.

.. code-block::

    pip install sfqmc-autohf

For ab-initio solids calculations, see `CoQuí <https://github.com/AbInitioQHub/coqui>`__ for installation instructions.

For chemistry, SAFIRE can interface with various quantum chemistry codes. If you want to use the PySCF backend, install

.. code-block::

   pip install pyscf

Next Steps
----------

You should now be ready for your first calculation. Continue to the :ref:`tutorials`.
