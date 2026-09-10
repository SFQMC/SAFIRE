.. _afqmctools:

afqmctools
==========

afqmctools is a companion Python package that provides a set of tools for working with SAFIRE calculations, including utilities for 
writing Hamiltonian and trial wavefunction input HDF5 files,
generating input files, and analyzing output data.
It supplies both a command line interface (CLI) and a Python library that can be 
used directly in custom scripts.

Installation
------------

The afqmctools library, and the accompanying scripts, can be installed via the `pip` package manager.
It is recommended that afqmctools be installed in a virtual environment as shown below.
Some of the afqmctools library requires a parallel build of hdf5 and the h5py library.
Instructions for building the parallel version of h5py can be found in the official
`h5py documentation <https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5>`_.
Parallel h5py should be installed before running `pip install .` below so that h5py is not installed
via pip (not built with mpi).
For an editable development build of afqmctools, replace the last line below with `pip install -e .`

.. code-block:: bash

    $ git clone [ afqmc code repo ]
    $ cd /path/to/venvs
    $ python -m venv afqmctools
    $ . afqmctools/bin/activate
    # (recommended!) install parallel h5py
    $ cd [ afqmc code repo ]/utils
    $ pip install .


All dependencies should be automatically installed via pip, and the convenience
scripts provided can be directly used from the command line from within the python virtual
environment. For example, 

.. code-block:: bash

    $ . afqmctools/bin/active
    (afqmctools) $ write_afqmc_json --help


will display the help menu for `write_afqmc_json`.

.. code-block:: text

    usage: write_afqmc_json [-h] [--fout FOUT] [--fwfn FWFN] [--fham FHAM] [--verbose] [--steps STEPS] [--timestep TIMESTEP] [--n_walkers N_WALKERS] [--mixed_est] [--time_bp TIME_BP] [--num_bp NUM_BP]
                            [--path_restoration PATH_RESTORATION]

    Writes an AFQMC json input file based on the settings provided.

    options:
    -h, --help            show this help message and exit
    --fout FOUT, -o FOUT
    --fwfn FWFN, -i FWFN
    --fham FHAM, -ih FHAM
    --verbose, -v
    --steps STEPS, -s STEPS
    --timestep TIMESTEP, -ts TIMESTEP
    --n_walkers N_WALKERS, -nw N_WALKERS
                            number of walkers per MPI rank
    --mixed_est, -me
    --time_bp TIME_BP, -tbp TIME_BP
                            maximum back propagation time
    --num_bp NUM_BP, -nbp NUM_BP
    --path_restoration PATH_RESTORATION, -pr PATH_RESTORATION
                            path restoration type "0", "1", or "e" for no, yes, and extra
    

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

The afqmctools package has several optional dependencies which enable additional functionality.
Optional dependencies are organized into groups (see below) and can be installed via:

.. code-block:: bash

    $ pip install .[GROUPNAME]

where GROUPNAME should be replaced by the name of the one of the groups below (but retaining the square brackets).

Current groups:

DEV TODO: update the following table to reflect current state of optional dependencies

============ ========================== ============================================================================================================== 
 Group Name           installs                                        high-level overview of which features are enabled                                
============ ========================== ============================================================================================================== 
   AUTOHF     jax,optax,jaxlib, jaxopt                                    enables the autoHF lattice model HF Solver                                   
   PYSCF            pyscf>=2.4.0         enables converters to/from PySCF formats, building wavefunctions/Hamiltonians in Quantum Chemistry basis sets  
   TESTING      pytest-html, coverage                             enables .html reports of pytest results and of code coverage                           
============ ========================== ============================================================================================================== 


Command Line Interface (CLI)
-----------------------------

afqmctools provides several command line utilities for common tasks. These scripts are automatically
installed when you install the package and can be used directly from the command line.

.. toctree::
    :maxdepth: 2

    afqmctools_cli


Lattice Models
--------------

Unlike quantum chemistry, and ab initio solids, there are few mature software packages
for setting up lattice model Hamiltonians and trial wavefunctions.
safiretools provides a lattice builder, a general model Hamiltonian builder, and
trial wavefunction classes.
The Hamiltonian builder takes a general lattice instance as input and constructs
one of several model Hamiltonian terms including up to the Hubbard-Kanamori model.
See the following for more details.

.. toctree::
  :maxdepth: 2

  lattice_models

API reference
-------------

.. toctree::
  :maxdepth: 2

  api/afqmctools.rst


