# AFQMCTOOLS

A collection of python tools to aid in the running SAFIRE and alayzing the output.

## Python Library

The afqmctools library contains useful functions to help with generating the input for
AFQMC simulations. 
Currently, many converters use inputs from PySCF, but adaptations for
other software packages should be straightforward.


## Command-Line Tools

In addition to the afqmctools Python package,
 there are several scripts to automate the generation of simulation input (listed below in alphabetical order).
Each script has a help menu that can be
displayed using `$ [script name] --help`.

See the examples in SAFIRE/examples/afqmc for more details on using these scripts or pass
-h/--help to the scripts themselves.

### Supported CLI Tools

- `afqmc_to_fcidump` : converts from the SAFIRE format Hamiltonian format (in an HDF5 file) to FCIDUMP. Converts only the k-point factorized Hamltonian or the generic dense Hamiltonian format.
- `aimbes_to_2nd_quant` : Convert AIMBES Hamiltonians saved in an HDF5 checkpoint file
        to either an FCIDUMP file or an SAFIRE file.
- `dice_to_hdf5` : Convert from DICE wavefunction to a SAFIRE readable wavefunction format.
- `fcidump_spat2spin` : Convert FCIDUMP expressed in a spatial orbital basis to a FCIDUMP expressed in a spinor basis.
- `fcidump_to_afqmc` : generates an SAFIRE Hamiltonian from a plain text FCIDUMP. The integrals are assumed to be real and 8-fold symmetric.
- `make_model_ham` : Builds a lattice model Hamiltonian in SAFIRE format based on the settings in a .toml input file and saves to HDF5.
- `observables_stats` : Analyze stochastic samples of observables output by SAFIRE
- `pyscf_to_afqmc` : generates Hamiltonian and Wavefunction for SAFIRE input from a pyscf scf calculation.
- `scalar_stats` : Analyze stochastic samples of scalar data output by SAFIRE. One major use case is the analysis of the AFQMC energy, but other scalar data can be analyzed as well.
- `test_afqmc_input` : Sanity checks SAFIRE Hamiltonian saved in HDF5.
- `write_afqmc_json` : Writes an AFQMC json input file based on the settings provided.

### Deprecated CLI Tools

- `aimbes_to_afqmc` : (deprecated) : use `aimbes_to_2nd_quant` instead : Convert AIMBES Hamiltonians saved in an HDF5 checkpoint file to an SAFIRE file.


## Installation

***If you are on Rusty see below for easier installation!***

The afqmctools library, and the acompanying scripts, can be installed via the `pip` package manager.
It is recommended that afqmctools be installed in a virtual environmen as shown below.
Some of the afqmctools library requires a parallel build of hdf5 and the h5py library.
Insctructions for building the parrallel version of h5py can be found in the official 
[h5py documentation](https://docs.h5py.org/en/stable/mpi.html#building-against-parallel-hdf5).
Parrelel h5py should be installed before running `pip install .` below so that h5py is not installed
via pip (not built with mpi).
For an editable development build of afqmctools, replace the last line below with `pip install -e .`

```bash
$ git clone [ afqmc code repo ]
$ cd /path/to/venvs/
$ python -m venv afqmctools
$ . afqmctools/bin/activate
# (recommended!) install parallel h5py
$ cd [ afqmc code repo ]/utils
$ pip install .
```

All dependencies should be automatically installed via pip, and the convenience
scripts provided can be directly used from the command line from within the python virtual
environment. For example, 

```bash
$ . afqmctools/bin/active
(afqmctools) $ write_afqmc_json --help
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
```

will display the help menu for `write_afqmc_json`.

### Optional Depedencies

The afqmctools package has several optional depencies which enable additonal functionality.
Optional dependencies are organized into groups (see below) and can be installed via:
```
pip install .[GROUPNAME]
```
where GROUPNAME should be replaced by the name of the one of the groups below (but retaining the square brackets).

Current groups

| Group Name | installs | high-level overview of which features are enabled |
|---|---|---|
| PYSCF | pyscf>=2.4.0 | enables converters to/from PySCF formats, building waveunctions/Hamiltonians in Quantum Chemistry basis sets |
| TESTING | pytest-html, coverage | enables .html reports of pyest results and of code coverage |
| DOCS | sphinx, sphinxcontrib-bibtex, sphinx-rtd-theme, myst_nb, sphinx-autodoc-typehints, numpydoc | enables compiling docs with sphinx |

### Rusty Installation

If you are connected to Rusty, you can do the following (tested with `modules-2.3`) 
beginning from the root directory of this repo.

```bash
export VENV_DIR=~/venvs  # replace this with a good directory to install a virtual enviroment
export AFQMC_ROOT_DIR=$(pwd)
module load openmpi hdf5 python-mpi/3.11
cd $VENV_DIR
python -m venv --system-site-packages afqmctools
source afqmctools/bin/activate
cd $AFQMC_ROOT_DIR/utils
pip install . AutoHF # add -e to make editable
# optional, install jax with gpu support
# pip install -U "jax[cuda12]" 
```

## afqmctools as a library

Additionally, the afqmctools library can be imported and used in Python:

```python
from afqmctools.hamiltonian.mol import write_hamil_mol

... 
```

## Tests

To run all unit tests,  navigate to `/utils`, and run `$ pytest`.
In addition to command line output, an html report will be generated in `utils/.htmlpytest/pytest.html` which can be viewed in a browser.
Unit tests are grouped and specific groups can be turned on/off from the 
command line using the `-m` switch.
For example, all tests related to PySCF can be run using `$ pytest -m pyscf`,
and we can avoid running slow tests using `$ pyest -m "not slow"`.
Test groups can be selected / deselected using `and` / `or` as well.
For example, `$ pytest -m "pwscf and not slow"` would run tests related to `pyscf`, but
would not run those that are also marked as `slow`

Currently, the following test groups exists:

- `slow` : tests that run slowly
- `pyscf` : tests that rely on PySCF - useful for confirming that afqmctools is compatible with the currently installed pyscf version.
- `mpi` : marks tests that require MPI at the Python level
- `past` : tests related to backwards-compatibility with older i/o formats
- `functional` : marks tests as related to functional requiements
- `dev` : marks tests that are in developement
- `debug` : used only for debugging tests

Additional marks exits to specify the test frequency:

- `push` : tests to run every push
- `weekly` : tests to run evry week

## Model Hamiltonian Builder

TODO: this is in the docs now, and is too specific for here. ensure that everything is in the docs and remove!

TODO: explain the combined (lattice,band) indices

The afqmctools Python module includes tooling to build general Hubbard-Kanamori Hamiltonians of the type:

$$
\hat{H} = 
\sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'} \\
+ \sum_{i} U_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} \\
+ \sum_{i<j} U_{ij}^1 (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} ) \\
+ \sum_{i<j} U_{ij}^2 (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} ) \\
+ \sum_{i<j} J_{ij} ( 
  \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
  +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} 
  +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
  +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
  ),
$$
where $i$,$j$ are combined lattice and band indices,
$\hat{c}^\dagger_{i\sigma}$, $\hat{c}_{j\sigma'}$ create/anihilate and electron on the site (and band) corresponding to $i$/$j$ with spin $\sigma$/$\sigma'$,
$\hat{n}_{i\sigma}$ is the number operator,
$t^{\sigma \sigma'}_{ij}$ includes all one-body terms ($n^{th}$-order neighbor hopping, spin-orbit coupling, etc.),
$U$ is the traditional on-site hubbard interaction, 
$U^1$ is a density-density interaction (typically between bands on the same lattice site).

Currently, SAFIRE can use any form of this Hamiltonian. For convenience, we supply tooling to build the most common forms of the Hamiltonain described above. Only on-site, but inter-band, $U$,$U^1$,$U^2$, and $J$ are implemented; however, a motivated user can build a custom Hamiltonian and save it in the format described below in the section, "input file format"

### Building a Model Hamiltonian

This section explains how to build a model Hamiltonian using the included Python tooling.
Input format/conventions are described in more detail
in the section [Model Hamiltonian Builder Input Conventions](#model-hamiltonian-builder-input-conventions)

A Hamiltonian may be built from a Python `dict` as follows:

```python
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

params = {
    'hamiltonian' : {
        't' : 1.0,
        'U' : 2.0,
        'U1' : 1.5,
        'U2' : 1.0,
        'J' : 0.5
  },
  'lattice' : {
      'L1' : 6,
      'L2' : 1,
      'boundary1' : 'PBC',
  }
}

hamiltonian = HamiltonianBuilder.from_input(
    source = params
).hamiltonian

...

```

the Hamiltonian can then be saved in the afqmc format with:

```python
...

from afqmctools.utils.io import write_model_hamiltion

write_model_hamiltion(
    hamiltonian=hamiltonian,
        fname='afqmc.h5',
        nelec=(6,6)
)
```

Alternatively, the cli includes a tool to automatically build and save a Hamiltonian from parameters saved in an input file (toml format).
If `afqmctools` was installed as described in [Installation](#installation), a model Hamiltonian can be built using:

```bash
$ make_model_ham -i model_ham_params.json -o afqmc.h5 
```

For example, the following input files will all make the same Hamiltonian as above.

In TOML:

```toml
[hamiltonian]
nbands = 2
t = 1.0
U = 2.0
U1 = 1.5
U2 = 1.0
J = 0.5

[lattice]
L1 = 6
L2 = 1
boundary1 = "PBC"

[misc_params]
nelec = [6,6]
```

### Model Hamiltonian Builder Input Conventions

Model Hamiltonian parameters can be specified either directly
as a Python dict, or via an input file in toml format.
All cases share the following conventions.

First, the input is organized into a `hamiltonian` block - where Hamiltionan parameters are specified - and a `lattice` block - where the lattice that the Hamiltonian is defined on is specified.

The `hamiltonian` block consists of the following fields.
In the following, $i,j$ are combined indices including both lattice indices ($\mu$,$\nu$), and band indices ($m$,$m'$)

- `t` : specifies hopping terms, $\sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'}$. `t` may be any of:
    - None : no hopping : $t^{\sigma \sigma'}_{ij} = 0$
    - a single number : interpretted as the nearest-neighbor hopping. Assumes no hopping between band : $t^{\sigma \sigma'}_{ij} = t\delta_{mm`}\delta_{\nu,\mu\pm1}\delta_{\sigma\sigma`}$
    - a 1-D array of length $n-1$ : interpretted as multiple hopping terms where `t[n-1]` is the hopping strength between $n^{th}$-order neighbors. Assumes no hopping between bands : $t^{\sigma \sigma'}_{ij} = \sum_{n=1} t_n \delta_{mm`}\delta_{\nu,\mu\pm n}\delta_{\sigma\sigma`}$
    - a 2-D array with shape $N_{bands}$ x $N_{bands}$  : defines hopping between bands and/or allows different bands to use different hopping strengths. In this case, the same `t` array is used for all lattice sites. $t^{\sigma \sigma'}_{ij} = t_{mm`} \delta_{\nu,\nu\pm 1}\delta_{\sigma\sigma`}$
- `U` : specifies the in-band, on-site Hubbard interaction. May be any of:
    - None : no hubbard U : $ U_i = 0$
    - a single number : interpretted as a constant U across lattice sites and bands $ U_i = U \delta_{mm`}\delta_{\mu\nu} $
    - a 1-D array of length $N_{bands}$ : intepretted as $U_m$ where $m$ is the band index, but otherwise constant in lattice index. $ U_i = U_m \delta_{\mu\nu} $
    - a 1-D array of length $N_{sites}$ : intepretted as $U_\mu$ where $\mu$ is the lattice site index, but otherwise constant in band index. $ U_i = U_\mu \delta_{mm`}$
- `U1` : specifies the interband, on-site, density-density Hubbard interaction. $\sum_{i<j} U_{ij}^1 (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} )$
May be any of:
    - None : no density-density Hubbard interaction $U_{ij}^1 = 0$
    - a single number : interpretted as a constant U1 for all valid interband interactions, on the same site. $U_{ij}^1 = U^1 \delta_{\mu\nu} \forall m,m`$
    - a 2-D array with shape $N_{bands}$ x $N_{bands}$ : interpretted as the onsite, interband $U^1$ interaction which is the same for all sites. The diagonal is ignored since it should be specified via by the parameter `U`. $U_{ij}^1 = U^1_{mm`} \delta_{\mu\nu}$
- `U2` : specifies the interband, on-site spin-spin Hubbard interaction. It follows similar input conventions as `U1`, but is understood to describe interactions between electrons of the same spin. See `U1` for more. $\sum_{i<j} U_{ij}^2 (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} )$
- `J` : specifies the on-site Hund's interaction. It follows similar input conventions as `U1`, but is understood to describe interactions between electrons of the same spin. See `U1` for more. $\sum_{i<j} J_{ij} ( 
  \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
  +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} 
  +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
  +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
  )$

The `lattice` block consists of the following fields.
In the following, $\hat{a}_1$ and $\hat{a}_2$ are the lattice
vectors.

- `L1` : the number of lattice sites along $\hat{a}_1$.
- `L2` : the number of lattice sites along $\hat{a}_2$.
- (optional) `type` : either `square` or `triangular` : use either a square lattice or a triangular lattice.
- (optional) `boundary1` : either `'open'` or `'pbc'`. the type of boundary to use in the direction perpendicular to $\hat{a}_2$. `open` is the default.
- (optional) `boundary2` : either `'open'` or `'pbc'`. the type of boundary to use in the direction perpendicular to $\hat{a}_1$. `open` is the default.

### input file conventions


