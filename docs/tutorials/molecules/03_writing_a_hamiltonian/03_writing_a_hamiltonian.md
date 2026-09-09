---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: ipython3
  name: python3
---

+++ {"id": "ETj3fTltyCNc"}

# Writing a Hamiltonian File

<b>Goal:</b>
Become acquainted with how to write a Hamiltonian to the SAFIRE HDF5 format.

## What you will learn

1.  How to write a Hamiltonian to the SAFIRE HDF5 format starting from one-body, and two-body integrals
2.  How to convert from a FCIDUMP file to the SAFIRE HDF5 format both using the CLI tools and within a Python script
3.  (For convenience) How to use PySCF and safiretools to generate and write the Hamiltonian

## Introduction

In previous tutorials, we learned how to run SAFIRE, and how
to post process the results to arrive at the AFQMC energy.
Now, we will learn how to write a Hamiltonian in SAFIRE's HDF5 format.

<!--
<div>
<img src="./figs/QChemWorkflow_v2.png" width="1000"/>
</div>
-->

![](files/TopLevelFlowChart_v3.png)

In this tutorial,
we will compute the ground state energy of
the minimal basis hydrogen dimer with a bond length of 1.4 Bohr radii. 

<!--
<div>
<img src="./figs/HyrdogenMolecule.png" width="500"/>
</div>
--->

![](files/HyrdogenMolecule.png)

+++ {"id": "Xg9PlaJByCNd"}

**Run the following code block to setup the tutorial.**

```{code-cell} ipython3
:id: zMSdfeQqyCNd

from pathlib import Path


from safiretools import MolecularHamiltonian
from safiretools import NOMSDWavefunction
from afqmctools.inputs.from_hdf import write_json

from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "J8ieYi3YyCNe"}

##  Setting up

As we have seen in previous tutorials, the SAFIRE executable is implemented for generic second quantized Hamiltonians, and wavefunctions.
Before we can run an AFQMC calculation, we must generate:

1. A Hamiltonian
2. A trial wavefunction
3. A json input file

Implicit in items 1. and 2. is the need for an orthonormal basis in
which to compute matrix elements of the Hamiltonian,
and to represent the trial wavefunction in terms of.

In this tutorial, we will use the set of canonical Hartree-Fock orbitals
from RHF as a basis.

+++ {"id": "RsbyXg-OyCNe"}

## The Hamiltonian

We are often interested in the electronic ground state of the Born-Oppenheimer Hamiltonian which is given in first quantization
by
$$
H_\mathrm{Born-Oppenheimer} = \sum_p^{N_e}\left(  -\frac{1}{2} \nabla^2_p + \sum_A^{N_{A}} \frac{Z_A}{ | \vec{R}_A - \vec{r}_p | }  \right) + \frac{1}{2} \sum^{N_e}_{q \neq p} \left(\frac{1}{r_{pq}}\right) + \frac{1}{2}\sum_{A' \neq A}^{N_A} \frac{Z_{A} Z_{A'}}{| R_A - R_{A'} |},
$$
where,
$N_e$ is the number of electrons,
$\vec{r}_p$ is the position of electron $p$,
$N_{A}$ is the number of atomic nuclei,
and $Z_A$ and $\vec{R}_A$ are the atomic number and position, respectively, of atomic nuclei, $A$.

AFQMC is formulated in terms of the second quantized Hamiltonian.
In second quantization, we express the Hamiltonian in terms of electronic creation/annihilation operators
, $\hat{c}^{\dagger}_i$ / $\hat{c}_i$, which create/annihilate an electron in the chosen basis set orbitals,
$\{ \phi_i (\vec{r})\}$.
The Hamiltonian takes the form,
$$
\hat{H} = H^0 + \sum_{ij} H^1_{ij} \hat{c}^{\dagger}_i\hat{c}_j + \frac{1}{2}\sum_{j \neq i} H^2_{ijkl} \hat{c}^{\dagger}_i \hat{c}^{\dagger}_j \hat{c}_k \hat{c}_l
$$
where
$$
H^0 = \frac{1}{2}\sum_{A' \neq A}^{N_A} \frac{Z_{A} Z_{A'}}{| R_A - R_{A'} |},
$$
is the constant nuclear repulsion energy,
$$
H^1_{ij} = \int d\vec{r}_p  \phi^*_i (\vec{r}_p) \left(  -\frac{1}{2} \nabla^2_p + \sum_A^{N_{atom}} \frac{Z_A}{ | \vec{R}_A - \vec{r}_p | }  \right) \phi_j (\vec{r}_p),
$$
contains the "one-body" kinetic term and electron-nuclei interactions,
and
$$
H^2_{ijkl} = (il|jk) = \frac{1}{2} \int \int d\vec{r}_p d\vec{r}_q  \left( \phi^*_i (\vec{r}_p)  \phi_l (\vec{r}_p)  \frac{1}{|\vec{r}_p - \vec{r}_q|}  \phi^*_j (\vec{r}_q) \phi_k (\vec{r}_q)  \right)
$$
is the electron-electron Coulomb interaction.

Typically, an external quantum chemistry code will be used to generate integrals.
For this tutorial, we will use the minimal basis Hydrogen dimer since it can be easily written down by hand.
See, for example, section 3.5.2 in ref. 1.

The code block below contains the Hamiltonian for the $H_2$ molecule at a
bond length of $\delta_{H-H} = 1.4$ Bohr radii in a standard sto-3g basis.
**Run the code block below to load the Hamiltonian into memory**

1. Szabo, A., & Ostlund, N. S. (1996). Modern quantum chemistry: Introduction to advanced electronic structure theory. Dover Publications.

```{code-cell} ipython3
:id: gCidZeInyCNe

import numpy as np

from safiretools import MolecularHamiltonian

number_of_electrons = (1,1) # up, down
number_of_orbitals = 2

# all energy units are E_Hartree

# nuclear repulsion energy
H0 = 0.714285714285714

# One-Body Integrals in the basis of canonical RHF orbitals
H1_ij = np.array([[-1.25279706e+00,  0.0           ],
                  [ 0.0           , -4.75602313e-01]])


# two-body integrals in the basis of canonical RHF orbitals
H2_ijkl = np.zeros((2,2,2,2))
H2_ijkl[0,0,:,:] = np.array([[ 6.74976525e-01, 0.0],
                             [0.0, 6.63461552e-01]])

H2_ijkl[0,1,:,:] = np.array([[ 0.0,  1.81606342e-01],
                             [ 1.81606342e-01, 0.0]])
# equivalent by symmetry
H2_ijkl[1,0,:,:] = H2_ijkl[0,1,:,:]
H2_ijkl[1,1,:,:] = np.array([[6.63461552e-01, 0.0],
                             [0.0, 6.96162129e-01]])


# For reference, the RHF orbitals in the STO-3G basis are
# phi_i(\vec{r}) = \sum_\mu C_{\mu i} g_\mu (\vec{r})
phi_hf = np.array([[ 0.54893404,  1.21146407],
                   [ 0.54893404, -1.21146407]])

# and the STO-3G basis overlap matrix for $\delta_{H-H}=1.4 Bohr$ is
# Sij = <g_\mu | g_\nu>
Sij = np.array([[1.,         0.65931821],
                [0.65931821, 1.        ]])
```

+++ {"id": "grY1VTDdyCNe"}

### Save the Hamiltonian

`safiretools` provides Hamiltonian classes that write themselves into an HDF5 file that can be read by the SAFIRE executable, as
shown in the code block below.
AFQMC uses a factorized form of the electron-electron interaction tensor,

$$
 H^2_{ijkl} = \sum_\gamma L^{\gamma}_{il}L^{\gamma}_{jk}.
$$
Any decomposition of this form can be used in AFQMC.
Density fitting integrals, for example can be written in this form.
A typical strategy in the AFQMC literature is to perform a Cholesky decomposition of the super-matrix $H^2_{(i,l),(j,k)}$
using a Cholesky tolerance of $\delta_{Chol}$ where a smaller
value of $\delta_{Chol}$ produces a more accurate representation.

The Cholesky decomposition is automatically performed within
`MolecularHamiltonian.from_integrals()` if you provide it with the electron Coulomb
interaction tensor via the `eri` argument.
Alternatively, 3-index integrals (from density fitting, for example) can be provided to use
directly with no further modifications.

**Run the following codeblock to output the Hamiltonian to an HDF5 file**.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: JyCauZJfyCNe
outputId: 5e7b4b17-4e40-4189-ac2e-633cfb904805
---
hamiltonian = MolecularHamiltonian.from_integrals(
    hcore=H1_ij,
    eri=H2_ijkl,
    enuc=H0,
    nelec=number_of_electrons
)

hamiltonian.to_hdf5(scratch_dir/"hamil.h5")
```

+++ {"id": "FL2jWTrTyCNf"}

## The Trial Wavefunction

We will explore how to write trial wavefunctions in {doc}`../04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction`.
In general, the trial wavefunction is a linear combination of Slater determinants,
$$
| \Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle,
$$
where $C_n$ is a coefficient,
and $|\Phi_n\rangle$ are Slater determinants which are not necessarily
orthogonal to each other.

**For now, simply run the next code block to save the RHF ground state
in the same HDF5 file as the Hamiltonian.**

```{code-cell} ipython3
:id: Jn_ITkJiyCNf

import numpy as np

from safiretools import NOMSDWavefunction
from afqmctools.inputs.from_hdf import write_json

# RHF orbtials represented in the basis of RHF orbitals
orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)

phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "wfn.h5")

write_json(scratch_dir / "afqmc.json", scratch_dir / "wfn.h5", scratch_dir / "hamil.h5", exec_opts=dict(timestep=0.005, steps=20000))
```

+++ {"id": "CXQ3jT_pyCNf"}

## Your Turn: Compute the AFQMC energy

At this point, you should run an AFQMC calculation using SAFIRE
as we covered earlier in {doc}`../01_hello_safire/01_hello_safire`,
and use `afqmctools` to obtain the AFQMC energy.

As a reminder, you can run SAFIRE in any of the following ways:

1. the `run_afqmc()` convenience wrapper - we've included a template in the next code cell.
2. run from the command line as `$ mpirun -np [number of MPI tasks] safire --filenames [input file].json`
3. (if you are running on a computing cluster) submit a job via a runscript.

As an additional reminder, the `analyze_scalar_data` tool can be invoked by either

1. calling it from the CLI as `$ scalar_stats [id].s[series].scalar.dat`
2. in a Python script / notebook with `analyze_scalar_data()` - we've also included a template in a code block below that you can use for this exercise.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: VhPYIHxeyCNf
outputId: f8c272ce-fe42-4727-be4e-2abd163c3a79
---
# Use this to run SAFIRE
run_afqmc(
    run_dir=scratch_dir,
    input_file = "afqmc.json",#"your_input_file.json",
    np=16,             # number of MPI tasks
    output_file=None,
)
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 904
id: heaiDcvRyCNg
outputId: 703d5b1c-4596-4fdd-e544-3708040007de
---
from stats.scalar_dat import analyze_scalar_data

settings = dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",
    nequil = 5,     # remember to check the equilibration!
    trace = True
)

E, dE = analyze_scalar_data(settings)
E_exact = -1.1372759436170439
(E-E_exact)/dE
```

+++ {"id": "U9KLdBoByCNg"}

### Check your result

Your AFQMC energy should agree with $-1.137024 \pm 0.000195$ Ha
to within 1-2 $\sigma$.

For comparison, the FCI energy is $-1.1372759436170439$ Ha.

Our AFQMC result agrees FCI to within 0.0003(2) Ha which
is well within chemical accuracy.

+++ {"id": "jnupX3Uz1l4q"}

## Starting from a FCIDUMP

`afqmctools` provides tools to convert from a Hamiltonian in a standard
FCIDUMP file to the format used by SAFIRE.

We will continue with the example of a Hydrogen dimer in an STO-3G basis to demonstrate the tooling.

![](files/HyrdogenMolecule.png)

Instructions for generating a FCIDUMP are beyond the scope of this tutorial;
however, quantum chemistry codes that can easily generate FCIDUMP files are ubiquitous.

For simplicity, we provide a FCIDUMP file which, starting from the GitHub repo root, can be found in,

```bash
    tutorial/molecules/03_writing_a_hamiltonian/files/H2_FCIDUMP
```

+++ {"id": "qYRe4bEt1xSN"}

## CLI: Covert from FCIDUMP to the SAFIRE format.

The `fcidump_to_afqmc` CLI script can be used to directly convert
from a FCIDUMP file to the HDF5 format used by SAFIRE.
If you follow the official installation instructions for `afqmctools`, then
`fcidump_to_afqmc` should already be installed.

It can be used as

```{code-cell} ipython3
!fcidump_to_afqmc -i files/H2_FCIDUMP  -t 1.0e-5 -o data/H2_Hamiltonian.h5
```

+++ {"id": "qYRe4bEt1xSN"}

which reads the "input" (`-i`) Hamiltonian from `H2_FCIDUMP`,
performs a Cholesky decomposition with a tolerance (`-t`) of `1.e-5`,
and saves the Hamiltonian to the "output" (`-o`) HDF5 file, ` H2_Hamiltonian.h5`.

Additionally, you can use the `--help` option to see what other options are available.

**Your Turn: Run the next code block to run see the help message**

```{code-cell} ipython3
:id: 7qfvZqBL14Vt

!fcidump_to_afqmc --help
```

+++ {"id": "pX0yckNd2j9A"}

### CLI: Also write a trial wavefunction

The next input that we will need for AFQMC is a trial wavefunction.
The `fcidump_to_afqmc` CLI script can also write either an RHF- or ROHF-like
trial wavefunction to the SAFIRE HDF5 file.

The wavefunction is a single Slater determinant in which the first $N_{\uparrow}$/$N_{\downarrow}$ *by index*
orbitals are occupied.
$N_{\uparrow}$ and $N_{\downarrow}$ are determined by the information in the header of the FCIDUMP file.
It is also possible to explicitly specify orbital occupancies with the `--occ_up` and `--occ_down` switches.

### Your Turn
Run the following codeblock to output the Wavefunction to an HDF5 file via the CLI.

```{code-cell} ipython3
:id: 8JgG5DbH24kR

# Use the Command Line Interface (CLI) to convert the Hamiltonian AND write a trial wavefunction
!fcidump_to_afqmc -i files/H2_FCIDUMP -o {scratch_dir}/H2.h5 --add_wfn --occ_up 0 --occ_down 0
```

+++ {"id": "7NLzntps2CS7"}

## Python: Covert from FCIDUMP to the SAFIRE format.

The `safiretools` Python package provides the following, which allow us to convert from FCIDUMP to
the SAFIRE format.

- `read_fcidump()`, re-exported at the top level as `safiretools.read_fcidump`
- `MolecularHamiltonian.from_integrals()`, followed by `.to_hdf5()`

### read_fcidump()

The `read_fcidump()` function reads the Hamiltonian from the FCIDUMP file
with name `filename`, and separately returns the 1-body, 2-body, and constant Hamiltonian
terms as well as the number of electrons.

```python
    from safiretools import read_fcidump

    H1_ij, H2_ijkl, E0, _ = read_fcidump(
        filename="H2_FCIDUMP"
    )
```

see the {doc}`Hamiltonian reference <../../../afqmctools/lattice_models>` for more.

### MolecularHamiltonian.from_integrals()

As we saw above, `MolecularHamiltonian.from_integrals()` builds a Hamiltonian that
`.to_hdf5()` writes in the form the SAFIRE executable reads.
It will automatically generate a Cholesky decomposed form of the interaction if electron-repulsion integrals are provided via the `eri` keyword argument.
Alternatively, any 3-index factorized form of the electron interaction can be provided
via the `chol` input parameter.
Exactly one of `chol` or `eri` must be provided.

```python
    import numpy as np

    from safiretools import MolecularHamiltonian


    # transopose to match the eri convention from_integrals() expects
    H2_ijkl = numpy.transpose(H2_ijkl,(0,1,3,2))

    MolecularHamiltonian.from_integrals(
        hcore=H1_ij,
        eri=H2_ijkl,
        enuc=H0,
        cholesky_tol=1.0e-5
    ).to_hdf5(scratch_dir/"H2_hamiltonian.h5")
```

see the {doc}`Hamiltonian reference <../../../afqmctools/lattice_models>` for more.

### Your Turn: Run the following code block to convert from the provided FCIDUMP file to the SAFIRE format

```{code-cell} ipython3
:id: 6ej77YPU2JmI

import numpy as np
from safiretools import MolecularHamiltonian, read_fcidump

H1_ij, H2_ijkl, E0, _ = read_fcidump(
    filename="files/H2_FCIDUMP"
)

# transopose to match the eri convention from_integrals() expects
H2_ijkl = np.transpose(H2_ijkl,(0,1,3,2))

MolecularHamiltonian.from_integrals(
    hcore=H1_ij,
    eri=H2_ijkl,
    enuc=E0,
    cholesky_tol=1.0e-5
).to_hdf5(scratch_dir/"H2_hamiltonian.h5")
```

+++ {"id": "jNDV-Ersu9ul"}

## Generate and write the Hamiltonian using PySCF and safiretools

Instead of defining the Hamiltonian and the wavefunction from scratch, one can also utilize `PySCF` to obtain the Hamiltonian in a given basis (in this case, the molecular orbital basis) and the trial wavefunction.
Here, we use `PySCF` to run RHF for the hydrogen dimer and export to a chkfile `h2_rhf.h5`, then use `safiretools` to obtain the Hamiltonian and the trial wavefunction from the chkfile.
After this, you may follow the same steps as above to run AFQMC.

```{code-cell} ipython3
:id: TMmTH-7SvFd8

from pyscf import gto, scf

# build the H2 dimer
xyz = "H 0. 0. 0.; H 0. 0. 1.4"
mol = gto.Mole(unit = "Bohr",
            atom = xyz,
            precision = 1e-6,
            basis = "sto-3g")

mol.build()

# run RHF
mf = scf.RHF(mol)
mf.chkfile = f"{scratch_dir}/h2_rhf.h5"
mf.kernel()

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from safiretools import Wavefunction
from safiretools import MolecularHamiltonian

# reload the mf data
mf_data = load_from_pyscf_chk_mol(f"{scratch_dir}/h2_rhf.h5", 'scf')

# dump the Hamiltonian and the wavefunction to the same h5df
fout = f"{scratch_dir}/h2_from_pyscf.h5"
MolecularHamiltonian.from_pyscf(
    mf_data,
    chol_cut=1e-6,
    real_chol=True,
    verbose=True
).to_hdf5(fout)

Wavefunction.from_pyscf(
    mf_data,
    basis_scf_data=mf_data
).to_hdf5(fout)
```

+++ {"id": "ZX0-iQdzyCNg"}

## Summary

In this tutorial, you set up and ran a quantum chemistry AFQMC calculation in which we computed the ground state energy of the Hydrogen dimer at a fixed bond length.

The AFQMC code needs a Hamiltonian, a trial wavefunction, and a file containing settings as input, and it outputs stochastic samples as specified in the input file.

### What you learned

1. How to write a Hamiltonian to the SAFIRE HDF5 format starting from one-body, and two-body integrals
2. How to convert from a FCIDUMP file to the SAFIRE HDF5 format both using the CLI tools and within a Python script

+++ {"id": "NaDbzk1jSEB1"}

### Sample Script - Full workflow

Below is a single script based on what you have learned.

```{code-cell} ipython3
:id: 8FhPN8ETyCNg

from pathlib import Path

import numpy as np

from safiretools import MolecularHamiltonian
from safiretools import NOMSDWavefunction
from afqmctools.inputs.from_hdf import write_json
from stats.scalar_dat import analyze_scalar_data

from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

number_of_electrons = (1,1) # up, down
number_of_orbitals = 2

# all energy units are E_Hartree

# nuclear repulsion energy
H0 = 0.714285714285714

# One-Body Integrals in the basis of canonical RHF orbitals
H1_ij = np.array([[-1.25279706e+00,  0.0           ],
                  [ 0.0           , -4.75602313e-01]])

# two-body integrals in the basis of canonical RHF orbitals
H2_ijkl[0,0,:,:] = np.array([[ 6.74976525e-01, 0.0],
                             [0.0, 6.63461552e-01]])

H2_ijkl[0,1,:,:] = np.array([[ 0.0,  1.81606342e-01],
                             [ 1.81606342e-01, 0.0]])
# equivalent by symmetry
H2_ijkl[1,0,:,:] = H2_ijkl[0,1,:,:]
H2_ijkl[1,1,:,:] = np.array([[6.63461552e-01, 0.0],
                             [0.0, 6.96162129e-01]])



# For reference, the RHF orbitals in the STO-3G basis are
# phi_i(\vec{r}) = \sum_\mu C_{\mu i} g_\mu (\vec{r})
phi_hf = np.array([[ 0.54893404,  1.21146407],
                   [ 0.54893404, -1.21146407]])

# and the STO-3G basis overlap matrix for $\delta_{H-H}=1.4 Bohr$ is
# Sij = <g_\mu | g_\nu>
Sij = np.array([[1.,         0.65931821],
                [0.65931821, 1.        ]])

# Save the Hamiltonian to HDF5
MolecularHamiltonian.from_integrals(
    hcore=H1_ij,
    eri=H2_ijkl,
    enuc=H0,
    nelec=number_of_electrons
).to_hdf5(scratch_dir/"H2_hamiltonian.h5")

# make and save a trial wavefunction
orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)

phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0
NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/"H2_rhf_wfn.h5")  # this is the name of the file that will be created

# make an input file

afqmc_execution_options = {
    "timestep": 0.01,
    "steps": 10000,
    "population_control_interval" : 10, # in units of steps
    "measure_interval_multiplier": 1,   # in units of population_control_interval
    "walker_ortho_interval" : 10 ,      # in units of steps
    "n_walkers_per_mpi_task": 200,
    "seed" : 42
}

write_json(
    scratch_dir/ "H2_afqmc_input.json",
    fham0=scratch_dir / "H2_hamiltonian.h5",
    fwfn0=scratch_dir / "H2_rhf_wfn.h5",
    exec_opts=afqmc_execution_options
)

# run afqmc
run_afqmc(
    run_dir=scratch_dir,
    run_mode="local_cpu",
    input_file="H2_afqmc_input.json",
    np=12,             # number of MPI tasks
    output_file=None   # optionally direct output to file instead of here
)

# post process

settings = dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",
    nequil = 5,
    trace = True
)

_ = analyze_scalar_data(settings)
```

```{code-cell} ipython3
:id: 1dykyPFNyCNg


```
