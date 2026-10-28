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

+++ {"id": "-CSom7nd5xcl"}

# AutoHF Crash Course

<b>Goal:</b>
Learn the basics of using AutoHF to generate Hartree-Fock trial wavefunctions for SAFIRE.

## What You Will Learn

1. How to find the Hartree-Fock solution of a lattice model Hamiltonian with AutoHF
2. How to turn the AutoHF result into a SAFIRE trial wavefunction
3. How to check the charge and spin densities of the Hartree-Fock solution

## Software Preliminaries

For this tutorial, you need to install AutoHF (see {ref}`install`).

```bash
pip install sfqmc-autohf
```

+++ {"id": "dfgyQ5OC5xcn"}

## Set Up the Model

In this tutorial, we will perform Hartree-Fock calculations using AutoHF for the Hubbard model on
a 4×4 lattice, which we learned to set up in {doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`.

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

```{code-cell} ipython3
---
id: tNji__B25xcn
colab:
  base_uri: https://localhost:8080/
  height: 979
outputId: 20f4b568-da4f-4137-fb6f-575426ee0d09
---
from safiretools import Lattice, LatticeHamiltonian
from safiretools import vis

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))

vis.plot_lattice(lattice)

hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t=1.0,
    U=4.0,
)
```

+++ {"id": "tsMzr8175xcp"}

## Obtaining a Hartree-Fock Solution with AutoHF

AutoHF takes two inputs: the Hamiltonian, converted from the safiretools format by `AutoHFHamiltonian`, and a dictionary of settings.
Here, `nelec` puts 5 spin-up and 5 spin-down electrons on the 16 sites, and `ansatz='SD'` together with `noncollinear=True` optimizes a single Slater determinant whose orbitals may mix spin up and spin down.
Because the optimization can get stuck in local minima, AutoHF runs `batch_size` independent trials from randomly perturbed starting points and keeps the one with the lowest energy.

```{code-cell} ipython3
from autohf import solve_hf, AutoHFHamiltonian

# convert SAFIRE Hamiltonian to the AutoHF Hamiltonian format
hamiltonian_autohf = AutoHFHamiltonian(hamiltonian)

# define settings
hf_settings = dict(
    ansatz = 'SD',
    steps = 100,
    nelec = (5,5),
    batch_size = 30,
    seed = 42,
    measure_spin = True,
    random_initial_phi0 = False,
    noncollinear = True,
    verbose = False
)

results = solve_hf(
    hamiltonian=hamiltonian_autohf,
    settings=hf_settings
)
print("E_HF =", results[0]["E"])
```

You should get a Hartree-Fock energy of $-17.75\,t$ (to within about $10^{-8}\,t$).

+++ {"id": "4TZMwRGlXmA_"}

## Turning the AutoHF Result into a SAFIRE Wavefunction

```{code-cell} ipython3
:id: fFMgfc_zU4oW

import numpy as np
from safiretools import Wavefunction

# the noncollinear orbitals are spinors, spin-up components first
orbitals, = results[0]["orbitals"]
wavefunction = Wavefunction.from_single_determinant(
    np.asarray(orbitals).reshape(2, lattice.N_sites, sum(hf_settings["nelec"]))
)
```

+++ {"id": "SHSd-ZC2Zopm"}

## Analyzing the HF Wavefunction

It is very important to analyze the HF solution
before using it as a trial wavefunction to ensure that it has the
desired properties. Here, we plot its charge and spin densities.

```{code-cell} ipython3
---
id: SbsGsL0h5xcp
colab:
  base_uri: https://localhost:8080/
  height: 701
outputId: 1bc6a04c-37b4-4a2c-d2bc-c82f193ee745
---
# get the density from autohf
state_results, functions = results

makeRDMs = functions['makeRDMs']
state = state_results['state']

rdm = makeRDMs(state)

# get lattice dimensions from the Lattice instance
L = lattice.L

# the noncollinear 1RDM has the spin-up block first
M = lattice.N_sites
rho_up = rdm[:M,:M].diagonal().reshape(*L)
rho_down = rdm[M:,M:].diagonal().reshape(*L)

rho_charge = rho_up + rho_down
rho_spin = rho_up - rho_down

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    vmin=0.0,
    vmax=2.0
)
```

```{code-cell} ipython3
---
id: BjfjakIz5xcp
colab:
  base_uri: https://localhost:8080/
  height: 690
outputId: c19b35c5-7840-45a6-e527-d9f4f27d12b5
---
vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    vmin=-0.5,
    vmax=0.5
)
```

## Summary

In this tutorial, you learned the basics of using AutoHF to generate Hartree-Fock trial wavefunctions for SAFIRE.

### What You Learned

1. How to find the Hartree-Fock solution of a lattice model Hamiltonian with AutoHF
2. How to turn the AutoHF result into a SAFIRE trial wavefunction
3. How to check the charge and spin densities of the Hartree-Fock solution

## Further Reading

The examples in {ref}`examples_models` contain many more advanced applications of AutoHF, including models that show stripe ground states and how to apply pinning fields. The [AutoHF documentation](https://safire.flatironinstitute.org/autohf/dev/) is a good resource to learn more about the detailed settings of AutoHF.
