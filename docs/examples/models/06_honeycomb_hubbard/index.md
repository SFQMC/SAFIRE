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

+++ {"id": "jFoBNSA0D-an"}

# Hubbard Model on a Honeycomb Lattice

## Set Up the Hamiltonian

```{code-cell} ipython3
:id: djut2ei2D6d-

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

# setup the Hamiltonian
from safiretools import Lattice
from safiretools import vis
from safiretools import LatticeHamiltonian

lattice = Lattice.from_type(
    "honeycomb", (4, 4), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)

# plot the lattice for reality checks!
vis.plot_lattice(lattice)

# make the Hamiltonian
hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=6.0)
```

```{code-cell} ipython3
:id: bYkD4o3sHwP9

# run autoHF and save wavefunction as trial wavefunction
import numpy as np

from autohf import solve_hf, AutoHFHamiltonian
from safiretools import Wavefunction

# Typically, we want to use a Ueff to generate a trial wavefunction
#    since HF's critical U is lower than the correct critical U
Ueff = 3.0

# make the effective Hamiltonian
effective_hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=Ueff)

# convert safiretools Hamiltonian to AutoHFHamiltonian
autohf_hamiltonian = AutoHFHamiltonian(effective_hamiltonian)

# set the number of electrons!!
nelec=(16-1,16-1)

settings = dict(
    ansatz = 'SD_ROT',
    steps = 100,
    nelec = nelec,
    batch_size = 100,
    seed = 42
)

results = solve_hf(hamiltonian=autohf_hamiltonian,settings=settings)

wavefunction = Wavefunction.from_single_determinant(
    tuple(np.asarray(o) for o in results[0]["orbitals"])
)
```

+++ {"id": "vYmHYe-nIm_Q"}

## Write Input File

For more information about the input file, see {doc}`../../../tutorials/models/02_understanding_the_input_file/02_understanding_the_input_file`.

```{code-cell} ipython3
:id: pm-N_EFxInNK

from safiretools import write_jobfile

# write input file
write_jobfile(
    "afqmc.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            "timestep": 0.01,
            "steps": 10000,
            "equilibration_steps": 500,
            "measure_interval": 10,             # in units of steps
            "population_control_interval": 10,  # in units of steps
            "walker_ortho_interval": 10,        # in units of steps
            "num_walkers": 1600
        }
    }
)
```

+++ {"id": "2umfVnQDJcfa"}

## Run AFQMC

```{code-cell} ipython3
---
id: xvJVj1DGJeW0
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

+++ {"id": "9YE2JAdlJesB"}

## Analyze the Results

```{code-cell} ipython3
:id: eADuTKxlJgfx

# analyze results
from safiretools import Results

results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
```
