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

```{code-cell} ipython3
from pathlib import Path
from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "0WrZ0hFmBbdm"}

# Multi-Slater determinant trial wavefunction

In this example, we will compute the AFQMC energy of the Hubbard model with U=8 on a 6x6x square lattice with open boundary conditions
at half-filling.
We will use a trial wavefunction consisting of two Slater determinants.
Specifically, we will use

$$
| \Psi_\mathrm{T} \rangle = \frac{1}{\sqrt{2}} \left( |\Phi_{0}\rangle + |\Phi_{1}\rangle \right)
$$

where

$$
|\Phi_{0}\rangle = |\Phi^{UHF}\rangle = |\Phi^{\uparrow}\rangle \otimes |\Phi^{\downarrow}\rangle
$$

is the UHF ground state
and $|\Phi_{1}\rangle$ is the spin-reversed UHF ground state,

$$
|\Phi_{1}\rangle = |\Phi^{\downarrow}\rangle \otimes |\Phi^{\uparrow}\rangle.
$$

Since the Hamiltonian is spin-independent, this trial wavefunction should have the same energy as the UHF ground state.

```{code-cell} ipython3
---
id: 1rz5sWn3Ecsc
outputId: 51107f5c-4186-4817-9e97-990b46c79acc
---
# setup the Hamiltonian
import numpy as np

from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.visualize import plot_lattice
from afqmctools.hamiltonian.model.director import HamiltonianDirector
import afqmctools.utils.io as io

lattice = get_lattice(
    params=dict(
        L1 = 6,
        L2 = 6,
        boundary1 = "open",
        boundary2 = "open"
    )
)

# plot the lattice for reality checks!
plot_lattice(lattice)

# define Hamiltonian parameters
hamiltonian_params = {
    'U': 8.0
}

params = {
    'hamiltonian': hamiltonian_params
}

# make the Hamiltonian
hamiltonian = HamiltonianDirector(
    source=params,
    lattice=lattice
).build()

# save for AFQMC
io.write_model_hamiltonian(hamiltonian,fname=scratch_dir/"afqmc.h5")
```

```{code-cell} ipython3
---
id: dCFJvp57FBV6
---
# run autoHF to get UHF solution
from autohf.hamiltonian import AutoHFHamiltonian

Ueff = 2.75

# define effective Hamiltonian parameters
hamiltonian_params = {
    'U': 2.75
}

params = {
    'hamiltonian': hamiltonian_params
}

# make the Hamiltonian
effective_hamiltonian = HamiltonianDirector(
    source=params,
    lattice=lattice
).build()

# convert afqmctools Hamitlonian to AutoHFHamiltonian
autohf_hamiltonian = AutoHFHamiltonian(effective_hamiltonian)
```

```{code-cell} ipython3
---
id: ZitUaYnfV2p8
---
from autohf import lattice_hf

nelec=(18,18)

settings = dict(
    ansatz = 'SD_ROT',
    numSteps = 100,
    nelec = nelec,
    numTrials = 100,
    seed = 42
)

results = lattice_hf(hamiltonian=autohf_hamiltonian,settings=settings)
```

+++ {"id": "BQRI3y5nygEF"}

## Construct and Save the trial wavefunction

A trial wavefunction can be written to AuxiliaryField's HDF5 format using the `write_wfn()` function from the `afqmctools.wavefunction.common` submodule of afqmctools.
In general, the trial wavefunction in AFQMC is a linear combination of Slater determinants,

$$
|\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle,
$$

where

$C_n$

is a complex-valued coefficient,
and $|\Phi_n\rangle$ are Slater determinants which are not necessarily
orthogonal to each other.
Here, we will specify Slater determinants as Slater matrices

$$
\Phi_{ip} ≐ \begin{bmatrix}
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0N} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1N} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M0} & \bar{C}_{M1} & \bar{C}_{M2} & \dots  & \bar{C}_{MN}
\end{bmatrix},
$$

where $i$ is the basis index, $p$ is the electron index, and
 $M,N$ are the number of basis functions and electrons, respectively.
<b>
SAFIRE uses the convention that orbitals are columns of the Slater matrices.</b>
For collinear Slater determinants, the first $N_{\uparrow}$ columns are the spin-up electrons and the last $N_{\downarrow}$ are the spin-down electrons.

In the code block below, we manually construct Slater matrices from the UHF orbitals that we got from autohf above.

```{code-cell} ipython3
---
id: amcCyPWmBtoW
outputId: 7849db3e-aaec-42e5-dfb0-fafe361b3a39
---
# construct the multi-Slater trial AND save
import numpy as np

from afqmctools.wavefunction.common import write_wfn

# get the orbitals from the AutoHF results.
orbitals_up = results[0]['orbitals'][0]
orbitals_down = results[0]['orbitals'][1]


# The columns of the Slater matrices are the HF orbitals
# Each Slater matrix has shape (Nsites x Nelec) where Nelec is N_up + N_down.
# For UHF-like Slater matrices
#     the first Nup columns are the spin-up orbitals,
#     the last Ndown columns are the spin-down orbitals
#
phi_up = orbitals_up[:,:nelec[0]]
phi_down = orbitals_down[:,:nelec[1]]


# numpy.hstack is a convenient way to concatenate the Slater matrices
phi_0 = np.hstack((phi_up, phi_down)) # |phi_0> = |phi_up> X |phi_down>

# now make the spin-reversed Slater determinant.
phi_1 = np.hstack((phi_down, phi_up)) # |phi_0> = |phi_down> X |phi_up>


C_0 = 1.0 / np.sqrt(2)

wfn = ( np.array([C_0,C_0]), np.array([phi_0,phi_1]))

write_wfn(
    filename=scratch_dir/ "afqmc.h5",
    wfn=wfn,
    walker_type="uhf",
    nelec=nelec,
    norb=lattice.N_sites
)
```

```{code-cell} ipython3
:id: OgN60HHqFJlO

from afqmctools.inputs.from_hdf import write_json

# write input file
afqmc_execution_options = {
    "timestep": 0.01,
    "steps": 10000,
    "measure_interval_multiplier": 1,   # in units of the population control interval
    "population_control_interval" : 10, # in units of steps
    "walker_ortho_interval" : 10 ,      # in units of steps
    "n_walkers_per_mpi_task": 100,
    "seed" : 42
}

write_json(
    scratch_dir/ "afqmc.json",
    fwfn0=scratch_dir / "afqmc.h5",
    exec_opts=afqmc_execution_options
)
```

```{code-cell} ipython3
---
id: SJ-EP0itFLgE
---
# run SAFIRE
run_afqmc(
    run_dir=scratch_dir,
    run_mode="local_cpu",
    input_file="afqmc.json",
    np=16,             # number of MPI tasks
    output_file=None   # optionally direct output to file instead of here
)
```

```{code-cell} ipython3
---
id: IZizKYGAFNvS
---
# analyze results
from stats.scalar_dat import analyze_scalar_data

_ = analyze_scalar_data(dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",
    nequil = 5,
    trace = True
))
```
