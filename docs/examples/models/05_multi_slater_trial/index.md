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

+++ {"id": "0WrZ0hFmBbdm"}

# Multi-Slater Determinant Trial Wavefunction

In this example, we will compute the AFQMC energy of the Hubbard model with U=8 on a 6×6 square lattice with open boundary conditions
at half-filling.
We will use a trial wavefunction consisting of two Slater determinants.
Specifically, we will use

$$
| \Psi_\mathrm{T} \rangle = \frac{1}{\sqrt{2}} \left( |\Phi_{0}\rangle + |\Phi_{1}\rangle \right)
$$

where

$$
|\Phi_{0}\rangle = |\Phi^\mathrm{UHF}\rangle = |\Phi^{\uparrow}\rangle \otimes |\Phi^{\downarrow}\rangle
$$

is the UHF ground state
and $|\Phi_{1}\rangle$ is the spin-reversed UHF ground state,

$$
|\Phi_{1}\rangle = |\Phi^{\downarrow}\rangle \otimes |\Phi^{\uparrow}\rangle.
$$

Since the Hamiltonian is spin-independent, this trial wavefunction should have the same energy as the UHF ground state.

```{code-cell} ipython3
:id: 1rz5sWn3Ecsc
:outputId: 51107f5c-4186-4817-9e97-990b46c79acc

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

# setup the Hamiltonian
from safiretools import Lattice, LatticeHamiltonian
from safiretools import vis

lattice = Lattice.from_type(
    "square", (6, 6), boundaries=("open", "open"), twists=(0.0, 0.0)
)

# plot the lattice for reality checks!
vis.plot_lattice(lattice)

# make the Hamiltonian
hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=8.0)
```

```{code-cell} ipython3
:id: dCFJvp57FBV6

# run autoHF to get UHF solution
from autohf import AutoHFHamiltonian

Ueff = 2.75

# make the effective Hamiltonian
effective_hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=Ueff)

# convert safiretools Hamiltonian to AutoHFHamiltonian
autohf_hamiltonian = AutoHFHamiltonian(effective_hamiltonian)
```

```{code-cell} ipython3
:id: ZitUaYnfV2p8

from autohf import solve_hf

nelec=(18,18)

settings = dict(
    ansatz = 'SD_ROT',
    steps = 100,
    nelec = nelec,
    batch_size = 100,
    seed = 42
)

results = solve_hf(hamiltonian=autohf_hamiltonian,settings=settings)
```

+++ {"id": "BQRI3y5nygEF"}

## Construct and Save the Trial Wavefunction

A trial wavefunction can be written to SAFIRE's HDF5 format by building a `NOMSDWavefunction` from `safiretools` and calling its `to_hdf5()` method.
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
A collinear Slater determinant is a pair of Slater matrices, one for the $N_{\uparrow}$ spin-up electrons and one for the $N_{\downarrow}$ spin-down electrons.

In the code block below, we manually construct Slater matrices from the UHF orbitals that we got from autohf above.

```{code-cell} ipython3
:id: amcCyPWmBtoW
:outputId: 7849db3e-aaec-42e5-dfb0-fafe361b3a39

# construct the multi-Slater trial AND save
import numpy as np

from safiretools import NOMSDWavefunction

# get the orbitals from the AutoHF results.
orbitals_up = results[0]['orbitals'][0]
orbitals_down = results[0]['orbitals'][1]


# The columns of the Slater matrices are the HF orbitals
# A UHF-like Slater determinant is a pair of Slater matrices
#     the spin-up orbitals, with shape (Nsites × Nup),
#     the spin-down orbitals, with shape (Nsites × Ndown)
#
phi_up = orbitals_up[:,:nelec[0]]
phi_down = orbitals_down[:,:nelec[1]]


phi_0 = (phi_up, phi_down) # |phi_0> = |phi_up> X |phi_down>

# now make the spin-reversed Slater determinant.
phi_1 = (phi_down, phi_up) # |phi_1> = |phi_down> X |phi_up>


C_0 = 1.0 / np.sqrt(2)

# numpy.stack is a convenient way to stack the determinants of each spin channel
wfn = NOMSDWavefunction(
    coeffs=np.array([C_0,C_0]),
    dets=(np.stack([phi_0[0], phi_1[0]]), np.stack([phi_0[1], phi_1[1]]))
)
```

```{code-cell} ipython3
:id: OgN60HHqFJlO

from safiretools import write_jobfile

# write input file, which also writes the wavefunction and the Hamiltonian to HDF5
write_jobfile(
    "afqmc.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": wfn,
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

```{code-cell} ipython3
:id: SJ-EP0itFLgE
:tags: [scroll-output]

!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
:id: IZizKYGAFNvS

# analyze results
from safiretools import Results

results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
```
