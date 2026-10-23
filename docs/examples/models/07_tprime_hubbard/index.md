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

+++ {"id": "d0c647d7-35b0-48c9-bc58-2a2498c94350"}

# Hubbard Model with $t'$

Author: Kyle Eskridge

In this tutorial, we will perform AFQMC calculations for the Hubbard model with $t'$ on
a 4×8 lattice.
The Hamiltonian is given by,
$$
\hat{H} = -t \sum_{\langle i,j\rangle,\sigma} \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma} - t' \sum_{\langle\langle i,j\rangle\rangle,\sigma} \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma} + U \sum_{i} \hat{n}_{i\uparrow} \hat{n}_{i\downarrow},
$$
where $\langle i,j\rangle$ and $\langle\langle i,j\rangle\rangle$ sum over the nearest- and next-nearest-neighbor bonds.
We will use the DMRG and AFQMC results of reference [1] for this system as reference data throughout
this tutorial.

See Table I of reference [1].
The AFQMC energy using a free-electron trial wavefunction is, for $U=4$ on a 4×8 square lattice
with $N_\uparrow = N_\downarrow = 16$, $-27.271(1)$ versus $-27.6924$ from DMRG.
(Better AFQMC results exist there when using a self-consistent procedure).

References:

1. https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.013065

+++ {"id": "b8153eeb-0f28-4151-ad09-af65db81f358"}

## Setup - Make a Lattice

```{code-cell} ipython3
:id: 254dc31a-b763-4cb3-9867-5129279b9c2b
:outputId: c065dd3d-6070-4a49-80ef-e02029ba7b13

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

from safiretools import Lattice
from safiretools import vis

lattice = Lattice.from_type("square", (4, 8), boundaries=("periodic", "open"), twists=(0.0, 0.0))

vis.plot_lattice(lattice)
```

+++ {"id": "91688c0b-bca3-4a6b-9d0f-008ab9c943ff"}

## Setup - Make the Hamiltonian

```{code-cell} ipython3
:id: 364ee811-8242-45ce-88ee-69fcaa6901fd
:outputId: b64b276a-cae9-4c47-a315-e9c015fc264a

from safiretools import LatticeHamiltonian, PinningField

# AFM pinning field of ±0.25 on the open edges y=1 and y=Ly, with opposite
# signs on the two edges.
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t=[1.0, 0.3],
    U=4,
    pinning=(None, PinningField.afm(0.5, staggered=True)),
)
```

+++ {"id": "81220323-65d6-4438-91db-87cc44e71047"}

## Setup - Trial Wavefunction

```{code-cell} ipython3
# First, let's compute the free-electron trial wavefunction
from safiretools import Wavefunction

nelec = (16,16)

wfn = Wavefunction.from_free_electron(
    source=hamiltonian,
    nelec=nelec
)

# get lattice dimensions from the Lattice instance
L = lattice.L
```

```{code-cell} ipython3
:id: 20127a9c-5c29-4c69-8a81-bd497f042406
:outputId: 55190e94-44ed-4b7c-9f28-7cdcd5d151ba

import numpy as np

# one-rdm of the free-electron determinant
alpha, beta = wfn.determinant(0)
rdm = np.array([alpha.conj() @ alpha.T, beta.conj() @ beta.T])

# collinear
rho_up = rdm[0].diagonal().reshape(*L)
rho_down = rdm[1].diagonal().reshape(*L)

rho_charge = rho_up + rho_down
#rho_spin = 0.5*(rho_up - rho_down)

vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    vmin=0.0,
    vmax=2.0
)
```

```{code-cell} ipython3
:id: 64c43a4e-6a9d-4d59-86dc-082074725094
:outputId: 73b71447-c369-4346-a6f7-0008001b59e3

rho_spin = 0.5*(rho_up - rho_down)
vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    vmin=-0.5,
    vmax=0.5
)
```

+++ {"id": "a42d9aa5-0a4d-45f4-a2cf-96e3949afb8d"}

## Setup - Write a SAFIRE Input File

```{code-cell} ipython3
# make a JSON job file
from safiretools import write_jobfile

bp_lengths = [200, 400, 600]

write_jobfile("afqmc.json", {
    "seed": 42,                          # for reproducibility
    "execute": {
        "wavefunction": wfn,
        "hamiltonian": hamiltonian,
        "timestep": 0.01,
        "steps": 12000,
        "num_walkers": 480,
        "population_control_interval": 5,
        "walker_ortho_interval": 5,
        "measure_interval": 10,
        "equilibration_steps": 2400,
        "propagator": {
            "use_cp_constraint": True,
            "project_force_bias": True
        },
        "estimators": {
            "backprop": {
                "walker_ortho_interval": 5,
                "propagation_steps": bp_lengths,
                "onerdm": {}
            }
        }
    }
})
```

+++ {"id": "5831b8a6-6262-46ce-9bef-f51e20838015"}

### Running AFQMC

```{code-cell} ipython3
:id: c5ec3cf1-78cc-428e-8436-c3ddef25451e
:outputId: 3544767c-4beb-4497-a895-92af2c2becc8
:scrolled: true
:tags: [scroll-output]

!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

+++ {"id": "3cb4eb7a-0137-4f7c-b73a-adae570c2cd7"}

## Analysis - Compute the AFQMC Energy

We compare the AFQMC energy to the value $-27.271(1)$ from reference [1].

```{code-cell} ipython3
:id: 4d782d47-030b-4baa-97f7-d5727575dca7
:outputId: 9141e891-875b-46db-9a23-ed9581f6442e

import matplotlib.pyplot as plt
from safiretools import Results

E_ref = -27.271 # reference [1]

results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
plt.axhline(E_ref, color="crimson")
```

+++ {"id": "da4dfc0c-047c-43b7-9568-5494c48067b2"}

## Analysis - Compute the Average One-Body Reduced Density Matrix

To check for convergence here, we will compute the spin density from the back-propagated one-body reduced density matrix for each back-propagation length $m_\mathrm{BP}$ saved in the
output HDF5 file, and plot it along a line cut of the lattice for each $m_\mathrm{BP}$.
We should see the density matrix converge for some value of $m_\mathrm{BP}$.

```{code-cell} ipython3
:id: 46011340-4ae4-41a9-ae29-e97568264d04

ix = 3

def spin_cut(rdm):
    """S^z along the line cut at ix, from a collinear one-rdm."""
    rho_spin = 0.5*(rdm[0].diagonal() - rdm[1].diagonal()).real
    return rho_spin.reshape(lattice.L)[ix]

rho_spin, rho_spin_err = {}, {}
for steps in bp_lengths:
    rho_spin[steps], rho_spin_err[steps] = results.evaluate(spin_cut, [f"BackPropEstimator/Steps={steps}/OneRDM"])
```

```{code-cell} ipython3
:id: bd4b242e-81b2-40d0-86fe-ec6b25866a45
:outputId: b0b06b69-d141-45b0-fafb-1bb6092d9d21

iy = np.arange(lattice.L[1])

for steps in bp_lengths:
    plt.errorbar(iy,rho_spin[steps],yerr=rho_spin_err[steps],fmt='o-',label=f"{steps} steps")
plt.xlabel("Site")
plt.ylabel("Spin Density")
plt.legend()
plt.show()
```
