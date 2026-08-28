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
a 4x8 lattice.
The Hamiltonian is given by,
$$
\hat{H} = -t \sum_{\langle i,j\rangle^{(1)}} \hat{c}^\dagger_i \hat{c}_j -t' \sum_{\langle i,j\rangle^{(2)}} \hat{c}^\dagger_i \hat{c}_j + U \sum_{i} \hat{n}^{\uparrow}_i \hat{n}^{\downarrow}_i,
$$
where $i,j$ are lattice site indices, angle brackets with superscript $(n)$ indicate
that sums are constrained to $n^{th}$-order neighbors,
$\hat{c}^\dagger_i$/$\hat{c}_i$ are electronic creation/annihilation operators,
and $\hat{n}^{\sigma}_i$ are spin-resolved number operators corresponding to site $i$.
The 4x4 lattice is small enough to make exact diagonalization possible.
We will use exact results for this system as reference data throughout
this tutorial.

See Table I of reference [1]
The AFQMC energy using a free-electron Trial wavefunction is, for $U=4$ on a 4x8 square lattice
with $N_{\uparrow} = N_{\downarrow} = 16$, $−27.271⁢(1)$ versus $−27.6924$ from DMRG.
(Better AFQMC results exist there when using a self-consistent procedure).

References:

1. https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.3.013065

```{code-cell} ipython3
:id: 418bd9e4-4da9-4612-889f-6f365fc667a6

# simple setup
from tutorial_utils import run_afqmc
from pathlib import Path

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "b8153eeb-0f28-4151-ad09-af65db81f358"}

## Setup - Make a lattice

```{code-cell} ipython3
:id: 254dc31a-b763-4cb3-9867-5129279b9c2b
:outputId: c065dd3d-6070-4a49-80ef-e02029ba7b13

from safiretools import Lattice
from afqmctools.utils.visualize import plot_lattice

lattice_params = {
    "L1" : 4,
    "L2" : 8,
    "boundary1" : "pbc",
    "boundary2" : "open"
}

lattice = Lattice.from_dict(lattice_params)

plot_lattice(lattice,show_coords=False)
```

+++ {"id": "91688c0b-bca3-4a6b-9d0f-008ab9c943ff"}

## Setup - Make the Hamiltonian

...

```{code-cell} ipython3
:id: 364ee811-8242-45ce-88ee-69fcaa6901fd
:outputId: b64b276a-cae9-4c47-a315-e9c015fc264a

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import SpinSymm
from afqmctools.utils.io import write_model_hamiltonian

#TODO: need AFM (staggered) pinning at y=1 and  y=Ly with h_pin = 0.25 only for AFQMC!!

hamiltonian_params = {
    "hamiltonian" : {
        "t" : [1.0,0.3],
        "U" : 4
    }
}

# afm_pinning() needs axis=1 and pin_type="same", which HamiltonianBuilder.from_input()
# cannot express, so we call the build steps directly instead and finalize once at the end
builder = HamiltonianBuilder(lattice=lattice,spin_symm=SpinSymm.COLLINEAR)
builder.nth_neighbor_hopping(hamiltonian_params["hamiltonian"]["t"])
builder.onsite_hubbard(hamiltonian_params["hamiltonian"]["U"])
builder.afm_pinning(h_afm_pin=0.25,axis=1,pin_type="same")
builder.finalize()

hamiltonian = builder.hamiltonian
write_model_hamiltonian(hamiltonian,fname=scratch_dir/"Hubbard_tprime0.4_U4.0.h5")
```

+++ {"id": "81220323-65d6-4438-91db-87cc44e71047"}

## Setup - Trial Wavefunction.

```{code-cell} ipython3
# First, let's compute the Free-Electron trial wavefunction
from afqmctools.wavefunction.free_electron import free_electron
from afqmctools.wavefunction.common import write_wfn

nelec = (16,16)

input_params = dict(
    lattice = lattice_params,                           # from step 1. above
    hamiltonian = hamiltonian_params["hamiltonian"]     # from step 2. above
)

# twist from ref 1:
twist = (0.0,0.0) #(0.01,0.01) # this twist was used for 4x16 with t' = 0.3t
wfn,_,results = free_electron(
    source=input_params,
    nelec=nelec,
    twist=twist,                          # (optional) using the default small twist
    return_autohf = True
)

# get lattice dimensions from the Lattice instance
L = lattice.L

write_wfn(scratch_dir/"free_elec_wfn.h5", wfn, walker_type='collinear', norb=L[0]*L[1], nelec=nelec)
```

```{code-cell} ipython3
:id: 20127a9c-5c29-4c69-8a81-bd497f042406
:outputId: 55190e94-44ed-4b7c-9f28-7cdcd5d151ba

import afqmctools.utils.visualize as vis

# convert the 'results' from autohf to a charge density
makeRDMs = results[1]['makeRDMs']
state = results[0]['state']

rdm = makeRDMs(state)

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

## Setup - Write an SAFIRE input file

```{code-cell} ipython3
# make a json input file
from afqmctools.inputs.from_hdf import write_json

afqmc_params = {
    "timestep": 0.01,
    "steps": 12000,
    "n_walkers_per_mpi_task": 30,
    "population_control_interval": 5,
    "walker_ortho_interval": 5,
    "measure_interval_multiplier": 2,
    "seed" : 42,                          # for reproducibility
    "propagator": {
        "use_cp_constraint": True,
        "use_real_vbias": True
    },
    "estimator": {
        "name":"back_propagation",
        "path_restoration": True,
        "bp_walker_ortho_interval": 5,
        "measure_interval_multiplier": [40, 80, 120, 160],
        "equil_multiplier":480,
        "onerdm":{
            "name":"onerdm"
        }
    }
}

write_json(
    scratch_dir/"afqmc.json",
    fwfn0=scratch_dir/"free_elec_wfn.h5",
    fham0=scratch_dir/"Hubbard_tprime0.4_U4.0.h5",
    exec_opts=afqmc_params
)
```

+++ {"id": "5831b8a6-6262-46ce-9bef-f51e20838015"}

### Running AFQMC

Below, we provide a cell that will run AFQMC here; however, for this calculation, you might prefer to submit the job to Slurm.
You can copy the runscript from `/mnt/home/beskridge/ceph/software/SAFIRE/local_scripts/run_afqmc_cpu.sh` to your scratch
directory for this tutorial and use

```bash
sbatch run_afqmc_cpu.sh
```

to submit the job (perhaps adjusting the run time).

```{code-cell} ipython3
:id: c5ec3cf1-78cc-428e-8436-c3ddef25451e
:outputId: 3544767c-4beb-4497-a895-92af2c2becc8

run_afqmc(
    run_dir=scratch_dir,
    output_file=None,       # this will dump the output from AFQMC here
    np=16,                   # set number of MPI tasks
    timeout_mins=20
)
```

+++ {"id": "3cb4eb7a-0137-4f7c-b73a-adae570c2cd7"}

## Analysis - compute the AFQMC energy

we're looking for −27.271⁢(1) from reference [1]

I'm currently at
-26.778445 +/- 0.009127.

Let's double check the paper.

```{code-cell} ipython3
:id: 4d782d47-030b-4baa-97f7-d5727575dca7
:outputId: 9141e891-875b-46db-9a23-ed9581f6442e

from stats.scalar_dat import analyze_scalar_data
nequil = 2.0 #t^{-1}

analysis_settings = dict(
    fname = scratch_dir/"qmc.s000.scalar.dat",
    xaxis = "time",     # use units of imaginary time for equilibration
    nequil = nequil,    # length of equilibration phase (in units of imaginary time)
    trace = True,       # plots a trace of the scalar data
)

E,dE = analyze_scalar_data(analysis_settings)

print(f"The AFQMC energy is {E:.6f} +/- {dE:.6f} Hartree")
```

+++ {"id": "da4dfc0c-047c-43b7-9568-5494c48067b2"}

## Analysis - compute the Average 1-rdm

To check for convergence here, we will compute the average 1-rdm for each "average" saved in the
output HDF5 file, and plot `\rho(i,j)` vs $N_{steps}$ for a line cut of the lattice.
We should see the average 1-rdm converge for some value of $N_{steps}$.

```{code-cell} ipython3
:id: 46011340-4ae4-41a9-ae29-e97568264d04

from afqmctools.analysis.rdm import average_afqmc_rdm

naverages = 4

rho_avg, delta_rho = average_afqmc_rdm(rdm_file=scratch_dir/"qmc.s000.stat.h5")

rho_up = rho_avg[:,0,:,:].diagonal().reshape((naverages,4,8))
delta_rho_up = delta_rho[:,0,:,:].diagonal().reshape((naverages,4,8))

rho_down = rho_avg[:,1,:,:].diagonal().reshape((naverages,4,8))
delta_rho_down = delta_rho[:,1,:,:].diagonal().reshape((naverages,4,8))
```

```{code-cell} ipython3
:id: bd4b242e-81b2-40d0-86fe-ec6b25866a45
:outputId: b0b06b69-d141-45b0-fafb-1bb6092d9d21

import matplotlib.pyplot as plt
import numpy as np

ix = 3
iy = np.array(range(8))

rho_spin = 0.5*(rho_up - rho_down)

plt.plot(iy,rho_spin[0,ix,iy])
plt.show()
```

+++ {"id": "12337ed8-7b39-4712-b99a-154aac89a650"}

## Improved Trial via Natural orbitals

```{code-cell} ipython3
:id: cf8d2b64-4ac4-4322-a9d1-b5360d9795e0


```
