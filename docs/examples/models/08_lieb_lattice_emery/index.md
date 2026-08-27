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

+++ {"id": "OOwNuHohKYWT"}

# Emery model on the Lieb lattice

Status:
- we can reproduce Adam's energy for the 4x4 "tilted" Lieb lattice (2 Cu per unit cell)!!

```{code-cell} ipython3
:editable: true
:id: c40c0266-bc19-4216-a70a-44d62234e9fd

from pathlib import Path

from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "d13e8127-0310-46e7-8546-f7e2d7bed54e"}

## Set up the lattice

```{code-cell} ipython3
:id: 3f6aeee9-53bd-41a8-be26-f00472aaf8f5
:outputId: 41c1c867-f098-44c7-d47d-dfcb8232728e

from afqmctools.systems.lattice import get_lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    type = 'square',
    basis = [
        [0.,0.],    # Cu 3d_{x^2 - y^2}
        [0.5,0.0],  # O 2p_x
        [0.0,0.5]   # O 2p_y
    ]
)

lattice = get_lattice(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Lieb Lattice",show_labels=False)
```

+++ {"id": "492caf81-ff32-4a5a-aac3-c2b957feb84e"}

## Set up the Hamiltonian

```{code-cell} ipython3
:id: 3e9badc9-ba8b-4de9-b71a-b663683060e6
:outputId: 31ab047a-5303-4822-c604-715a042ec8fe

import numpy as np

# definning model parameters
tdp = 1.2
tpp = 0.7
epsilon_d = -7.6
epsilon_p = -3.2
Ud = 8.4
Up = 2.0

# for the Emery model, we have to define sign conventions related to the
#   overlap of the +/- lobes of the Co 3d_{x^2 - y^2} orbital and the O 2p_x / 2p_y orbitals
#
#   For tpd:
#       we'll use a '+' sign for the -x, +y direction
#       we'll use a '-' sign for the +x, -y direction
#   For tpp:
#       we'll use a '+' sign for the 1/sqrt(2) (-x, +y), and 1/sqrt(2) (+x,-y) direction
#       we'll use a '-' sign for the 1/sqrt(2) (-x, -y), and 1/sqrt(2) (+x,+y) direction

# define unit vectors
x_hat = np.array([1.0,0.0])
y_hat = np.array([0.0,1.0])

r_x_minus_y_hat = (1/np.sqrt(2))*np.array([1.0,-1.0])  # \hat{r}_{x-y} = \hat{x} - \hat{y}
r_xy_hat = (1/np.sqrt(2))*np.array([1.0,1.0])         # \hat{r}_{x+y} = \hat{x} + \hat{y}

t_pd_signs = { () }

# TODO: make the hopping matrix taking care of the +/- sign
hopping_shape = (lattice.N_sites, lattice.N_sites)

hopping = np.zeros(hopping_shape,dtype=np.complex128)

for pair in lattice.get_nth_neighbors(n=1):
    i,j = pair.i,pair.j
    if j <= i:
        continue

    if np.dot(pair.r_relative,x_hat) > 0 or np.dot(pair.r_relative,-y_hat) > 0:
        hopping[i,j] = -tdp
    elif np.dot(pair.r_relative,-x_hat) > 0 or np.dot(pair.r_relative,y_hat) > 0:
        hopping[i,j] = tdp
    else:
        raise ValueError(f"Unexpected relative distance for nearest neighbors. {pair.r_relative}")

for pair in lattice.get_nth_neighbors(n=2):
    i,j = pair.i,pair.j
    if j <= i:
        continue

    if np.dot(pair.r_relative,r_x_minus_y_hat) != 0.0:
        hopping[i,j] = tpp
    elif np.dot(pair.r_relative,r_xy_hat) != 0.0:
        hopping[i,j] = -tpp
    else:
        raise ValueError(f"Unexpected relative distance for next-nearest neighbors. {pair.r_relative}")

# add in the j<=i part of the hopping matrix
hopping = hopping + hopping.T.conj()


# add in on-site energies
U_emery = np.zeros((lattice.N_sites),dtype=np.complex128)
for i in range(lattice.N_sites):
    if i % 3 == 0:
        hopping[i,i] = epsilon_d
        U_emery[i] = Ud
    elif i % 3 == 1 or i % 3 == 2:
        hopping[i,i] = epsilon_p
        U_emery[i] = Up
```

```{code-cell} ipython3
:id: f59857be
:outputId: 5555c96d-6bfe-445d-89a3-8fc90e952449

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
import afqmctools.utils.io as io

hamiltonian_params = {
    'hamiltonian' : dict(
        t = 0.0,
        U = U_emery,
        custom_one_body = hopping,
    )
}

hamiltonian = HamiltonianBuilder.from_input(
    lattice=lattice,
    source=hamiltonian_params
).hamiltonian

io.write_model_hamiltonian(
    hamiltonian=hamiltonian,
    fname=scratch_dir/"afqmc.h5"
)
```

+++ {"id": "ef2ed76f-26fe-4818-9c90-8e74476d2bef"}

## Run Hartree-Fock (HF)

(sometimes jax doesn't initialize properly on the first run; just try running again!)

```{code-cell} ipython3
:id: 2529be21-ed17-4e89-8871-32335a476603
:outputId: 7e83f104-6cba-4c4c-b97a-b5bb35424437

import jax
jax.config.update('jax_platform_name', 'cpu')

from autohf import lattice_hf, AutoHFHamiltonian

from afqmctools.inputs.from_autohf import autohf_to_afqmc

nelec = (8,8)

hf_settings = dict(
    ansatz = 'SD',
    numSteps = 400,
    nelec = nelec,
    numTrials = 20,
    seed = 42,
    measure_spin = True,
    noncollinear = True
)

results = lattice_hf(
    hamiltonian=AutoHFHamiltonian(hamiltonian),
    settings=hf_settings
)

autohf_to_afqmc(
    results,
    output_fname = scratch_dir/"ghf_wfn.h5"
)
```

+++ {"id": "f21331fb-142c-42e4-87d5-40a61cd7757b"}

##  Analyze the HF results

Before proceeding, it's important to check that the HF solution is correct.
First, check that AutoHF actually converged.

We will also want to check the charge and spin density on the lattice.
You can you the lattice model visualizer `
vis.plot_lattice(lattice,title="Lieb Lattice",density=...)` to plot an arbitrary density
on the lattice.

```{code-cell} ipython3
:id: e325e318-5bee-4f93-91e7-4f72aa1f0edc
:outputId: a48e8eab-7dfd-4f42-d206-077cf3ca1539

import jax.numpy as jnp

# now look at the density
autohf_results = results[0]
autohf_funcs = results[1]
makeRDMs = autohf_funcs['makeRDMs']
state = autohf_results['state']
rdm = makeRDMs(state)

L = lattice.L
# noncollinear
M = lattice.N_sites
new_shape = (L[0],L[1],lattice.nb)
rho_up = rdm[:M,:M].diagonal().reshape(new_shape)
rho_down = rdm[M:,M:].diagonal().reshape(new_shape)

rho_charge = rho_up + rho_down
rho_spin = 0.5*(rho_up - rho_down)

print(f"Integrated charge: {rho_charge.real.sum()}")
print(f" mean charge density: {rho_charge.real.mean()} +/- {np.std(rho_charge.real)}")
print(f" mean spin density: {rho_spin.real.mean()} +/- {np.std(rho_spin.real)}")

Cu_inds =  jnp.array([i*lattice.nb for i in range(lattice.N_sites//lattice.nb)])
Opx_inds = jnp.array([i*lattice.nb+1 for i in range(lattice.N_sites//lattice.nb)])
Opy_inds = jnp.array([i*lattice.nb+2 for i in range(lattice.N_sites//lattice.nb)])

Cu_density_up = rdm[:M,:M].diagonal()[Cu_inds]
Opx_density_up = rdm[:M,:M].diagonal()[Opx_inds]
Opy_density_up = rdm[:M,:M].diagonal()[Opy_inds]

Cu_density_down  = rdm[M:,M:].diagonal()[Cu_inds]
Opx_density_down = rdm[M:,M:].diagonal()[Opx_inds]
Opy_density_down = rdm[M:,M:].diagonal()[Opy_inds]

Cu_density = Cu_density_up + Cu_density_down
Opx_density = Opx_density_up + Opx_density_down
Opy_density = Opy_density_up + Opy_density_down

print(f"For Cu sublattice: mean charge density: {np.mean(Cu_density.real)} +/- {np.std(Cu_density.real)}")
print(f"For O px sublattice: mean charge density: {np.mean(Opx_density.real)} +/- {np.std(Opx_density.real)}")
print(f"For O py sublattice: mean charge density: {np.mean(Opy_density.real)} +/- {np.std(Opy_density.real)}")


vis.plot_lattice(
    lattice,
    density=rho_charge.real,
    density_label="charge density",
    cmap = 'bwr',
    show_labels=False
)

vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    cmap = 'bwr',
    show_labels=False
)
```

+++ {"id": "54616d5a-2831-45da-bd5f-b68621efc8f1"}

## 🚧 Old Scratch work below!! 🚧

```{code-cell} ipython3
:id: 68d02ad8-c469-49b2-a54c-f8611c4ecb68

import numpy as np

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
import afqmctools.utils.io as io
from afqmctools.wavefunction.free_electron import free_electron
from afqmctools.inputs.from_hdf import write_json
import afqmctools.utils.visualize as vis


# define the problem
nelec = (5,5)
nbands = 3

#t_dd = 1.0
t_pd = t = 1.0
t_pp = tprime = 0.5

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = 'PBC',
    boundary2 = 'PBC'
)

lattice = get_lattice(
    params=lattice_params
)

nearest_neighbors = lattice.get_nth_neighbors(n=1)
sites = lattice.sites

nbasis = nbands*lattice.N_sites

hopping_intrasite = np.array([
    [0.0,   t_pd,  t_pd],
    [t_pd,   0.0,  t_pp],
    [t_pd,  t_pp,   0.0]
])

hopping_neatest_x =  np.array([
    [0.0,   t_pd,  0.0],
    [t_pd,   0.0,  t_pp],
    [0.0,  t_pp,   0.0]
])

hopping_neatest_y =  np.array([
    [0.0,    0.0,  t_pd],
    [0.0,   0.0,  t_pp],
    [t_pd,  t_pp,   0.0]
])

hopping = np.zeros((nbasis,nbasis))

# add in intrasite
for site in sites:
    site_slice = slice(site.index*nbands,(site.index+1)*nbands)
    hopping[ site_slice, site_slice ] = hopping_intrasite

# add in nearest-neighbor
for pair in nearest_neighbors:
    i_slice = slice(pair.i*nbands, (pair.i+1)*nbands)
    j_slice = slice(pair.j*nbands, (pair.j+1)*nbands)

    # hopping from Cu to O px/py bands is direction-dependent!
    if (pair.r_relative[0] == 1.0 or pair.r_relative[0] == -1.0):
        hopping[i_slice,j_slice] = hopping_neatest_x
    elif (pair.r_relative[1] == 1.0 or pair.r_relative[1] == -1.0):
        hopping[i_slice,j_slice] = hopping_neatest_y
    else:
        raise ValueError(f"Invalid relative position {pair.r_relative} for a square lattice")
```

```{code-cell} ipython3
:id: 982159fc-56da-40e4-b552-422ff9d8b336

import matplotlib.pyplot as plt

plt.matshow(hopping)
plt.colorbar()
```

```{code-cell} ipython3
:id: f050037a-c4b8-4944-a0b2-b8a7b9fde92b

hamiltonian_params = {
    'hamiltonian' : dict(
        nbands = nbands,
        t = hopping,
        U = 8.0
    )
}

hamiltonian = HamiltonianBuilder.from_input(
    lattice=lattice,
    source=hamiltonian_params
).hamiltonian

io.write_model_hamiltion(
    hamiltonian=hamiltonian,
    fname=scratch_dir/"lieb.h5",
    nelec=nelec
)
```

```{code-cell} ipython3
:id: 541c8b73-0c77-44aa-b9a8-341673ffa6b5


```

+++ {"id": "uGiiaPd0K_YG"}

### Tilted Lieb Lattice : Annealing

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: gl_wuN6FLDxG
outputId: ef58caef-bf77-47a0-abdb-73e4c7df5ed9
---
"""
This is a benchmarking script, so we'll keep things basic
"""
import numpy as np
import h5py as h5

import afqmctools.systems.lattice as lat
import afqmctools.utils.io as io
import afqmctools.utils.visualize as vis
from afqmctools.wavefunction.free_electron import free_electron
from afqmctools.wavefunction.common import modified_gram_schmidt

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import HamiltonianComponent,SpinSymm

import faulthandler; faulthandler.enable()

import scipy.sparse as sps

import jax.numpy as jnp
import jax

import jax
# Set the default device to CPU
#jax.config.update('jax_platform_name', 'cpu')

print("JAX devices : ",jax.devices())


import matplotlib.pyplot as plt
import pdb

#### Some input params ######
recompute = True
show_lattice_before = True

annealing_steps = 10
annealing_amplitude = 0.01

L = (4,4)
Cu_per_cell = 2
h = 1/8

#expected HF energy
Ehf_ref = -313.792

#############################

# this is Nelec = (h+1) * Ncell where Ncell is the number of primitive cells
# BUT, be careful. the tilted lattice has two primitive cells per unit cell!
nelec_total = (h+1) * L[0] * L[1] * Cu_per_cell

assert nelec_total % 2 == 0, "must have even number of electrons"

nelec = (int(nelec_total//2),int(nelec_total//2))

print(f" ==== Emery model run parameters ==== ")
print(f"   - hole doping, h = {h}")
print(f"   - nelectrons = {nelec}")


def make_emery(lattice, show_mats=False):
    """
    """

    # 2. define / build the model
    tdp = 1.2
    tpp = 0.7
    epsilon_d = -7.6
    epsilon_p = -3.2
    Ud = 8.4
    Up = 2.0

    x_hat = np.array([1.0,0.0])
    y_hat = np.array([0.0,1.0])

    r_x_minus_y_hat = (1/np.sqrt(2))*np.array([1.0,-1.0])  # \hat{r}_{x-y} = \hat{x} - \hat{y}
    r_xy_hat = (1/np.sqrt(2))*np.array([1.0,1.0])         # \hat{r}_{x+y} = \hat{x} + \hat{y}

    lattice = lattice

    hopping_shape = (lattice.N_sites, lattice.N_sites)

    hopping = np.zeros(hopping_shape)

    for pair in lattice.get_nth_neighbors(n=1):
        i,j = pair.i,pair.j
        if j <= i:
            continue

        # NOTE: hard code the hopping phase between sublattices for now.
        tdp_pq = tdp*np.array(
            [[0.0,1.0,1.0],
            [-1.0,0.0,1.0],
            [-1.0,1.0,0.0]]
        ) # starting on sublattice from row, ending on sublattece from column
        p = pair.i % 3 # sublattice of i
        q = pair.j % 3 # sublattice of j

        # TODO: save origin and destination sublattice to NeighborPair object
        if np.dot(pair.r_relative,x_hat) > 0 or np.dot(pair.r_relative,-y_hat) > 0:
            hopping[i,j] = -tdp_pq[p,q]
        elif np.dot(pair.r_relative,-x_hat) > 0 or np.dot(pair.r_relative,y_hat) > 0:
            hopping[i,j] = tdp_pq[p,q]
        else:
            raise ValueError(f"Unexpected relative distance for nearest neighbors. {pair.r_relative}")

    for pair in lattice.get_nth_neighbors(n=2):
        i,j = pair.i,pair.j
        if j <= i:
            continue

        if np.dot(pair.r_relative,r_x_minus_y_hat) != 0.0:
            hopping[i,j] = tpp
        elif np.dot(pair.r_relative,r_xy_hat) != 0.0:
            hopping[i,j] = -tpp
        else:
            raise ValueError(f"Unexpected relative distance for next-nearest neighbors. {pair.r_relative}")


    # add in the j<=i part of the hopping matrix
    hopping = hopping + hopping.T.conj()


    # add in on-site energies
    U = np.zeros((lattice.N_sites,lattice.N_sites))
    for i in range(lattice.N_sites):
        if i % 3 == 0:
            hopping[i,i] = epsilon_d
            U[i,i] = Ud
        elif i % 3 == 1 or i % 3 == 2:
            hopping[i,i] = epsilon_p
            U[i,i] = Up

    if show_mats:
        plt.matshow(hopping.real)
        plt.title("Hopping matrix")
        plt.colorbar()
        plt.grid(True)
        plt.show()

    if show_mats:
        plt.matshow(U)
        plt.title("U matrix")
        plt.colorbar()
        plt.grid(True)
        plt.show()

    hopping = np.block(
        [[hopping,np.zeros(hopping_shape)]
        ,[np.zeros(hopping_shape),hopping]]
    )

    # 3. save the Hamiltonian
    builder = HamiltonianBuilder(lattice=lattice)
    hopping = sps.csr_matrix(hopping)
    custom_one_body = HamiltonianComponent(
        csr_array=hopping,
        model_type='one_body',
        spin_symm=SpinSymm.NONCOLLINEAR
    )

    # manually add the custom term
    builder.hamiltonian["tij"] = custom_one_body
    custom_hubbard = HamiltonianComponent(
        csr_array=U,
        model_type='hubbard_u',
        hst_type='discrete_spin'
    )
    builder.hamiltonian["Uij"] = custom_hubbard
    builder.finalize()

    hamiltonian = builder.hamiltonian

    io.write_model_hamiltonian(
        hamiltonian=hamiltonian,
        fname="afqmc.h5",
        nelec=nelec
    )

    return hamiltonian

# 1. define the lattice
basis = [ np.array(delta) for delta in [(0,0),(0.5,0),(0,0.5),(1.0,0),(1.5,0),(1.0,0.5)] ]

lattice = lat.CustomLattice(
    L=(4,4),
    a1=np.array((1.0,-1.0)),
    a2=np.array((1.0,1.0)),
    basis=basis,
    axis1_boundary=lat.PBCBoundary,
    axis2_boundary=lat.PBCBoundary,
)

if show_lattice_before:
    vis.plot_lattice(
        lattice,
        show_labels=True,
        show_lattice_vecs=False,
        title="Lieb lattice"
    )

hamiltonian = make_emery(lattice,show_mats=True)


# for initial guess - we need another H with a small twist!
THETA_X = 1/np.sqrt(592560607) # 592560607 is prime
THETA_Y = 1/np.sqrt(47603)     # 47603 is prime
twist = np.array((THETA_X,THETA_Y))

lattice_wt_twist = lat.CustomLattice(
    L=(4,4),
    a1=np.array((1.0,-1.0)),
    a2=np.array((1.0,1.0)),
    basis=basis,
    twist=twist
)

hamiltonian2 = make_emery(lattice_wt_twist)

wfn,spin_symm = free_electron(
    source=hamiltonian2,
    nelec=nelec,
    lattice=lattice,
    spin_symm=SpinSymm.NONCOLLINEAR
)

# add noise to the initial guess
wfn = (wfn[0],wfn[1] + annealing_amplitude*np.random.randn(*wfn[1].shape))

# 4. run autohf
from autohf import lattice_hf, AutoHFHamiltonian
best_E_final = None
for a in range(annealing_steps):
    print(f"Annealing step {a}")

    slater_det = wfn[1][0]

    hf_settings = dict(
        ansatz = 'SD',#'SD_Rot',
        steps = 500, #numSteps = 500,
        #output = f"ghf_wfn_step_{a}.h5",
        nelec = nelec,
        batch_size = 20, #numTrials = 20,
        #seed = 42,
        measure_spin = False,
        noncollinear =  True,
        gpu = False,
        state0_scale = 1.0e-3,
    )

    results, functions = lattice_hf(
        hamiltonian=AutoHFHamiltonian(hamiltonian),
        lattice=lattice,
        settings=hf_settings,
        jaxoptargs=dict(tol=1e-8),
        initial_guess=slater_det,
    )



    # now look at the density
    makeRDMs = functions['makeRDMs']
    state = results['state']
    rdm = makeRDMs(state)

    with h5.File("dev_checkpoint.h5","w") as f:
        f.create_dataset(f"rdm_{a}",data=rdm)

    # get a wavefunction for the next annealing step
    orbitals = results['orbitals'][0]

    slater_det = np.array(orbitals[:,:sum(nelec)])
    # orthonormalize!
    slater_det = modified_gram_schmidt(slater_det)
    slater_det = slater_det + annealing_amplitude*np.random.randn(*slater_det.shape)

    wfn = (wfn[0],[slater_det])

    L = lattice.L

    # noncollinear
    M = lattice.N_sites
    new_shape = (L[0],L[1],lattice.nb)
    rho_up = rdm[:M,:M].diagonal().reshape(new_shape)
    rho_down = rdm[M:,M:].diagonal().reshape(new_shape)

    rho_charge = rho_up + rho_down
    rho_spin = 0.5*(rho_up - rho_down)



    if best_E_final is None:
        best_E_final = results['E']
        best_rho_charge = rho_charge
        best_rho_spin = rho_spin
        best_rdm = rdm

    if results['E'] < best_E_final:
        print("Found a better energy: ",results['E'])
        best_E_final = results['E']
        best_rho_charge = rho_charge
        best_rho_spin = rho_spin
        best_rdm = rdm

    if recompute:
        with h5.File("dev_checkpoint.h5","a") as f:
            f.create_dataset("E",data=best_E_final)
            f.create_dataset(f"best_rho_charge",data=best_rho_charge)
            f.create_dataset(f"best_rho_spin",data=best_rho_spin)


print(f"\nBest Energy = {best_E_final} t")
print(f"            = {best_E_final / (L[0] * L[1] * Cu_per_cell)} t / Cu site\n\n")

rdm = best_rdm
rho_charge = best_rho_charge
rho_spin = best_rho_spin

print(f"Integrated charge: {rho_charge.real.sum()}")

print(f" mean charge density: {rho_charge.real.mean()} +/- {np.std(rho_charge.real)}")
print(f" mean spin density: {rho_spin.real.mean()} +/- {np.std(rho_spin.real)}")

Cu_inds =  jnp.array([i*lattice.nb for i in range(lattice.N_sites//lattice.nb)])
Opx_inds = jnp.array([i*lattice.nb+1 for i in range(lattice.N_sites//lattice.nb)])
Opy_inds = jnp.array([i*lattice.nb+2 for i in range(lattice.N_sites//lattice.nb)])

Cu_density_up = rdm[:M,:M].diagonal()[Cu_inds]
Opx_density_up = rdm[:M,:M].diagonal()[Opx_inds]
Opy_density_up = rdm[:M,:M].diagonal()[Opy_inds]

Cu_density_down  = rdm[M:,M:].diagonal()[Cu_inds]
Opx_density_down = rdm[M:,M:].diagonal()[Opx_inds]
Opy_density_down = rdm[M:,M:].diagonal()[Opy_inds]

Cu_density = Cu_density_up + Cu_density_down
Opx_density = Opx_density_up + Opx_density_down
Opy_density = Opy_density_up + Opy_density_down

print(f"For Cu sublattice: mean charge density: {np.mean(Cu_density.real)} +/- {np.std(Cu_density.real)}")
print(f"For O px sublattice: mean charge density: {np.mean(Opx_density.real)} +/- {np.std(Opx_density.real)}")
print(f"For O py sublattice: mean charge density: {np.mean(Opy_density.real)} +/- {np.std(Opy_density.real)}")


if True:
    vis.plot_lattice(
        lattice,
        density=rho_charge.real,
        density_label="charge density",
        cmap = 'bwr',
        show_labels=False,
        save="charge.png"
    )

    vis.plot_lattice(
        lattice,
        density=rho_spin.real,
        density_label="spin density",
        cmap = 'bwr',
        show_labels=False,
        save="spin.png"
    )
```

```{code-cell} ipython3
:id: 80e2be32-8159-4ea0-a1e7-d228a9a91c11


```
