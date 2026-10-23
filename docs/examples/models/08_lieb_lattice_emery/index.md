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

# Emery Model on the Lieb Lattice

+++ {"id": "d13e8127-0310-46e7-8546-f7e2d7bed54e"}

## Set Up the Lattice

```{code-cell} ipython3
:id: 3f6aeee9-53bd-41a8-be26-f00472aaf8f5
:outputId: 41c1c867-f098-44c7-d47d-dfcb8232728e

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

from safiretools import Lattice
from safiretools import vis

lattice = Lattice(
    lattice_vectors=[
        [1.0,0.0],  # a1
        [0.0,1.0]   # a2
    ],
    basis=[
        [0.,0.],    # Cu 3d_{x^2 - y^2}
        [0.5,0.0],  # O 2p_x
        [0.0,0.5]   # O 2p_y
    ],
    Ls=(4, 4),
    boundaries=('open', 'open'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Lieb Lattice",show_labels=False)
```

+++ {"id": "492caf81-ff32-4a5a-aac3-c2b957feb84e"}

## Set Up the Hamiltonian

```{code-cell} ipython3
:id: 3e9badc9-ba8b-4de9-b71a-b663683060e6
:outputId: 31ab047a-5303-4822-c604-715a042ec8fe

import numpy as np

# defining model parameters
tdp = 1.2
tpp = 0.7
epsilon_d = -7.6
epsilon_p = -3.2
Ud = 8.4
Up = 2.0

# for the Emery model, we have to define sign conventions related to the
#   overlap of the +/- lobes of the Cu 3d_{x^2 - y^2} orbital and the O 2p_x / 2p_y orbitals
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

from safiretools import LatticeHamiltonian

hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t = 0.0,
    U = U_emery,
    custom_one_body = hopping,
)

hamiltonian.to_hdf5("afqmc.h5")
```

+++ {"id": "ef2ed76f-26fe-4818-9c90-8e74476d2bef"}

## Run Hartree-Fock (HF)

```{code-cell} ipython3
:id: 2529be21-ed17-4e89-8871-32335a476603
:outputId: 7e83f104-6cba-4c4c-b97a-b5bb35424437

import jax
jax.config.update('jax_platform_name', 'cpu')

from autohf import solve_hf, AutoHFHamiltonian

from safiretools import Wavefunction

nelec = (8,8)

hf_settings = dict(
    ansatz = 'SD',
    steps = 400,
    nelec = nelec,
    batch_size = 20,
    seed = 42,
    measure_spin = True,
    noncollinear = True
)

results = solve_hf(
    hamiltonian=AutoHFHamiltonian(hamiltonian),
    settings=hf_settings
)

# the noncollinear orbitals are spinors, spin-up components first
orbitals, = results[0]["orbitals"]
ghf_wfn = Wavefunction.from_single_determinant(
    np.asarray(orbitals).reshape(2, -1, sum(nelec))
)
```

+++ {"id": "f21331fb-142c-42e4-87d5-40a61cd7757b"}

## Analyze the HF Results

Before proceeding, it's important to check that the HF solution is correct.
First, check that AutoHF actually converged.

We will also want to check the charge and spin density on the lattice.
You can use the lattice model visualizer `vis.plot_lattice(lattice, density=...)`
to plot an arbitrary density on the lattice.

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
    show_labels=False
)

vis.plot_lattice(
    lattice,
    density=rho_spin.real,
    density_label="spin density",
    show_labels=False
)
```
