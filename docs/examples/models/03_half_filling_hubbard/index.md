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

+++ {"id": "a9e290f7-823f-4240-8398-1e11040ab402"}

# The square Hubbard Model at half filling

Author: Ryan Levy

```{code-cell} ipython3
:id: 6d86eb21-7fcb-4c98-9780-3d657dff85a2

import os
# This are in case you have a GPU enabled JAX but are running on CPU.
os.environ['JAX_PLATFORMS'] = 'cpu'
# If you want to use GPU enabled Jax in multiply notebooks with the same
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# use x64 numbers
from jax import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

# all the imports used later in the tutorial, but put here for convenience

import afqmctools.systems.lattice as lat
import afqmctools.utils.visualize as vis
import afqmctools.utils.io as io
import afqmctools.hamiltonian.model.builder as ham

from afqmctools.wavefunction.converter import read_wavefunction
from afqmctools.wavefunction.model import write_free_electron_wfn, make_free_elec,write_wfn
from afqmctools.inputs.from_hdf import write_json
from afqmctools.analysis.rdm import average_afqmc_rdm

from afqmctools.observables.greens import greens_1body
import afqmctools.observables.spin as spobs

from stats.scalar_dat import analyze_scalar_data

from tutorial_utils import run_afqmc

import autohf
from afqmctools.inputs.from_autohf import autohf_to_afqmc

from pathlib import Path
scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"editable": true, "id": "f91cf26c-5663-416e-9802-aeb1aa5a77c6"}

## Physics Background

The Hubbard model at Half Filling ($n=1$) has a special "symmetry" known as particle-hole symmetry, that allows auxiliary field methods to be sign free. However, care must be taken to make sure that phaseless/constrained path AFQMC is used properly to exploit said symmetry.

[1]: _Benchmark study of the two-dimensional Hubbard model with auxiliary-field quantum Monte Carlo method_  
M Qin, H Shi, S Zhang Phys. Rev. B 94, 085103 4 August, 2016  
DOI: https://doi.org/10.1103/PhysRevB.94.085103

+++ {"id": "678e6d9b-4d8f-4f74-87ee-2420b8588c57"}

## Lattice and Hamiltonian Setup

```{code-cell} ipython3
:id: 132741fa-48b2-46a2-a980-3d58740b78d3

# take from 10.1103/PhysRevB.45.10741, U => E_0
E0_n1_lookup = {4 : -13.62185,
                8 : -8.46888}
```

```{code-cell} ipython3
:id: 55293f57-e436-47c6-bef7-1b095b690f57

U = 4.0
lattice_dims = (4,4)

lattice_params = {
    'L1' : lattice_dims[0],
    'L2' : lattice_dims[1],
    'boundary1' : 'PBC',
    'boundary2' : 'PBC'
}
```

```{code-cell} ipython3
:id: b2a30f8a-5c48-4272-b0df-14d9c2062e59
:outputId: 9a69fa7d-1093-4d5a-8245-f57f3639017c

lattice = lat.get_lattice(params=lattice_params)

Ne = int(lattice.N_sites)//2
nelec = (Ne,Ne)
nelec
```

```{code-cell} ipython3
:id: 9a18f6e5-68f2-46cd-ae51-3bf8826dc679
:outputId: 47dcfa86-4f49-4688-b813-cc11bcc1c244

vis.plot_lattice(lattice)
plt.show()
```

+++ {"id": "5f7f1b9d-6230-41a0-8037-da334866defb"}

The Hamiltonian will be

$$
\begin{aligned}
H &= H_t + H_U \\
H_t &= -t \sum_{\langle i j\rangle \sigma} c^\dagger_{i\sigma} c_{j\sigma} \\
H_U &= U\sum_i n_{i\uparrow}n_{i\downarrow}
\end{aligned}
$$

```{code-cell} ipython3
:id: 7416bbce-9541-46bb-bc52-1e6a42e2d001
:outputId: 8bcb041d-4e1e-4a10-9bd2-d5a7a2385793

builder = ham.HamiltonianBuilder(
          lattice=lattice,
          spin_symm="collinear" # we have no spin-flip terms
              )
# add standard Hubbard terms
builder.nth_neighbor_hopping(1.0)
builder.onsite_hubbard(U)
builder.finalize()
```

```{code-cell} ipython3
:id: 3670ecc2-0aa9-4a34-ab5b-b903f7fddc08
:outputId: 70b88144-c728-4f1c-d8de-70a78fa2914e

builderHF = ham.HamiltonianBuilder(
          lattice=lattice,
          spin_symm="collinear" # we have no spin-flip terms
              )
# add standard Hubbard terms
builderHF.nth_neighbor_hopping(1.0)
builderHF.onsite_hubbard(4) # if you want an effective U, you can change this
builderHF.finalize()
```

```{code-cell} ipython3
:id: 67700d97-b76c-4d89-a7f9-0b814e411497

# get the 1 body Hamiltonian
T = builderHF.hamiltonian.get_one_body().toarray().reshape(2,lattice.N_sites,lattice.N_sites)
```

+++ {"id": "d96b5df9-c26f-4dcc-87f6-bee7ed16e205"}

## Version 1: GHF trial

+++ {"id": "1f18437f-875d-486e-8ecc-22c8ec5f5ca8"}

Because of the open shell at half filling, we'll follow Ref. 1 and use a GHF trial w/ AFM constrained to the X-Y plane (eq 14)

$$
H_\mathrm{GHF} = -t\sum_{\langle ij\rangle \sigma} c^\dagger_{i\sigma}c_{j\sigma} + \text{h.c.} + \sum_{i}M_i c^\dagger_{i\uparrow}c_{i\downarrow} + \text{h.c.}
$$
where $M_i= (-1)^i\Delta_i$

+++ {"id": "95189cf3-a880-41d1-aa34-dfd8a9af4713"}

### Aside: custom Hartree-Fock solver

We'll want to optimize $M$ directly, and to do this we'll use AutoHF's custom ansatz power to build a modification of the `DIAG` ansatz

To do that we'll need:
* `orbitalFunc` This takes a `state` and produces the orbitals
* `rdmFunc` this takes a `state` and produces the rdm $\langle c^\dagger_{\sigma i}c_{\sigma^\prime j}\rangle$
* `state0` the initial state
* `state0_ref` the initial state that should correspond to e.g. $U=0$

```{code-cell} ipython3
:id: 37602e67-9b8b-4b66-8a1a-33b184a670bc

N = lattice.N_sites # will be captured by closure
Ne = N//2 # n_up = n_dn = Ne

# get the (-1)^i which is really the checkerboard pattern for a 4x4
signs = np.array([ (-1)**np.sum(s.coord) for s in lattice.sites])

# convert T -> GHF format
# TODO this should be in afqmctools
TGHF_ = np.block([[T[0],np.zeros_like(T[1])],
               [np.zeros_like(T[0]),T[1]]])
TGHF_ = jnp.array(TGHF_)


def orbitalFunc(state):
    # take the hopping matrix TGHF, and "set" M on the off-diagonal blocks
    TGHF = TGHF_+0.0
    TGHF = TGHF.at[N+np.arange(N),np.arange(N)].set(signs*state)
    TGHF = TGHF.at[np.arange(N),N+np.arange(N)].set(signs*state)
    lams,vecs = jnp.linalg.eigh(TGHF)
    return jnp.stack([vecs[:,:2*Ne]])

def rdmFunc(state):
    o = orbitalFunc(state)
    return autohf.solver.sd_to_rdm_noncollinear(o)
```

```{code-cell} ipython3
:id: 09cfee9a-7cef-4cf5-8af0-ed664049e3f0

def checkPH(state):
    '''check the spectrum of the 'faux' HF state is PH symmetric'''
    # take the hopping matrix TGHF, and "set" M on the off-diagonal blocks
    TGHF = TGHF_+0.0
    TGHF = TGHF.at[N+np.arange(N),np.arange(N)].set(signs*state)
    TGHF = TGHF.at[np.arange(N),N+np.arange(N)].set(signs*state)
    lams,vecs = jnp.linalg.eigh(TGHF)
    # verify symmetry
    return np.allclose(lams,-lams[::-1])
```

```{code-cell} ipython3
:id: bf76815f-d862-4604-8bbe-8f83c166ff73
:outputId: 2085ab00-6551-458a-bf8e-018e6cd372f4

hf_settings = {
    "verbose": False,
    "plot": False,
    "steps": 200,
    "seed": 1479,
    "noncollinear": True,
    "gpu": False,
    "ansatz": "CUSTOM",
    "opt_method": "lbfgs",
    "batch_size": 16,
    "nelec": [Ne, Ne],
}

rng = np.random.default_rng(42)
state0_ref = 0.0 * np.arange(N).reshape(N)
state0 = state0_ref + rng.normal(scale=0.01, size=(hf_settings["batch_size"], N))


dataHFC = autohf.solver.lattice_hf(
    hamiltonian=autohf.AutoHFHamiltonian(builderHF.hamiltonian),
    lattice=lattice,
    settings=hf_settings,
    state2orbitals=orbitalFunc,
    state2rdm=rdmFunc,
    state0=state0,
    state0_ref=state0_ref,
)
```

+++ {"id": "4df95593-82d2-41e7-95aa-262bf3fb7eb8"}

Let's check out what M values we got

```{code-cell} ipython3
:id: e650b2a3-278d-4bed-8849-bd9952a2d028
:outputId: 0380df75-7c1d-4fca-8576-44afe3681c1b

dataHFC[0]["state"]
```

```{code-cell} ipython3
:id: 5186a0f6-63c1-4eb9-ab05-33c912a696b5
:outputId: 489164a5-5f87-437d-9711-54d8adb6eb94

plt.imshow(dataHFC[0]["state"].reshape(4,4))
```

+++ {"id": "7956d434-41e8-4d69-b88e-4d4dc2f364cb"}

Hmm... This looks like we haven't fully converged into a translationally invariant solution, but's close. Let's increase the precision!

**⚠️ Warning** This notebook is setup for the CPU, the following cell takes a while (the GPU is a bit faster if you have one)

```{code-cell} ipython3
:id: 6c40673b-bdb8-4091-a6c6-26d0308a32db

wfn_fname = "autoHF_wfn.h5"
```

```{code-cell} ipython3
:id: bf090735-faac-4b21-90da-72283a85641f
:outputId: ca57afd7-0a3b-4d87-c125-5fe04e926178

hf_settings = {
    "verbose": False,
    "plot": False,
    "steps": 2000,  # increase
    "seed": 1479,
    "noncollinear": True,
    "gpu": False,
    "ansatz": "CUSTOM",
    "opt_method": "lbfgs",
    "num_batches": 16,
    "nelec": [Ne, Ne],
}

rng = np.random.default_rng(42)
state0_ref = 0.0 * np.arange(N).reshape(N)
state0 = state0_ref + rng.normal(scale=0.01, size=(hf_settings["num_batches"], N))


dataHFC2 = autohf.solver.lattice_hf(
    hamiltonian=autohf.AutoHFHamiltonian(builderHF.hamiltonian),
    lattice=lattice,
    settings=hf_settings,
    state2orbitals=orbitalFunc,
    state2rdm=rdmFunc,
    state0=state0,
    state0_ref=state0_ref,
    jaxoptargs=dict(tol=1e-14),  # secret sauce
)

autohf_to_afqmc(
    dataHFC2,
    output_fname = scratch_dir / wfn_fname
)
```

```{code-cell} ipython3
:id: 1a7cd7e0-e3b2-404b-9fbf-27a2effea563
:outputId: e1c36aad-d05a-47f2-e5b4-9dcb49bf3561

dataHFC2[0]["state"]
```

```{code-cell} ipython3
:id: 53b7ce8c-b10a-4376-ba29-bfaabbf18024
:outputId: 69f5278f-29f3-403b-8a76-5ba41ed39688

checkPH(dataHFC2[0]["state"])
```

```{code-cell} ipython3
:id: 1818b009-35b7-497f-b3bb-2fee961a09cd
:outputId: c8e409c6-a625-440d-ccaa-644ddee64dc4

plt.imshow(dataHFC2[0]["state"].reshape(4,4))
```

+++ {"id": "0bbfe20c-66db-4b99-9f69-dfa45a29f0d1"}

Let's check that this is a better energy than the previous energy

```{code-cell} ipython3
:id: c489bd73-3ae7-4943-b8a6-90705f5e3654
:outputId: 64057b24-c950-4eea-8870-709650f095cc

dataHFC2[0]["E"] < dataHFC[0]["E"]
```

+++ {"id": "d7142e2a-a771-496a-8300-8547e0eec68d"}

Great!

Now let's check if we can guess a better state, by removing translation invariance

```{code-cell} ipython3
:id: 8f623772-48d7-4b53-a629-70d668e830c1
:outputId: 00efbe7f-e45a-48bd-f6ec-77c17b0813b1

dataHFC2[1]["energy_func"](dataHFC2[0]["state"]) < dataHFC2[1]["energy_func"](dataHFC2[0]["state"]/np.abs(dataHFC2[0]["state"]))
```

```{code-cell} ipython3
:id: b02b905f-2789-4a5d-bac2-5847467d31a4
:outputId: 693f28ca-8e55-4eb7-e17e-1a95e26d77ad

uni_state = np.ones(lattice.N_sites)*np.mean(np.abs(dataHFC2[0]["state"]))
np.isclose(dataHFC2[1]["energy_func"](dataHFC2[0]["state"]),dataHFC2[1]["energy_func"](uni_state))
```

+++ {"id": "8944c9a6-fc6c-4cb0-9e07-8b8712bb5f45"}

So we see we've reached a minimum, and that a constant $M$ is the same as our nearly constant optimized $M_i$

+++ {"id": "5afb0278-218c-4d88-9bea-b1c0f2ce8b15"}

### Constructing the input file

+++ {"id": "6a2b6690-bb9c-4704-bd6c-91979a34c480"}

First we'll manually create a non-interacting state as our RHF initial state for the walkers

```{code-cell} ipython3
:id: 85b8e2f7-d035-4a12-bd53-a2eccf5ae40e

ham_fname = f"hamU{U}_afqmc.h5"
io.write_model_hamiltonian(builder.hamiltonian, scratch_dir / ham_fname,
                        nelec=nelec,spin_symm="collinear")
```

```{code-cell} ipython3
:id: 10617b29-48f0-4a65-9cd2-40f6199035c1

afqmc_params = {
        "timestep": 0.005,
        "steps": 10000,
        "population_control_interval": 2,
        "walker_ortho_interval": 2,
        "n_walkers_per_mpi_task": 40,
        "measure_interval_multiplier": 1,    
        "estimator": {
            "name": "energy",
            "overwrite": True,
            "print_components": True
        },
    }

write_json(
    scratch_dir / "afqmc.json",
    fwfn0=scratch_dir / wfn_fname,
    fham0=scratch_dir / ham_fname,
    exec_opts=afqmc_params,
    series=0
)
```

```{code-cell} ipython3
:id: 33a63a5d-ea54-4b9c-bbdd-14a281bb5d73
:outputId: cc66b225-6079-444f-d976-160bae2fe148

run_afqmc(
    run_dir = scratch_dir,
    input_file = "afqmc.json",
    np = 16,
    output_file = None,
)
```

```{code-cell} ipython3
:id: 66d213fc-1f02-41e4-adc3-e1449a04b403
:outputId: b9242712-fde9-4c91-a990-61cc5cf271f0

!energy_stats {scratch_dir / "qmc.s000.scalar.dat"} -t -e 20.0
```

```{code-cell} ipython3
:id: bb38d13d-6e93-4bcb-b6f2-e3a02fbc0827
:outputId: 02f67500-9f6f-4c12-a4d9-a5a206ddc890

settings = dict(
    fname = scratch_dir / "qmc.s000.scalar.dat",
    nequil = 14,
    trace = True,
    return_data= True,
    verbose = False,
)

data = analyze_scalar_data(settings)
E = data["mean_real"]
err = data["error_real"]
```

```{code-cell} ipython3
:id: 13292e5d-d485-4504-aae8-d8ae1f1ee6df
:outputId: 680bd92e-3786-4192-bdef-4302e9c420c9

E0_n1_lookup[U]
```

```{code-cell} ipython3
:id: 328d849f-f076-4442-8f44-cec585e8c2cc
:outputId: 3803d445-a19f-4f7c-92fe-87e4e1a1a57a

plt.errorbar([0],[E],yerr=[err],capsize=6,fmt='o-')
plt.axhline(y=E0_n1_lookup[U],ls='--')
plt.ylim(-13.63,-13.6)
plt.show()
```

+++ {"id": "70543110-d3bc-44ab-a30b-0bff04ba81ea"}

### For further thinking

* What if we didn't use the correct signs ($M_i = (-1)^i\Delta_i$) for the HF ansatz? Could we construct an even more constrained wave function?
* What do the local spin/charge observables of the optimized ansatz look like?
* What happens for $U=8$? Do we need a different wave function?
* What happens if you start from a free electron state instead of the HF state? (`(coeffs,free_wfn),spin_symm = make_free_elec(ham_fname,nelec,spin_symm="nc")`)

+++ {"id": "f7eab26a-3624-402e-83de-00ee38daf267"}

## Version 2: multi-Slater trial

+++ {"id": "ec6849d2-44e2-4e02-89c1-826a54fe9719"}

To take advantage of particle hole symmetry at half filling with a spin decomposition, besides using a GHF in the X-Y plane, we could use two particle-hole symmetric determinants

```{code-cell} ipython3
:id: 5144095e-d93d-4614-8fc0-27ed2ea6ce99

# TODO!
```

+++ {"id": "382b53bd-c986-497e-b362-e72bcd29fc23"}

## Version 3: Charge decomposition

+++ {"id": "27b1d4ad-ffd1-47ff-b31a-ac2bb5a40f37"}

The last alternative is to use a particle hole symmetry trial with the charge decomposition. Because we are focusing on positive $U$ interactions, the charge decomposition will create a complex phase. At half-filling, due to the particle symmetry the walker weights will end up being all real and positive so we don't have to worry about this

```{code-cell} ipython3
:id: 6b4b0c59-0f4c-40cf-8063-71eda6e6d018

# TODO!
```
