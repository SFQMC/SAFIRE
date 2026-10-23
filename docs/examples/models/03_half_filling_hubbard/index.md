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

# The Square Hubbard Model at Half Filling

Author: Ryan Levy

```{code-cell} ipython3
:id: 6d86eb21-7fcb-4c98-9780-3d657dff85a2

import os
# This is in case you have a GPU-enabled JAX but are running on CPU.
os.environ['JAX_PLATFORMS'] = 'cpu'
# Don't preallocate GPU memory, so that several notebooks can share the same GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# use x64 numbers
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt
import numpy as np

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

+++ {"editable": true, "id": "f91cf26c-5663-416e-9802-aeb1aa5a77c6"}

## Physics Background

The Hubbard model at half filling ($n=1$) has a special symmetry known as particle-hole symmetry, which allows auxiliary-field methods to be sign-free. However, care must be taken to make sure that phaseless/constrained-path AFQMC is used properly to exploit said symmetry.

[1] M. Qin, H. Shi, and S. Zhang, _Benchmark study of the two-dimensional Hubbard model with auxiliary-field quantum Monte Carlo method_, Phys. Rev. B **94**, 085103, published 4 August 2016.  
DOI: https://doi.org/10.1103/PhysRevB.94.085103

+++ {"id": "678e6d9b-4d8f-4f74-87ee-2420b8588c57"}

## Lattice and Hamiltonian Setup

```{code-cell} ipython3
:id: 132741fa-48b2-46a2-a980-3d58740b78d3

# taken from 10.1103/PhysRevB.45.10741, U => E_0
E0_n1_lookup = {4 : -13.62185,
                8 : -8.46888}
```

```{code-cell} ipython3
:id: 55293f57-e436-47c6-bef7-1b095b690f57

U = 4.0
```

```{code-cell} ipython3
:id: b2a30f8a-5c48-4272-b0df-14d9c2062e59
:outputId: 9a69fa7d-1093-4d5a-8245-f57f3639017c

from safiretools import Lattice

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))

Ne = lattice.N_sites//2
nelec = (Ne,Ne)
nelec
```

```{code-cell} ipython3
:id: 9a18f6e5-68f2-46cd-ae51-3bf8826dc679
:outputId: 47dcfa86-4f49-4688-b813-cc11bcc1c244

from safiretools import vis

vis.plot_lattice(lattice)
plt.show()
```

+++ {"id": "5f7f1b9d-6230-41a0-8037-da334866defb"}

The Hamiltonian will be

$$
\begin{aligned}
H &= H_t + H_U \\
H_t &= -t \sum_{\langle i j\rangle \sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + \mathrm{h.c.}\right) \\
H_U &= U\sum_i n_{i\uparrow}n_{i\downarrow}
\end{aligned}
$$

```{code-cell} ipython3
:id: 7416bbce-9541-46bb-bc52-1e6a42e2d001
:outputId: 8bcb041d-4e1e-4a10-9bd2-d5a7a2385793

from safiretools import LatticeHamiltonian

hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=U)
```

```{code-cell} ipython3
:id: 3670ecc2-0aa9-4a34-ab5b-b903f7fddc08
:outputId: 70b88144-c728-4f1c-d8de-70a78fa2914e

hamiltonian_hf = LatticeHamiltonian.from_lattice(
    lattice,
    U=U, # if you want an effective U, you can change this
)
```

```{code-cell} ipython3
:id: 67700d97-b76c-4d89-a7f9-0b814e411497

# get the one-body Hamiltonian
T = hamiltonian_hf.get_one_body().toarray().reshape(2,lattice.N_sites,lattice.N_sites)
```

+++ {"id": "d96b5df9-c26f-4dcc-87f6-bee7ed16e205"}

## GHF Trial

+++ {"id": "1f18437f-875d-486e-8ecc-22c8ec5f5ca8"}

Because of the open shell at half filling, we'll follow Ref. [1] and use a GHF trial with AFM constrained to the X-Y plane (Eq. 14),

$$
H_\mathrm{GHF} = -t\sum_{\langle ij\rangle \sigma} \left(c^\dagger_{i\sigma}c_{j\sigma} + \text{h.c.}\right) + \sum_{i} \left(M_i c^\dagger_{i\uparrow}c_{i\downarrow} + \text{h.c.}\right),
$$
where $M_i= (-1)^i\Delta_i$.

+++ {"id": "95189cf3-a880-41d1-aa34-dfd8a9af4713"}

### Aside: Custom Hartree-Fock Solver

We'll want to optimize $M$ directly, and to do this we'll use AutoHF's custom ansatz support to build a modification of the `DIAG` ansatz.

To do that we'll need:
* `orbitalFunc`, which takes a `state` and produces the orbitals
* `rdmFunc`, which takes a `state` and produces the rdm $\langle c^\dagger_{\sigma i}c_{\sigma^\prime j}\rangle$
* `state0`, the initial state
* `state0_ref`, the initial state that should correspond to e.g. $U=0$

```{code-cell} ipython3
:id: 37602e67-9b8b-4b66-8a1a-33b184a670bc

import jax.numpy as jnp

from autohf import solve_hf, AutoHFHamiltonian
from autohf.solver import sd_to_rdm_noncollinear

# get the (-1)^i which is really the checkerboard pattern for a 4×4
signs = np.array([ (-1)**np.sum(s.coord) for s in lattice.sites])
N = lattice.N_sites
# convert T -> GHF format
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
    return sd_to_rdm_noncollinear(o)
```

```{code-cell} ipython3
:id: 09cfee9a-7cef-4cf5-8af0-ed664049e3f0

def checkPH(state):
    '''check that the spectrum of the faux HF state is PH symmetric'''
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
    "nelec": nelec,
}

rng = np.random.default_rng(42)
state0_ref = 0.0 * np.arange(N).reshape(N)
state0 = state0_ref + rng.normal(scale=0.01, size=(hf_settings["batch_size"], N))


dataHFC = solve_hf(
    hamiltonian=AutoHFHamiltonian(hamiltonian_hf),
    settings=hf_settings,
    state2orbitals=orbitalFunc,
    state2rdm=rdmFunc,
    state0=state0,
    state0_ref=state0_ref,
)
```

+++ {"id": "4df95593-82d2-41e7-95aa-262bf3fb7eb8"}

Let's check out what $M$ values we got.

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

Hmm... This looks like we haven't fully converged into a translationally invariant solution, but it's close. Let's increase the precision!

**⚠️ Warning** This notebook is set up for the CPU; the following cell takes a while (the GPU is a bit faster if you have one).

```{code-cell} ipython3
:id: bf090735-faac-4b21-90da-72283a85641f
:outputId: ca57afd7-0a3b-4d87-c125-5fe04e926178

hf_settings = {
    "verbose": False,
    "plot": False,
    "steps": 2000,  # increase
    "seed": 149,
    "noncollinear": True,
    "gpu": False,
    "ansatz": "CUSTOM",
    "opt_method": "lbfgs",
    "batch_size": 16,
    "nelec": nelec,
}

rng = np.random.default_rng(42)
state0_ref = 0.0 * np.arange(N).reshape(N)
state0 = state0_ref + rng.normal(scale=0.01, size=(hf_settings["batch_size"], N))


dataHFC2 = solve_hf(
    hamiltonian=AutoHFHamiltonian(hamiltonian_hf),
    settings=hf_settings,
    state2orbitals=orbitalFunc,
    state2rdm=rdmFunc,
    state0=state0,
    state0_ref=state0_ref,
    jaxoptargs=dict(tol=1e-8), # tune to get closer to the minimum
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

Let's check that this is a better energy than the previous energy.

```{code-cell} ipython3
:id: c489bd73-3ae7-4943-b8a6-90705f5e3654
:outputId: 64057b24-c950-4eea-8870-709650f095cc

dataHFC2[0]["E"] < dataHFC[0]["E"]
```

+++ {"id": "d7142e2a-a771-496a-8300-8547e0eec68d"}

Great!

Now let's check if we can guess a better state by explicitly restoring translation invariance.

```{code-cell} ipython3
:id: b02b905f-2789-4a5d-bac2-5847467d31a4
:outputId: 693f28ca-8e55-4eb7-e17e-1a95e26d77ad

uni_state = np.ones(lattice.N_sites)*np.mean(np.abs(dataHFC2[0]["state"]))
np.isclose(dataHFC2[1]["energy_func"](dataHFC2[0]["state"]),dataHFC2[1]["energy_func"](uni_state))
```

+++ {"id": "8944c9a6-fc6c-4cb0-9e07-8b8712bb5f45"}

So we see we've reached a minimum, and that a constant $M$ is the same as our nearly constant optimized $M_i$.

+++ {"id": "5afb0278-218c-4d88-9bea-b1c0f2ce8b15"}

### Preparing the SAFIRE Input Files

+++ {"id": "6a2b6690-bb9c-4704-bd6c-91979a34c480"}

Next, we turn the AutoHF orbitals into a SAFIRE trial wavefunction.

```{code-cell} ipython3
:id: 85b8e2f7-d035-4a12-bd53-a2eccf5ae40e

from safiretools import Wavefunction

wavefunction = Wavefunction.from_single_determinant(np.array(dataHFC2[0]["orbitals"][0].reshape(2, N, -1)))
```

```{code-cell} ipython3
:id: 10617b29-48f0-4a65-9cd2-40f6199035c1

from safiretools import write_jobfile

write_jobfile(
    "afqmc.json",
    {
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            "timestep": 0.005,
            "steps": 20000,
            "equilibration_steps": 1000,
            "num_walkers": 640,
        }
    }
)
```

```{code-cell} ipython3
:id: 33a63a5d-ea54-4b9c-bbdd-14a281bb5d73
:outputId: cc66b225-6079-444f-d976-160bae2fe148
:tags: [scroll-output]

!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
:id: bb38d13d-6e93-4bcb-b6f2-e3a02fbc0827
:outputId: 02f67500-9f6f-4c12-a4d9-a5a206ddc890

from safiretools import Results
results = Results("afqmc.results.h5")

vis.plot_timeseries(results, skip=250)
E_ref = E0_n1_lookup[U]
plt.axhline(E_ref, color="crimson")
```

+++ {"id": "70543110-d3bc-44ab-a30b-0bff04ba81ea"}

### For Further Thinking

* What if we didn't use the correct signs ($M_i = (-1)^i\Delta_i$) for the HF ansatz? Could we construct an even more constrained wavefunction?
* What do the local spin/charge observables of the optimized ansatz look like?
* What happens for $U=8$? Do we need a different wavefunction?
* What happens if you start from a free-electron state instead of the HF state? (`free_wfn = Wavefunction.from_free_electron(hamiltonian, nelec, spin_symm="nc")`; this warns, since the untwisted lattice is open-shell at half filling.)
