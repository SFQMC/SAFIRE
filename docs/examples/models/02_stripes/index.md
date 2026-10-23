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

+++ {"id": "af3003d3-6a86-4f3b-aedd-d2fcc19a22bc"}

# Stripes and the Square Hubbard Model

Author: Ryan Levy

```{code-cell} ipython3
:id: 9b2b92e5-990e-47bf-b235-3a37edbf4a0d

import os
# This is in case you have a GPU-enabled JAX but are running on CPU.
os.environ['JAX_PLATFORMS'] = 'cpu'
# Don't preallocate GPU memory, so that several notebooks can share the same GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# use x64 numbers
from jax import config
config.update("jax_enable_x64", True)
```

```{code-cell} ipython3
:id: f6e937d2-fde5-4f1b-afd9-6afb3798244a

import matplotlib.pyplot as plt
import numpy as np

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

+++ {"editable": true, "id": "083d8f36-02de-4281-a56c-7cd3d5add077"}

## Physics Background

In the paper _Stripes and spin-density waves in the doped two-dimensional Hubbard model: Ground state phase diagram_ [1] we see that for the nearest-neighbor square Hubbard model, the ground state may have stable staggered stripe order.

![](files/fig9.png)

We'll explore how to see this with AFQMC and the role of the trial wavefunction.

+++ {"editable": true, "id": "8dc800f8-72bf-4591-911f-7d99b692c8f4"}

---
[1] Hao Xu, Hao Shi, Ettore Vitali, Mingpu Qin, and Shiwei Zhang, _Stripes and spin-density waves in the doped two-dimensional Hubbard model: Ground state phase diagram_, Phys. Rev. Research **4**, 013239, published 28 March 2022.  
DOI: https://doi.org/10.1103/PhysRevResearch.4.013239

+++ {"id": "e8420a88-5a19-4b32-996d-21b3d85aa58c"}

## Obtaining the Spin Density

Let's look at a 10×4 system with 0.2 doping at $U=6$ with a staggered AFM pinning field at the edges. The pinning breaks symmetry in order to understand the response of the system to the applied field.

When confronted with a new problem, the first thing we generally do is use a non-interacting (free-electron) wavefunction as the trial. Let's do that now.

```{code-cell} ipython3
:id: 9d8005b9-ae0e-42e9-9d87-7fa2d233c43c

U = 6.0
lattice_dims = (10,4)

from safiretools import Lattice

lattice = Lattice.from_type(
    'square', lattice_dims, boundaries=('open', 'periodic'), twists=(0.0, 0.0)
)

Ne = int(lattice.N_sites*(1-0.2))//2
nelec = (Ne,Ne)
nelec
```

```{code-cell} ipython3
from safiretools import vis

vis.plot_lattice(lattice)
```

+++ {"id": "a02dd178-90bb-463e-b5d6-75fccb288272"}

The Hamiltonian will be simple,

$$
\begin{aligned}
H &= H_t + H_U + H_\text{pin} \\
H_t &= -t \sum_{\langle i j\rangle \sigma} \left(c^\dagger_{i\sigma} c_{j\sigma} + \mathrm{h.c.}\right) \\
H_U &= U\sum_i n_{i\uparrow}n_{i\downarrow}\\
H_\text{pin} &= \sum_{i\in \text{edge}} (-1)^{i_x+i_y} h_i S^z_i
\end{aligned}
$$

where `PinningField.afm` adds the edge pinning. We'll assume $t=1$ and $h_i=0.5$, just like in the paper.

```{code-cell} ipython3
---
id: 53474a16-1c9e-4c9a-9d49-7022c8a659f1
outputId: 731af8f0-e234-49d5-cfc5-d38fb5483845
colab:
  base_uri: https://localhost:8080/
---
from safiretools import LatticeHamiltonian, PinningField

# staggered AFM pinning on the two open edges at the ends of axis 0
pinning = (PinningField.afm(0.5, staggered=True), None)
```

```{code-cell} ipython3
:id: f94e92dd-9802-46ea-a1db-0baffb7efbb4

# standard Hubbard terms; nearest-neighbor hopping with t=1 is included by default
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    U=U,
    pinning=pinning,
)
```

```{code-cell} ipython3
---
id: 5db7dbbb-0484-480c-8857-71986b9da4c4
outputId: a0cc1270-40df-46ca-fc5a-094aae93aa2b
colab:
  base_uri: https://localhost:8080/
---
from safiretools import Wavefunction

# get a trial wavefunction: First, let's try a free-electron (i.e. non-interacting) wavefunction.
# from_free_electron() takes the Hamiltonian itself.
#
# No twist is needed here: the open boundary along axis 0 and the pinning field
# already split the shells, so the determinant is well defined. On a fully
# periodic, unpinned lattice it would not be. See the 4×4 Hubbard example.
wavefunction = Wavefunction.from_free_electron(
    source=hamiltonian,
    nelec=nelec
)
```

+++ {"id": "ec2d1ea0-fc24-45f6-a799-5e9d14e39639"}

Now we'll write the job file to run AFQMC.

```{code-cell} ipython3
:id: 866c943b-b811-40b9-a8a5-e257e30bda24

from safiretools import write_jobfile

bp_lengths = [500]

# make an input file
afqmc_params = {
    "timestep": 0.005,
    "steps": 10000,
    "population_control_interval": 2,
    "walker_ortho_interval": 2,
    "num_walkers": 640,
    "measure_interval": 2,
    "equilibration_steps": 2000,
    "estimators": {
        "backprop": {
            "walker_ortho_interval": 2,
            "propagation_steps": bp_lengths,
            "onerdm": {}
        }
    },
    "propagator": {
        "use_cp_constraint": True,
        "project_force_bias": True
    }
}

write_jobfile(
    "afqmc.json",
    {
        "seed": 42,                          # just for reproducibility
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            **afqmc_params
        }
    }
)
```

```{code-cell} ipython3
---
id: 96ec3330-42f6-4b14-af45-7e18a5225f7f
outputId: 8c812e11-d1f3-454d-82c4-0322455b50bb
colab:
  base_uri: https://localhost:8080/
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

```{code-cell} ipython3
---
id: nn3u7u8k5xcp
colab:
  base_uri: https://localhost:8080/
  height: 921
outputId: 78be3a24-0f0c-4577-cb48-6ad867cb1f1b
---
from safiretools import Results

results = Results("afqmc.results.h5")
vis.plot_timeseries(results)
```

+++ {"id": "e5e2892a-59fe-412e-9d8b-42f42c05ff79"}

### Analysis

We'll now analyze the spin density of the output given the free-electron trial.

```{code-cell} ipython3
:id: 08eaa53d-eed0-4540-9bb8-08ea5aeab4c0

x, y = np.arange(lattice.L[0]), np.arange(lattice.L[1])
# staggered signs (-1)^(x+y), in the site order of the lattice
staggered_signs = ((-1)**(x[:, None] + y[None, :])).flatten()

def spin_density(rdm):
    """S^z on each site, from a collinear one-rdm."""
    return 0.5*(rdm[0].diagonal() - rdm[1].diagonal()).real

def column_sz(rdm):
    """Staggered S^z averaged over each column, from a collinear one-rdm."""
    return (staggered_signs*spin_density(rdm)).reshape(lattice.L).mean(1)

bp_rdm = f"BackPropEstimator/Steps={bp_lengths[-1]}/OneRDM"

# DMRG data of the form: site Sz Nup Ndn
SzDmrg = np.loadtxt("../files/data_10x4_Ne16U6.0_15360.dat",usecols=(1,))
dmrg_sz = -(staggered_signs*SzDmrg).reshape(lattice.L).mean(1)
```

```{code-cell} ipython3
:id: 6c223f77-f3f4-42a5-bacb-a0f84a868f2b

Zs, delta_Sz = results.evaluate(spin_density, [bp_rdm])
```

```{code-cell} ipython3
:id: 79ed1667-cd75-451d-b68a-5897ca697206
:outputId: fb38354c-9921-4f09-86ef-eb21cf200940

plt.imshow(Zs.reshape(*lattice.L).T)
plt.colorbar()
plt.show()

plt.imshow(delta_Sz.reshape(*lattice.L).T)
plt.colorbar()
plt.show()
```

```{code-cell} ipython3
:id: d2e9c723-0471-404c-acb5-30e862b7905f
:outputId: a53dc884-c7ab-43d4-9b39-1ad5c3e23957

# now we'll average over the columns
sz_cols, sz_cols_err = results.evaluate(column_sz, [bp_rdm])

plt.errorbar(np.arange(lattice.L[0]),sz_cols,yerr=sz_cols_err,
             fmt='o-',label="AFQMC Free Electron")
plt.plot(dmrg_sz,label="DMRG",c='k')

plt.xlabel("Col")
plt.ylabel("Staggered $S^z$")
plt.legend()
plt.show()
```

+++ {"id": "e9fbcc21-8216-4f25-a64e-54f84dc4023c"}

## Self-Consistency with Hartree-Fock

The idea behind self-consistent AFQMC is to have the trial wavefunction match as much as possible the output of AFQMC.

For this model, we can do this by introducing an effective Hubbard model, where $U$ is replaced with $U_\text{eff}$. By solving Hartree-Fock (mean-field theory) for this effective model and using the result as a trial wavefunction, we can scan through different $U_\text{eff}$ to find the one that matches the best.

We'll use AutoHF to explore creating a trial wavefunction for AFQMC. Let's do a few between $U_\text{eff}=1$ and $U_\text{eff}=4$.

_Note:_ The Hartree-Fock code is faster if you use a GPU.

```{code-cell} ipython3
:id: ee2b575c-7d27-4a1d-a9b9-e9b02a682f8f
:outputId: 743ad05d-6d2a-4eba-ec3c-92e78bcbe2e3

from autohf import solve_hf, AutoHFHamiltonian

Ueffs = [1,2,3,4]
trial_wfns = {}
for Ueff in Ueffs:

    hamiltonian_eff = LatticeHamiltonian.from_lattice(
        lattice,
        U=Ueff, # We want our HF solver to solve Ueff not U
        pinning=pinning,
    )

    hf_settings = dict(
        steps = 2000,
        opt_method="lbfgs",
        ansatz="SD_ROT",
        nelec = nelec,
        batch_size = 8,
        seed = 1,
        noncollinear = False
    )
    results = solve_hf(
        hamiltonian=AutoHFHamiltonian(hamiltonian_eff),
        settings=hf_settings,
    )
    # the trial solves the Ueff model, but AFQMC runs with the original Hamiltonian (U=6)
    trial_wfns[Ueff] = Wavefunction.from_single_determinant(
        tuple(np.asarray(o) for o in results[0]["orbitals"])
    )

    write_jobfile(
        f"afqmc_{Ueff}.json",
        {
            "seed": 42,
            "execute": {
                "wavefunction": trial_wfns[Ueff],
                "hamiltonian": hamiltonian,
                **afqmc_params
            }
        }
    )
    print(f"Wrote afqmc_{Ueff}.json")
```

```{code-cell} ipython3
:id: 3d68db2c-e000-4c4f-8b48-2f4bd5836749
:scrolled: true
:tags: [scroll-output]

for Ueff in Ueffs:
    !mpirun -n 16 $$AFQMC_EXEC afqmc_{Ueff}.json
```

```{code-cell} ipython3
:id: 9ac5439b-daa4-4cf4-a248-e2466acf32e5

# Ueff=0 stands for the free-electron trial of the first run
runs = {0: Results("afqmc.results.h5")}
for Ueff in Ueffs:
    runs[Ueff] = Results(f"afqmc_{Ueff}.results.h5")
```

```{code-cell} ipython3
:id: 7f7e1dc8-4378-4da3-a77a-8e2638ed3c0e

for Ueff,run in runs.items():
    Zs, delta_Sz = run.evaluate(spin_density, [bp_rdm])

    fig,axes = plt.subplots(1,2,figsize=(12,4))
    p = axes[0].matshow(Zs.reshape(*lattice.L).T)
    plt.colorbar(p,ax= axes[0], label = "$S^z$")


    p =axes[1].matshow(delta_Sz.reshape(*lattice.L).T)
    plt.colorbar(p,ax= axes[1], label=r"$\Delta S^z$")
    if Ueff==0:
        fig.suptitle("Free Electron")
    else:
        fig.suptitle(f"{Ueff=}")
    plt.show()
```

+++ {"id": "8445abd3-e472-4406-9c2a-9ca3a8c8d3a3"}

## Self-Consistency Condition

We'll now measure the error between the trial wavefunction's observable and the output from AFQMC.

```{code-cell} ipython3
:id: eff15403-1be8-4f08-8ed8-9ccf212e1d9f

trial_sz = {}
for Ueff in runs:
    if Ueff==0:
        trial = wavefunction
    else:
        trial = trial_wfns[Ueff]
    # determinant() gives the leading determinant's alpha and beta orbitals
    alpha, beta = trial.determinant(0)

    # one-rdm of a determinant with orthonormal orbitals
    rdm = np.array([alpha.conj() @ alpha.T, beta.conj() @ beta.T])
    trial_sz[Ueff] = column_sz(rdm)
```

```{code-cell} ipython3
:id: 2926392c-64b7-4c5b-be5a-d289adc38ff5

# now we'll average over the columns
for Ueff,run in runs.items():
    sz_cols, sz_cols_err = run.evaluate(column_sz, [bp_rdm])
    if Ueff==0:
        label = "Free Electron"
    else:
        label = f"HF Ueff={Ueff}"

    plt.errorbar(np.arange(lattice.L[0]),
                 sz_cols,yerr=sz_cols_err,
                 fmt='o--',label=label)
plt.plot(dmrg_sz,'x-',label="DMRG",c='k')

plt.xlabel("Col")
plt.ylabel("Staggered $S^z$")
plt.legend()
plt.show()
```

```{code-cell} ipython3
:id: 61bcaf57-b7bd-4671-801f-31ab5d6684e1

xs,ys,yerrs = [],[],[]
for Ueff,run in runs.items():
    deviation, deviation_err = run.evaluate(
        lambda rdm: np.mean(np.abs(column_sz(rdm)-dmrg_sz)), [bp_rdm]
    )

    xs.append(Ueff)
    ys.append(deviation)
    yerrs.append(deviation_err)

plt.errorbar(xs,ys,yerr=yerrs,fmt='o-',label="AFQMC",capsize=6)
plt.xticks(xs)
plt.xlabel("Ueff")
plt.ylabel(r"$\langle |$DMRG $S^z - $AFQMC $S^z|\rangle$",fontsize=12)
plt.legend()
plt.grid()
plt.show()


xs,ys,yerrs = [],[],[]
for Ueff,run in runs.items():
    deviation, deviation_err = run.evaluate(
        lambda rdm: np.mean(np.abs(column_sz(rdm)-trial_sz[Ueff])), [bp_rdm]
    )

    xs.append(Ueff)
    ys.append(deviation)
    yerrs.append(deviation_err)

plt.errorbar(xs,ys,yerr=yerrs,fmt='o-',label="AFQMC",capsize=6)
plt.xticks(xs)
plt.xlabel("Ueff")
plt.ylabel(r"$\langle |$Trial $S^z - $AFQMC $S^z|\rangle$",fontsize=12)
plt.legend()
plt.grid()
plt.show()
```

+++ {"id": "9e472093-a34e-4083-bb1a-3e04e07f19d2"}

So we can see that somewhere around $U_\text{eff}\approx 3$
would have the lowest error for self-consistency.
Despite the difference between DMRG and AFQMC being on average 0.01, there is a clear optimal trial wavefunction to use. For fine-tuning between $U_\text{eff}\in [2,4]$, we'll need more sophisticated methods than presented here.

Ref. [1] finds the optimal $U_\text{eff} \approx 2.77$.

+++ {"id": "672d9ba4-f659-466d-be57-8fa082fcfb0e"}

## Your Turn

### Part 1 – Recreate Fig. 1

+++ {"editable": true, "id": "0ac955ea-ab2d-49fb-b9dd-9a0fafdaf159"}

![](files/fig1.png)

+++ {"editable": true, "id": "27958cd8-4035-4720-82b8-c163b1a89e99"}

Let's recreate Fig. 1 of Xu et al. [1] by changing

- Lattice size 10 × 4 to 20 × 4
- Doping 0.2 to 0.1 hole doping
- $U = 6$ ✓

+++ {"id": "edc3a318-40a1-40cb-86bf-b85ae366fc9b"}

### Part 2 – Try Different Settings

Let's use this as a testbed to explore how AFQMC responds to different settings. Some questions to consider:

- Use $U_\text{eff} = U$, the bare $U$ Hartree-Fock trial. How does the quality of the results change?
- We compared spin densities; what about the hole density? This is effectively the same as comparing the charge density.
- Use a trial with a small number of steps so that the solver doesn't converge. How do the results change? How could you tell this trial isn't good?
- The free-electron trial isn't a valid RHF state (why?). What would be the corresponding RHF solution? What does that output look like?
- Explore the parameters of back-propagation: how does the one-body reduced density matrix change as a function of the back-propagation length? You can give several lengths in `propagation_steps` to see the results converge.
