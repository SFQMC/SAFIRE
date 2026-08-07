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

+++ {"id": "m5yYPGDk-NIu"}

# Observables: Pair Correlation functions

<b>Goal:</b>
At the end of this tutorial, you will know how to measure pair correlation functions using SAFIRE for a lattice models.

## What you will learn

1. How to set up the HDF5 and json input files to request pair correlation functions
2. how to post process the correlation functions

+++ {"id": "7FBZ0COY-NIw"}

## Introduction

By pair correlation functions, we mean

$$
P^{\alpha \beta}_{ij} ≡ ⟨ Δ^†_{i,\alpha}Δ_{j,\beta} ⟩,
$$
where  $i,j$ are basis set indices,
$α, β$ are spatial offset indices, and

$$
Δ^†_{i,\alpha} \equiv \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i⇑} c^{†}_{i + e_α ⇓} -  c^{\dagger}_{i⇓} c^{†}_{i + e_α ⇑} \right),
$$

where $e_α$ is the *index offset* associated with the spatial offset.
For example, $e_\alpha$ may correspond to a $+x$ offset.
For convenience, we define,

$$
\bar{i}_α \equiv i + e_\alpha,
$$

such that,

$$
Δ^†_{i,\alpha} \equiv \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i⇑} c^{†}_{\bar{i}_α ⇓} -  c^{\dagger}_{i⇓} c^{†}_{\bar{i}_α ⇑} \right),
$$

and

$$
Δ_{j,\beta} \equiv \frac{1}{\sqrt{2}} \left( c_{\bar{j}_\beta⇓} c_{j ⇑}  - c_{\bar{j}_\beta ⇑} c_{j ⇓} \right),
$$

Using this language, the pair correlation function can be written as,

$$
P^{\alpha \beta}_{ij} ≡ \frac{1}{2} ⟨ \left(c^{\dagger}_{i⇑} c^{†}_{\bar{i}_α ⇓} -  c^{\dagger}_{i⇓} c^{†}_{\bar{i}_α ⇑} \right)  \left( c_{\bar{j}_\beta⇓} c_{j ⇑}  - c_{\bar{j}_\beta ⇑} c_{j ⇓} \right) ⟩.
$$

SAFIRE implements this form of pair correlation function.
It requires a set of $\bar{i}_\alpha = i + e_\alpha$, in the HDF5 input file.
Additionally, specific pairings may be selected in the json input file.
This allows the user to skip some of the offsets saved in the HDF5 file if desired.
We will explore this in more detail below.

+++

## Defining the Hamiltonian

```{code-cell} ipython3
from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.visualize import plot_lattice
from afqmctools.hamiltonian.model.director import HamiltonianDirector
from pathlib import Path

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

lattice = get_lattice(
    params=dict(
        L1 = 4,
        L2 = 4,
        boundary1 = "PBC",
        boundary2 = "PBC"
    )
)

# plot the lattice for reality checks!
plot_lattice(lattice)

params = dict(
    hamiltonian=dict(
        t=1.0,
        U=4.0
    )
)

hamiltonian = HamiltonianDirector(source=params, lattice=lattice).build()

from afqmctools.utils.io import write_model_hamiltonian

write_model_hamiltonian(
    hamiltonian=hamiltonian,
    fname=scratch_dir/"afqmc.h5"
)
```

+++ {"id": "Np-zIkGusNIP"}

## Saving index offsets in the HDF5 input

The Lattice class from afqmctools can be directly used to find the pairs that we need via the `.get_directed_pairs()` function.
`.get_directed_pairs()` needs a list of "directions", and it will return a dictionary where the keys are the requested directions, and the values are a list of index offsets, $\bar{i}_\alpha [i]$, such that $\bar{i}_{\alpha} [i] = i + e_\alpha$.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 907
id: VjaOtDOltPAj
outputId: 30ac79d1-e76e-4b8e-8c2c-6341210f4b4f
---

pairs_dict=lattice.get_directed_pairs(
  directions=["s","+x","-x","+y","-y"]
)

for key,value in pairs_dict.items():
  print(key, value)
```

afqmctools also provides a function for saving these pairs to HDF5 in the appropriate format.

```{code-cell} ipython3
%load_ext autoreload
%autoreload
from afqmctools.utils.io import write_pair_correlators

write_pair_correlators(scratch_dir / "afqmc.h5",pairs_dict)
```

+++ {"id": "lkkZMHMWvFAU"}

### Adding Custom Pairings

We note that custom pairings can be added by simply inserting a list of offsets with a new key.
For example, we can manually add "+x+y" pairing to the `pairs_dict` dictionary as

```{code-cell} ipython3
pairs_dict["xy"] = [ 5, 6 , 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0]

# update hdf5
write_pair_correlators(scratch_dir / "afqmc.h5",pairs_dict)
```

+++ {"id": "iwp3CeLownyV"}

## Requesting the pair correlation function in the json input file

The pair correlation function observable may be requested as any other observable within an estimator block.
However, the pair correlation function observable has a few unique parameters that must be set.
They are:

- "filename" which is the name of the HDF5 file where the pairing offsets were saved. These can saved in the same file as the Wavefunction, and/or Hamiltonian or in their own independent input file. Whichever you prefer.
- "pair_type" is a list of the names of the types of pairing you want to measure. For example, to measure all of the types of pairing that we defined above, we would use `"pair_type" : ["s","+x","-x","+y","-y","xy"]`.

<div class="alert alert-block alert-info">
<b>Note:</b>
  SAFIRE will measure all combinations of pairings that you request.
  i.e. it will compute $P^{\alpha \beta}_{ij}$ where both $\alpha$ and $\beta$ run over the full set of pairing definitions.
</div>

Since the pair correlation functions do not generally commute with the Hamiltonian, we should use a back-propagated estimator.
A back-propagated estimator block with pair correlation functions would look like the following,

```json
"estimator": {
    "name": "back_propagation",
    "path_restoration": true,
    "bp_walker_ortho_interval": 10,
    "equil": 200,
    "pair_correlators" : {
        "name" : "my_pair_correlators",
        "filename" : "afqmc.h5",
        "pair_type" : ["s","+x","-x","+y","-y","xy"]
    }
  }
```

If we decided we only wanted s-wave pair correlations, we could use

```json
"pair_correlators" : {
  "name" : "my_pair_correlators",
  "filename" : "afqmc.h5",
  "pair_type" : ["s","+x","-x","+y","-y","xy"]
}
```

instead of the above.

### Full sample input file

```json
{
  "afqmc": {
    "project": {
      "id": "qmc",
      "series": 0
    },
    "execute": {
      "walker_set": {
        "walker_type": "NONCOLLINEAR"
      },
      "wavefunction": {
        "filename": "afqmc.h5"
      },
      "timestep": 0.01,
      "steps": 12000,
      "n_walkers_per_mpi_task": 80,
      "measure_interval_multiplier": 1,
      "population_control_interval": 10,
      "walker_ortho_interval": 10,
      "seed": 42,
      "estimator": {
        "name": "back_propagation",
        "path_restoration": true,
        "bp_walker_ortho_interval": 10,
        "equil": 200,
        "pair_correlators" : {
            "name" : "my_pair_correlators",
            "filename" : "afqmc.h5",
            "pair_type" : ["s","+x","-x","+y","-y","xy"]
        }
      }
    }
  }
}
```

```{code-cell} ipython3
from afqmctools.inputs.from_hdf import write_json
from afqmctools.inputs.from_autohf import autohf_to_afqmc
import autohf

nelec = (8, 8)

hf_settings = {
    "verbose": False,
    "plot": False,
    "steps": 2000,
    "seed": 1479,
    "noncollinear": True,
    "gpu": False,
    "ansatz": "SD_ROT",
    "seed": 42,
    "batch_size": 8,
    "nelec": nelec,
}
    
wfn = autohf.solver.lattice_hf(
    hamiltonian=autohf.AutoHFHamiltonian(hamiltonian),
    lattice=lattice,
    settings=hf_settings,
)

autohf_to_afqmc(
    wfn,
    output_fname = scratch_dir / "afqmc.h5"
)


afqmc_params = {
    "timestep": 0.005,
    "steps": 10000,
    "population_control_interval": 2,
    "walker_ortho_interval": 2,
    "n_walkers_per_mpi_task": 40,
    "measure_interval_multiplier": 1,    
    "estimator": {
        "name": "back_propagation",
        "path_restoration": True,
        "bp_walker_ortho_interval": 10,
        "equil": 200,
        "pair_correlators" : {
            "name" : "my_pair_correlators",
            "filename" : "afqmc.h5",
            "pair_type" : ["s","+x","-x","+y","-y","xy"]
        }
    }
}

write_json(
    scratch_dir / "afqmc.json",
    fwfn0=scratch_dir / "afqmc.h5",
    fham0=scratch_dir / "afqmc.h5",
    exec_opts=afqmc_params,
    series=0
)
```

```{code-cell} ipython3
:id: SrM8syoYzUeS

from tutorial_utils import run_afqmc
run_afqmc(
        run_dir = scratch_dir,
        input_file = "afqmc.json",
        np = 16,
        output_file = None
    )   
```

+++ {"id": "WIxdjnH-zopo"}

## Post-processing pair correlations functions

```{code-cell} ipython3
:id: eh8JuNVdag-o

import numpy as np
from warnings import warn

from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.visualize import plot_lattice
from afqmctools.analysis.average import average_pair_correlation


P,dP,correlator_names = average_pair_correlation("qmc.s000.stat.h5")

for i in range(len(correlator_names)):
    print(f"correlator name: {correlator_names[i]}")

a = correlator_names.index("s")
b = correlator_names.index("+x")
iav = 3

n_correlators = len(correlator_names)

P_ab_ij = P[iav].reshape(n_correlators,n_correlators,lattice.N_sites,lattice.N_sites)

# take the s, +x correlator
P_s_x_ij = P_ab_ij[a,b,:,:].reshape(lattice.N_sites,lattice.N_sites)
# use the first site as the reference site
P_s_x_0j = P_s_x_ij[0,:].reshape(lattice.L)

plot_lattice(
    lattice,
    density=np.abs(P_s_x_0j.real),
    norm_type="log",
    cmap="plasma"
)
```

+++ {"id": "gGT0fdst-NIz"}

## Summary

### What you learned

1. How to set up the HDF5 and json input files to request pair correlation functions
2. how to post process the correlation functions
