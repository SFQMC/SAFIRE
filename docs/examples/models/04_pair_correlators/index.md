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

# Observables: Pair Correlation Functions

<b>Goal:</b>
At the end of this tutorial, you will know how to measure pair correlation functions using SAFIRE for a lattice models.

## What You Will Learn

1. How to set up the HDF5 and JSON job files to request pair correlation functions
2. how to post process the correlation functions

+++ {"id": "7FBZ0COY-NIw"}

## Introduction

By pair correlation functions, we mean the singlet pair correlation functions

$$
\mathtt{PairCorr/}\alpha\mathtt{\_}\beta[i, j] = \langle \Delta^\dagger_{i,\alpha} \Delta_{j,\beta} \rangle
$$

between the pair creation and annihilation operators

$$
\Delta^\dagger_{i,\alpha} = \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{\bar{i}_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{\bar{i}_\alpha \uparrow} \right), \qquad
\Delta_{j,\beta} = \frac{1}{\sqrt{2}} \left( c_{\bar{j}_\beta\downarrow} c_{j \uparrow}  - c_{\bar{j}_\beta \uparrow} c_{j \downarrow} \right).
$$

Here $i$ and $j$ are orbitals, and $\alpha$ and $\beta$ are *pair maps* that assign every orbital $i$ a partner orbital $\bar{i}_\alpha$.
On a lattice, a pair map typically pairs each site with its neighbor at a fixed offset such as $+x$, and the pair map $\bar{i}_\alpha = i$ gives on-site (s-wave) pairs.
Written out, the pair correlation function is

$$
\mathtt{PairCorr/}\alpha\mathtt{\_}\beta[i, j] = \frac{1}{2} \left\langle \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{\bar{i}_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{\bar{i}_\alpha \uparrow} \right) \left( c_{\bar{j}_\beta\downarrow} c_{j \uparrow}  - c_{\bar{j}_\beta \uparrow} c_{j \downarrow} \right) \right\rangle.
$$

SAFIRE reads the pair maps from the HDF5 input file.
It measures every pair map saved in the group the JSON job file points to, so a subset of pair maps is selected by saving it to a group of its own.
We will explore this in more detail below.

+++

## Defining the Hamiltonian

```{code-cell} ipython3
from safiretools import Lattice
from safiretools import vis
from safiretools import LatticeHamiltonian

# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

lattice = Lattice.from_type(
    "square", (4, 4), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)

# plot the lattice for reality checks!
vis.plot_lattice(lattice)

hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0)
```

+++ {"id": "Np-zIkGusNIP"}

## Saving Pair Maps in the HDF5 Input

The Lattice class from safiretools can be directly used to find the pair maps that we need via the `.get_directed_pairs()` function.
`.get_directed_pairs()` needs a list of `directions`, and it returns a `DirectedPairs` whose `.offsets` maps each requested direction $\alpha$ to its pair map, the list of partner orbitals $\bar{i}_\alpha$ indexed by $i$.
The direction `"s"` is the on-site pair map $\bar{i}_\alpha = i$.

Every direction has to run along a periodic axis. Stepping off an open edge would wrap around to the far side and pair sites that are not neighbours at all, so that is rejected rather than quietly measured.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 907
id: VjaOtDOltPAj
outputId: 30ac79d1-e76e-4b8e-8c2c-6341210f4b4f
---
pairs=lattice.get_directed_pairs(
  directions=["s","+x","-x","+y","-y"]
)

for key,value in pairs.offsets.items():
  print(key, value)
```

`DirectedPairs` writes itself to HDF5 in the format the estimator reads. It may go in the same file as the Wavefunction and Hamiltonian, or in one of its own.

```{code-cell} ipython3
pairs.to_hdf5("afqmc.h5")
```

+++ {"id": "lkkZMHMWvFAU"}

### Adding Custom Pair Maps

We note that custom pair maps can be added by simply inserting a list of partner orbitals with a new key.
For example, we can manually add the $+x+y$ pair map to `pairs.offsets` as

```{code-cell} ipython3
pairs.offsets["xy"] = [ 5, 6 , 7, 4, 9, 10, 11, 8, 13, 14, 15, 12, 1, 2, 3, 0]

# update hdf5 - the whole map is rewritten, so every direction is still there
pairs.to_hdf5("afqmc.h5")
```

+++ {"id": "iwp3CeLownyV"}

## Requesting the Pair Correlation Function in the JSON Job File

The pair correlation function observable may be requested as any other observable within an estimator block.
However, the pair correlation function observable has one unique parameter that must be set,
`pairs`, which locates the pair maps with

- `filename`, the name of the HDF5 file where the pair maps were saved. These can be saved in the same file as the Wavefunction, and/or Hamiltonian or in their own independent input file. Whichever you prefer.
- `group`, the HDF5 group holding them. `DirectedPairs.to_hdf5()` writes them to `PairCorrelator/orbital_map`.

<div class="alert alert-block alert-info">
<b>Note:</b>

SAFIRE will measure all combinations of the pair maps in the group,
i.e. every ordered pair of maps $(\alpha, \beta)$ is measured and stored as its own $(M, M)$ array `PairCorr/<α>_<β>`, where $M$ is the number of orbitals.

</div>

Since the pair correlation functions do not generally commute with the Hamiltonian, we should use a back-propagated estimator.
A back-propagated estimator block with pair correlation functions would look like the following,

```json
"estimators": {
  "backprop": {
    "walker_ortho_interval": 10,
    "paircorr": {
      "pairs": {
        "filename": "afqmc.h5",
        "group": "PairCorrelator/orbital_map"
      }
    }
  }
}
```

If we decided we only wanted s-wave pair correlations, we could save only that pair map to a file of its own,

```python
lattice.get_directed_pairs(directions=["s"]).to_hdf5("pairs_s.h5")
```

and use

```json
"paircorr": {
  "pairs": {
    "filename": "pairs_s.h5",
    "group": "PairCorrelator/orbital_map"
  }
}
```

instead of the above.

### Full Sample Input File

```json
{
  "output_name": "qmc",
  "seed": 42,
  "execute": {
    "wavefunction": {
      "filename": "afqmc.h5"
    },
    "hamiltonian": {
      "filename": "afqmc.h5"
    },
    "timestep": 0.01,
    "steps": 12000,
    "num_walkers": 200,
    "measure_interval": 10,
    "population_control_interval": 10,
    "walker_ortho_interval": 10,
    "estimators": {
      "backprop": {
        "walker_ortho_interval": 10,
        "propagation_steps": [100],
        "paircorr": {
          "pairs": {
            "filename": "afqmc.h5",
            "group": "PairCorrelator/orbital_map"
          }
        }
      }
    }
  }
}
```

```{code-cell} ipython3
import numpy as np
from safiretools import Wavefunction, write_jobfile
from autohf import solve_hf, AutoHFHamiltonian

nelec = (8, 8)
bp_lengths = [100]

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
    
results = solve_hf(
    hamiltonian=AutoHFHamiltonian(hamiltonian),
    lattice=lattice,
    settings=hf_settings,
)

# the noncollinear orbitals are spinors, spin-up components first
orbitals, = results[0]["orbitals"]
wavefunction = Wavefunction.from_single_determinant(
    np.asarray(orbitals).reshape(2, lattice.N_sites, sum(nelec))
)

write_jobfile(
    "afqmc.json",
    {
        "execute": {
            "wavefunction": wavefunction,
            "hamiltonian": hamiltonian,
            "timestep": 0.005,
            "steps": 10000,
            "population_control_interval": 2,
            "walker_ortho_interval": 2,
            "num_walkers": 640,
            "measure_interval": 2,
            "estimators": {
                "backprop": {
                    "walker_ortho_interval": 10,
                    "propagation_steps": bp_lengths,
                    "paircorr": {
                        "pairs": {
                            "filename": "afqmc.h5",
                            "group": "PairCorrelator/orbital_map"
                        }
                    }
                }
            }
        }
    }
)
```

```{code-cell} ipython3
:id: SrM8syoYzUeS
:scrolled: true
:tags: [scroll-output]

!mpirun -n 16 $AFQMC_EXEC afqmc.json
```

+++ {"id": "WIxdjnH-zopo"}

## Post-Processing Pair Correlation Functions

```{code-cell} ipython3
:id: eh8JuNVdag-o

from safiretools import Results

results = Results("afqmc.results.h5")

for name in results.observable_names():
    if "PairCorr" in name:
        print(f"correlator name: {name}")

# take the s, +x correlator
P_s_x_ij, _ = results.average(f"BackPropEstimator/Steps={bp_lengths[-1]}/PairCorr/s_+x")
# use the first site as the reference site
P_s_x_0j = P_s_x_ij[0,:]

vis.plot_lattice(
    lattice,
    density=np.abs(P_s_x_0j.real),
    show_labels=False,
    norm_type="log",
    cmap="viridis"
)
```

+++ {"id": "gGT0fdst-NIz"}

## Summary

### What You Learned

1. How to set up the HDF5 and JSON job files to request pair correlation functions
2. how to post process the correlation functions
