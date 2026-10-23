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

+++ {"id": "82e12143-bd3a-4c82-9a51-bcf96abd6950"}

# Computing General Observables

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-intro -->
:end-before: <!-- END observables-intro -->
```

+++ {"id": "4d4d8135-bc2d-4caa-bbaf-c88ab045176c"}

## Set Up the Test System

We will again use Si in its primitive cell at equilibrium represented in a basis of Kohn-Sham bands as a toy case to run
concrete calculations on.

![](files/Si_primitive.png)

The trial wavefunction is also derived from the DFT solution.
We have provided HDF5 files containing the Hamiltonian and trial wavefunctions, respectively.

We will need the Hamiltonian again to evaluate the energy from the measured observables, so we read it here.

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

import h5py
import numpy as np

with h5py.File("../files/hamiltonian.h5", "r") as f:
    system = f["System"].attrs
    M = int(system["number_of_bands"])
    # constant energy: nuclear repulsion and the Madelung self-interaction of the electrons
    econst = system["nuclear_energy"] - system["number_of_elec"] * system["madelung_constant"]

    H0 = f["System/H0"][0, 0]  # spin 0, Gamma point
    h = H0[..., 0] + 1j * H0[..., 1]

    # Cholesky vectors of momentum transfer Q = 0 at the Gamma point
    Vq0 = f["Interaction/Vq0"][:, 0, 0]
    L = np.moveaxis(Vq0[..., 0] + 1j * Vq0[..., 1], 0, -1)  # (M, M, nchol)
```

+++ {"id": "dSJ9e4tRBxxx"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-mixed -->
:end-before: <!-- END observables-mixed -->
```

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-mixed-demo -->
:end-before: <!-- END observables-mixed-demo -->
```

### Run the Code Block Below to Write a JSON Job File with a Mixed Estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: QcoDuWPjAZDx
outputId: f3692576-0e97-4c3f-8440-8ed09dfeb738
---
from safiretools import write_jobfile

write_jobfile(
    "afqmc_mixed.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": {
                "filename": "../files/wavefunction.h5"
            },
            "hamiltonian": {
                "filename": "../files/hamiltonian.h5"
            },
            "timestep": 0.01,
            "steps": 7000,
            "equilibration_steps": 1000,
            "measure_interval": 10,
            "population_control_interval": 10,
            "walker_ortho_interval": 10,
            "num_walkers": 800,
            "estimators": {
                "mixed": {
                    "onerdm": {},
                    "twordm": {}
                }
            }
        }
    }
)
```

+++ {"id": "TrVqnMi4AxuT"}

Now, we can run SAFIRE with this configuration. **This will probably take a while!** To finish the tutorial more quickly, remove the `twordm` block from the input and just compute the one-body energy.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 6fnE-_O_HUgs
outputId: 3345a1e5-a201-4ba0-f6c0-ea04425aa41c
scrolled: true
tags: [scroll-output]
---
!mpirun -n 16 $AFQMC_EXEC afqmc_mixed.json
```

+++ {"id": "fgJ_Pp2EA7oA"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-extraction -->
:end-before: <!-- END observables-extraction -->
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 631
id: c9LgOOxOA_xI
outputId: b3c1c8c5-e7ac-4ffb-c212-1e7aec371f79
---
from safiretools import Results


def onebody_energy(onerdm):
    # the CoQuí trial is collinear, so the walkers are too: sum the spin-up and spin-down blocks
    return np.einsum("ij,sij->", h, onerdm)

results = Results("afqmc_mixed.results.h5")

rebinsize = 60 # chosen automatically by default. only required here to get exact numerical agreement between different observables

onerdm, onerdm_err = results.average("MixedEstimator/OneRDM", rebinsize=rebinsize)
E1_rdm = onebody_energy(onerdm)
E1_rdm += econst

E1_default, E1_default_err = results.average("OnebodyEnergy", rebinsize=rebinsize)

print(f"one-body energy, mixed estimator:  {E1_rdm.real:.8f}")
print(f"one-body energy, energy estimator: {E1_default.real:.8f} ± {E1_default_err.real:.8f}")

# to get the correct errorbar use results.evaluate()!
E1_jackknife, E1_jackknife_err = results.evaluate(onebody_energy, ["MixedEstimator/OneRDM"], rebinsize=rebinsize)
E1_jackknife += econst
print(f"one-body energy, mixed estimator (jackknife):  {E1_jackknife.real:.8f} ± {E1_jackknife_err.real:.8f}")
```

+++ {"id": "wDMvF0aT04_x"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-mixed-2rdm -->
:end-before: <!-- END observables-mixed-2rdm -->
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 929
id: afFNaJpQdymb
outputId: b681facb-f520-49ed-a2c2-cfceff7c2d40
---
# Because the Hamiltonian in this tutorial is defined in the supercell,
# the full interaction tensor can be recovered from the Q=0 cholesky operator.
# v_ijkl = sum_g L^g_ik (L^g_lj)^*, stored in the [i, k, j, l] index order of the TwoRDM
v = np.einsum("ikg,ljg->ikjl", L, L.conj())

def twobody_energy(twordm):
    # collinear walkers store the up-up, up-down and down-down blocks;
    # the down-up block contributes the same as the up-down block
    uu, ud, dd = np.einsum("ikjl,bikjl->b", v, twordm)
    return 0.5 * (uu + dd) + ud

E2_default, E2_default_err = results.evaluate(lambda eH, eF: eH + eF, ["CoulombEnergy", "ExchangeEnergy"], rebinsize=rebinsize)

############
# YOUR TURN: Calculate the twobody-energy using results.evaluate
E2_rdm, E2_rdm_err = 0.0, 0.0
############

print(f"two-body energy, mixed estimator: {E2_rdm.real:.8f} ± {E2_rdm_err.real:.8f}")
print(f"two-body energy, energy estimator: {E2_default.real:.8f} ± {E2_default_err.real:.8f}")
```

+++ {"id": "NjutJDKY4B_m"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-averaging -->
:end-before: <!-- END observables-averaging -->
```

+++ {"id": "6a1baf12-288e-4f8f-afaf-6f9d83e899d4"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-bp -->
:end-before: <!-- END observables-bp -->
```

+++

### Run the Code Block Below to Write a JSON Job File with a BP Estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 41e0454a
outputId: adbb5a65-99ad-49a9-d8a9-3e9b0a166bad
---
# Example: Setting up a BP calculation with multiple BP lengths
dt = 0.01
bp_lengths = [40, 50, 60]  # Multiple BP lengths, in steps
# Set up AFQMC parameters with back-propagation
write_jobfile(
    "afqmc_bp.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": {
                "filename": "../files/wavefunction.h5"
            },
            "hamiltonian": {
                "filename": "../files/hamiltonian.h5"
            },
            "timestep": dt,
            "steps": 10000,
            "measure_interval": 10,
            "equilibration_steps": 1000,
            "population_control_interval": 10,
            "walker_ortho_interval": 10,
            "num_walkers": 1600,
            "estimators": {
                "backprop": {
                    "walker_ortho_interval": 10,
                    "propagation_steps": bp_lengths,
                    "onerdm": {}
                }
            }
        }
    }
)
```

### Run SAFIRE with BP

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 918df5d4
outputId: af566232-d9fd-434f-f6ad-951e8242c35f
scrolled: true
tags: [scroll-output]
---
# Run AFQMC with back-propagation
print("Running AFQMC with back-propagation estimator...")
print("This will take longer than mixed estimators due to the additional back-propagation steps.")
!mpirun -n 16 $AFQMC_EXEC afqmc_bp.json
```

### Run the Code Below to Evaluate the One-Body Energy from the One-RDM and Compare with the BP Estimator

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: 39c23cf1
outputId: 9db77855-7355-4009-f299-6dd441e3e317
---
import matplotlib.pyplot as plt

results_bp = Results("afqmc_bp.results.h5")

bp_onebody_energy, bp_onebody_energy_err = np.transpose([
   results_bp.evaluate(onebody_energy, [f"BackPropEstimator/Steps={bp_length}/OneRDM"])
   for bp_length in bp_lengths
])
bp_onebody_energy += econst
plt.errorbar(np.array(bp_lengths) * dt, bp_onebody_energy, bp_onebody_energy_err, label="BackProp Estimator")
plt.axhline(y=E1_default, color='red', linestyle='--', alpha=0.7, label="Mixed Estimator")
plt.xlabel(r"BP length $β_\mathrm{BP}$ (Ha${}^{-1}$)")
plt.ylabel("$E_1$ (Ha)")
plt.legend()
```

+++ {"id": "d0f6abfa"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-summary -->
:end-before: <!-- END observables-summary -->
```
