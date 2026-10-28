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

## Set Up the System

For this tutorial, our starting point will be to use safiretools to generate a Hamiltonian and AutoHF to generate a trial wavefunction for the Hubbard model with $U/t = 4$ on a 4×4 lattice with periodic boundary conditions and $n_\uparrow = n_\downarrow = 5$. For details on the set up steps, refer to the previous tutorials.

![](files/Hubbard_4x4_tutorial.png)

### Run the Code Block to Set Up the Hamiltonian and Wavefunction

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: c844616d-45c5-498a-b6f9-c995027965e6
outputId: 855ebc93-90d5-4e84-94ff-e974f19587fc
---
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data

import numpy as np

from safiretools import Lattice, LatticeHamiltonian, NOMSDWavefunction

from autohf import solve_hf, AutoHFHamiltonian

# some jax initialization for autohf
import jax
jax.config.update('jax_platform_name', 'cpu')

# define the system
nelec = (5,5)

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))

hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0, spin_symm="closed")

# AutoHF needs the Hamiltonian in collinear form, so build it again and convert it
hamiltonian_for_autohf = AutoHFHamiltonian(
    LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0, spin_symm="collinear")
)


hf_settings = dict(
    ansatz = 'SD',
    steps = 200,
    nelec = nelec,
    batch_size = 50,
    seed = 42,
    measure_spin = True,
    random_initial_phi0 = False,
    noncollinear = False,
    verbose = False
)

hf_data, _ = solve_hf(hamiltonian=hamiltonian_for_autohf, settings=hf_settings)

# AutoHF returns the orbitals as (spin, orbital, electron)
alpha, beta = np.asarray(hf_data["orbitals"])
wfn = NOMSDWavefunction(
    coeffs=np.array([1.0]),
    dets=(alpha[None], beta[None])
)
```

+++ {"id": "dSJ9e4tRBxxx"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-mixed -->
:end-before: <!-- END observables-mixed -->
```

### Pedagogical Example: Calculating the Energy from the Mixed Estimator

Below, we will use a mixed estimator to compute the 1-rdm, and the diagonal 2-rdm.
We can then directly compute each contribution to the AFQMC energy,

$$\langle H_1 \rangle = \sum_{ij} h_{ij} \langle c^\dagger_i c_j \rangle$$

$$\langle H_2 \rangle = U \sum_{i} D^{\uparrow\downarrow}_{ii}$$

where the indices of the one-body term are combined orbital and spin indices,
and compare with the results from the energy estimator. The two should agree exactly.

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
            "wavefunction": wfn,
            "hamiltonian": hamiltonian,
            "timestep": 0.01,
            "steps": 10000,
            "equilibration_steps": 1000,
            "measure_interval": 10,
            "population_control_interval": 10,
            "walker_ortho_interval": 10,
            "num_walkers": 3000,
            "estimators": {
                "mixed": {
                    "onerdm": {},
                    "diag_twordm": {}
                }
            }
        }
    }
)
```

+++ {"id": "TrVqnMi4AxuT"}

Now, we can run SAFIRE with this configuration.

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
  height: 611
id: c9LgOOxOA_xI
outputId: b3c1c8c5-e7ac-4ffb-c212-1e7aec371f79
---
from safiretools import Results


h = hamiltonian.get_one_body().toarray()
M = h.shape[0]

def onebody_energy(onerdm):
    return np.einsum("ij,sij->", h, onerdm)

results = Results("afqmc_mixed.results.h5")

rebinsize = 90 # chosen automatically by default. only required here to get exact numerical agreement between different observables

onerdm, onerdm_err = results.average("MixedEstimator/OneRDM", rebinsize=rebinsize)
E1_rdm = onebody_energy(onerdm)

E1_default, E1_default_err = results.average("OnebodyEnergy", rebinsize=rebinsize)

print(f"one-body energy, mixed estimator:  {E1_rdm.real:.8f}")
print(f"one-body energy, energy estimator: {E1_default.real:.8f} ± {E1_default_err.real:.8f}")

# to get the correct errorbar use results.evaluate()!
E1_jackknife, E1_jackknife_err = results.evaluate(onebody_energy, ["MixedEstimator/OneRDM"], rebinsize=rebinsize)
print(f"one-body energy, mixed estimator (jackknife):  {E1_jackknife.real:.8f} ± {E1_jackknife_err.real:.8f}")
```

+++ {"id": "wDMvF0aT04_x"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-1rdm-conclusion -->
:end-before: <!-- END observables-1rdm-conclusion -->
```

### Your Turn: Extract the Diagonal 2-RDM and Compute the Two-Body Energy

In this section it is your turn to repeat the above for the two-body energy from the diagonal 2-rdm.

We provide code for setting up the appropriate transform function in the code block below.
Note that `DiagonalTwoRDM` is stored as a flat array that holds only the upper triangle of the same-spin blocks (see {ref}`diag_twordm_obs`).

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 929
id: afFNaJpQdymb
outputId: b681facb-f520-49ed-a2c2-cfceff7c2d40
---
U = hamiltonian.get_U().diagonal()
# position of D^{up,down}_{ii} in the flat DiagonalTwoRDM: row i holds M-1-i same-spin
# elements followed by the M elements of the up-down block
ud_diagonal = np.array([i * (2 * M - 1) - i * (i - 1) // 2 + M - 1 for i in range(M)])

def twobody_energy(diag_twordm):
    return np.sum(U * diag_twordm[ud_diagonal])

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
bp_lengths = [50, 80, 100]  # Multiple BP lengths, in steps
# Set up AFQMC parameters with back-propagation
write_jobfile(
    "afqmc_bp.json",
    {
        "seed": 42,
        "execute": {
            "wavefunction": wfn,
            "hamiltonian": hamiltonian,
            "timestep": dt,
            "steps": 10000,
            "measure_interval": 50,
            "equilibration_steps": 400,
            "population_control_interval": 10,
            "walker_ortho_interval": 10,
            "num_walkers": 3000,
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
plt.errorbar(np.array(bp_lengths) * dt, bp_onebody_energy, bp_onebody_energy_err, label="BackProp Estimator")
plt.axhline(y=E1_default, color='red', linestyle='--', alpha=0.7, label="Mixed Estimator")
plt.xlabel(r"BP length $β_\mathrm{BP}$ ($t^{-1}$)")
plt.ylabel("$E_1$ ($t$)")
plt.legend()
```

+++ {"id": "d0f6abfa"}

```{include} /tutorials/molecules/05_computing_observables/05_computing_observables.md
:start-after: <!-- BEGIN observables-summary -->
:end-before: <!-- END observables-summary -->
```
