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

+++ {"id": "P8_OXxiRE_dA"}

# Writing a Trial Wavefunction

<b>Goal:</b>
Become acquainted with how to write a trial wavefunction to the SAFIRE HDF5 format.

## What You Will Learn

1. How to write a single Slater determinant trial wavefunction given the Slater Matrix, for closed-shell, collinear, and noncollinear spin
2. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3. How to compute a free-electron trial wavefunction for a lattice model

## Setup

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

+++ {"id": "vt9PbuQhpw4u"}

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-introduction -->
:end-before: <!-- END wfn-introduction -->
```

<div class="alert alert-block alert-info">
<b>Note:</b>

PHMSD trial wavefunctions are more commonly used for molecules and we will not cover them further in this tutorial.
Most lattice model calculations are performed with a single Slater determinant or a handful of non-orthogonal ones.

</div>

+++ {"id": "RXzexuh_sQyL"}

## Single Slater Determinant Trial Wavefunction

The simplest trial wavefunction is a single Slater determinant.

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-single-det -->
:end-before: <!-- END wfn-single-det -->
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 385
id: locAjSqZsYSI
outputId: b67f9653-4cfe-4ef4-ad16-1b9b5502444f
---
import numpy as np

from safiretools import NOMSDWavefunction, Wavefunction

number_of_electrons = (2,2)
number_of_orbitals = 4

# four orthonormal orbitals, one per column
orbitals = 0.5 * np.array(
    [[1.0, 1.0, 1.0, 1.0],
     [1.0,-1.0, 1.0,-1.0],
     [1.0, 1.0,-1.0,-1.0],
     [1.0,-1.0,-1.0, 1.0]]
)

# shorthand
wavefunction = Wavefunction.from_single_determinant(orbitals[:, :number_of_electrons[0]])
print(wavefunction.spin_symm.label, wavefunction.nelec)

# under the hood: NOMSD with 1 determinant and 1 expansion coefficient
phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0

wavefunction = NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0
)

# write to HDF5
wavefunction.to_hdf5("closed_wfn.h5")
```

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-single-det-constructor -->
:end-before: <!-- END wfn-single-det-constructor -->
```

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-collinear -->
:end-before: <!-- END wfn-collinear -->
```

```{code-cell} ipython3
phi_up = orbitals[:, :3]
phi_down = orbitals[:, [0, 2]]

wavefunction = Wavefunction.from_single_determinant((phi_up, phi_down))
print(wavefunction.spin_symm.label, wavefunction.nelec)

wavefunction.to_hdf5("collinear_wfn.h5")
```

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-noncollinear -->
:end-before: <!-- END wfn-noncollinear -->
```

```{code-cell} ipython3
theta = np.array([0.0, np.pi/2, np.pi])

phi = np.array([
    orbitals[:, :3] * np.cos(theta/2),  # spin-up components
    orbitals[:, :3] * np.sin(theta/2),  # spin-down components
])

wavefunction = Wavefunction.from_single_determinant(phi)
print(wavefunction.spin_symm.label, wavefunction.nelec)

wavefunction.to_hdf5("noncollinear_wfn.h5")
```

+++ {"id": "VJfGuez4rsE-"}

```{include} /tutorials/molecules/04_writing_a_trial_wavefunction/04_writing_a_trial_wavefunction.md
:start-after: <!-- BEGIN wfn-nomsd -->
:end-before: <!-- END wfn-nomsd -->
```

```{code-cell} ipython3
:id: 8Y0sDP2wr1vk

# two nonorthogonal closed-shell determinants with two orbitals each
phi_nomsd = np.array([
    orbitals[:, [0, 1]],
    np.column_stack([orbitals[:, 0], (orbitals[:, 1] + orbitals[:, 2])/np.sqrt(2)]),
])

C_0 = 1.0
C_1 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0,C_1]),
    dets=phi_nomsd
).to_hdf5("NOMSD_wfn.h5")
```

+++ {"id": "NYbUINg6HQyx"}

## Free Electron Trial Wavefunction

For lattice models, a common trial wavefunction is the free-electron ground state, i.e. the ground state of the one-body part of the Hamiltonian.
{py:meth}`safiretools.Wavefunction.from_free_electron` computes it directly from a `LatticeHamiltonian`.
Below, we build the Hubbard model on a 4 × 4 square lattice with periodic boundary conditions and $U/t = 4$, and write the free-electron trial wavefunction with 5 spin-up and 5 spin-down electrons, which fill a closed shell.

```{code-cell} ipython3
:id: XDDPCMhvHKRc

from safiretools import Lattice, LatticeHamiltonian

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'))

# nearest-neighbor hopping with t = 1 is included by default
hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=4.0)

wavefunction = Wavefunction.from_free_electron(
    source=hamiltonian,
    nelec=(5, 5)
)

wavefunction.to_hdf5("wfn.h5")
```

Mean-field trial wavefunctions, which often capture the physics of the interacting system better, are the topic of {doc}`../06_autohf_crash_course/06_autohf_crash_course`.

+++ {"id": "FoovlUVcFqxX"}

## Summary

In this tutorial, you became acquainted with how to write a trial wavefunction to the SAFIRE HDF5 format.

### What You Learned

1. How to write a single Slater determinant trial wavefunction given the Slater Matrix, for closed-shell, collinear, and noncollinear spin
2. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3. How to compute a free-electron trial wavefunction for a lattice model
