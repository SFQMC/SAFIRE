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
Become acquainted with how to write a Trial wavefunction to the SAFIRE HDF5 format.

## What you will learn

1.  How to write a single Slater determinant trial wavefunction given the Slater Matrix
2.  How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3.  How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings

+++

## ▶️ Run the cell below to setup the tutorial

```{code-cell} ipython3
# Run me (shift+enter or click the play button) to setup the tutorial!
from pathlib import Path

# simple setup
from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "vt9PbuQhpw4u"}

## Introduction

In general, the trial wavefunction in AFQMC is a linear combination of Slater determinants,

$$
|\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle,
$$

where

$C_n$

is a complex-valued coefficient,
and $|\Phi_n\rangle$ are Slater determinants which are not necessarily
orthogonal to each other.
Of course, each Slater determinant consists of some set of single-particle orbitals, $\{ \psi_p \}$, such that,
$$
\psi_{p} = \sum_i \bar{C}_{ip} \phi_i,
$$
where $\{\phi_i\}$ are the chosen orthonormal basis
set orbitals.
Slater determinants can either be represented explicitly as a Slater matrix,  

$$
\Phi ≐ \begin{bmatrix}
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0N} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1N} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M0} & \bar{C}_{M1} & \bar{C}_{M2} & \dots  & \bar{C}_{MN}
\end{bmatrix},
$$

where $M,N$ are the number of basis functions and electrons, respectively,
or in terms of an "occupation vector", which is simply a list of orbital indices which should be occupied.
    

SAFIRE implements two basic types of trial wavefunction.

1. <b>"particle-hole" multi-Slater determinant (ph-mSD) trial wavefunctions</b>
which is a configuration interaction-like wavefunction
where  $\langle \Phi_n |\Phi_m\rangle = \delta_{nm} \forall n,m$. In this case,
the Slater determinants are specified in terms of a set of orbitals, and a list
of $C_n$ and corresponding occupancy vectors. 

2. <b> non-orthogonal multi-slater Determinant (NOMSD) trial wavefunction </b>
where **strictly** $\langle \Phi_n |\Phi_m\rangle \neq 0 \forall n,m$.
In this case, the Slater determinants are specified as a list
of $C_n$ and corresponding Slater matrices. 

We'll explore each of these wavefunction types in more detail below.

<!--
<div class="alert alert-block alert-warning">
<b>Important:</b>
When using NOMSD trial wavefunctions, it is important
that each Slater determinant has a non-zero overlap with all
other Slater determinants to avoid singular values.
Use ph-mSD for orthogonal expansions of Slater determinants.
</div>

-->

+++ {"id": "RXzexuh_sQyL"}

## Single Slater determinant trial wavefunction

We've already seen examples of writing a single Slater determinant trial wavefunction in previous tutorials.
Here, we will explain the code blocks that you have already seen which we have reproduced below.

Technically, we have been writing single Slater determinant trial wavefunctions as a non-orthogonal multi-Slater determinant (NOMSD) expansion with only a single determinant.
To accomplish this, we wrote down a set of orbitals in the chosen orthonormal orbital basis as,

```python
import numpy as np

orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)
```

where we are using the RHF orbitals as a basis to represent the RHF orbitals.

Next, we constructed the Slater matrix, $\Phi_{ip}$, and wrote down a trivial expansion coefficient with,

```python
phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0
```

Finally, we build a `NOMSDWavefunction` and write it to an HDF5 file in the SAFIRE format with `to_hdf5()` as,

```python
from safiretools import NOMSDWavefunction

NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "rhf_wavefunction.h5")
```

Note that `coeffs` is a 1-dimensional array with the CI coefficient, and `dets` is a 3 dimensional array where the first index corresponds to the Slater determinant index, the second index corresponds to the basis set index, and the third index corresponds to the electron index.

We put all of this together in the following code block.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 385
id: locAjSqZsYSI
outputId: b67f9653-4cfe-4ef4-ad16-1b9b5502444f
---
import numpy as np

from safiretools import NOMSDWavefunction

number_of_electrons = (1,1)
number_of_orbitals = 2

orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)

phi_0 = np.array([
    orbitals[:, :number_of_electrons[0]]
])
C_0 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0]),
    dets=phi_0,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "rhf_wfn.h5")
```

we can use the `h5dump` utility to inspect the contents of the file that we just generated. 
`h5dump -n [filename]` will show us which datasets exist, and we can use `h5dump -d [dataset] [filename]` to
view the actual data.

Note that the NOMSD format is a compressed sparse row (CSR) format.
See the wavefunction formats in the [user manual](https://users.flatironinstitute.org/~beskridge/safire/user_manual/wavefunctions.html) for
details on the expected data.

```{code-cell} ipython3
!h5dump -n {scratch_dir}/rhf_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/data_ {scratch_dir}/rhf_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/jdata_ {scratch_dir}/rhf_wfn.h5
```

+++ {"id": "VJfGuez4rsE-"}

## NOMSD Trial Wavefunction

In the previous example, we learned how to specify a trial wavefunction with a single Slater determinant. Here, we will see how to we can write a suitable non-orthogonal multi-Slater trial wavefunction for the AFQMC code.

Similarly to the single Slater determinant case, we begin by specifying a set of orbitals. In this case, however, we chose a set of non-orthogonal orbitals represented in a common basis:

```python
import numpy as np

orbitals = np.array(
    [[1.0,0.7071],
    [0.0,0.7071]]
)
```
Note that each column of the `orbitals` object are normalized, but they are not orthogonal to each other.

Next, we construct two determinants with a single electron each. The first determinant will be specified by the orbital, whereas the second orbital will be used in the other determinant.

```python
phi_nomsd = np.array([
    orbitals[:, :number_of_electrons[0]], # Representing the first determinant, with the first orbital occupied
    orbitals[:, 1:number_of_electrons[0]], # Representing the second determinant, with the second orbital occupied
])
C_0 = 1.0
C_1 = 1.0
```

The dimensions of `phi_nomsd` should be (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\mathrm{elec}$), as previously indicated in the single determinant case.

Finally, we build a `NOMSDWavefunction` and write it to an HDF5 file in the SAFIRE format with `to_hdf5()` as,

```python
from safiretools import NOMSDWavefunction

NOMSDWavefunction(
    coeffs=np.array([C_0,C_1]),
    dets=phi_nomsd,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "NOMSD_wfn.h5")
```

We put all of this together in the code block below.

```{code-cell} ipython3
:id: 8Y0sDP2wr1vk

number_of_electrons = (1,1)
number_of_orbitals = 2

orbitals = np.array(
    [[1.0,0.7071],
    [0.0,0.7071]]
)

phi_nomsd = np.array([
    orbitals[:, :number_of_electrons[0]],                                                # Representing the first determinant, with the first orbital occupied
    orbitals[:, number_of_electrons[0]:number_of_electrons[0] + number_of_electrons[1]], # Representing the second determinant, with the second orbital occupied
])
C_0 = 1.0
C_1 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0,C_1]),
    dets=phi_nomsd,
    spin_symm="rhf",
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "NOMSD_wfn.h5")
```

```{code-cell} ipython3
!h5dump -n {scratch_dir}/NOMSD_wfn.h5
```

notice that there are two Slater determinants in the HDF5 file! We note that
, in the collinear format, the spin up and spin down Slater determinants are stored as separate data sets with even (odd) indices
corresponding to up (down) sector Slater determinants.

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_0/data_ {scratch_dir}/NOMSD_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/NOMSD/PsiT_1/data_ {scratch_dir}/NOMSD_wfn.h5
```

+++ {"id": "YtBpL-q8rlGT"}

## ph-mSD Trial Wavefunctions.

In the previous sections, we learned (i) how to construct trial wavefunctions with a single Slater determinant, (ii) how to construct non-orthogonal multi-Slater determinant trial wavefunctions (NOMSD). Here, we will look at how to use CI-like expansions, also called particle-hole multi Slater (ph-mSD) wavefunctions, as trial wavefunctions for AFQMC.

In contrast with the NOMSD case where each Slater determinant had its own set of molecular orbitals, ph-mSD wavefunctions are characterized by a single set of molecular orbitals. Orthogonal Slater Determinants are generated by occupying different orbitals contained in the set. For instance, let's take the following set of orbitals $ \{\phi_g, \phi_u \}$, with $\langle \phi_g | \phi_u \rangle = 0 $, and construct two Slater determinants
$$
|\Psi_0 \rangle = a_{g,\downarrow}^\dagger a_{g,\uparrow}^\dagger |0\rangle
$$
and
$$
|\Psi_1 \rangle = a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger |0\rangle
$$

You can verify that these two Slater determinants are also orthogonal to each other $\langle \Psi_0 | \Psi_1 \rangle = 0$. Taking $|\Psi_0 \rangle$ to be our reference state, we can re-express the other determinant as
$$
| \Psi_1 \rangle = a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger a_{g,\downarrow} a_{g,\uparrow} |\Psi_0 \rangle
$$
where we explicitly create holes (through the $a_{g,\downarrow} a_{g,\uparrow}$ operators) and particles (through the $a_{u,\downarrow}^\dagger a_{u,\uparrow}^\dagger$) operators on top of a reference state.

For the tooling side of the AFQMC, to write a ph-mSD trial wavefunction we need to specify an array with the occupied orbitals in each Slater determinant. $α$ and $β$ electrons should be specified separately. Finally, we need to provide an array containing the normalized weights/coefficients associated with each determinant in our trial wavefunction.
Combining these ingredients (array of normalized coefficients, array of occupied $α$ orbitals and array of occupied $β$ orbitals) into a `PHMSDWavefunction` allows us to write the ph-mSD wavefunction as an hdf5 file for the AFQMC code

In our example, if we enumerate the orbitals in set starting with 0, we have:

```{code-cell} ipython3
:id: 9u7rGYAqrlrV

import numpy as np

from safiretools import PHMSDWavefunction

number_of_electrons = (1,1)
number_of_orbitals = 2

# Specify an array containing the normalized coefficients for each
# Slater Determinant.
Cn = np.array([0.9, np.sqrt(1 - 0.9**2)])

# Specify arrays for the α and β occupied orbitals for Ψ0.
occa_0 = np.array([0])
occb_0 = np.array([0])

# Specify arrays for the α and β occupied orbitals for Ψ1.
occa_1 = np.array([1])
occb_1 = np.array([1])

# Combine the α and β occupied arrays for each determinant in a single object
# for each spin channel.
occa = np.array((occa_0, occa_1))
occb = np.array((occb_0, occb_1))

# Build a PHMSDWavefunction from the coefficients and the α and β occupied
# arrays, and write it in the appropriate hdf5 format.
PHMSDWavefunction(
    coeffs=Cn,
    occa=occa,
    occb=occb,
    nelec=number_of_electrons,
    nmo=number_of_orbitals
).to_hdf5(scratch_dir/ "PHMSD_wfn.h5")
```

```{code-cell} ipython3
!h5dump -n {scratch_dir}/PHMSD_wfn.h5
```

here, the interesting information is the orbital occupancies in `/Wavefunction/PHMSD/occs` and
the CI coefficients in `/Wavefunction/PHMSD/ci_coeffs`.

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/occs {scratch_dir}/PHMSD_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/ci_coeffs {scratch_dir}/PHMSD_wfn.h5
```

+++ {"id": "FoovlUVcFqxX"}

## Summary

In this tutorial, you became acquainted with how to write a Trial wavefunction to the SAFIRE HDF5 format.

### What you learned

1. How to write a single Slater determinant trial wavefunction given the Slater Matrix
2. How to write a single Slater determinant given orbital occupancies
3. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
4. How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings

```{code-cell} ipython3
:id: qVBQ6CAhF3Xn


```
