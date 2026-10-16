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

## What You Will Learn

1.  How to write a single Slater determinant trial wavefunction given the Slater Matrix
2.  How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3.  How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings

## Setup

```{code-cell} ipython3
%mkdir -p data
%cd -q data
```

+++ {"id": "vt9PbuQhpw4u"}

## Introduction

In general, the trial wavefunction in AFQMC is a linear combination of Slater determinants,

$$
|\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle,
$$

where $C_n$ are complex-valued coefficients, and $|\Phi_n\rangle$ are Slater determinants which are not necessarily
orthogonal to each other. Each Slater determinant consists of some set of single-particle orbitals, $\{ \psi_p \}$, such that,
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
or in terms of an *occupation vector*, which is simply a list of orbital indices which should be occupied.
    

SAFIRE implements two basic types of trial wavefunction.

1. <b>particle-hole multi-Slater determinant (PHMSD) trial wavefunctions</b>
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
Use PHMSD for orthogonal expansions of Slater determinants.
</div>

-->

+++ {"id": "RXzexuh_sQyL"}

## Single Slater Determinant Trial Wavefunction

We have already seen examples of writing a single Slater determinant trial wavefunction using {py:meth}`safiretools.Wavefunction.from_single_determinant` in previous tutorials.
Under the hood this creates a NOMSD wavefunction with a single determinant, as the code block below shows.

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

number_of_electrons = (1,1)
number_of_orbitals = 2

orbitals = np.array(
    [[1.0,0.0],
    [0.0,1.0]]
)

# shorthand
wavefunction = Wavefunction.from_single_determinant(orbitals[:, :number_of_electrons[0]])

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
wavefunction.to_hdf5("rhf_wfn.h5")
```

In the `NOMSDWavefunction` constructor, `coeffs` is a 1-dimensional array of expansion coefficients, and `dets` is a 3-dimensional array where the first index corresponds to the Slater determinant index, the second index corresponds to the basis set index, and the third index corresponds to the electron index.

We can use the `h5dump` utility to inspect the contents of the file that we just generated. 
`h5dump -n [filename]` will show us which datasets exist, and we can use `h5dump -d [dataset] [filename]` to
view the actual data.

Note that the NOMSD format is a compressed sparse row (CSR) format.
See {ref}`Wavefunction-classes` for details on the expected data.

```{code-cell} ipython3
!h5dump -n rhf_wfn.h5
```

+++ {"id": "VJfGuez4rsE-"}

## NOMSD Trial Wavefunction

In the previous example, we learned how to specify a trial wavefunction with a single Slater determinant. Here, we will see how to we can write a suitable non-orthogonal multi-Slater trial wavefunction for the AFQMC code.

We take two determinants with one electron per spin, each occupying a different orbital. The two orbitals are normalized but not orthogonal to each other.
As in the single determinant case, `phi_nomsd` has the dimensions (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\mathrm{elec}$).

```{code-cell} ipython3
:id: 8Y0sDP2wr1vk

# two nonorthogonal RHF determinants with one orbital each
phi_nomsd = np.array([
    [[1.0], [0.0]],
    [[1/np.sqrt(2)], [1/np.sqrt(2)]],
])

C_0 = 1.0
C_1 = 1.0

NOMSDWavefunction(
    coeffs=np.array([C_0,C_1]),
    dets=phi_nomsd
).to_hdf5("NOMSD_wfn.h5")
```

```{code-cell} ipython3
!h5dump -n NOMSD_wfn.h5
```

notice that there are two Slater determinants in the HDF5 file, `PsiT_0` and `PsiT_1`! In this closed-shell example, both spins share the same determinant.
For collinear wavefunctions, the spin up and spin down Slater determinants are stored as separate data sets with even (odd) indices
corresponding to up (down) sector Slater determinants.

+++ {"id": "YtBpL-q8rlGT"}

## PHMSD Trial Wavefunctions

In the previous sections, we learned (i) how to construct trial wavefunctions with a single Slater determinant, (ii) how to construct non-orthogonal multi-Slater determinant trial wavefunctions (NOMSD). Here, we will look at how to use CI-like expansions, also called particle-hole multi Slater (PHMSD) wavefunctions, as trial wavefunctions for AFQMC.

In contrast with the NOMSD case where each Slater determinant had its own set of molecular orbitals, PHMSD wavefunctions are characterized by a single set of molecular orbitals. Orthogonal Slater Determinants are generated by occupying different orbitals contained in the set. For instance, let's take the following set of orbitals $ \{\phi_\mathrm{g}, \phi_\mathrm{u} \}$, with $\langle \phi_\mathrm{g} | \phi_\mathrm{u} \rangle = 0 $, and construct two Slater determinants
$$
|\Psi_0 \rangle = a_{\mathrm{g},\downarrow}^\dagger a_{\mathrm{g},\uparrow}^\dagger |0\rangle
$$
and
$$
|\Psi_1 \rangle = a_{\mathrm{u},\downarrow}^\dagger a_{\mathrm{u},\uparrow}^\dagger |0\rangle
$$

You can verify that these two Slater determinants are also orthogonal to each other $\langle \Psi_0 | \Psi_1 \rangle = 0$. Taking $|\Psi_0 \rangle$ to be our reference state, we can re-express the other determinant as
$$
| \Psi_1 \rangle = a_{\mathrm{u},\downarrow}^\dagger a_{\mathrm{u},\uparrow}^\dagger a_{\mathrm{g},\downarrow} a_{\mathrm{g},\uparrow} |\Psi_0 \rangle
$$
where we explicitly create holes (through the $a_{\mathrm{g},\downarrow} a_{\mathrm{g},\uparrow}$ operators) and particles (through the $a_{\mathrm{u},\downarrow}^\dagger a_{\mathrm{u},\uparrow}^\dagger$) operators on top of a reference state.

For the tooling side of the AFQMC, to write a PHMSD trial wavefunction we need to specify an array with the occupied orbitals in each Slater determinant. $α$ and $β$ electrons should be specified separately. Finally, we need to provide an array containing the normalized weights/coefficients associated with each determinant in our trial wavefunction.
Combining these ingredients (array of normalized coefficients, array of occupied $α$ orbitals and array of occupied $β$ orbitals) into a `PHMSDWavefunction` allows us to write the PHMSD wavefunction as an hdf5 file for the AFQMC code

In our example, if we enumerate the orbitals in set starting with 0, we have:

```{code-cell} ipython3
:id: 9u7rGYAqrlrV

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
).to_hdf5("PHMSD_wfn.h5")
```

```{code-cell} ipython3
!h5dump -n PHMSD_wfn.h5
```

here, the interesting information is the orbital occupancies in `/Wavefunction/PHMSD/occs` and
the CI coefficients in `/Wavefunction/PHMSD/ci_coeffs`.

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/occa PHMSD_wfn.h5
```

```{code-cell} ipython3
!h5dump -d /Wavefunction/PHMSD/ci_coeffs PHMSD_wfn.h5
```

+++ {"id": "FoovlUVcFqxX"}

## Summary

In this tutorial, you became acquainted with how to write a Trial wavefunction to the SAFIRE HDF5 format.

### What You Learned

1. How to write a single Slater determinant trial wavefunction given the Slater Matrix
2. How to write a single Slater determinant given orbital occupancies
3. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
4. How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings

```{code-cell} ipython3
:id: qVBQ6CAhF3Xn


```
