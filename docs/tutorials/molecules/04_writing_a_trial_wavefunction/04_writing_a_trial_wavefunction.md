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

<!--
This is the main version of the Writing a Trial Wavefunction tutorial. The lattice-model version
includes the regions between the "BEGIN wfn-NAME" and "END wfn-NAME" markers below,
so keep those regions free of system-specific content and code cells. Code cells and
system-specific text go between the regions, and every version repeats them.
-->

<b>Goal:</b>
Become acquainted with how to write a trial wavefunction to the SAFIRE HDF5 format.

## What You Will Learn

1.  How to write a single Slater determinant trial wavefunction given the Slater Matrix, for closed-shell, collinear, and noncollinear spin
2.  How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3.  How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings

## Setup

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

+++ {"id": "vt9PbuQhpw4u"}

<!-- BEGIN wfn-introduction -->

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
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0,N-1} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1,N-1} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M-1,0} & \bar{C}_{M-1,1} & \bar{C}_{M-1,2} & \dots  & \bar{C}_{M-1,N-1}
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

2. <b> non-orthogonal multi-Slater determinant (NOMSD) trial wavefunction </b>
where **strictly** $\langle \Phi_n |\Phi_m\rangle \neq 0 \forall n,m$.
In this case, the Slater determinants are specified as a list
of $C_n$ and corresponding Slater matrices. 

<!-- END wfn-introduction -->

We'll explore each of these wavefunction types in more detail below.

+++ {"id": "RXzexuh_sQyL"}

## Single Slater Determinant Trial Wavefunction

We have already seen examples of writing a single Slater determinant trial wavefunction using {py:meth}`safiretools.Wavefunction.from_single_determinant` in previous tutorials.

<!-- BEGIN wfn-single-det -->

{py:meth}`safiretools.Wavefunction.from_single_determinant` takes the Slater matrix, whose columns are the occupied orbitals, so it has shape $M \times N$ with generally more basis functions $M$ than electrons $N$.
How the Slater matrix is passed decides the spin symmetry of the wavefunction, which, together with that of the Hamiltonian, decides the walker type (see {ref}`Walker-classes`).
All examples below use the same set of four orthonormal orbitals.

### Closed Shell

In a closed-shell determinant, spin-up and spin-down electrons occupy the same orbitals, so a single $M \times N_\uparrow$ array describes both spin sectors.
Here, we put two electrons of each spin into the lowest two of four orbitals.
Under the hood, this creates a NOMSD wavefunction with a single determinant, as the code block below shows.

<!-- END wfn-single-det -->

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

<!-- BEGIN wfn-single-det-constructor -->

In the `NOMSDWavefunction` constructor, `coeffs` is a 1-dimensional array of expansion coefficients, and `dets` is a 3-dimensional array where the first index corresponds to the Slater determinant index, the second index corresponds to the basis set index, and the third index corresponds to the electron index.
See {ref}`Wavefunction-classes` for details on the file format.

<!-- END wfn-single-det-constructor -->

<!-- BEGIN wfn-collinear -->

### Collinear

In a collinear determinant, spin-up and spin-down electrons occupy different orbitals, and their numbers may differ.
The Slater matrix is passed as a tuple of an $M \times N_\uparrow$ and an $M \times N_\downarrow$ array.
It has to be a tuple: a list is rejected, and a single stacked array of shape $2 \times M \times N$ is read as a noncollinear determinant.
Here, three spin-up electrons occupy the lowest three orbitals, while the two spin-down electrons occupy orbitals 0 and 2.

<!-- END wfn-collinear -->

```{code-cell} ipython3
phi_up = orbitals[:, :3]
phi_down = orbitals[:, [0, 2]]

wavefunction = Wavefunction.from_single_determinant((phi_up, phi_down))
print(wavefunction.spin_symm.label, wavefunction.nelec)

wavefunction.to_hdf5("collinear_wfn.h5")
```

<!-- BEGIN wfn-noncollinear -->

### Noncollinear

If spin is not a good quantum number, e.g. in the presence of spin-orbit coupling or noncollinear magnetic order, each orbital is a two-component spinor.
The Slater matrix is then passed as a single $2 \times M \times N$ array, where the first index selects the spin-up or spin-down component and $N$ is the total number of electrons.
Here, three spinors occupy the lowest three orbitals, with their spins rotated away from the $z$ axis by $\theta = 0, \pi/2, \pi$ in the $xz$ plane.
A noncollinear wavefunction reports its electron count as `nelec = (3, 0)`.

<!-- END wfn-noncollinear -->

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

<!-- BEGIN wfn-nomsd -->

## NOMSD Trial Wavefunction

In the previous example, we learned how to specify a trial wavefunction with a single Slater determinant. Here, we will see how we can write a suitable non-orthogonal multi-Slater trial wavefunction for the AFQMC code.

We take two closed-shell determinants with two electrons per spin in the four orbitals from above. Both occupy the first orbital; the second determinant replaces the second orbital by an equal mix of the second and third, so the two determinants are normalized but not orthogonal to each other.
As in the single determinant case, `phi_nomsd` has the dimensions (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\mathrm{elec}$).
Collinear and noncollinear expansions work the same way: each array of the corresponding single-determinant layout gains a leading N$_\mathrm{dets}$ axis, giving a tuple of (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\uparrow$) and (N$_\mathrm{dets}$,N$_\mathrm{basis}$,N$_\downarrow$) arrays or a single (N$_\mathrm{dets}$,2,N$_\mathrm{basis}$,N$_\mathrm{elec}$) array, respectively.

<!-- END wfn-nomsd -->

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

+++ {"id": "YtBpL-q8rlGT"}

## PHMSD Trial Wavefunctions

In the previous sections, we learned (i) how to construct trial wavefunctions with a single Slater determinant, (ii) how to construct non-orthogonal multi-Slater determinant trial wavefunctions (NOMSD). Here, we will look at how to use CI-like expansions, also called particle-hole multi Slater (PHMSD) wavefunctions, as trial wavefunctions for AFQMC.

In contrast with the NOMSD case where each Slater determinant had its own set of molecular orbitals, PHMSD wavefunctions are characterized by a single set of molecular orbitals. Orthogonal Slater determinants are generated by occupying different orbitals contained in the set. For instance, let's take the following set of orbitals $ \{\phi_\mathrm{g}, \phi_\mathrm{u} \}$, with $\langle \phi_\mathrm{g} | \phi_\mathrm{u} \rangle = 0 $, and construct two Slater determinants
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
Combining these ingredients (array of normalized coefficients, array of occupied $α$ orbitals and array of occupied $β$ orbitals) into a `PHMSDWavefunction` allows us to write the PHMSD wavefunction as an hdf5 file for the AFQMC code.

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

+++ {"id": "FoovlUVcFqxX"}

## Summary

In this tutorial, you became acquainted with how to write a trial wavefunction to the SAFIRE HDF5 format.

### What You Learned

1. How to write a single Slater determinant trial wavefunction given the Slater Matrix, for closed-shell, collinear, and noncollinear spin
2. How to write a non-orthogonal multi-Slater determinant trial wavefunction given the expansion coefficients and Slater matrices
3. How to write a configuration-interaction (CI) type trial wavefunction given CI coefficients and a corresponding occupancy strings
