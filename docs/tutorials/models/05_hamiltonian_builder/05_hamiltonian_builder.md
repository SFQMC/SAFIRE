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

+++ {"id": "029b98c5-fdd9-45e9-bd82-9a8acee63e5c"}

# The Hamiltonian Builder

The Hamiltonian builder requires a Lattice instance to build the Hamiltonian on
(see {doc}`../03_setting_up_a_lattice/03_setting_up_a_lattice` for details on the lattice parameters).
We assume familiarity with the Lattice class in this tutorial.

The Hamiltonian builder (`HamiltonianBuilder` class in afqmctools) is
responsible for performing "build steps".
A "build step" consists of constructing a Hamiltonian component (i.e. term),
and adding it to the `Hamiltonian` that it has.
As we saw in the previous tutorial ({doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`),
`HamiltonianBuilder.from_input()` can be used to build a lattice model Hamiltonian by
automatically running the build steps implied by a set of input Hamiltonian parameters.
This is the recommended way of building lattice model Hamiltonians;
however, the Hamiltonian builder may be used directly
to gain more control over the build steps.

TODO: add in figure with the builder highlighted.

## Part I: Hamiltonian builder Overview

"Build steps" are implemented as methods of the `HamiltonianBuilder` class
and correspond to a specific type of Hamiltonian term.
We'll cover which specific build steps are implemented below.

Let's begin with a quick overview of how the build steps work:

1. calling a build step will construct the corresponding Hamiltonian term and add it to the Hamiltonian. For example, the following code snippet will add nearest neighbor hopping with t=1.0 to the Hamiltonian.

    ```python
    builder = HamiltonianBuilder(lattice=lattice)
    builder.nth_neighbor_hopping(1.0, nth_neighbor=1)
    ```
    
2. calling the same build step multiple times will add one term per call without overwriting any terms. So, the following code snippet will add both t, and t' to the Hamiltonian.

    ```python
    builder = HamiltonianBuilder(lattice=lattice)
    builder.nth_neighbor_hopping(1.0, nth_neighbor=1)
    builder.nth_neighbor_hopping(0.5, nth_neighbor=2)
    ```

3. While each build step has it's own respective input parameters and conventions, there are a few general rules that apply to nearly all build steps.

    - Each term's primary input parameters can generally be provided as one of the following input "units":
        - a floating point number, implying that it should be applied uniformly
        - a 2d-array with shape nband x nband, implying a band-dependent Hamiltonian term.
    - For each term, the input parameter can be either:
        - a single "unit" as defined above
        - a list of "units"
    
    In the of case of a list of input units, the input is interpreted as multiple Hamiltonian terms where the list
    position is the "neighbor order" that the term will be constructed for. For example, the code snippet below will
    generate both nearest (t=1.0), and next-nearest (t'=0.5) neighbor-hopping.

    ```python
    builder = HamiltonianBuilder(lattice=lattice)
    builder.nth_neighbor_hopping([1.0,0.5])
    ```

    This also applies to many of the interactions, for example, the extended Hubbard model nearest-neighbor V =2.0 term can
    be added to the Hamiltonian along with a next-nearest neighbor V' = 1.0 term using

    ```python
    builder = HamiltonianBuilder(lattice=lattice)
    builder.nth_order_hubbard_Vij([2.0,1.0])
    ```

The terms for which rule 3 do not currently apply are:

- onsite_hubbard()
- custom_one_body()
- rashba_soc()
- pinning field terms:
    - afm_pinning()
    - fm_pinning()
    - charge_pinning()
    - edge_pinning()

+++ {"id": "2ce4ac1b-1c09-48e4-84da-a294502421a9"}

## Part II: Using the Hamiltonian builder

The Hamiltonian builder must be used in conjunction with a Lattice instance.
In addition, afqmctools provides a function that can directly write the Hamiltonian
that is built by the Hamiltonian builder into an HDF5 file that can be used by SAFIRE.
Therefore, a typical use of the `HamiltonianBuilder` class in a Python script will look like the
code block below.

Here, we construct the Hubbard model on a 4x4 square lattice with $t=1.0$, $t'=0.5$, and $U=4.0$.

$$
\hat{H} = \sum_{\langle i,j\rangle^1} -(1.0) \hat{c}^\dagger_i \hat{c}_j + \sum_{\langle i,j\rangle^2} -(0.5) \hat{c}^\dagger_i \hat{c}_j + \sum_i (4.0) \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}
$$

where the notation $\langle i,j\rangle^n$ indicates that only pairs of indices corresponding to $n^{th}$-order neighbors
are included.

```{code-cell} ipython3
:id: fdf7cd0e-b553-4ac1-8f0f-1730a3faf403

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.io import write_model_hamiltonian
from pathlib import Path

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)

lattice = get_lattice(
    params={
        'L1' : 4,
        'L2' : 4,
        'boundary1' : 'PBC',
        'boundary2' : 'PBC',
        'type' : 'square'
    }
)

builder = HamiltonianBuilder(lattice=lattice)

builder.nth_neighbor_hopping([1.0,0.5])
builder.onsite_hubbard(4.0)

# highly recommended to call finalize!
builder.finalize()

# save for latter use
write_model_hamiltonian(
    hamiltonian=builder.hamiltonian,
    fname=scratch_dir / 'hamiltonian.h5'
)
```

+++ {"id": "d95754b7-05ef-40a0-9632-c660f535d095"}

we note that calling the `.finalize()` function after calling all build steps is recommended.
It performs various clean up tasks, such as combining all Hamiltonian terms that can be safely combined
into a single term.

+++ {"id": "ac73ed88-8485-4885-ae6b-ca20955c0b51"}

## Part III: Hamiltonian build steps

Now that we understand the role of the Hamiltonian builder, and how it will typically be used in a Python script,
we will cover the specific terms that can be built.
We will also cover any term specific details that have not been covered so far.

**We note that the same input conventions for parameters are used by `HamiltonianBuilder.from_input()` since it simply forwards parameters to these build steps.**

Each term is explored in it's own respective subsection.
A minimal list of sections that should be covered are,

- hopping matrix
- applying a twist
- on-site hubbard interaction
- applying a pinning field
- extended Hubbard V

Some more advanced sections are as follows. Each is written to be independent such that an interested learner can go through the sections that are specific interest to them.

- Hubbard-Kanamori U1, U2, and J
- rashba SOC
- custom one body terms

+++ {"id": "ea07187d-0c31-41d9-9a7c-b3087af2b501"}

## Adding Hopping

To cover:
- nth-order hopping
- band-dependent hopping (with 1st-nth order hopping)
- sublattice-dependent hopping (with 1st-nth order hopping)
- hopping with non-trivial sublattice and band degrees of freedom.
- using no hopping

`afqmctools` can generate nth-order neighbor hopping.

```{code-cell} ipython3
:id: 8b6da842-a3a2-4a30-a408-b3af7a185ad2

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

hamiltonian_params = {
    "lattice" : dict(
        L1 = 4,
        L2 = 4,
        boundary1 = "pbc",
        boundary2 = "pbc"
    ),
    "hamiltonian" : dict(
        t = [1.0,0.5,0.1],
        U = 4.0
    )
}

hamiltonian = HamiltonianBuilder.from_input(hamiltonian_params).hamiltonian
```

+++ {"id": "668e3150-2f46-4f71-81c2-16845b15188e"}

## Adding Interactions

to cover

- onsite U
    - uniform
    - band/sublattice degrees of freedom (individual and combined)
- U1,U2,J
    - uniform
    - band/sublattice degrees of freedom (individual and combined)
- hst over-rides (expert-mode)

```{code-cell} ipython3
:id: 91b862e0-881c-4e53-a5ee-cd23ede58378


```

+++ {"id": "3c8dd617-521c-49b5-86d5-1c78e323e9ec"}

## Adding Rashba SOC

to cover:
- basic inclusion
- interaction with the hopping term.

```{code-cell} ipython3
:id: 62397d4e-beb3-4f2d-b984-4d1cbcc65a81


```

+++ {"id": "ddd63dd5-fc60-4bb0-9eea-b80758e17024"}

## How to handle more general Hamiltonians

to cover:
- combining `from_input()`-style build steps with fully custom build steps

```{code-cell} ipython3
:id: da9c7b01-bc46-448d-b4dd-9d057bfac5b7

import random
import scipy.sparse as sps
import numpy as np
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import SpinSymm
from afqmctools.systems.lattice import get_lattice

lattice = get_lattice(
    params=dict(
        L1 = 4,
        L2 = 4,
        boundary1 = "pbc",
        boundary2 = "pbc"
    )
)

# equivalent to HamiltonianBuilder.from_input() with hamiltonian=dict(t=1.0, U=4.0),
# but invoked step-by-step so that we can add a custom term below
builder = HamiltonianBuilder(lattice=lattice, spin_symm=SpinSymm.COLLINEAR)
builder.nth_neighbor_hopping(1.0)
builder.onsite_hubbard(4.0)

# make a custom disorder term
nsites = builder.hamiltonian.nsites
nbands = builder.hamiltonian.nbands
nbasis = nsites*nbands

# adding disorder to a random site, uniform across bands
random_site = random.randint(0, nsites-1)

print("adding disorder to site", random_site)

data = [0.1]*nbands
row = np.arange(nbands) + random_site
col = np.arange(nbands) + random_site

disorder_up = sps.csr_array((data, (row, col)), shape=(nbasis, nbasis))

random_site = random.randint(0, nsites-1)

print("adding disorder to site", random_site)

data = [0.1]*nbands
row = np.arange(nbands) + random_site
col = np.arange(nbands) + random_site

disorder_down = sps.csr_array((data, (row, col)), shape=(nbasis, nbasis))

# add the disorder term to the Hamiltonian
builder.custom_one_body(sps.vstack([disorder_up, disorder_down]))

# finalize once, after all build steps (both from_input()-style and custom) have been invoked
builder.finalize()

hamiltonian = builder.hamiltonian
```

+++ {"id": "31a5cea3"}

### Some Notes

Since custom build steps are just method calls on the same `HamiltonianBuilder` instance as
`from_input()`-style ones, they can be freely interleaved in any order you like.
The only rule is to call `.finalize()`
once, after every build step has been invoked.

```{code-cell} ipython3
:id: fead54f6

import matplotlib.pyplot as plt

plt.matshow(hamiltonian.get_one_body().toarray())
plt.colorbar()
```

```{code-cell} ipython3
:id: 6c70ce9c


```
