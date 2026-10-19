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

+++ {"id": "e9d26b92-e165-4baf-8fc1-dd312330f9d0"}

# Lattice Class Tutorial

In this tutorial, we will learn how to set up a lattice to define a Model Hamiltonian (See __LATTICE_HAMIL_TUTORIAL__ for details on building a Hamiltonian).
The `safiretools` Python package implements a Lattice class which is responsible for answering geometric questions about a specific lattice such as:

- what are the boundary conditions
- what are the $n^\mathrm{th}$-nearest neighbors
- which of those neighbors are direct neighbors, and which are image neighbors

and more.
The `Lattice` constructor builds a lattice from any unit cell, given by its lattice vectors and basis;
the `Lattice.from_type()` classmethod supplies the unit cell of one of the built-in lattice types.
In this tutorial, we will explore how the Lattice class works, and the available options for building a Lattice.

## Lattice Formalism

We consider lattices with primitive lattice vectors $\vec{a}_1$, and $\vec{a}_2$ (given explicitly or by a built-in lattice type) which may have a basis, $\{ \delta_p \}$ where $p$ runs over sub-lattice degrees of freedom, and the full cell has dimension $L_1$ x $L_2$.
The positions of lattice sites on sublattice $p$ are given by  
$$
\vec{r}_p = n_1 \vec{a}_1 + n_2 \vec{a}_2  + \vec{\delta}_p,
$$
where $n_1$, and $n_2$ are integers.
Additionally, each boundary is labelled by the primitive vector along which it is crossed.
For example, *boundary 1* is crossed along $\vec{a}_1$; it is the pair of edges at $0 * \vec{a}_1$ and $L_1 * \vec{a}_1$, which run parallel to $\vec{a}_2$.
Finally, when boundaries are crossed, a twist angle $\vec{\theta} = (\theta_1,\theta_2)$
is used to apply a phase where $\theta_1$/$\theta_2$ is applied when boundary 1 / boundary 2 are crossed, respectively.

<div class="alert alert-block alert-info">
<b>Important:</b>
    Only 2-dimensional lattices are currently implemented / supported.
    Contact the developers if 3-dimensional lattices would be useful to you.
</div>

TODO add a figure illustrating this.

## Building a Lattice

The constructor `Lattice(lattice_vectors, basis, Ls, *, boundaries, twists=(0.0, 0.0))` takes

- `lattice_vectors` : the lattice vectors $\vec{a}_1$ and $\vec{a}_2$, one per row of a 2 × 2 array
- `basis` : the list of site positions $\vec{\delta}_p$ within the unit cell
- `Ls` : the lattice dimensions $(L_1, L_2)$ along $\vec{a}_1$ and $\vec{a}_2$, in units of the unit cell
- `boundaries` : the boundary *types* along $\vec{a}_1$ (at $0 * \vec{a}_1$ and $L_1 * \vec{a}_1$)
  and along $\vec{a}_2$ (at $0 * \vec{a}_2$ and $L_2 * \vec{a}_2$)
- `twists` : the twist angles to apply, given as $\vec{\theta} = (\theta_1,\theta_2)$ in units of radians;
  optional, no twist by default

`boundaries` and `twists` are keyword-only. `boundaries` has no default, so every lattice states it explicitly.

For the built-in lattice types, `Lattice.from_type(type, Ls, *, boundaries, twists=(0.0, 0.0))` fills in
`lattice_vectors` and `basis` for you. It takes the same `Ls`, `boundaries` and `twists`, plus

- `type` : the lattice type, one of `'square'`, `'triangular'`, `'honeycomb'` or `'kagome'`

The supercell is the parallelogram spanned by $L_1 \vec{a}_1$ and $L_2 \vec{a}_2$.

We'll explore these options individually below.
Here is a minimal example of building a square lattice with open boundaries.
We also use `vis.plot_lattice()` so that we can visualize what we have made.

```{code-cell} ipython3
:id: 68d02ad8-c469-49b2-a54c-f8611c4ecb68

from safiretools import Lattice
from safiretools import vis

lattice = Lattice.from_type(
    'square',
    (4, 4),
    boundaries=('open', 'open'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice)
```

+++ {"id": "16542f56-b4f4-455e-b117-cc003b665648"}

As we can see, we have constructed a 4x4 square lattice.
Nearest-neighbors in the direct cell are indicated with a solid black line by default.
The lattice vectors, $\vec{a}_1$ and $\vec{a}_2$,
the lattice coordinate $n_1,n_2$,
and the lattice site index, $i$,
are shown for convenience as well.

### Exercise 1: Play with Dimensions

Rerun the cell above with different lattice dimensions.
Try:
- 12 x 4
- 2 x 4
- 4 x 7
- 1 x 10
- anything else you would like to try

+++ {"id": "e1134f16-cfc2-4089-9d00-034f0e063781"}

## Boundary Conditions

In this section, we will explore the available options for boundary conditions.
The boundary types are selected by `boundaries`, a pair of strings describing the boundary type
along $\vec{a}_1$ and $\vec{a}_2$.
The strings are case insensitive.

The example above used open boundaries,

```python
boundaries = ("open", "open")
```

alternatively, periodic boundary conditions can be used by setting

```python
boundaries = ("periodic", "periodic")
```

or mixed boundary conditions can be used by setting, for example,

```python
boundaries = ("open", "periodic")
```

<div class="alert alert-block alert-warning">
<b>Important:</b>
    When using periodic boundary conditions,
    lattice sites are not allowed to be their own
    image neighbors. So, for example, a 1-dimensional
    lattice can only be periodic along the long direction.
    A ValueError will be raised if any sites are found to be
    image neighbors of themselves.
</div>

### Exercise 2: Play with Boundary Conditions

Try setting different combinations of boundary conditions
in `boundaries` in the code block below.
The `vis.plot_lattice()` function will indicate the position of nearest image
neighbors with a black dotted line.

```{code-cell} ipython3
:id: 08d214dc-f42a-48ca-9215-f17f86f481fa

lattice = Lattice.from_type(
    'square',
    (4, 4),
    boundaries=('open', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice)
```

+++ {"id": "6fe94bfc-e5f9-400e-8cdd-5583342be096"}

### Twists

When using periodic boundary conditions, a twist angle, $\theta = (\theta_1,\theta_2)$ can be set such that, for image neighbors $i,j$,
$$
c^{\dagger}_i c_j = e^{i \theta_1} c^{\dagger}_i c_j
$$
when crossing boundary 1 in the $+\vec{a}_1$ direction
and
$$
c^{\dagger}_i c_j = e^{-i \theta_1} c^{\dagger}_i c_j
$$
when crossing boundary 1 in the $-\vec{a}_1$ direction.
Similar relations apply for boundary 2 and $\theta_2$.
The Lattice class does not directly use the Twist angle, but
stores the relative phase between pairs of lattice site neighbors
if they are set.

All of the following are valid twist inputs:

```python
twists = (0.0, 0.0)
twists = (3.14, 0.0)
twists = (0.0, 0.5)
twists = ("1/2 Pi", "3/2 Pi")
```

The twists are saved as the relative phase between image neighbors
consistent with the boundaries.

For example, in the code block

```python
lattice = Lattice.from_type(
    'square',
    (4, 4),
    boundaries=('open', 'periodic'),
    twists=('1/2 Pi', '1/2 Pi')
)
```

although we have provided a twist for both boundary 1 and boundary 2,
only the twist for boundary 2 will be applied since
boundary 1 is not periodic.

+++ {"id": "94ad1d70-c561-4211-b09b-eb5db0de05ac"}

## Lattice Types

We provide a few lattice types which can be selected by passing one of `"square"`,
`"triangular"`, `"honeycomb"`, or `"kagome"` as the `type` of `Lattice.from_type()`.

Try building a few of these in the code block below.

```{code-cell} ipython3
:id: 590508fb-00bd-4bb6-9e3a-03b95a19ba01

lattice = Lattice.from_type(
    'honeycomb',
    (4, 4),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Honeycomb lattice",show_labels=False)
```

+++ {"id": "f4903dd8-53d6-4fff-86b9-66e75098fd22"}

## Lattice Basis

The lattice vectors $\vec{a}_1$, $\vec{a}_2$ and the basis $\{ \delta_p \}$ are
fixed by the lattice `type` — they are what *makes* a type that type.
A `square` lattice is square precisely because its lattice vectors are the unit
x- and y-vectors and it has one site per cell; the honeycomb and kagome lattices
are the triangular lattice with a predefined 2- and 3-site basis, respectively.

Consequently, `Lattice.from_type()` takes no lattice vectors or basis; the lattice
it builds exposes them as `lattice.A`, whose columns are $\vec{a}_1$ and $\vec{a}_2$,
and `lattice.basis`:

```{code-cell} ipython3
lattice = Lattice.from_type('honeycomb', (4, 4), boundaries=('periodic', 'periodic'), twists=(0.0, 0.0))
print("A =", lattice.A)
print("basis =", lattice.basis)
```

To choose your own lattice vectors and basis, pass them to the `Lattice`
constructor as described in the next section.

### Exercise

Play around with setting a basis, using the `Lattice` constructor below.

+++ {"id": "67ef466a-2411-4d21-8fa1-8a2516db44e4"}

## Custom Lattices

If the lattice that you want is not one of the built-in types, pass its lattice
vectors and basis to the `Lattice` constructor directly.

```{code-cell} ipython3
:id: 1f3486b9-62ab-4687-a322-295de51b8ae8
:outputId: 220c8539-682a-473b-ce17-592734b4f775

import numpy as np

basis = [ np.array(delta) for delta in [(0,0),(0.5,0),(0,0.5),(1.0,0),(1.5,0),(1.0,0.5)] ]

lattice = Lattice(
    lattice_vectors=np.array([(1.0, -1.0), (1.0, 1.0)]),
    basis=basis,
    Ls=(4, 8),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)
vis.plot_lattice(lattice,title="Lieb lattice with tilted cell",show_labels=False)
```

+++ {"id": "2431a7fd-8a39-4172-9f68-b9bf3f7294c2"}

## Visualizing nth-Order Neighbors

The `vis.plot_lattice()` function can be used to visualize which pairs of sites
are $n^\mathrm{th}$-order neighbors.
As you will see in the {doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`,
$n^\mathrm{th}$-order neighbor hopping and interactions are available in the lattice model Hamiltonian builder tools.
Visualizing the neighbors can be helpful or ensuring that the desired hopping / interactions are being included,
especially when dealing with complicated lattices.

```{code-cell} ipython3
:id: b5d6e08c-4c41-4b9c-820f-4b480c4c3bc1
:outputId: 2e47ab4f-58e4-47f6-e39a-25726050c235


basis = [ np.array(delta) for delta in [(0,0),(0.5,0),(0,0.5),(1.0,0),(1.5,0),(1.0,0.5)] ]

lattice = Lattice(
    lattice_vectors=np.array([(1.0, -1.0), (1.0, 1.0)]),
    basis=basis,
    Ls=(4, 4),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)
vis.plot_lattice(
    lattice,
    title="Lieb lattice with tilted cell",
    show_labels=False,
    nth_neighbor=2
)
```

+++ {"id": "c6c6bd33-4fce-439d-bc76-c6663031b838"}

## Archive of Possible Examples

### A Water Molecule

TODO: write an intro. The goal here is to shift everyone's brains to thinking
about mapping atoms in a solid onto a lattice model without the complication of multiple sites.

```{code-cell} ipython3
:id: 3c519903-9583-4236-9e64-192a8026b087

lattice = Lattice(
    lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
    basis=[[0., 0.], [-0.5, -0.5], [0.5, -0.5]],
    Ls=(1, 1),
    boundaries=('open', 'open'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Water Molecule",show_labels=False)
```

+++ {"id": "2fafdcad-d356-47a6-80a4-930ba8a3a840"}

### Lieb Lattice

The Cuprates are a class of high-temperature superconductors which consists of 2-dimensional layers of CuO$_2$ and other materials.
The Cu and O atoms are arranged in the Lieb lattice as shown below.

```{code-cell} ipython3
:id: d4955fae-dac3-4197-a3b5-f35bf51b103e

lattice = Lattice(
    lattice_vectors=[[1.0, 0.0], [0.0, 1.0]],
    basis=[[0., 0.], [0.5, 0.0], [0.0, 0.5]],
    Ls=(4, 4),
    boundaries=('open', 'open'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Lieb Lattice",show_labels=False)
```

+++ {"id": "325db840-f820-4f10-bb2f-c346f51e8b63"}

### Kagome Lattice

```{code-cell} ipython3
:id: 3eac1b2a-f2e9-4416-81bb-88d02fb5952b

lattice = Lattice.from_type(
    'kagome',
    (4, 4),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Kagome Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: ae7e9e7c-896b-4ce4-86c7-98f6951756f7

lattice = Lattice.from_type(
    'triangular',
    (4, 4),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Triangular Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 9767452c-454d-4e3f-99a6-42a35f501b35

lattice = Lattice.from_type(
    'honeycomb',
    (4, 4),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Honeycomb Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 669f3165-f665-4fbc-940b-5783f111b1a9

lattice = Lattice.from_type(
    'square',
    (8, 8),
    boundaries=('periodic', 'periodic'),
    twists=(0.0, 0.0)
)

vis.plot_lattice(lattice,title="Square Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 733459f3-7200-4a86-89e2-6220bd503536


```
