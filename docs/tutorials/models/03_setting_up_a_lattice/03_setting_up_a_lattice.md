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

# Defining a Lattice

In this tutorial, we will learn how to set up a lattice using the {py:class}`safiretools.Lattice` class. The lattice is a purely geometrical description of the connectivity of a lattice model. It forms the main ingredient of a lattice model Hamiltonian file for SAFIRE, which we will build in the next tutorial {doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`.

## Lattice Formalism

We consider lattices with primitive lattice vectors $\vec{a}_1$ and $\vec{a}_2$ which may have a basis, $\{ \vec{\delta}_p \}$ where $p$ runs over sub-lattice degrees of freedom, and the supercell has dimension $L_1 \times L_2$.
The basis is given in fractional coordinates, $\vec{\delta}_p = (\delta_{p,1}, \delta_{p,2})$, i.e. in units of the lattice vectors.
The positions of lattice sites on sublattice $p$ are given by  
$$
\vec{r}_{n_1,n_2,p} = (n_1 + \delta_{p,1}) \vec{a}_1 + (n_2 + \delta_{p,2}) \vec{a}_2,
$$
where $n_1$, and $n_2$ are integers.

These positions span a supercell with sides $L_1 \vec{a}_1$ and $L_2 \vec{a}_2$ with two boundary conditions 1 and 2, determining how $L_1 \vec{a}_1$ and $L_2 \vec{a}_2$ wrap around, respectively. These boundaries may have a twist angle $\vec{\theta} = (\theta_1,\theta_2)$ so that electrons crossing them pick up a phase.

<div class="alert alert-block alert-info">
<b>Important:</b>

Only 2-dimensional lattices are currently implemented.
Contact the developers if 3-dimensional lattices would be useful to you.

</div>

## Creating a Lattice

The simplest way to build a lattice is to use one of the built-in lattice types with the {py:meth}`safiretools.Lattice.from_type` constructor. The following snippet creates a 4 × 4 square lattice with open boundary conditions.

```{code-cell} ipython3
:id: 68d02ad8-c469-49b2-a54c-f8611c4ecb68

import numpy as np
from safiretools import Lattice
from safiretools import vis

lattice = Lattice.from_type(
    'square',
    (4, 4),
    boundaries=('open', 'open'),
)

vis.plot_lattice(lattice)
```

The built-in lattice types are `'square'`, `'triangular'`, `'honeycomb'`, or `'kagome'` and the second argument is the lattice dimensions $(L_1, L_2)$.

+++ {"id": "16542f56-b4f4-455e-b117-cc003b665648"}

As we can see, we have constructed a 4 × 4 square lattice.
Nearest neighbors in the direct cell are indicated with a solid black line by default.
The lattice vectors, $\vec{a}_1$ and $\vec{a}_2$,
the lattice coordinate $n_1,n_2$,
and the lattice site index, $i$,
are shown for convenience as well.

### Exercise 1: Play with Dimensions

Rerun the cell above with different lattice dimensions and lattice types.
Try:
- 12 × 4 square lattice
- 4 × 7 honeycomb lattice
- 1 × 10 triangular lattice

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
    boundaries=('open', 'periodic')
)

vis.plot_lattice(lattice)
```

+++ {"id": "6fe94bfc-e5f9-400e-8cdd-5583342be096"}

### Twists

When using periodic boundary conditions, a twist angle, $\vec{\theta} = (\theta_1,\theta_2)$ can be set such that, for neighbors $(i,j)$ where going from $i$ to $j$ crosses boundary 1 in the $+\vec{a}_1$ direction,
$$
c^{\dagger}_i c_j  + c^{\dagger}_j c_i \to e^{i \theta_1} c^{\dagger}_i c_j + e^{-i \theta_1} c^{\dagger}_j c_i.
$$
Similar relations apply for boundary 2 and $\theta_2$.

The twist $(\theta_1, \theta_2)$ is specified by the `twists` argument:

```{code-cell} ipython3
lattice = Lattice.from_type(
    'square',
    (4, 4),
    boundaries=('open', 'periodic'),
    twists=(np.pi/2, np.pi/2)
)
```

+++

Note that in the above example, due to the open boundary condition along $\vec{a}_1$, $\theta_1$ has no effect.

+++ {"id": "67ef466a-2411-4d21-8fa1-8a2516db44e4"}

## Custom Lattices

If the lattice that you want is not one of the built-in types, pass its lattice
vectors (as the rows of `lattice_vectors`) and basis to the {py:class}`safiretools.Lattice` constructor directly. The basis is given in fractional coordinates of the lattice vectors.

```{code-cell} ipython3
:id: 1f3486b9-62ab-4687-a322-295de51b8ae8
:outputId: 220c8539-682a-473b-ce17-592734b4f775

basis = [(0, 0), (0.25, 0.25), (-0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.25, 0.75)]

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
As you will see in {doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`,
$n^\mathrm{th}$-order neighbor hopping and interactions are available in the lattice model Hamiltonian builder tools.
Visualizing the neighbors can be helpful for ensuring that the desired hopping / interactions are being included,
especially when dealing with complicated lattices.

```{code-cell} ipython3
:id: b5d6e08c-4c41-4b9c-820f-4b480c4c3bc1
:outputId: 2e47ab4f-58e4-47f6-e39a-25726050c235


basis = [(0, 0), (0.25, 0.25), (-0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (0.25, 0.75)]

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

## Further Examples

+++ {"id": "2fafdcad-d356-47a6-80a4-930ba8a3a840"}

### Lieb Lattice

The cuprates are a class of high-temperature superconductors which consist of 2-dimensional layers of CuO$_2$ and other materials.
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

### Triangular Lattice

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
