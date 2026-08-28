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

# Lattice class tutorial

In this tutorial, we will learn how to set up a lattice to define a Model Hamiltonian (See __LATTICE_HAMIL_TUTORIAL__ for details on building a Hamiltonian).
The `safiretools` Python package implements a Lattice class which is responsible for answering geometric questions about a specific lattice such as:

- what are the boundary conditions
- what are the $n^{th}$-nearest neighbors
- which of those neighbors are "direct" neighbors, and which are image neighbors

and more.
`Lattice` is an abstract base class with one concrete subclass per lattice type; its
`Lattice.from_dict()` classmethod builds the right one for you given a set of geometric parameters.
In this tutorial, we will explore how the Lattice class works, and the available options for building a Lattice via `Lattice.from_dict()`.

## Lattice formalism

We consider lattices with primitive lattice vectors $\vec{a}_1$, and $\vec{a}_2$ (as determined by the lattice "type") which may have a basis, $\{ \delta_p \}$ where $p$ runs over sub-lattice degrees of freedom, and the full cell has dimension $L_1$ x $L_2$.
The positions of lattice sites on sublattice $p$ are given by  
$$
\vec{r}_p = n_1 \vec{a}_1 + n_2 \vec{a}_2  + \vec{\delta}_p,
$$
where $n_1$, and $n_2$ are integers.
Additionally, boundaries are defined normal to the primitive vectors and are labelled according to the primitive vector which they are normal to.
For example, "boundary 1" is normal to $\vec{a}_1$.
Finally, when boundaries are crossed, a "twist" angle $\vec{\theta} = (\theta_1,\theta_2)$
is used to apply a phase where $\theta_1$/$\theta_2$ is applied when boundary 1 / boundary 2 are crossed, respectively.

<div class="alert alert-block alert-info">
<b>Important:</b>
    Only 2-dimensional lattices are currently implemented / supported.
    Contact the developers if 3-dimensional lattices would be useful to you.
</div>

TODO add a figure illustrating this.

## Summary of Lattice.from_dict()

The `Lattice.from_dict()` classmethod takes a Python dictionary containing "lattice parameters" as an argument and returns a Lattice instance based on the provided "lattice parameters".

We'll explore the available options individually, but possible lattice parameters are:

- `type` : specifying the lattice type; `square` by default
- `L1` : the lattice dimension along the lattice vector $\vec{a}_1$
- `L2` : the lattice dimension along the lattice vector $\vec{a}_2$
- `boundary1` : the boundary *type* of the boundary normal to $\vec{a}_1$ (at $0 * \vec{a}_1$ and $L_1 * \vec{a}_1$ ). Defaults to an open boundary if not specified.
- `boundary2` : the boundary *type* of the boundary normal to $\vec{a}_2$ (at $0 * \vec{a}_2$ and $L_2 * \vec{a}_2$ ). Defaults to an open boundary if not specified.
- `twist` : the twist angle to apply, given as $\vec{\theta} = (\theta_1,\theta_2)$ in units of radians. Defaults to (0.0,0.0)
- `a1`, `a2`, `basis` : the lattice vectors and basis. **Only for `type='custom'`** — every other type
  defines its own geometry, and setting these for one of them is an error. See
  [Lattice basis](#lattice-basis) and [Custom lattices](#custom-lattices) below.
- `cyl_mode` : reshape the unit cell to be rectangular, `'XC'` or `'YC'`. **Only for
  `type='triangular'`**, since the reshaping is hardcoded for hexagonal geometry.

Below is a minimal example of building a lattice allowing most parameters to take on default values.
We also use `afqmctool`'s `plot_lattice()` function so that we can visualize what we have made.

```{code-cell} ipython3
:id: 68d02ad8-c469-49b2-a54c-f8611c4ecb68

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4
)

lattice = Lattice.from_dict(
    params=lattice_params
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

### Exercise 1: Play with dimensions

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
To select a boundary type, set `boundary1` and/or `boundary2` in the `lattice_params`
dictionary to a string describing the boundary type.
This is a case insensitive parameter.

As we saw previously, the default boundaries are open.
This can be made explicit by setting

```python
lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1= "open",
    boundary2= "open"
)
```

alternatively, periodic boundary conditions (PBC) can be used by setting

```python
lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1= "pbc",
    boundary2= "pbc"
)
```

or mixed boundary conditions can be used by setting, for example,

```python
lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1= "open",
    boundary2= "pbc"
)
```

<div class="alert alert-block alert-warning">
<b>Important:</b>
    When using periodic boundary conditions,
    lattice sites are not allowed to be their own
    image neighbors. So, for example, a 1-dimensional
    lattice can only have PBCs in the "long" direction.
    A ValueError will be raised if any sites are found to be
    image neighbors of themselves.
</div>

### Exercise 2: Play with boundary conditions

Try setting different combinations of boundary conditions
in `lattice_params` in the code block below.
The `plot_lattice()` function will indicate the position of nearest image
neighbors with a black dotted line.

```{code-cell} ipython3
:id: 08d214dc-f42a-48ca-9215-f17f86f481fa

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "open",
    boundary2 = "pbc",
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice)
```

+++ {"id": "6fe94bfc-e5f9-400e-8cdd-5583342be096"}

### Twists

When using periodic boundary conditions (PBCs), a "twist" angle, $\theta = (\theta_1,\theta_2)$ can be set such that, for image neighbors $i,j$,
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
twist = [0.0,0.0]
twist = [3.14,0.0]
twist = [0.0, 0.5]
twist = ["1/2 Pi", "3/2 Pi"]
```

If a twist is provide in `lattice_params`, then
it will be saved as the relative phase between image neighbors
consistent with the boundaries.

For example, in the code block

```python
from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "open",
    boundary2 = "pbc",
    twist = ["1/2 Pi", "1/2 Pi"]
)

lattice = Lattice.from_dict(
    params=lattice_params
)
```

although we have provided a twist for both boundary 1 and boundary 2,
only the twist for boundary 2 will be applied since
boundary 1 is not periodic.

+++ {"id": "94ad1d70-c561-4211-b09b-eb5db0de05ac"}

## Lattice Types

We provide a few lattice types which can be selected by setting `"type"` to one of `"square"`,
`"triangular"`, `"honeycomb"`, or `"kagome"` in `lattice_params`

Try building a few of these in the code block below.

```{code-cell} ipython3
:id: 590508fb-00bd-4bb6-9e3a-03b95a19ba01

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = "honeycomb"
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Honeycomb lattice",show_labels=False)
```

+++ {"id": "f4903dd8-53d6-4fff-86b9-66e75098fd22"}

## Lattice basis

The lattice vectors $\vec{a}_1$, $\vec{a}_2$ and the basis $\{ \delta_p \}$ are
fixed by the lattice `type` — they are what *makes* a type that type.
A `square` lattice is square precisely because its lattice vectors are the unit
x- and y-vectors and it has one site per cell; the honeycomb and kagome lattices
are the triangular lattice with a predefined 2- and 3-site basis, respectively.

Consequently, setting `a1`, `a2` or `basis` in `lattice_params` for any of these
types raises an error rather than being silently ignored:

```{code-cell} ipython3
from safiretools import Lattice

try:
    Lattice.from_dict(dict(L1=4, L2=4, type="square", basis=[[0.0,0.0],[0.5,0.5]]))
except ValueError as e:
    print("ValueError:", e)
```

To choose your own lattice vectors and basis, use the `custom` lattice type
described in the next section — that is exactly what it is for.

<div class="alert alert-block alert-info">
<b>Note:</b>
    Earlier versions of this tooling accepted <code>a1</code>/<code>a2</code>/<code>basis</code>
    for the built-in types and then discarded them, so a lattice could quietly
    differ from the one you asked for. Those keys are now rejected.
</div>

### Exercise

Play around with setting a basis, using the `custom` type below.

+++ {"id": "67ef466a-2411-4d21-8fa1-8a2516db44e4"}

## Custom lattices

If the lattice that you want is not specifically implemented, you can use the
"custom" lattice.
Defining the unit cell yourself is exactly what makes a lattice "custom", so this
is the one type for which `a1`, `a2` and `basis` are accepted — and `a1` and `a2`
are required.
`basis` is optional and defaults to a single site at the cell origin.

```{code-cell} ipython3
:id: 1f3486b9-62ab-4687-a322-295de51b8ae8
:outputId: 220c8539-682a-473b-ce17-592734b4f775

import numpy as np
from safiretools import Lattice
import afqmctools.utils.visualize as vis

basis = [ np.array(delta) for delta in [(0,0),(0.5,0),(0,0.5),(1.0,0),(1.5,0),(1.0,0.5)] ]

lattice_params = dict(
    L1 = 4,
    L2 = 8,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = "custom",
    basis = basis,
    a1=np.array((1.0,-1.0)),
    a2=np.array((1.0,1.0)),
)

lattice = Lattice.from_dict(
    lattice_params
)
vis.plot_lattice(lattice,title="Lieb lattice with tilted cell",show_labels=False)
```

+++ {"id": "2431a7fd-8a39-4172-9f68-b9bf3f7294c2"}

## Visualizing nth-order neighbors

The `vis.plot_lattice()` function can be used to visualize which pairs of sites
are $n^{th}$-order neighbors.
As you will see in the {doc}`../04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`,
$n_{th}$-order neighbor hopping and interactions are available in the lattice model Hamiltonian builder tools.
Visualizing the neighbors can be helpful or ensuring that the desired hopping / interactions are being included,
especially when dealing with complicated lattices.

```{code-cell} ipython3
:id: b5d6e08c-4c41-4b9c-820f-4b480c4c3bc1
:outputId: 2e47ab4f-58e4-47f6-e39a-25726050c235


basis = [ np.array(delta) for delta in [(0,0),(0.5,0),(0,0.5),(1.0,0),(1.5,0),(1.0,0.5)] ]

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = "custom",
    basis = basis,
    a1=np.array((1.0,-1.0)),
    a2=np.array((1.0,1.0)),
)

lattice = Lattice.from_dict(
    lattice_params
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

### a water molecule:

TODO: write an intro. The goal here is to shift everyone's brains to thinking
about mapping atoms in a solid onto a lattice model without the complication of multiple sites.

```{code-cell} ipython3
:id: 3c519903-9583-4236-9e64-192a8026b087

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 1,
    L2 = 1,
    type = 'custom',
    a1 = [1.0,0.0],
    a2 = [0.0,1.0],
    basis = [[0.,0.],[-0.5,-0.5],[0.5,-0.5]]
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Water Molecule",show_labels=False)
```

+++ {"id": "2fafdcad-d356-47a6-80a4-930ba8a3a840"}

### Lieb Lattice

The Cuprates are a class of high-temperature superconductors which consists of 2-dimensional layers of CuO$_2$ and other materials.
The Cu and O atoms are arranged in the Lieb lattice as shown below.

```{code-cell} ipython3
:id: d4955fae-dac3-4197-a3b5-f35bf51b103e

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    type = 'custom',
    a1 = [1.0,0.0],
    a2 = [0.0,1.0],
    basis = [[0.,0.],[0.5,0.0],[0.0,0.5]]
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Lieb Lattice",show_labels=False)
```

+++ {"id": "325db840-f820-4f10-bb2f-c346f51e8b63"}

### Kagome Lattice

```{code-cell} ipython3
:id: 3eac1b2a-f2e9-4416-81bb-88d02fb5952b

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = 'kagome'
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Kagome Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: ae7e9e7c-896b-4ce4-86c7-98f6951756f7

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = 'triangular'
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Triangular Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 9767452c-454d-4e3f-99a6-42a35f501b35

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 4,
    L2 = 4,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = 'honeycomb'
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Honeycomb Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 669f3165-f665-4fbc-940b-5783f111b1a9

from safiretools import Lattice
import afqmctools.utils.visualize as vis

lattice_params = dict(
    L1 = 8,
    L2 = 8,
    boundary1 = "pbc",
    boundary2 = "pbc",
    type = 'square'
)

lattice = Lattice.from_dict(
    params=lattice_params
)

vis.plot_lattice(lattice,title="Square Lattice",show_labels=False)
```

```{code-cell} ipython3
:id: 733459f3-7200-4a86-89e2-6220bd503536


```
