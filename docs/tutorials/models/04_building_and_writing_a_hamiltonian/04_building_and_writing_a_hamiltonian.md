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

+++ {"id": "kEXtY-HpOK_T"}

# Building Lattice Model Hamiltonians

In {docs}`TODO: ref to previous tutorial`, we learned how to create a {py:class}`safiretools.Lattice` object. The `Lattice` object only describes the geometry of a lattice. To build a lattice model Hamiltonian from a lattice, we need to supply additional information about the physical couplings between electrons on the lattice sites. In this tutorial, we will see how {py:meth}`safiretools.LatticeHamiltonian.from_lattice` can be used to do this and create lattice model Hamiltonian inputs for SAFIRE.


SAFIRE can handle model Hamiltonians of the form
AGENT TODO: refer to the overview page for the hamiltonian.



### Example: Standard Hubbard Model

Below, we construct a Hubbard model on a 4×4 square lattice with both nearest, and next-nearest neighbor hopping by providing a list of values for `t`.
We plot the hopping matrix to demonstrate that the Hamiltonian includes both terms.
Note that the default hopping matrix shape is $2N_\mathrm{sites} \times N_\mathrm{sites}$ where the first $N_\mathrm{sites}$ rows correspond to the spin-up sector and the last $N_\mathrm{sites}$ rows correspond to the spin-down sector.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 979
id: 6QgfLv7kOK_V
outputId: c3552c28-5791-45f0-ae57-39f29435e27d
---
%mkdir -p data
%cd -q data

from safiretools import Lattice, LatticeHamiltonian
from safiretools import vis

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'))
vis.plot_lattice(lattice)

hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0)
```

+++ {"id": "_7Wg-_UGQuXr"}

## Writing Hamiltonians to HDF5

In the code block below, we demonstrate how to save a lattice model Hamiltonian
to an HDF5 file that can be read by SAFIRE.
Every Hamiltonian class in safiretools has a `.to_hdf5()` method that writes the
format SAFIRE reads for that kind of Hamiltonian, so this works the same way for
all of them. We will not repeat this below.

```{code-cell} ipython3
:id: 1ugXKZ5QQ7ug

hamiltonian.to_hdf5("hamiltonian.h5")
```

+++

Now, let's check the contents of the scratch directory to see that the Hamiltonian has been generated.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xrykuJS4SnBR
outputId: bfe2522e-2996-43d9-88b3-ed2cd7b84350
---
!ls -l
```

+++ {"id": "HEbY-vNOOK_W"}

## Adding More Terms to the Hamiltonian

### Adding Hopping

`LatticeHamiltonian.from_lattice()` supports nth-order neighbor hopping and band-dependent hopping.
$n$ is limited by the condition that sites are not allowed to be their own neighbors via
periodic boundary conditions.
In this case, an error will be raised.

The simplest case,
nearest-neighbor hopping with $t$ = 1, is included in the Hamiltonian by default.
For multiband Hamiltonians, the hopping is uniform across bands and it is assumed that
there is no intrasite hopping.
This is handles by a different Hamiltonian term, which we refer to as $\epsilon$.
You can change this value by providing a different value of $t$,

```python
hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=0.5)
```

hopping can be turned off by setting $t$ to either 0 or `None`.

nth-order neighbor hopping can be included by setting $t$ to a list of length $n$
where the nth-entry is interpreted as the hopping strength for nth-order neighbors.
Formally, if $t_n$ are the elements of the list, $t$,
the following term will be added,
$$
\hat{H}_t = \sum_n - t_n \sum_{\langle i,j \rangle^n} \hat{c}^\dagger_i \hat{c}_j
$$
where $\langle i,j \rangle^n$ indicates that the sum is constrained to run over nth-order neighbors.

> **NOTE**:
> nth-order neighbors are determined by spatial distance. This is especially important to keep in mind when attempting to use a lattice with a basis.

<div class="alert alert-block alert-warning">
<b>CAUTION:</b>
    nth-order neighbors are determined by spatial distance. This is especially important to keep in mind when attempting to use a lattice with a basis.
</div>

+++ {"id": "Tk2-PveROK_X"}

#### Example: Hubbard Model with $t'$

Below, we construct a Hubbard model on a 4×4 square lattice with both nearest, and next-nearest neighbor hopping by providing a list of values for `t`.
We plot the hopping matrix to demonstrate that the Hamiltonian includes both terms.
Note that the default hopping matrix shape is $2N_\mathrm{sites} \times N_\mathrm{sites}$ where the first $N_\mathrm{sites}$ rows correspond to the spin-up sector and the last $N_\mathrm{sites}$ rows correspond to the spin-down sector.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: yAa96ZdbOK_X
outputId: 731d316b-7dbb-4990-f273-0ad6da75081a
---
import matplotlib.pyplot as plt

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'))

# this will make a Hubbard model on a 4×4 lattice with periodic boundaries
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    U=4.0,
    t=[1.0, 0.5] # implicit minus sign!
)

hopping = hamiltonian.get_one_body()
plt.matshow(hopping.toarray())
plt.colorbar()
plt.show()
```

+++ {"id": "09Ir9dWeOK_W"}

### Adding Onsite Hubbard U

SAFIRE supports arbitrary onsite interactions of the from

$$
\hat{H}_U =  \sum_{\mu} U_{\mu} \hat{n}_{\mu\uparrow} \hat{n}_{\mu\downarrow},
$$

where, again, the combined index $\mu$ may include both lattice site and band indices.
We've already seen how to add a basic nearest-neighbor onsite Hubbard U, now we see see how to provide a site-, and/or band-dependent onsite interaction.
<b>This input parameter, $U$, is in general an array. The band/site dependence of the interaction is deduced based on the shape of the $U$ array</b>.

#### Band- and Site-Dependence Deduction Rules

1. if U is a scalar (or an array with a single entry), then the onsite Hubbard interaction term is included with strength U and applied to all sites and bands uniformly.

2. if U is 1-dimensional with length nbands (i.e. [U1,U2,…,Um] where for an m-band model), then the onsite Hubbard interaction term is included with strength U_i applied to band i and uniformly across sites.

3. if U is 1-dimensional with length nsites (i.e. [U1,U2,…,Un] where n is the number of sites), then the onsite Hubbard interaction term is included with strength U_i applied to site i and uniformly across bands.

4. If nbands and nsites are equal, then nbands take priority (i.e. rule 2 applies and not rule 3).

+++ {"id": "BskmJL_Kn9Fm"}

#### Example: Site-Dependent U

In the example below, we build a site-dependent U with a checkerboard pattern.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 909
id: i0l8VxorOK_W
outputId: 4113cab3-7b9d-4366-edb5-ff8b22bd89fd
---
lattice = Lattice.from_type('square', (2, 4), boundaries=('periodic', 'periodic'))


# this will make a Hubbard model on a 2×4 lattice with periodic boundaries
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=1,
    U=[1.0,2.0,1.0,2.0,2.0,1.0,2.0,1.0]
)

# let's take a look at the U term:
U = hamiltonian.get_U().toarray().diagonal().reshape(lattice.L)

vis.plot_lattice(
    lattice,
    density=U,
    vmin=0.0,
    vmax=3.0
)
```

+++ {"id": "r53E1XbMoJM8"}

#### Example: Band-Dependent U

Now we will construct a band-dependent U which is *uniform* across lattice sites.
We will use two bands and the first band will have $U/t = 1$
and the second will have $U/t = 2$.

As a bonus, we also demonstrate how to include a band-dependent hopping.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: jN42bp5cOK_W
outputId: d257ba45-6561-41df-a581-8a19e1f73a2a
---
import numpy as np

lattice = Lattice.from_type('square', (2, 4), boundaries=('periodic', 'periodic'))


# this will make a Hubbard model on a 2×4 lattice with periodic boundaries
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=2,
    t=[np.array([[1.0, -0.2], [-0.2, 1.0]])], # bonus: band-dependent hopping
    U=[1.0,2.0]
)

# let's take a look at the U term:
U_shape = (lattice.L[0],lattice.L[1],hamiltonian.nbands)
U = hamiltonian.get_U().toarray().diagonal().reshape(U_shape)
vis.plot_lattice(
    lattice,
    density=U[:,:,0],
    title='U for band 0',
    vmin=0.0,
    vmax=3.0
)
vis.plot_lattice(
    lattice,
    density=U[:,:,1],
    title='U for band 1',
    vmin=0.0,
    vmax=3.0
)
```

```{code-cell} ipython3
:id: _6grG3FOOK_X

lattice = Lattice.from_type('square', (2, 4), boundaries=('periodic', 'periodic'))


# this will make a Hubbard model on a 2×4 lattice with periodic boundaries
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=2,
    t=[np.array([[1.0, -0.2], [-0.2, 1.0]])], # bonus: band-dependent hopping
    U=[1.0,2.0]
)

# let's take a look at the U term:
U_shape = (lattice.L[0],lattice.L[1],hamiltonian.nbands)
U = hamiltonian.get_U().toarray().diagonal().reshape(U_shape)
vis.plot_lattice(
    lattice,
    density=U[:,:,0],
    title='U for band 0',
    vmin=0.0,
    vmax=3.0
)
vis.plot_lattice(
    lattice,
    density=U[:,:,1],
    title='U for band 1',
    vmin=0.0,
    vmax=3.0
)
```

```{code-cell} ipython3
:id: G1F4jHEyOK_X


```

+++ {"id": "pqLN_brHOK_X"}

### Adding Density-Density Interaction

SAFIRE supports general interactions of the form,

$$
\hat{H}_{U1} =  \sum_{\mu<\nu} U_{\mu\nu}^1 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\downarrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\uparrow} ),
$$

where $\mu = (i,m)$ / $\nu = (i,p)$ are a composite lattice, $i$ / $j$, and band $m$, $p$ indices.

safiretools is able to generate a subset of interactions of this type;
The $U_{\mu\nu}^1$ matrix is constructed based on what information
the user provides.
specifically, it handles the following cases.

#### Case 1: Uniform Intrasite Interaction

The user provides a single value for $U^1$ and keeps the default $n=0$.

In this case, the density-density interaction takes the following form.

$$
\hat{H}_{U1} = U^1 \sum_{i} \sum_{mp} (\hat{n}_{(i,m)\uparrow} \hat{n}_{(i,p)\downarrow} + \hat{n}_{(i,m)\downarrow} \hat{n}_{(i,p)\uparrow} ),
$$

where we have replaced the composite indices with site and band indices for clarity.

#### Case 2: Band-Dependent Intrasite Interaction

The user provides a matrix with shape $nbands \times nbands$,

$$
U^1 = U^1_{mp},
$$

and keeps the default $n = 0$ for the neighbor order parameter.

In this case, the density-density interaction takes the following form.

$$
\hat{H}_{U1} =  \sum_{i} \sum_{mp} U_{mp}^1 (\hat{n}_{(i,m)\uparrow} \hat{n}_{(i,p)\downarrow} + \hat{n}_{(i,m)\downarrow} \hat{n}_{(i,p)\uparrow} ),
$$

where we have replaced the composite indices with site and band indices for clarity.

#### Case 3: Uniform nth-Order Neighbor Interaction

#### Case 4: Band-Dependent nth-Order Neighbor Interaction

The user provides the neighbor order, $n$, and a
 matrix with shape $nbands \times nbands$,

$$
U^1 = U^1_{mp}.
$$

In this case, the density-density interaction takes the following form.

$$
\hat{H}_{U1} =  \sum_{\langle i,j \rangle^n} \sum_{mp} U_{mp}^1 (\hat{n}_{(i,m)\uparrow} \hat{n}_{(i,p)\downarrow} + \hat{n}_{(i,m)\downarrow} \hat{n}_{(i,p)\uparrow} ),
$$

where we have replaced the composite indices with site and band indices for clarity.

#### Case 5: A List of Inputs from the Cases Above

If a user provides a list of valid inputs, then safiretools will generate each
term provided <b>using the list index / position for the neighbor order $n$.</b>
For example, if a user provides

```python
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=2,
    U1=[2.0,
        [[0.5, 1.0],
         [1.0, 0.5]]]
)
```

Both a uniform intrasite interaction with U1 = 2.0 (n=0 / onsite),
and a band-dependent nearest-neighbor (n=1) interaction will be added to the
Hamiltonian.

+++ {"id": "_BXfk70iudQJ"}

### Adding Spin-Spin Interaction

SAFIRE supports general interactions of the form,

$$
\hat{H}_{U2} = \sum_{\mu<\nu} U_{\mu\nu}^2 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\uparrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\downarrow} ),
$$

where $\mu = (i,m)$ / $\nu = (i,p)$ are a composite lattice, $i$ / $j$, and band $m$, $p$ indices.

safiretools is able to generate a subset of interactions of this type;
The $U_{\mu\nu}^2$ matrix is constructed based on what information
the user provides similarly to the density-density (i.e. the $U^1$) term.
See the density-density section above for more details.

+++ {"id": "L-D4VJuBOK_X"}

### Adding a Hund's J Term

SAFIRE supports general interactions of the form,

$$
\hat{H}_J = \sum_{\mu<\nu} J_{\mu\nu} (
     \hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\nu\uparrow}
     +\hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\nu\uparrow}
     +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\mu\uparrow}
     +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\mu\uparrow}
     ).
$$

$\mu$ / $\nu$ are compound indices, $\mu = (i,p,m)$ / $\mu = (j,q,n)$, that include the lattice site index, ($i$ / $j$), sublattice, ($p$ / $q$) and band ($m$ / $n$) indices.

safiretools is able to generate a subset of interactions of this type;
The $J_{\mu\nu}$ matrix is constructed based on what information
the user provides similarly to the density-density (i.e. the $U^1$ term).
See the density-density section above for more details.

+++ {"id": "CIbAdbiaOK_X"}

#### Example: 3-Band Hubbard-Kanamori Hamiltonian

Below we construct a 3-band Hubbard-Kanamori model using uniform values for U,U1,U2, and J as in Table I of [PRB 99, 235142 (2019)](https://doi.org/10.1103/PhysRevB.99.235142)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: D72pRPc6OK_X
outputId: b6f1c99c-e33b-4574-d108-3a2ea7f66d89
---
lattice = Lattice.from_type('square', (6, 1), boundaries=('periodic', 'open'))
vis.plot_lattice(lattice)

hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=3,
    t=1.0,
    U=4.0,
    U1=1.5,
    U2=1.0,
    J=0.5
)
```

+++ {"id": "W-TfLNDJOK_W"}

### Adding an Extended Hubbard V Interaction

The $n^\mathrm{th}$-order neighbor extended Hubbard interaction,

$$
\hat{H}_V = \sum_{\langle i<j \rangle^n} \sum_{\sigma \sigma'} V_{\mu \nu} \hat{n}_{\mu \sigma } \hat{n}_{\nu \sigma'},
$$

where $\mu = (i,m)$ / $\nu = (i,p)$ are a composite lattice, $i$ / $j$, and band $m$, $p$ indices, can be expressed in terms of $\hat{H}_{U1}$ and $\hat{H}_{U2}$.

$$
\hat{H}_V = \sum_{\langle i<j \rangle^n} V_{\mu \nu} (\hat{n}_{\mu \uparrow} \hat{n}_{\nu \downarrow} + \hat{n}_{\mu \downarrow} \hat{n}_{\nu \uparrow})
+ \sum_{\langle i<j \rangle^n} V_{\mu \nu} (\hat{n}_{\mu \uparrow} \hat{n}_{\nu \uparrow} + \hat{n}_{\mu \downarrow} \hat{n}_{\nu \downarrow}),
$$

which we can identify as,

$$
\hat{H}_V = \hat{H}_{U1}[U1 = V] + \hat{H}_{U2}[U2 = V],
$$

for $n^\mathrm{th}$-order neighbors.e

safiretools is able to generate a subset of interactions of this type using the
same input conventions as for $\hat{H}_{U1}$, and $\hat{H}_{U2}$.

+++ {"id": "_MEnzhH0OK_X"}

#### Example: Extended Hubbard Model

Below, we build an extended Hubbard model including nearest-neighbor V.

If you inspect the output, you will see that behind the scenes, we use a U1 and U2 term to represent the extended Hubbard V since those terms are directly implemented in SAFIRE.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lAw6245ub5GZ
outputId: c5bfa756-3d14-4865-ddb6-2c4d9fac99e0
---
lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'))

hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0, V=1.0)
```

```{code-cell} ipython3
:id: awcSeCyfon9U


```
