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

In {doc}`../03_setting_up_a_lattice/03_setting_up_a_lattice`, we learned how to create a {py:class}`safiretools.Lattice` object. The `Lattice` object only describes the geometry of a lattice. To build a lattice model Hamiltonian from a lattice, we need to supply additional information about the physical couplings between electrons on the lattice sites. In this tutorial, we will see how {py:meth}`safiretools.LatticeHamiltonian.from_lattice` can be used to do this and create lattice model Hamiltonian inputs for SAFIRE.


SAFIRE can handle general Hubbard-Kanamori Hamiltonians; their form is given in {ref}`lattice_model_hamiltonian`. Below, we go through the terms `from_lattice` can build, first for a single band per site. Their multiband extensions are covered at the end, in the Multiband Models section.

## Example: Standard Hubbard Model

Below, we construct a Hubbard model on a 4×4 square lattice.

$$
\hat{H} = - t \sum_{\langle i,j \rangle,\sigma} \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma} + \text{h. c.} + U \sum_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow}
$$

```{code-cell} ipython3
# create scratch directory in notebook path
%mkdir -p {_dh[0]}/data
%cd -q {_dh[0]}/data
```

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 979
id: 6QgfLv7kOK_V
outputId: c3552c28-5791-45f0-ae57-39f29435e27d
---
from safiretools import Lattice, LatticeHamiltonian
from safiretools import vis
import matplotlib.pyplot as plt
import numpy as np

lattice = Lattice.from_type('square', (4, 4), boundaries=('periodic', 'periodic'))

hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0)

# plot the hopping matrix
# hamiltonian.get_one_body() returns a (2*Nsites, Nsites) spin-stacked sparse matrix.
plt.matshow(hamiltonian.get_one_body().todense())
plt.colorbar()
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

Now, let's check the contents of the `data` directory to see that the Hamiltonian has been generated.

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

### $n^\text{th}$-Nearest Neighbor Hopping

Nearest-neighbor hopping with $t = 1$ is included by default; `t=None` turns it off.
{py:meth}`safiretools.LatticeHamiltonian.from_lattice` supports higher-order hoppings by passing a list for `t`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: yAa96ZdbOK_X
outputId: 731d316b-7dbb-4990-f273-0ad6da75081a
---
# The 4×4 t-t' Hubbard model
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    U=4.0,
    t=[1.0, 0.2] # implicit minus sign!
)

plt.matshow(hamiltonian.get_one_body().todense())
plt.colorbar()
plt.show()
```

+++

<div class="alert alert-block alert-warning">
<b>Caution:</b>

$n^\mathrm{th}$-order neighbors are determined by spatial distance. This is especially important to keep in mind when attempting to use a lattice with a basis.

</div>

### On-Site Energy

An on-site energy,

$$
\hat{H}_\epsilon = \epsilon \sum_{i\sigma} \hat{n}_{i\sigma},
$$

is added with the `epsilon` argument. For example, a chemical potential $\mu$ corresponds to `epsilon=-μ`.

```{code-cell} ipython3
hamiltonian = LatticeHamiltonian.from_lattice(lattice, t=1.0, U=4.0, epsilon=-2.0)

# the on-site energy sits on the diagonal of the one-body matrix
plt.matshow(hamiltonian.get_one_body().todense())
plt.colorbar()
plt.show()
```

+++ {"id": "09Ir9dWeOK_W"}

### On-Site Hubbard U

A Hubbard $U$ interaction,

$$
\hat{H}_U =  \sum_{i} U_{i} \hat{n}_{i\uparrow} \hat{n}_{i\downarrow},
$$

is added using the `U` argument, as we have seen previously. By passing a list with one entry per site, we can make $U_i$ site dependent.

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

+++ {"id": "pqLN_brHOK_X"}

### Density-Density Interaction

SAFIRE also supports density-density interactions between opposite spins on different sites,

$$
\hat{H}_{U^{(1)}} = \sum_{n} U^{(1)}_n \sum_{\langle i,j \rangle^n} (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} ),
$$

where $\langle i,j \rangle^n$ runs over each pair of $n^\text{th}$-order neighbors once, the $i<j$ of {ref}`lattice_model_hamiltonian`.
The `U1` argument takes a list $[U^{(1)}_0, U^{(1)}_1, \dots]$ whose index is the neighbor order $n$, *starting at $n = 0$*, the on-site term.
With a single band, the on-site term is the Hubbard $U$ and `from_lattice` rejects a nonzero $U^{(1)}_0$, so a nearest-neighbor interaction is `U1=[0.0, 0.5]`. Zero entries are skipped.

+++ {"id": "_BXfk70iudQJ"}

### Spin-Spin Interaction

The spin-spin interaction between equal spins,

$$
\hat{H}_{U^{(2)}} = \sum_{n} U^{(2)}_n \sum_{\langle i,j \rangle^n} (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} ),
$$

is set by the `U2` argument, with the same input conventions as `U1`.

+++ {"id": "L-D4VJuBOK_X"}

### Hund's J

The Hund's coupling,

$$
\hat{H}_J = \sum_{n} J_n \sum_{\langle i,j \rangle^n} (
     \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
     +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow}
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow}
     ),
$$

is set by the `J` argument, again with the same input conventions as `U1`.
It is mostly used between the bands of one site, see Multiband Models below.

+++ {"id": "W-TfLNDJOK_W"}

### Extended Hubbard V

The extended Hubbard interaction,

$$
\hat{H}_V = \sum_{n} V_n \sum_{\langle i,j \rangle^n} \hat{n}_i \hat{n}_j, \qquad \hat{n}_i = \hat{n}_{i\uparrow} + \hat{n}_{i\downarrow},
$$

is the sum of a density-density and a spin-spin interaction of equal strength,

$$
\hat{H}_V = \hat{H}_{U^{(1)}}[U^{(1)} = V] + \hat{H}_{U^{(2)}}[U^{(2)} = V].
$$

The `V` argument takes a scalar for nearest neighbors or a list $[V_1, V_2, \dots]$ whose index starts at $n = 1$.

+++ {"id": "_MEnzhH0OK_X"}

### Example: Extended Hubbard Model

Below, we build an extended Hubbard model including nearest-neighbor V.

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

### Rashba Spin-Orbit Coupling

Rashba spin-orbit coupling,

$$
\hat{H}_\mathrm{R} = i \lambda_\mathrm{R} \sum_{n} t_n \sum_{(i,j)^n} \sum_{\sigma\sigma'} (\vec{\sigma} \times \vec{r}_{ij})_z^{\sigma\sigma'} \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma'},
$$

where $\vec{\sigma}$ is the vector of Pauli matrices, $\vec{r}_{ij}$ is the displacement from site $i$ to site $j$, $t_n$ is the $n^\text{th}$-order hopping, and $(i,j)^n$ runs over both orderings of each pair of $n^\text{th}$-order neighbors,
is added with the `rashba_lambda` argument. Currently Rashba Spin-Orbit coupling is only tested for 1-band models.
Because the Rashba term flips spins, the one-body matrix becomes a $(2 N_\text{sites}, 2 N_\text{sites})$ spinor matrix whose off-diagonal blocks couple up and down spins.

```{code-cell} ipython3
hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t=1.0,
    U=4.0,
    rashba_lambda=0.3
)

# the Rashba term is imaginary and sits in the spin-flip blocks
plt.matshow(hamiltonian.get_one_body().toarray().imag)
plt.colorbar()
plt.show()
```

### Pinning Fields

Pinning fields are on-site potentials on the edges of a lattice, usually open ones, that break a symmetry so that the order it induces can be measured.
They are added with the `pinning` argument, a pair with one {py:class}`safiretools.PinningField` or `None` per lattice axis, acting on the two edges at the ends of that axis.
`PinningField.afm`, `PinningField.fm` and `PinningField.charge` give antiferromagnetic, ferromagnetic and charge fields $\pm h/2$; with `staggered=True`, the two edges have opposite signs.
Other patterns can be given as a function `(n, edge) -> (h_up, h_down)` passed to `PinningField`.

Below, we pin the two open edges at the ends of axis 0 antiferromagnetically, as in {doc}`/examples/models/02_stripes/index`.

```{code-cell} ipython3
from safiretools import PinningField

open_lattice = Lattice.from_type('square', (6, 4), boundaries=('open', 'periodic'))

hamiltonian = LatticeHamiltonian.from_lattice(
    open_lattice,
    t=1.0,
    U=4.0,
    pinning=(PinningField.afm(0.5, staggered=True), None)
)

# the spin-down potential is the negative of the spin-up one
vis.plot_lattice(
    open_lattice,
    density=hamiltonian.get_one_body().diagonal(),
    density_label='spin-up potential'
)
```

### Custom One-Body Terms

A one-body term that none of the arguments above cover can be passed as a matrix with `custom_one_body`. It is added to the one-body Hamiltonian as is, without an implicit sign, and must be Hermitian.
Its rows and columns are the site indices $i$ shown by `vis.plot_lattice`, and it may have one of three shapes:

- $(N_\text{sites}, N_\text{sites})$: the same matrix for both spins,
- $(2 N_\text{sites}, N_\text{sites})$: the spin-up matrix stacked on the spin-down matrix, e.g. `sps.vstack([h, -h])` for a Zeeman field $h$,
- $(2 N_\text{sites}, 2 N_\text{sites})$: a full spinor matrix.

Below, we add a staggered potential,

$$
\hat{H}_\Delta = \Delta \sum_{i\sigma} (-1)^{x_i + y_i} \hat{n}_{i\sigma},
$$

where $(x_i, y_i)$ is the position of site $i$. With $\Delta = 0.5$, this is the ionic Hubbard model.

```{code-cell} ipython3
import scipy.sparse as sps

x, y = lattice.get_positions().T
stagger = (-1.0) ** np.round(x + y)

hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    t=1.0,
    U=4.0,
    custom_one_body=sps.diags_array(0.5 * stagger)
)

# the diagonal of the spin-up block is the on-site potential
vis.plot_lattice(
    lattice,
    density=hamiltonian.get_one_body().diagonal(),
    density_label='on-site potential'
)
```

## Multiband Models

By passing `nbands` to {py:meth}`safiretools.LatticeHamiltonian.from_lattice`, every lattice site carries $n_\text{bands}$ orbitals, and the operators $\hat{c}_{im\sigma}$ get a band index $m$ (or $p$) next to the site index $i$.
In the matrices returned by `get_one_body()` and `get_U()`, the band is the fastest index, `site * nbands + band`.
Each argument above extends to bands as follows.

### One-Body Terms

A scalar `t` corresponds to $t_{mp} = t \delta_{mp}$, i.e. uniform band-diagonal hopping. An $n_\text{bands} \times n_\text{bands}$ matrix $t_{mp}$ also hops between bands on neighboring sites,

$$
\hat{H}_t = -\sum_{\langle i,j \rangle} \sum_{mp\sigma} t_{mp}\, \hat{c}^\dagger_{im\sigma} \hat{c}_{jp\sigma} + \text{h. c.},
$$

and a list of such matrices gives higher-order neighbors as before. Hopping never couples bands on the same site; that is the job of `epsilon`.
A scalar `epsilon` is the same on-site energy for every band. A Hermitian $n_\text{bands} \times n_\text{bands}$ matrix $\epsilon_{mp}$ holds band energies on its diagonal and on-site hybridization off it,

$$
\hat{H}_\epsilon = \sum_{i} \sum_{mp\sigma} \epsilon_{mp}\, \hat{c}^\dagger_{im\sigma} \hat{c}_{ip\sigma}.
$$

### On-Site Hubbard U

A list of length $n_\text{bands}$ gives a band-dependent $U_m$ that is uniform across sites,

$$
\hat{H}_U = \sum_{i} \sum_{m} U_m \hat{n}_{im\uparrow} \hat{n}_{im\downarrow}.
$$

If $n_\text{bands} = n_\text{sites}$, a list is read as band-dependent.

### Interband Interactions

`U1`, `U2` and `J` share one convention. For each neighbor order $n$, starting at $n = 0$, the amplitude is either a scalar or an $n_\text{bands} \times n_\text{bands}$ matrix. For `U1`,

$$
\hat{H}_{U^{(1)}} = \sum_{n} \sum_{\langle i,j \rangle^n} \sum_{mp} U^{(1)}_{n,mp} (\hat{n}_{im\uparrow} \hat{n}_{jp\downarrow} + \hat{n}_{im\downarrow} \hat{n}_{jp\uparrow} ),
$$

where $\langle i,j \rangle^0$ means $i = j$, and `U2` and `J` follow the same pattern. With several bands, the on-site term $n = 0$ is allowed, and a scalar is accepted directly, so `U1=1.5` is a uniform on-site interband interaction.

- On-site ($n = 0$), each pair of bands interacts once, so only $m < p$ enters; a scalar fills all pairs $m < p$. A matrix should only set its upper triangle: SAFIRE ignores the lower triangle, and the diagonal is ignored for `U2` and `J` but added to the Hubbard $U$ for `U1`. The interaction within one band is the Hubbard $U$.
- Between sites ($n \geq 1$), band $m$ on site $i$ and band $p$ on site $j$ is a different interaction from $p$ on $i$ and $m$ on $j$, so every $(m, p)$ enters; a scalar fills the whole matrix.

A list must be all scalars or all matrices. For example, an on-site interaction between two bands plus a band-dependent nearest-neighbor interaction is

```python
U1=[[[0.0, 2.0],
     [0.0, 0.0]],
    [[0.5, 1.0],
     [1.0, 0.5]]]
```

### Extended Hubbard V

`V` takes a scalar or an $n_\text{bands} \times n_\text{bands}$ matrix $V_{mp}$ per neighbor order, starting at $n = 1$, like the intersite part of `U1`.

+++ {"id": "r53E1XbMoJM8"}

### Example: Band-Dependent U

Now we will construct a band-dependent U which is *uniform* across lattice sites.
We will use two bands and the first band will have $U/t = 1$
and the second will have $U/t = 2$.
We also include a band-dependent hopping and raise the energy of the second band with `epsilon`.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: jN42bp5cOK_W
outputId: d257ba45-6561-41df-a581-8a19e1f73a2a
---
lattice = Lattice.from_type('square', (2, 4), boundaries=('periodic', 'periodic'))

hamiltonian = LatticeHamiltonian.from_lattice(
    lattice,
    nbands=2,
    t=[np.array([[1.0, -0.2], [-0.2, 1.0]])],
    epsilon=np.diag([0.0, 0.5]),
    U=[1.0, 2.0]
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

+++ {"id": "CIbAdbiaOK_X"}

### Example: 2-Band Hubbard-Kanamori Hamiltonian

Below we construct the 2-band, 6×1 Hubbard-Kanamori model with uniform values for `U`, `U1`, `U2` and `J` from the first row of Table I in [PRB 99, 235142 (2019)](https://doi.org/10.1103/PhysRevB.99.235142).

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
    nbands=2,
    t=1.0,
    U=2.0,
    U1=1.5,
    U2=1.0,
    J=0.5
)
```

## Summary

In this tutorial, you learned how to build lattice model Hamiltonians with {py:meth}`safiretools.LatticeHamiltonian.from_lattice` and write them to the SAFIRE HDF5 format.

### What You Learned

1. How to build a Hubbard model on a lattice and write it to an HDF5 file that SAFIRE can read
2. How to add higher-order hopping, on-site energies, density-density, spin-spin, Hund's and extended Hubbard interactions, Rashba spin-orbit coupling, pinning fields and custom one-body terms
3. How these terms extend to multiband models
