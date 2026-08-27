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

# Building Lattice Model Hamiltonians Tutorial

SAFIRE can handle model Hamiltonians of the form

$$
\hat{H} = \hat{H}_t + \hat{H}_U + \hat{H}_J,
$$
where
$$
\hat{H}_t = \sum_{\mu\nu,\sigma \sigma'}t^{\sigma \sigma'}_{\mu\nu}\hat{c}^\dagger_{\mu\sigma}\hat{c}_{\nu\sigma'},
$$

$$
\hat{H}_U =  \sum_{\mu} U_i \hat{n}_{\mu\uparrow} \hat{n}_{\mu\downarrow}
+ \sum_{\mu<\nu} U_{\mu\nu}^1 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\downarrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\uparrow} )
+ \sum_{\mu<\nu} U_{\mu\nu}^2 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\uparrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\downarrow} ),
$$

and,

$$
\hat{H}_J = \sum_{\mu<\nu} J_{\mu\nu} (
     \hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\nu\uparrow}
     +\hat{c}^\dagger_{\mu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\nu\uparrow}
     +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\mu\downarrow}\hat{c}_{\nu\downarrow}\hat{c}_{\mu\uparrow}
     +\hat{c}^\dagger_{\nu\uparrow}\hat{c}^\dagger_{\nu\downarrow}\hat{c}_{\mu\downarrow}\hat{c}_{\mu\uparrow}
     ).
$$

$\mu$ / $\nu$ are compound indices, $\mu = (i,p,m)$ / $\mu = (j,q,n)$, that include the lattice site index, ($i$ / $j$), sublattice, ($p$ / $q$) and band ($m$ / $n$) indices.
This is known as the Hubbard-Kanamori Hamiltonian.
Many standard model Hamiltonians can be written in this language including the standard Hubbard model,
extended Hubbard with nearest-neighbor $V$, etc.
SAFIRE is agnostic to the details of the compound indices.
It simply accepts Hamiltonian "components" which are interpreted as above.
This provides great flexibility in specifying Hamiltonians, however care must be taken outside of SAFIRE to ensure that consistent index conventions are used.

afqmctools provides a framework for building lattice model Hamiltonians that can generate broad classes of lattice model Hamiltonians
on a variety of lattices, using consistent conventions for indexing.
The framework consists of a Lattice class which is responsible for geometry (see {doc}`../03_setting_up_a_lattice/03_setting_up_a_lattice`),
and a Hamiltonian "Builder" which is responsible for generating specific Hamiltonian terms given a specific Lattice instance.
The `HamiltonianBuilder.from_input()` classmethod accepts a set of input Hamiltonian parameters (as a Python
dict or a toml file) and automatically runs the corresponding build steps, which covers most use cases.
Users who want direct control over the Hamiltonian build steps can instead construct a `HamiltonianBuilder`
and invoke build steps directly. See {doc}`../05_hamiltonian_builder/05_hamiltonian_builder` for
more detail.

<b>In this tutorial, we focus on building Hamiltonian terms via `HamiltonianBuilder.from_input()`.</b>
We will explore the possible input parameters and their respective meanings/conventions.

+++ {"id": "pGSnNnm7Rr54"}

## ▶️ Setup : Run me

Run the cell below to setup a scratch directory.

```{code-cell} ipython3
:id: bD2Kcg2dRq0C

# setup scratch dir
from pathlib import Path
from tutorial_utils import run_afqmc

scratch_dir = Path("data")
scratch_dir.mkdir(parents=True, exist_ok=True)
```

+++ {"id": "WcOH6y7xOK_V"}

## Building Hamiltonians from Input Parameters

The `afqmctools` Python package provides a framework for generating lattice model Hamiltonians.
`HamiltonianBuilder.from_input()` is the highest-level interface to this framework.
It may be invoked as a command line tool using an input file, or a directly within a Python script.
We will briefly demonstrate the command line tool, but we will focus on using `HamiltonianBuilder.from_input()` within a Python script for most of this tutorial.

`HamiltonianBuilder.from_input()` is meant to handle building Hamiltonians which can be expressed in terms of a few simple parameters on relatively simple lattices.
While SAFIRE supports any combination of lattice, sublattice, and band degrees-of-freedom in the Hamiltonian, it is not always feasible to map such general models onto a few simple parameters.
For these cases, we provide lower-level classes to assist as much as possible. See {doc}`../03_setting_up_a_lattice/03_setting_up_a_lattice` for example.
For example, the Hubbard model only needs a single parameter, $U/t$, and a lattice / boundary conditions in order to be fully specified.
More general Hamiltonians can also be built via `from_input()`; however, it may be useful to some users to work directly with the Lattice and Hamiltonian Builder in a Python script. See {doc}`../05_hamiltonian_builder/05_hamiltonian_builder` for more advance uses.

In this tutorial, we will cover how to request specific Hamiltonian terms via `HamiltonianBuilder.from_input()`.
We will also cover the details of the input conventions for each of the terms needed to build these Hamiltonians.

+++ {"id": "DvPH8LBbOK_V"}

## Invoking the Hamiltonian Builder via the CLI

In addition to the ability to invoke the Hamiltonian builder directly within a Python script, afqmctools provides a command line tool that invokes it.
If you installed `afqmctools` using the official instructions, you can run the following to see possible options

```bash
$ make_model_ham --help

usage: make_model_ham [-h] [--input input] [--output outfile] [--free-elec] [--slater-type {closed,collinear,noncollinear}] [--plotlat] [--plot1b] [--plot2b] [--saveplots] [--pair-corr] [--hubbardU HUBBARDU]

options:
  -h, --help            show this help message and exit
  --input input, -i input
                        Input file containing Hamiltonan params (toml)
  --output outfile, -o outfile
                        Name of HDF5 file to save model Hamiltonian in
  --free-elec           Save free-electron wavefunction as well
  --slater-type {closed,collinear,noncollinear}
                        type of free-electron Slater determinant (closed,collinear,noncollinear)
  --plotlat, -pl        plot the lattice
  --plot1b              plot the 1-body Hamiltonian
  --plot2b              plot the 2-body Hamiltonian
  --saveplots, -sp      save plots instead of displaying (useful for remote connections)
  --pair-corr, -pc      Add pair-correlators to AFQMC input file (s,+x,-x,+y,-y)
  --hubbardU HUBBARDU, -U HUBBARDU
                        Hubbard U, if given and '--free-elec' is set, the hubbard U energy is evaluated using the free-electron wavefunction (Note: this is only for reference - does not effect the output wavefunction!)
```

The primary option that we are interested in here is `-i` which allows an input file (in .toml format) containing Hamiltonian parameters to be specified.
The parameter input conventions are the same whether we use the `make_model_ham` CLI tool, or use `HamiltonianBuilder.from_input()` within a Python script.

Here is a sample input file which will generate a Hubbard model with nearest-neighbor hopping, on a 4x4 square lattice with periodic boundary conditions.

```toml
[hamiltonian]
t = 1.0
U = 4.0

[lattice]
L1 = 4
L2 = 4
boundary1 = "PBC"
boundary2 = "PBC"
```

Notice that the Hamiltonian parameters and the lattice parameters are separate.
We can then build the Hamiltonian with

```bash
make_model_ham -i input.toml
```
which generates the following output,

```bash

$ make_model_ham -i input.toml

No lattice instance supplied: building from parameters
running build step nth_neighbor_hopping((1.0,))
computing and storing 1th-nearest neighbors
Computing distance matrix of lattice
Reading lattice site positions from Lattice
computing and storing 1th-nearest image neighbors
Using same twists for up and down spins
running build step onsite_hubbard((4,))
Building Hubbard U term with U=4
Building onsite hubbard with positive U values: 4
Combining terms of the same type
Combining Hubbard U, U1, and U2 terms where possible
Max spin symmetry is  2

```

Note that the builder will echo what build steps are invoked using which parameters.
For example, we can see from the line,

```bash
running build step nth_neighbor_hopping((1.0,))
```

that the builder called the nth_neighbor_hopping() build step with an argument of `1.0`.
We will see below that this generates nearest-neighbor hopping with t=1.0 (minus sign is implicit).
We also see that an onsite Hubbard interaction with $U = 4$ was also generated.

By default, `make_model_ham` saves the Hamiltonian in a file called `afqmc.h5`, but you can change this with the `-o [output name]` option.
For example,
```bash
make_model_ham -i input.toml -o example.h5
```
will save the Hamiltonian in `example.h5`.

We note that the in Python, `HamiltonianBuilder.from_input()` can be given either the name of an input file or a Python dictionary containing the Hamiltonian parameters.
It will interpret the parameters using the same conventions regardless of which input method is used.
We refer you to the [TOML specification](https://toml.io/en/v1.0.0) for clarification on the TOML format.
For the remainder of the tutorial, we will be working directly with Python within this notebook.

+++ {"id": "uincH8ATOK_X"}

### ✨ Example : Standard Hubbard Model

Below, we construct a Hubbard model on a 4x4 square lattice with both nearest, and next-nearest neighbor hopping by providing a list of values for "t".
We plot the hopping matrix to demonstrate that the Hamiltonian includes both terms.
Note that the default hopping matrix shape is $2N_{sites} \times N_{sites}$ where the first $N_{sites}$ rows correspond to the spin-up sector and the last $N_{sites}$ rows correspond to the spin-down sector.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 979
id: 6QgfLv7kOK_V
outputId: c3552c28-5791-45f0-ae57-39f29435e27d
---
from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

# Define the lattice parameters
lattice_params = {
    'L1': 4,
    'L2': 4,
    'boundary1': 'PBC',
    'boundary2': 'PBC'
}

# define Hamiltonian parameters
hamiltonian_params = {
    't': 1.0,
    'U': 4.0
}

params = {
    'hamiltonian': hamiltonian_params
}

lattice = get_lattice(lattice_params)
plot_lattice(lattice)

hamiltonian = HamiltonianBuilder.from_input(source=params, lattice=lattice).hamiltonian
```

+++ {"id": "_7Wg-_UGQuXr"}

## Writing Hamiltonians to HDF5

In the code block below, we demonstrate how to save a lattice model Hamiltonian
to an HDF5 file that can be read by SAFIRE.
This works the same way for all Hamiltonians generated by afqmctools.
We will not repeat this below.

```{code-cell} ipython3
:id: 1ugXKZ5QQ7ug

from afqmctools.utils.io import write_model_hamiltonian

write_model_hamiltonian(
    hamiltonian=hamiltonian,
    fname=scratch_dir/"hamiltonian.h5"
)
```

+++ {"id": "wgronaqPSgXY"}

Now, let's check the contents of the scratch directory to see that the Hamiltonian has been generated.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: xrykuJS4SnBR
outputId: bfe2522e-2996-43d9-88b3-ed2cd7b84350
---
! ls -l {scratch_dir}
```

+++ {"id": "HEbY-vNOOK_W"}

## Adding more terms to the Hamiltonian

### Adding Hopping

`HamiltonianBuilder.from_input()` supports nth-order neighbor hopping and band-dependent hopping.
$n$ is limited by the condition that sites are not allowed to be their own neighbors via
periodic boundary conditions.
In this case, an error will be raised.

The simplest case,
nearest-neighbor hopping with $t$ = 1, is included in the Hamiltonian by default.
For multiband Hamiltonians, the hopping is uniform across bands and it is assumed that
there is no intrasite hopping.
This is handles by a different Hamiltonian term, which we refer to as $\epsilon$.
You can change this value by providing a different value of $t$ as

```toml
[hamiltonian]
t = 0.5
```

or as a Python dictionary

```python
params = {
    'Hamiltonian' : {
        't' : 0.5
    }    
}
```

hopping can be turned off by either setting $t$ to 0, or (in Python only) setting t to `None`.

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

#### ✨ Example : Hubbard Model with $t'$

Below, we construct a Hubbard model on a 4x4 square lattice with both nearest, and next-nearest neighbor hopping by providing a list of values for "t".
We plot the hopping matrix to demonstrate that the Hamiltonian includes both terms.
Note that the default hopping matrix shape is $2N_{sites} \times N_{sites}$ where the first $N_{sites}$ rows correspond to the spin-up sector and the last $N_{sites}$ rows correspond to the spin-down sector.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: yAa96ZdbOK_X
outputId: 731d316b-7dbb-4990-f273-0ad6da75081a
---
import matplotlib.pyplot as plt

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

# define Hamiltonian parameters
params = {
    'lattice': {
        'L1': 4,
        'L2': 4,
        'boundary1': 'PBC',
        'boundary2': 'PBC'
    },
    'hamiltonian': {
        'U': 4.0,
        't' : [1.0,0.5] # implicit minus sign!
    }
}

# this will make a Hubbard model on a 4x4 lattice with PBCs
hamiltonian = HamiltonianBuilder.from_input(source=params).hamiltonian

hopping = hamiltonian.get_one_body()
plt.matshow(hopping.toarray())
plt.colorbar()
plt.show()
```

+++ {"id": "09Ir9dWeOK_W"}

### Adding onsite Hubbard U

SAFIRE supports arbitrary onsite interactions of the from

$$
\hat{H}_U =  \sum_{\mu} U_{\mu} \hat{n}_{\mu\uparrow} \hat{n}_{\mu\downarrow},
$$

where, again, the combined index $\mu$ may include both lattice site and band indices.
We've already seen how to add a basic nearest-neighbor onsite Hubbard U, now we see see how to provide a site-, and/or band-dependent onsite interaction.
<b>This input parameter, $U$, is in general an array. The band/site dependence of the interaction is deduced based on the shape of the $U$ array</b>.

#### Band- and site-dependence deduction rules

1. if U is a scalar (or an array with a single entry), then the onsite Hubbard interaction term is included with strength U and applied to all sites and bands uniformly.

2. if U is 1-dimensional with length nbands (i.e. [U1,U2,…,Um] where for an m-band model), then the onsite Hubbard interaction term is included with strength U_i applied to band i and uniformly across sites.

3. if U is 1-dimensional with length nsites (i.e. [U1,U2,…,Un] where n is the number of sites), then the onsite Hubbard interaction term is included with strength U_i applied to site i and uniformly across bands.

4. If nbands and nsites are equal, then nbands take priority (i.e. rule 2 applies and not rule 3).

+++ {"id": "BskmJL_Kn9Fm"}

#### ✨ Example: site-dependent U

In the example below, we build a site-dependent U with a checkerboard pattern.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 909
id: i0l8VxorOK_W
outputId: 4113cab3-7b9d-4366-edb5-ff8b22bd89fd
---
from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

import numpy as np

lattice_params = {
    'L1': 2,
    'L2': 4,
    'boundary1': 'PBC',
    'boundary2': 'PBC'
}

lattice = get_lattice(lattice_params)


# define Hamiltonian parameters
hamiltonian_params = {
    'nbands' : 1,
    'U': [1.0,2.0,1.0,2.0,2.0,1.0,2.0,1.0]
}

params = {
    'hamiltonian': hamiltonian_params
}

# this will make a Hubbard model on a 4x4 lattice with PBCs
hamiltonian = HamiltonianBuilder.from_input(source=params, lattice=lattice).hamiltonian

# let's take a look at the U term:
U = hamiltonian.get_U().toarray().diagonal().reshape(lattice.L)

plot_lattice(
    lattice,
    density=U,
    vmin=0.0,
    vmax=3.0
)
```

+++ {"id": "r53E1XbMoJM8"}

#### ✨ Example: band-dependent U

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
from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

import numpy as np

lattice_params = {
    'L1': 2,
    'L2': 4,
    'boundary1': 'PBC',
    'boundary2': 'PBC'
}
lattice = get_lattice(lattice_params)


# define Hamiltonian parameters
hamiltonian_params = {
    'nbands' : 2,
    't' : [np.array([[1.0, -0.2], [-0.2, 1.0]])], # bonus: band-dependent hopping
    'U': [1.0,2.0]
}

params = {
    'hamiltonian': hamiltonian_params
}

# this will make a Hubbard model on a 4x4 lattice with PBCs
hamiltonian = HamiltonianBuilder.from_input(source=params, lattice=lattice).hamiltonian

# let's take a look at the U term:
U_shape = (lattice.L[0],lattice.L[1],hamiltonian_params['nbands'])
U = hamiltonian.get_U().toarray().diagonal().reshape(U_shape)
plot_lattice(
    lattice,
    density=U[:,:,0],
    title='U for band 0',
    vmin=0.0,
    vmax=3.0
)
plot_lattice(
    lattice,
    density=U[:,:,1],
    title='U for band 1',
    vmin=0.0,
    vmax=3.0
)
```

```{code-cell} ipython3
:id: _6grG3FOOK_X

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

import numpy as np

lattice_params = {
    'L1': 2,
    'L2': 4,
    'boundary1': 'PBC',
    'boundary2': 'PBC'
}
lattice = get_lattice(lattice_params)


# define Hamiltonian parameters
hamiltonian_params = {
    'nbands' : 2,
    't' : [np.array([[1.0, -0.2], [-0.2, 1.0]])], # bonus: band-dependent hopping
    'U': [1.0,2.0]
}

params = {
    'hamiltonian': hamiltonian_params
}

# this will make a Hubbard model on a 4x4 lattice with PBCs
hamiltonian = HamiltonianBuilder.from_input(source=params, lattice=lattice).hamiltonian

# let's take a look at the U term:
U_shape = (lattice.L[0],lattice.L[1],hamiltonian_params['nbands'])
U = hamiltonian.get_U().toarray().diagonal().reshape(U_shape)
plot_lattice(
    lattice,
    density=U[:,:,0],
    title='U for band 0',
    vmin=0.0,
    vmax=3.0
)
plot_lattice(
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

### Adding Density-Density interaction

SAFIRE supports general interactions of the form,

$$
\hat{H}_{U1} =  \sum_{\mu<\nu} U_{\mu\nu}^1 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\downarrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\uparrow} ),
$$

where $\mu = (i,m)$ / $\nu = (i,p)$ are a composite lattice, $i$ / $j$, and band $m$, $p$ indices.

afqmctools is able to generate a subset of interactions of this type;
The $U_{\mu\nu}^1$ matrix is constructed based on what information
the user provides.
specifically, it handles the following cases.

#### Case 1: Uniform intrasite interaction

The user provides a single value for $U^1$ and keeps the default $n=0$.

In this case, the density-density interaction takes the following form.

$$
\hat{H}_{U1} = U^1 \sum_{i} \sum_{mp} (\hat{n}_{(i,m)\uparrow} \hat{n}_{(i,p)\downarrow} + \hat{n}_{(i,m)\downarrow} \hat{n}_{(i,p)\uparrow} ),
$$

where we have replaced the composite indices with site and band indices for clarity.

#### Case 2: Band-dependent intrasite interaction

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

#### Case 3: Uniform nth-order neighbor interaction

#### Case 4: Band-dependent nth-order neighbor interaction

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

#### Case 5: a list of inputs from the cases above

If a user provides a list of valid inputs, then afqmctools will generate each
term provided <b>using the list index / position for the neighbor order "$n$."</b>
For example, if a user provides

```toml
[hamiltonian]
nbands = 2
U1= [ 2.0,
      [[ 0.5, 1.0 ],
       [ 1.0, 0.5 ]]
    ]
```

Both a uniform intrasite interaction with U1 = 2.0 (n=0 / onsite),
and a band-dependent nearest-neighbor (n=1) interaction will be added to the
Hamiltonian.

+++ {"id": "_BXfk70iudQJ"}

### Adding Spin-Spin interaction

SAFIRE supports general interactions of the form,

$$
\hat{H}_{U2} = \sum_{\mu<\nu} U_{\mu\nu}^2 (\hat{n}_{\mu\uparrow} \hat{n}_{\nu\uparrow} + \hat{n}_{\mu\downarrow} \hat{n}_{\nu\downarrow} ),
$$

where $\mu = (i,m)$ / $\nu = (i,p)$ are a composite lattice, $i$ / $j$, and band $m$, $p$ indices.

afqmctools is able to generate a subset of interactions of this type;
The $U_{\mu\nu}^2$ matrix is constructed based on what information
the user provides similarly to the density-density (i.e. the $U^1$) term.
See the density-density section above for more details.

+++ {"id": "L-D4VJuBOK_X"}

### Adding a Hund's J term

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

afqmctools is able to generate a subset of interactions of this type;
The $J_{\mu\nu}$ matrix is constructed based on what information
the user provides similarly to the density-density (i.e. the $U^1$ term).
See the density-density section above for more details.

+++ {"id": "CIbAdbiaOK_X"}

#### ✨ Example: 3-band Hubbard-Kanamori Hamiltonian

Below we construct a 3-band Hubbard-Kanamori model using uniform values for U,U1,U2, and J as in Table I of [PRB 99, 235142 (2019)](https://doi.org/10.1103/PhysRevB.99.235142)

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
  height: 1000
id: D72pRPc6OK_X
outputId: b6f1c99c-e33b-4574-d108-3a2ea7f66d89
---
from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

# Define the lattice parameters
lattice_params = {
    'L1': 6,
    'L2': 1,
    'boundary1': 'PBC',
    'boundary2': 'open'
}

# define Hamiltonian parameters
hamiltonian_params = {
    'nbands' : 3,
    't' : 1.0,
    'U': 4.0,
    'U1': 1.5,
    'U2': 1.0,
    'J': 0.5
}

params = {
    'hamiltonian': hamiltonian_params
}

lattice = get_lattice(lattice_params)
plot_lattice(lattice)

hamiltonian = HamiltonianBuilder.from_input(source=params, lattice=lattice).hamiltonian
```

+++ {"id": "W-TfLNDJOK_W"}

### Adding an extended Hubbard V interaction

The $n^{th}$-order neighbor extended Hubbard interaction,

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

for $n^{th}$-order neighbors.e

afqmctools is able to generate a subset of interactions of this type using the
same input conventions as for $\hat{H}_{U1}$, and $\hat{H}_{U2}$.

+++ {"id": "_MEnzhH0OK_X"}

#### ✨ Example: Extended Hubbard Model

Below, we build an extended Hubbard model including nearest-neighbor V.

If you inspect the output, you will see that behind the scenes, we use a U1 and U2 term to represent the extended Hubbard V since those terms are directly implemented in SAFIRE.

```{code-cell} ipython3
---
colab:
  base_uri: https://localhost:8080/
id: lAw6245ub5GZ
outputId: c5bfa756-3d14-4865-ddb6-2c4d9fac99e0
---
import matplotlib.pyplot as plt

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.utils.visualize import plot_lattice

# define Hamiltonian parameters
params = {
    'lattice': {
        'L1': 4,
        'L2': 4,
        'boundary1': 'PBC',
        'boundary2': 'PBC'
    },
    'hamiltonian': {
        't' : 1.0,
        'U': 4.0,
        'V' : 1.0
    }
}

hamiltonian = HamiltonianBuilder.from_input(source=params).hamiltonian
```

```{code-cell} ipython3
:id: awcSeCyfon9U


```
