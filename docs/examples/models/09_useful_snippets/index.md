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

# Useful Snippets for Various Lattice Models

This example collects short recipes for setting up common lattice model Hamiltonians and their free-electron trial wavefunctions.

Each section builds a Hamiltonian and a free-electron trial wavefunction.
Each snippet illustrates one aspect of the setup, but they can be combined: for example, a pinning field can be added to the three-band model in the same way as to the single-band one.
The lattices are explained in {doc}`/tutorials/models/03_setting_up_a_lattice/03_setting_up_a_lattice`, and the Hamiltonian terms in {doc}`/tutorials/models/04_building_and_writing_a_hamiltonian/04_building_and_writing_a_hamiltonian`.

## Setup

```{code-cell} ipython3
import numpy as np

from safiretools import HamiltonianBuilder, Lattice, LatticeHamiltonian, PinningField, SpinSymm, Wavefunction, vis

default_twist = (1/np.sqrt(592560607), 1/np.sqrt(47603))
```

## Rectangular Hubbard Model

A Hubbard model on a 2 × 3 square lattice with periodic boundary conditions.

```{code-cell} ipython3
params = dict(t=1.0, U=2.0)

lattice = Lattice.from_type(
    "square", (2, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

# The trial wavefunction needs its own Hamiltonian, built on a lattice
#   carrying a small irrational twist.
#   The AFQMC run itself uses the untwisted Hamiltonian above.
fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(2, 2),
)
```

`Wavefunction.from_free_electron()` uses the Hamiltonian exactly as given and applies no twist of its own, so the twist belongs on the lattice you build that Hamiltonian from.
For an open-shell filling, an untwisted lattice leads to an arbitrary choice of which degenerate orbitals are occupied, and `from_free_electron()` warns when that happens.
The following snippets use the same pattern.

## Three-Band Hubbard Model

`nbands` sets the number of orbitals per lattice site.

```{code-cell} ipython3
params = dict(nbands=3, U=2.0)

lattice = Lattice.from_type(
    "square", (2, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(3, 3),
)
```

## Hubbard-Kanamori Model

A two-band Hubbard-Kanamori model on a 6 × 1 square lattice.
With periodic boundary conditions only along the long direction, the system is effectively one-dimensional.

```{code-cell} ipython3
params = dict(nbands=2, U=2.0, U1=1.5, U2=1.0, J=0.5)

lattice = Lattice.from_type(
    "square", (6, 1), boundaries=("periodic", "open"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(6, 6),
)
```

## Hubbard Model with nth-Order Hopping

A list `t` of length $n$ gives hopping up to the $n$th-order neighbors.
Here, `t=[1.0, 0.5]` adds a next-nearest-neighbor hopping $t' = 0.5$.

```{code-cell} ipython3
params = dict(t=[1.0, 0.5], U=2.0)

lattice = Lattice.from_type(
    "square", (2, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(2, 2),
)
```

## Pinning Fields

A $t$-$t'$ Hubbard model on an 8 × 4 square lattice with an antiferromagnetic pinning field on the two edges at the ends of axis 0.
The pinning field is a one-body term, so the trial wavefunction Hamiltonian has to carry it too.

```{code-cell} ipython3
params = dict(t=[1.0, 0.5], U=2.0, pinning=(PinningField.afm(0.5, staggered=True), None))

lattice = Lattice.from_type(
    "square", (8, 4), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(16, 16),
)
```

A charge pinning field $\frac{h}{2} \hat{n}_i$, constant along the edge, is set up the same way.

```{code-cell} ipython3
params = dict(t=[1.0, 0.5], U=2.0, pinning=(PinningField.charge(0.5, staggered=False), None))

lattice = Lattice.from_type(
    "square", (8, 4), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(16, 16),
)
```

## Twist Angles

`twists` holds the twist angles $\vec{\theta}=(\theta_1,\theta_2)$ in radians, with both components given even if one of them is 0.
Here, the lattice we want to simulate at the AFQMC level is already twisted, so one Hamiltonian serves both the trial wavefunction and the AFQMC run.

```{code-cell} ipython3
lattice = Lattice.from_type(
    "square", (4, 8), boundaries=("periodic", "periodic"), twists=(0.0, 7 * np.pi / 8)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, U=4.0)

wavefunction = Wavefunction.from_free_electron(
    source=hamiltonian,
    nelec=(24, 24),
)
```

## Overriding the Hubbard-Stratonovich Transformation Type

<div class="alert alert-block alert-warning">
<b>Warning:</b>

This functionality is for experts only. Use at your own risk.

</div>

By default, the Hamiltonian builder chooses a Hubbard-Stratonovich transformation (HST) type from the sign of each interaction.
For band-dependent interactions, it splits each interaction into two separate terms if there are both positive and negative interaction strengths and assigns an HST type to each.
`hst_types` forces a specific HST type for the interaction terms it names, regardless of the sign of the interaction strength.
The other terms keep their default.
The choices are `"discrete_spin"`, `"discrete_charge"`, `"continuous_spin"` and `"continuous_charge"`.

```{code-cell} ipython3
params = dict(
    nbands=2,
    U=4.0,      # defaults to "discrete_spin" for U > 0
    U1=-1.5,    # defaults to "discrete_charge" for U1 < 0
    U2=1.0,
    J=0.5,
    hst_types={"U1": "discrete_spin"},
)

lattice = Lattice.from_type(
    "square", (3, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(2, 2),
)
```

## Triangular Lattice

The first argument of `Lattice.from_type()` selects the unit cell: `"square"`, `"triangular"`, `"honeycomb"` or `"kagome"`.
Below, a Hubbard model on a 3 × 3 triangular lattice.

```{code-cell} ipython3
params = dict(t=1.0, U=2.0)

lattice = Lattice.from_type(
    "triangular", (3, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)
vis.plot_lattice(lattice)

hamiltonian = LatticeHamiltonian.from_lattice(lattice, **params)

fe_lattice = Lattice(**dict(hamiltonian.lattice_params, twists=default_twist))
fe_hamiltonian = LatticeHamiltonian.from_lattice(fe_lattice, **params)

wavefunction = Wavefunction.from_free_electron(
    source=fe_hamiltonian,
    nelec=(2, 2),
)
```

## Rashba Spin-Orbit Coupling

Rashba spin-orbit coupling mixes the spin sectors, so the Hamiltonian is built in noncollinear form.
Here, it is added step by step with a `HamiltonianBuilder`.

```{code-cell} ipython3
lattice = Lattice.from_type(
    "square", (3, 3), boundaries=("periodic", "periodic"), twists=(0.0, 0.0)
)

# hopping[0] is t, hopping[1] is t', and so on
hopping = [1.0, 0.5]

builder = HamiltonianBuilder(
    lattice=lattice,
    spin_symm=SpinSymm.NONCOLLINEAR
)
builder.nth_neighbor_hopping(t=hopping)
builder.onsite_hubbard(U=8.0)

# add Rashba SOC consistent with the hopping
builder.rashba_soc(rashba_lambda=0.3, t=hopping)
builder.finalize()

hamiltonian = builder.get_hamiltonian()

wavefunction = Wavefunction.from_free_electron(
    source=hamiltonian,
    nelec=(5, 5)
)
```
