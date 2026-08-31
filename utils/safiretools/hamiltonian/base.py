# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
The `Hamiltonian` abstract base class and the on-disk format detection that
`Hamiltonian.from_hdf5` dispatches on.

Hamiltonians split by *source domain* — `LatticeHamiltonian`,
`MolecularHamiltonian`, `PeriodicHamiltonian` — because those genuinely have
different storage formats, so each subclass implements its own
``to_hdf5``/``from_hdf5``. Spin symmetry is a plain attribute, not a subclass
axis.
"""

from abc import ABC, abstractmethod
from importlib import import_module

import h5py as h5

from safiretools.types import SpinSymm


def hamiltonian_format(path) -> str:
    """
    Identify the Hamiltonian format stored in the HDF5 file at `path`.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to inspect.

    Returns
    -------
    str
        One of ``'model'``, ``'dense'``, ``'kpoint'``, ``'thc'``, or
        ``'kpoint_coqui'``.

    Raises
    ------
    ValueError
        If the file matches none of the known formats.
    """
    with h5.File(path, 'r') as fh5:
        if 'Hamiltonian/ModelHamiltonian/number_of_components' in fh5:
            return 'model'
        if 'Hamiltonian/DenseFactorized/L' in fh5:
            return 'dense'
        if 'Hamiltonian/KPFactorized/L0' in fh5:
            return 'kpoint'
        if 'Hamiltonian/THC/Luv' in fh5:
            return 'thc'
        if 'Interaction/Vq0' in fh5:
            return 'kpoint_coqui'

    raise ValueError(f"'{path}' holds no Hamiltonian in a format safiretools recognizes")


_READERS = {
    'model': ('safiretools.hamiltonian.model.lattice_hamiltonian', 'LatticeHamiltonian'),
    'dense': ('safiretools.hamiltonian.molecular', 'MolecularHamiltonian'),
    'kpoint': ('safiretools.hamiltonian.periodic', 'PeriodicHamiltonian'),
}
"""Format name from `hamiltonian_format` -> the module and class that reads it."""


def _class_for_format(fmt: str):
    """
    The `Hamiltonian` subclass that reads format `fmt`.

    Resolved by import at call time, both so this module stays free of upward
    imports and so reading one format never imports another's dependencies.
    """
    if fmt not in _READERS:
        raise NotImplementedError(
            f"No safiretools reader for Hamiltonian format '{fmt}'"
        )

    module_name, class_name = _READERS[fmt]
    return getattr(import_module(module_name), class_name)


class Hamiltonian(ABC):
    """
    Base class for all SAFIRE Hamiltonians.

    Parameters
    ----------
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry of the Hamiltonian. Coerced through
        `SpinSymm.from_input`, so ``'uhf'``, ``'collinear'`` and
        ``SpinSymm.COLLINEAR`` are all accepted.

    Notes
    -----
    ``__init__`` takes already-formed, validated in-memory data. Build a
    Hamiltonian from an external description with one of the classmethod
    factories on the concrete subclasses (``from_dict``, ``from_hdf5``,
    ...).
    """

    def __init__(self, spin_symm=SpinSymm.CLOSED) -> None:
        self.spin_symm = spin_symm

    @property
    def spin_symm(self):
        """Spin symmetry of this Hamiltonian, as a `SpinSymm` (or None)."""
        return self._spin_symm

    @spin_symm.setter
    def spin_symm(self, value):
        self._spin_symm = None if value is None else SpinSymm.from_input(value)

    @abstractmethod
    def to_hdf5(self, path) -> None:
        """
        Write this Hamiltonian to the HDF5 file `path`, in the format the AFQMC
        executable reads for this Hamiltonian's domain.
        """

    @classmethod
    def from_hdf5(cls, path) -> "Hamiltonian":
        """
        Read a Hamiltonian from the HDF5 file `path`.

        Called on `Hamiltonian` itself, the concrete subclass is chosen from the
        format stored in the file. Called on a concrete subclass, the file must
        hold a format that subclass reads.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to read.

        Returns
        -------
        Hamiltonian
            An instance of the concrete subclass for the stored format.

        Raises
        ------
        ValueError
            If the file holds no recognized Hamiltonian, or holds one that the
            subclass this was called on does not read.
        """
        fmt = hamiltonian_format(path)
        target = _class_for_format(fmt)

        if cls is not Hamiltonian and not issubclass(target, cls):
            raise ValueError(
                f"'{path}' holds a '{fmt}' Hamiltonian, which "
                f"{cls.__name__} does not read; use {target.__name__}.from_hdf5 "
                "or the dispatching Hamiltonian.from_hdf5"
            )

        return target._read_hdf5(path, fmt)

    @classmethod
    @abstractmethod
    def _read_hdf5(cls, path, fmt: str) -> "Hamiltonian":
        """
        Read `path`, which `from_hdf5` has already identified as holding format
        `fmt`. Subclasses implement this rather than overriding `from_hdf5`, so
        that dispatch stays in one place.
        """
