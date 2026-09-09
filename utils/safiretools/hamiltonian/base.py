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
from contextlib import contextmanager
from importlib import import_module

import h5py as h5

from safiretools.types import HamiltonianFormat, SpinSymm

HAMILTONIAN_GROUP = 'Hamiltonian'
"""Top-level HDF5 group every Hamiltonian format writes into."""

TYPE_DATASET = f'{HAMILTONIAN_GROUP}/type'
"""Dataset a writer records its on-disk format in; see `write_hamiltonian_format`."""


def clear_hamiltonian(fh5) -> None:
    """
    Remove the Hamiltonian already in the open HDF5 file `fh5`, if there is one.

    A SAFIRE input file holds at most one Hamiltonian and at most one
    wavefunction, so writing a Hamiltonian replaces any Hamiltonian already
    present while leaving everything else — notably ``Wavefunction`` — alone.
    """
    if HAMILTONIAN_GROUP in fh5:
        del fh5[HAMILTONIAN_GROUP]


@contextmanager
def open_for_hamiltonian(path):
    """
    Open `path` for writing one Hamiltonian, creating the file if needed.

    Anything else already in the file is preserved; only a Hamiltonian already
    present is replaced. This is what lets a Hamiltonian and a wavefunction
    share one file in either order.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to write into.

    Yields
    ------
    h5py.File
        The open file, with no ``Hamiltonian`` group in it.

    Notes
    -----
    HDF5 unlinks rather than reclaims, so repeatedly rewriting a Hamiltonian
    into the same file grows it. Write to a fresh path if that matters.
    """
    with h5.File(path, 'a') as fh5:
        clear_hamiltonian(fh5)
        yield fh5


def write_hamiltonian_format(fh5, fmt) -> None:
    """
    Record `fmt` as the on-disk format of the Hamiltonian being written into
    `fh5`, so that a reader does not have to infer it from the layout.

    Parameters
    ----------
    fh5 : h5py.File
        Destination, open for writing. The ``Hamiltonian`` group is created if
        it does not exist yet, so this can be called before or after the rest of
        the Hamiltonian is written.
    fmt : HamiltonianFormat or str
        The format being written, or its safiretools name.

    Raises
    ------
    ValueError
        If `fmt` names no known format, or names one that is never recorded.

    Notes
    -----
    `fh5` must be a serially opened file: this writes a variable-length string,
    as ``spin_type`` does, and parallel HDF5 cannot write variable-length data.
    The one Hamiltonian written in parallel tags itself afterwards, from one
    rank — see `~safiretools.PeriodicHamiltonian.write_from_pyscf`.
    """
    fmt = HamiltonianFormat(fmt)
    if not fmt.tag:
        raise ValueError(f"the '{fmt}' format is never recorded in a file")

    if TYPE_DATASET in fh5:
        del fh5[TYPE_DATASET]

    fh5.create_dataset(TYPE_DATASET, data=fmt.tag)


def hamiltonian_format(path) -> str:
    """
    Identify the Hamiltonian format stored in the HDF5 file at `path`.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to inspect.

    Returns
    -------
    HamiltonianFormat
        The stored format. It is a string too, so it compares equal to its
        safiretools name (``'model'``, ``'dense'``, ...).

    Raises
    ------
    ValueError
        If the file records a format safiretools does not know, or records none
        and matches none of the known layouts.

    Notes
    -----
    A file written by `write_hamiltonian_format` says outright which format it
    holds. Files written before that key existed do not, so their format is
    inferred from which datasets are present — see `_format_from_layout`.
    """
    with h5.File(path, 'r') as fh5:
        if TYPE_DATASET not in fh5:
            return _format_from_layout(fh5, path)

        try:
            return HamiltonianFormat.from_tag(fh5[TYPE_DATASET].asstr()[()])
        except ValueError as error:
            raise ValueError(f"'{path}' records an {error}") from None


def _format_from_layout(fh5, path) -> "HamiltonianFormat":
    """
    Infer the format of a file that records none from the datasets it holds.

    What `hamiltonian_format` did for every file before writers began recording
    `TYPE_DATASET`, and still the only way to identify one written back then.
    """
    if 'Hamiltonian/ModelHamiltonian/number_of_components' in fh5:
        return HamiltonianFormat.MODEL
    if 'Hamiltonian/DenseFactorized/L' in fh5:
        return HamiltonianFormat.DENSE
    if 'Hamiltonian/KPFactorized/L0' in fh5:
        return HamiltonianFormat.KPOINT
    if 'Hamiltonian/THC/Luv' in fh5:
        return HamiltonianFormat.THC
    if 'Interaction/Vq0' in fh5:
        return HamiltonianFormat.KPOINT_COQUI

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


def _check_domain(cls, target, factory: str) -> None:
    """
    Guard a fixed-domain factory against being called on a subclass it can never
    return.

    The factories live on `Hamiltonian` and reach the subclasses by inheritance,
    so without this ``MolecularHamiltonian.from_dict(...)`` would quietly hand
    back a `LatticeHamiltonian`.

    Parameters
    ----------
    cls : type
        The class the factory was called on.
    target : type
        The concrete subclass this factory builds.
    factory : str
        Name of the factory, for the error message.

    Raises
    ------
    ValueError
        If `target` is not a `cls`.
    """
    if cls is not Hamiltonian and not issubclass(target, cls):
        raise ValueError(
            f"{factory} builds a {target.__name__}, which is not a "
            f"{cls.__name__}; call it on {target.__name__} or on the "
            "dispatching Hamiltonian"
        )


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

    @classmethod
    def from_dict(cls, source, **kwargs) -> "Hamiltonian":
        """
        Build a lattice-model Hamiltonian from a parameter dict or TOML file.
        Always a `safiretools.LatticeHamiltonian`.

        See `safiretools.LatticeHamiltonian.from_dict` for the parameters.
        """
        from safiretools.hamiltonian.model.lattice_hamiltonian import (
            LatticeHamiltonian)

        _check_domain(cls, LatticeHamiltonian, 'from_dict')

        return LatticeHamiltonian.from_dict(source, **kwargs)

    @classmethod
    def from_integrals(cls, hcore, **kwargs) -> "Hamiltonian":
        """
        Build a Hamiltonian from one- and two-body integrals. Always a
        `safiretools.MolecularHamiltonian`.

        See `safiretools.MolecularHamiltonian.from_integrals` for the parameters.
        """
        from safiretools.hamiltonian.molecular import MolecularHamiltonian

        _check_domain(cls, MolecularHamiltonian, 'from_integrals')

        return MolecularHamiltonian.from_integrals(hcore, **kwargs)

    @classmethod
    def from_pyscf(cls, scf_data, **kwargs) -> "Hamiltonian":
        """
        Build a Hamiltonian from a PySCF SCF calculation, molecular or periodic.

        The domain is taken from `scf_data`: a checkpoint loaded from a periodic
        calculation carries ``'cell'`` and yields a
        `safiretools.PeriodicHamiltonian`, while a molecular one carries
        ``'mol'`` and yields a `safiretools.MolecularHamiltonian`.

        Parameters
        ----------
        scf_data : dict
            Unpacked PySCF checkpoint.
        **kwargs
            Forwarded unchanged to `safiretools.MolecularHamiltonian.from_pyscf`
            or `safiretools.PeriodicHamiltonian.from_pyscf`, which document
            them. The two take different parameters — ``cas``, ``ortho_ao``,
            ``df`` and ``real_chol`` are molecular; ``comm``,
            ``kpoint_symmetry``, ``maxvecs`` and ``exxdiv`` are periodic — so a
            keyword aimed at the wrong domain raises `TypeError` from the
            concrete classmethod rather than being silently ignored.

        Returns
        -------
        Hamiltonian
            A `safiretools.MolecularHamiltonian` or a
            `safiretools.PeriodicHamiltonian`.

        Raises
        ------
        ValueError
            If `scf_data` carries neither key, or both.

        Notes
        -----
        The domain is decided by which **key** is present, not by the type of
        the object stored there: ``pyscf.pbc.gto.Cell`` is a subclass of
        ``pyscf.gto.Mole``, so an ``isinstance`` test would read a periodic cell
        as molecular.
        """
        from safiretools.hamiltonian.molecular import MolecularHamiltonian
        from safiretools.hamiltonian.periodic import PeriodicHamiltonian

        periodic = 'cell' in scf_data
        molecular = 'mol' in scf_data

        if periodic == molecular:
            found = sorted(k for k in ('cell', 'mol') if k in scf_data)
            raise ValueError(
                "cannot tell whether this is a molecular or a periodic "
                "calculation: scf_data must carry exactly one of 'cell' "
                f"(periodic) or 'mol' (molecular), found {found or 'neither'}"
            )

        target = PeriodicHamiltonian if periodic else MolecularHamiltonian
        _check_domain(cls, target, 'from_pyscf')

        return target.from_pyscf(scf_data, **kwargs)

    @classmethod
    def write_from_pyscf(cls, comm, scf_data, path, **kwargs) -> None:
        """
        Generate a periodic Hamiltonian and stream it to `path` over `comm`.
        Always a `safiretools.PeriodicHamiltonian`; this is the production path
        for large periodic systems, and the only one that runs in parallel.

        See `safiretools.PeriodicHamiltonian.write_from_pyscf` for the
        parameters.
        """
        from safiretools.hamiltonian.periodic import PeriodicHamiltonian

        _check_domain(cls, PeriodicHamiltonian, 'write_from_pyscf')

        return PeriodicHamiltonian.write_from_pyscf(comm, scf_data, path,
                                                    **kwargs)
