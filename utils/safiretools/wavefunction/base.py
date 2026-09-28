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
The `Wavefunction` abstract base class, and the on-disk format detection that
`Wavefunction.from_hdf5` dispatches on.

Wavefunctions split by *representation* — `NOMSDWavefunction` (coefficients plus
per-determinant orbital matrices) and `PHMSDWavefunction` (coefficients plus
occupation numbers) — not by the domain that built them. A free-electron
wavefunction and a PySCF UHF wavefunction are the same NOMSD representation,
built differently, so the domain is a classmethod factory rather than a
subclass. Spin symmetry is a plain attribute, not a subclass axis.

Unlike `Hamiltonian`, whose formats genuinely differ by domain, the wavefunction
format is one schema, so ``to_hdf5``/``from_hdf5`` are implemented here once;
subclasses supply only the representation-specific payload.
"""

from abc import ABC, abstractmethod

import numpy as np
import h5py as h5

from safiretools.hdf5 import replace_group
from safiretools.types import SpinSymm
from safiretools.wavefunction import io
from safiretools.wavefunction.slater import (
    ORTHONORMAL_TOL,
    format_spin_layout,
    parse_spin_layout,
)


def wavefunction_format(path) -> str:
    """
    Identify the wavefunction representation stored in the HDF5 file at `path`.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to inspect.

    Returns
    -------
    str
        ``'nomsd'`` or ``'phmsd'``.

    Raises
    ------
    ValueError
        If the file holds no wavefunction.
    """
    with h5.File(path, 'r') as fh5:
        if 'Wavefunction/NOMSD' in fh5:
            return 'nomsd'
        if 'Wavefunction/PHMSD' in fh5:
            return 'phmsd'

    raise ValueError(f"'{path}' holds no wavefunction safiretools recognizes")


def _check_representation(cls, target, factory: str) -> None:
    """
    Stop ``PHMSDWavefunction.from_free_electron(...)`` from quietly handing back
    an `NOMSDWavefunction`: the factories reach the subclasses by inheritance.
    """
    if cls is not Wavefunction and not issubclass(target, cls):
        raise ValueError(
            f"{factory} always builds a {target.__name__}, which is not a "
            f"{cls.__name__}; call it on {target.__name__} or on the "
            "dispatching Wavefunction"
        )


class Wavefunction(ABC):
    """
    Base class for all SAFIRE trial wavefunctions.

    Parameters
    ----------
    coeffs : array_like
        Determinant coefficients, ``(ndets,)``. Stored as ``complex128``.
    nelec : tuple(int, int)
        Physical electron counts ``(nup, ndown)``.
    nmo : int
        Number of *spatial* orbitals, even when noncollinear.
    spin_symm : SpinSymm or str or int
        Spin symmetry, coerced through `SpinSymm.from_input`.

    Raises
    ------
    ValueError
        If the electron counts contradict `spin_symm`.

    Notes
    -----
    ``__init__`` takes already-formed, validated in-memory data. Build a
    wavefunction from an external source with one of the classmethod factories
    on the concrete subclasses (``from_free_electron``, ``from_pyscf``,
    ``from_pbc_scf``, ``from_dice``, ``from_hdf5``, ...).
    """

    _HDF5_GROUP: str
    """Subgroup of ``Wavefunction`` this representation is written into."""

    def __init__(self, coeffs, nelec, nmo: int, spin_symm) -> None:
        self.coeffs = np.asarray(coeffs, dtype=np.complex128)
        if self.coeffs.ndim != 1:
            raise ValueError(
                f"coeffs must be one-dimensional, got shape {self.coeffs.shape}"
            )

        self.nmo = int(nmo)
        self.nelec = tuple(int(n) for n in nelec)
        if len(self.nelec) != 2:
            raise ValueError(f"nelec must be a (nup, ndown) pair, got {nelec!r}")

        self.spin_symm = spin_symm

    # ------------------------------------------------------------------
    # derived shape
    # ------------------------------------------------------------------

    @property
    def spin_symm(self) -> SpinSymm:
        """Spin symmetry of this wavefunction, as a `SpinSymm`."""
        return self._spin_symm

    @spin_symm.setter
    def spin_symm(self, value) -> None:
        spin_symm = SpinSymm.from_input(value)

        if spin_symm is SpinSymm.CLOSED and self.nelec[0] != self.nelec[1]:
            raise ValueError(
                f"a closed-shell wavefunction needs equal spin populations, "
                f"got nelec={self.nelec}; use 'collinear' instead"
            )

        self._spin_symm = spin_symm

    @property
    def nspin(self) -> int:
        """Number of independent spin channels; see `SpinSymm.nspin`."""
        return self.spin_symm.nspin

    @property
    def npol(self) -> int:
        """Spin polarizations per orbital; see `SpinSymm.npol`."""
        return self.spin_symm.npol

    @property
    def nelec_per_spin(self) -> tuple:
        """Electron count in each independent spin channel; see `SpinSymm.nelec_per_spin`."""
        return self.spin_symm.nelec_per_spin(self.nelec)

    @property
    def nrows(self) -> int:
        """Rows of an orbital matrix, ``npol * nmo``."""
        return self.npol * self.nmo

    @property
    def ndets(self) -> int:
        """Number of determinants in the expansion."""
        return self.coeffs.size

    @abstractmethod
    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "Wavefunction":
        """
        Return a copy whose Slater matrices have orthonormal columns.

        Each block goes through `safiretools.wavefunction.slater.orthonormalize`,
        so one already orthonormal to within `tol` is left exactly as it is. The
        instance this is called on is never modified.
        """

    def to_hdf5(self, path) -> None:
        """
        Write this wavefunction in the format the AFQMC executable reads.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist. A
            wavefunction already in the file is replaced; everything else —
            notably a ``Hamiltonian`` — is left alone, so a Hamiltonian and a
            wavefunction can share one file in either order.

        Notes
        -----
        The header (``spin_type``, ``ci_coeffs``) is the same for every
        representation and is written here; the subclass adds only its own
        payload.

        No initial walker is written: the AFQMC input chooses the wavefunction
        a walker set starts from (``walker_set.from``), and the executable
        derives the initial determinant from it.

        Nothing is repaired on the way out. Every Slater matrix that reaches
        disk has its overlap's condition number checked, and an ill-conditioned
        one is warned about — see
        `safiretools.wavefunction.io.warn_if_ill_conditioned`.
        """
        with h5.File(path, 'a') as fh5:
            group = replace_group(fh5, 'Wavefunction').create_group(
                type(self)._HDF5_GROUP)

            io.write_header(group, spin_symm=self.spin_symm, coeffs=self.coeffs)
            self._write_payload(group)

    @classmethod
    def from_hdf5(cls, path) -> "Wavefunction":
        """
        Read a wavefunction from the HDF5 file `path`.

        Called on `Wavefunction` itself, the concrete subclass is chosen from
        the representation stored in the file. Called on a concrete subclass,
        the file must hold that representation.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to read.

        Returns
        -------
        Wavefunction
            An instance of the concrete subclass for the stored representation.

        Raises
        ------
        ValueError
            If the file holds no wavefunction, or holds a representation that
            the subclass this was called on does not read.
        """
        # imported here, not at module scope, because both subclasses import
        #   this module
        from safiretools.wavefunction.nomsd import NOMSDWavefunction
        from safiretools.wavefunction.phmsd import PHMSDWavefunction

        fmt = wavefunction_format(path)
        target = NOMSDWavefunction if fmt == 'nomsd' else PHMSDWavefunction

        if cls is not Wavefunction and not issubclass(target, cls):
            raise ValueError(
                f"'{path}' holds a '{fmt}' wavefunction, which {cls.__name__} "
                f"does not read; use {target.__name__}.from_hdf5 or the "
                "dispatching Wavefunction.from_hdf5"
            )

        with h5.File(path, 'r') as fh5:
            group = fh5['Wavefunction'][target._HDF5_GROUP]
            return target._read_payload(group, io.read_header(group))

    @abstractmethod
    def _write_payload(self, group) -> None:
        """
        Write this representation's payload into `group`, whose shared header
        `to_hdf5` has already written.
        """

    @classmethod
    @abstractmethod
    def _read_payload(cls, group, header: dict) -> "Wavefunction":
        """
        Build an instance from `group` and the already-read shared `header`.
        Subclasses implement this rather than overriding `from_hdf5`, so that
        dispatch stays in one place.
        """

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_single_determinant(cls, det) -> "Wavefunction":
        """
        Build a single-determinant trial wavefunction from its occupied
        orbitals. Always a `safiretools.NOMSDWavefunction`.

        The layout of `det` decides the spin symmetry, and its shape the
        electron counts and the number of orbitals, so nothing else is needed.

        Parameters
        ----------
        det : numpy.ndarray or tuple of numpy.ndarray
            The occupied orbitals, as columns:

            - closed shell: an array ``(nmo, nup)``, shared by both spins;
            - collinear: a *tuple* of arrays ``(nmo, nup)`` and ``(nmo, ndown)``;
            - noncollinear: an array ``(2, nmo, nelec)`` of spinor orbitals,
              spin-up components first.

        Returns
        -------
        NOMSDWavefunction
            A one-determinant expansion with coefficient 1. A noncollinear one
            reports ``nelec == (nelec, 0)``, as the file format does.

        Raises
        ------
        TypeError
            If `det` is none of the three layouts — including a list, which is
            not read as collinear.
        ValueError
            If the two collinear channels span different numbers of orbitals, or
            this is called on a subclass other than `NOMSDWavefunction`.

        Notes
        -----
        A collinear determinant must be a tuple. An array holding both spin
        channels — such as PySCF's UHF ``mo_coeff[:, :, :nocc]`` — has the
        noncollinear shape ``(2, nmo, n)`` and is read as a noncollinear
        determinant; pass ``tuple(...)`` of it instead.

        Examples
        --------
        A closed-shell determinant occupying the lowest two of four orbitals:

        >>> import numpy as np
        >>> wavefunction = Wavefunction.from_single_determinant(np.eye(4)[:, :2])
        >>> wavefunction.spin_symm.label, wavefunction.nelec
        ('closed', (2, 2))

        The collinear determinant with one more spin-up electron:

        >>> orbitals = np.eye(4)
        >>> wavefunction = Wavefunction.from_single_determinant(
        ...     (orbitals[:, :3], orbitals[:, :2]))
        >>> wavefunction.nelec
        (3, 2)
        """
        from safiretools.wavefunction.nomsd import NOMSDWavefunction

        _check_representation(cls, NOMSDWavefunction, 'from_single_determinant')

        spin_symm, blocks = parse_spin_layout(det, name='det')
        dets = format_spin_layout(tuple(block[np.newaxis] for block in blocks),
                                  spin_symm)

        return NOMSDWavefunction(coeffs=np.array([1.0 + 0j]), dets=dets)

    @classmethod
    def from_free_electron(cls, source, nelec, spin_symm=None,
                           filling_strategy='aufbau',
                           shell_tol=None) -> "Wavefunction":
        """
        Build a free-electron trial wavefunction from a
        `safiretools.LatticeHamiltonian`. Always a
        `safiretools.NOMSDWavefunction`.

        See `safiretools.wavefunction.free_electron.from_free_electron` for the
        full parameter documentation. `shell_tol` defaults to that module's
        `SHELL_TOL` rather than being restated here, so the two cannot drift.
        """
        from safiretools.wavefunction.free_electron import (
            SHELL_TOL,
            from_free_electron,
        )
        from safiretools.wavefunction.nomsd import NOMSDWavefunction

        _check_representation(cls, NOMSDWavefunction, 'from_free_electron')

        return from_free_electron(
            source, nelec=nelec, spin_symm=spin_symm,
            filling_strategy=filling_strategy,
            shell_tol=SHELL_TOL if shell_tol is None else shell_tol,
        )

    @classmethod
    def from_pyscf(cls, mf, basis=None, active_space=None) -> "Wavefunction":
        """
        Build a single-determinant trial wavefunction from a molecular PySCF SCF
        object. Always a `safiretools.NOMSDWavefunction`.

        See `safiretools.wavefunction.pyscf.from_pyscf` for the full parameter
        documentation.
        """
        from safiretools.wavefunction.nomsd import NOMSDWavefunction
        from safiretools.wavefunction.pyscf import from_pyscf

        _check_representation(cls, NOMSDWavefunction, 'from_pyscf')

        return from_pyscf(mf, basis=basis, active_space=active_space)

    @classmethod
    def from_pyscf_cas(cls, mc, tol=1e-4, max_det=None) -> "Wavefunction":
        """
        Read a CASSCF/CASCI expansion from a PySCF ``mcscf`` object. Always a
        `safiretools.PHMSDWavefunction`.

        See `safiretools.wavefunction.pyscf.from_pyscf_cas` for the full
        parameter documentation.
        """
        from safiretools.wavefunction.phmsd import PHMSDWavefunction
        from safiretools.wavefunction.pyscf import from_pyscf_cas

        _check_representation(cls, PHMSDWavefunction, 'from_pyscf_cas')

        return from_pyscf_cas(mc, tol=tol, max_det=max_det)

    @classmethod
    def from_dice(cls, path, ndets, state=0) -> "Wavefunction":
        """
        Read a selected-CI expansion from Dice's output. Always a
        `safiretools.PHMSDWavefunction`.

        See `safiretools.wavefunction.dice.from_dice` for the full parameter
        documentation.
        """
        from safiretools.wavefunction.dice import from_dice
        from safiretools.wavefunction.phmsd import PHMSDWavefunction

        _check_representation(cls, PHMSDWavefunction, 'from_dice')

        return from_dice(path, ndets=ndets, state=state)

    @classmethod
    def from_pbc_scf(cls, kmf, basis=None, rediag=True,
                     low=0.1, high=0.95) -> "Wavefunction":
        """
        Build a single-determinant trial wavefunction from a periodic PySCF SCF
        object. Always a `safiretools.NOMSDWavefunction`.

        See `safiretools.wavefunction.pbc.from_pbc_scf` for the full parameter
        documentation.
        """
        from safiretools.wavefunction.nomsd import NOMSDWavefunction
        from safiretools.wavefunction.pbc import from_pbc_scf

        _check_representation(cls, NOMSDWavefunction, 'from_pbc_scf')

        return from_pbc_scf(kmf, basis=basis, rediag=rediag, low=low, high=high)
