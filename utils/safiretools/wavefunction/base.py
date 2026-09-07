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
from contextlib import contextmanager
from warnings import warn

import numpy as np
import h5py as h5

from safiretools.types import SpinSymm
from safiretools.wavefunction import io

WAVEFUNCTION_GROUP = 'Wavefunction'
"""Top-level HDF5 group every wavefunction is written into."""

ORTHONORMAL_TOL = 1e-10
"""How far an overlap matrix may stray from the identity and still count as
orthonormal."""


def clear_wavefunction(fh5) -> None:
    """
    Remove the wavefunction already in the open HDF5 file `fh5`, if there is one.

    A SAFIRE input file holds at most one Hamiltonian and at most one
    wavefunction, so writing a wavefunction replaces any wavefunction already
    present while leaving everything else — notably ``Hamiltonian`` — alone.
    """
    if WAVEFUNCTION_GROUP in fh5:
        del fh5[WAVEFUNCTION_GROUP]


@contextmanager
def open_for_wavefunction(path):
    """
    Open `path` for writing one wavefunction, creating the file if needed.

    Anything else already in the file is preserved; only a wavefunction already
    present is replaced. This is what lets a Hamiltonian and a wavefunction
    share one file in either order.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to write into.

    Yields
    ------
    h5py.File
        The open file, with no ``Wavefunction`` group in it.

    Notes
    -----
    HDF5 unlinks rather than reclaims, so repeatedly rewriting a wavefunction
    into the same file grows it. Write to a fresh path if that matters.
    """
    with h5.File(path, 'a') as fh5:
        clear_wavefunction(fh5)
        yield fh5


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
        if f'{WAVEFUNCTION_GROUP}/NOMSD/dims' in fh5:
            return 'nomsd'
        if f'{WAVEFUNCTION_GROUP}/PHMSD/dims' in fh5:
            return 'phmsd'

    raise ValueError(f"'{path}' holds no wavefunction safiretools recognizes")


def _check_representation(cls, target, factory: str) -> None:
    """
    Guard a fixed-representation factory against being called on a subclass it
    can never return.

    The factories live on `Wavefunction` and reach the subclasses by
    inheritance, so without this ``PHMSDWavefunction.from_free_electron(...)``
    would quietly hand back an `NOMSDWavefunction`.

    Parameters
    ----------
    cls : type
        The class the factory was called on.
    target : type
        The concrete subclass this factory always builds.
    factory : str
        Name of the factory, for the error message.

    Raises
    ------
    ValueError
        If `target` is not a `cls`.
    """
    if cls is not Wavefunction and not issubclass(target, cls):
        raise ValueError(
            f"{factory} always builds a {target.__name__}, which is not a "
            f"{cls.__name__}; call it on {target.__name__} or on the "
            "dispatching Wavefunction"
        )


# ----------------------------------------------------------------------
# orthonormality
# ----------------------------------------------------------------------

def modified_gram_schmidt(matrix, tol=1e-12):
    """
    Orthonormalize the columns of `matrix` by modified Gram-Schmidt.

    Parameters
    ----------
    matrix : numpy.ndarray
        Matrix ``(n, m)`` whose columns are orthonormalized in place order.
    tol : float, optional
        Smallest norm accepted before columns count as linearly dependent.
        Default 1e-12.

    Returns
    -------
    numpy.ndarray
        A new ``complex128`` matrix with orthonormal columns spanning the same
        column space.

    Raises
    ------
    ValueError
        If `matrix` is not two-dimensional, or a column is (near) linearly
        dependent on the previous ones.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise ValueError(f"expected a 2-dimensional array, got {matrix.ndim}D")

    orthonormal = np.zeros_like(matrix, dtype=np.complex128)

    for column in range(matrix.shape[1]):
        vector = np.array(matrix[:, column], dtype=np.complex128, copy=True)
        for previous in range(column):
            vector -= np.vdot(orthonormal[:, previous], vector) \
                * orthonormal[:, previous]

        norm = np.linalg.norm(vector)
        if norm < tol:
            raise ValueError(
                "linearly dependent vectors encountered during Gram-Schmidt "
                f"orthogonalization of column {column}"
            )

        orthonormal[:, column] = vector / norm

    return orthonormal


def is_orthonormal(matrix, tol=ORTHONORMAL_TOL) -> bool:
    """
    True when `matrix`'s columns are orthonormal, i.e. when
    :math:`M^\\dagger M` is the identity to within `tol`.

    A matrix with no columns is orthonormal: its overlap is the empty identity.
    """
    matrix = np.asarray(matrix)
    if matrix.shape[-1] == 0:
        return True

    overlap = matrix.conj().T @ matrix
    identity = np.eye(matrix.shape[-1], dtype=overlap.dtype)
    return bool(np.max(np.abs(overlap - identity)) < tol)


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
    psi0 : sequence of numpy.ndarray, optional
        Initial Slater determinant for the AFQMC walkers, one
        ``(npol*nmo, nelec_of_that_spin)`` block per spin channel. Derived from
        the wavefunction itself when omitted.

    Raises
    ------
    ValueError
        If the electron counts contradict `spin_symm`, or `psi0` has the wrong
        number of blocks or the wrong shape.

    Notes
    -----
    ``__init__`` takes already-formed, validated in-memory data. Build a
    wavefunction from an external source with one of the classmethod factories
    on the concrete subclasses (``from_free_electron``, ``from_pyscf``,
    ``from_pbc_scf``, ``from_dice``, ``from_hdf5``, ...).
    """

    _HDF5_GROUP: str
    """Subgroup of ``Wavefunction`` this representation is written into."""

    def __init__(self, coeffs, nelec, nmo: int, spin_symm, psi0=None) -> None:
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
        self.psi0 = psi0

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
        """Number of independent spin channels: 2 when collinear, else 1."""
        return 2 if self.spin_symm is SpinSymm.COLLINEAR else 1

    @property
    def npol(self) -> int:
        """Spin polarizations per orbital: 2 when noncollinear, else 1."""
        return 2 if self.spin_symm is SpinSymm.NONCOLLINEAR else 1

    @property
    def nelec_per_spin(self) -> tuple:
        """
        Electron count in each independent spin channel, of length `nspin`.

        ``(nup, ndown)`` when collinear; the alpha count alone when closed
        (the beta channel repeats it); the total when noncollinear (both
        polarizations share one channel).
        """
        if self.spin_symm is SpinSymm.COLLINEAR:
            return self.nelec
        if self.spin_symm is SpinSymm.NONCOLLINEAR:
            return (sum(self.nelec),)
        return (self.nelec[0],)

    @property
    def nelec_on_disk(self) -> tuple:
        """
        The ``(nup, ndown)`` pair ``dims[1:3]`` records.

        A noncollinear wavefunction reports ``(nup + ndown, 0)``: both
        polarizations live in one channel, so the split is not part of the
        format.
        """
        if self.spin_symm is SpinSymm.NONCOLLINEAR:
            return (sum(self.nelec), 0)
        return self.nelec

    @property
    def nrows(self) -> int:
        """Rows of an orbital matrix, ``npol * nmo``."""
        return self.npol * self.nmo

    @property
    def ndets(self) -> int:
        """Number of determinants in the expansion."""
        return self.coeffs.size

    @property
    def psi0(self) -> tuple:
        """
        Initial Slater determinant for the AFQMC walkers: one
        ``(npol*nmo, nelec_of_that_spin)`` block per spin channel.

        Derived from the wavefunction itself when none was supplied — see
        `_default_psi0` on the concrete subclass.
        """
        if self._psi0 is None:
            return self._default_psi0()
        return self._psi0

    @psi0.setter
    def psi0(self, value) -> None:
        if value is None:
            self._psi0 = None
            return

        blocks = tuple(np.asarray(block, dtype=np.complex128) for block in value)

        if len(blocks) != self.nspin:
            raise ValueError(
                f"psi0 must have one block per spin channel: expected "
                f"{self.nspin} for a {self.spin_symm.label} wavefunction, "
                f"got {len(blocks)}"
            )

        for ispin, (block, nelec) in enumerate(zip(blocks, self.nelec_per_spin)):
            if block.shape != (self.nrows, nelec):
                raise ValueError(
                    f"psi0 block {ispin} has shape {block.shape}, expected "
                    f"({self.nrows}, {nelec})"
                )

        self._psi0 = blocks

    @abstractmethod
    def _default_psi0(self) -> tuple:
        """
        The initial Slater determinant to use when the caller supplied none.
        """

    def _warn_about_default_psi0(self) -> None:
        """
        Warn, on write, about a defaulted `psi0` that will make for a poor
        initial walker. Nothing by default; a representation whose default is a
        poor choice overrides this.
        """

    @abstractmethod
    def orthonormalize(self, tol=ORTHONORMAL_TOL) -> "Wavefunction":
        """
        Return a copy whose Slater matrices have orthonormal columns.

        Blocks that are already orthonormal to within `tol` are left exactly as
        they are; the rest are orthonormalized by `modified_gram_schmidt`. The
        instance this is called on is never modified.
        """

    @abstractmethod
    def _slater_matrices(self):
        """
        Every Slater matrix this wavefunction will write, as
        ``(label, matrix)`` pairs, for the orthonormality check on write.
        """

    def _warn_if_not_orthonormal(self, tol=ORTHONORMAL_TOL) -> None:
        """
        Warn about each Slater matrix whose columns are not orthonormal.

        Writing does not orthonormalize — nothing here mutates the caller's
        data — so this is the last point at which a wavefunction that AFQMC
        will struggle with can be flagged. Call `orthonormalize` to fix it.
        """
        offenders = [label for label, matrix in self._slater_matrices()
                     if not is_orthonormal(matrix, tol=tol)]

        if offenders:
            warn(
                f"Slater matrices are not orthonormal: {', '.join(offenders)}. "
                "Call orthonormalize() before writing, or expect poor AFQMC "
                "behavior."
            )

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
        The header (``dims``, ``ci_coeffs``, ``Psi0_alpha``/``Psi0_beta``) is
        the same for every representation and is written here; the subclass adds
        only its own payload.

        Nothing is orthonormalized on the way out — a non-orthonormal Slater
        matrix is warned about, not silently repaired.
        """
        self._warn_if_not_orthonormal()

        if self._psi0 is None:
            self._warn_about_default_psi0()

        with open_for_wavefunction(path) as fh5:
            group = fh5.create_group(
                f'{WAVEFUNCTION_GROUP}/{type(self)._HDF5_GROUP}')

            io.write_header(
                group,
                nmo=self.nmo,
                nelec=self.nelec_on_disk,
                spin_symm=self.spin_symm,
                coeffs=self.coeffs,
                psi0=self.psi0,
            )
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
            group = fh5[f'{WAVEFUNCTION_GROUP}/{target._HDF5_GROUP}']
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
    def from_free_electron(cls, source, nelec, twist=None, spin_symm=None,
                           use_dense=True, lattice=None,
                           filling_strategy='aufbau', shell_tol=1e-6,
                           orthonormalize=True) -> "Wavefunction":
        """
        Build a free-electron trial wavefunction from a lattice model. Always a
        `safiretools.NOMSDWavefunction`.

        See `safiretools.wavefunction.free_electron.from_free_electron` for the
        full parameter documentation.
        """
        from safiretools.wavefunction.free_electron import from_free_electron
        from safiretools.wavefunction.nomsd import NOMSDWavefunction

        _check_representation(cls, NOMSDWavefunction, 'from_free_electron')

        return from_free_electron(
            source, nelec=nelec, twist=twist, spin_symm=spin_symm,
            use_dense=use_dense, lattice=lattice,
            filling_strategy=filling_strategy, shell_tol=shell_tol,
            orthonormalize=orthonormalize,
        )

    @classmethod
    def from_pyscf(cls, scf_data, basis_scf_data=None, ortho_ao=False, cas=None,
                   spin_symm=None, orthonormalize=True) -> "Wavefunction":
        """
        Build a single-determinant trial wavefunction from a molecular PySCF SCF
        calculation. Always a `safiretools.NOMSDWavefunction`.

        See `safiretools.wavefunction.pyscf.from_pyscf` for the full parameter
        documentation.
        """
        from safiretools.wavefunction.nomsd import NOMSDWavefunction
        from safiretools.wavefunction.pyscf import from_pyscf

        _check_representation(cls, NOMSDWavefunction, 'from_pyscf')

        return from_pyscf(
            scf_data, basis_scf_data=basis_scf_data, ortho_ao=ortho_ao, cas=cas,
            spin_symm=spin_symm, orthonormalize=orthonormalize,
        )

    @classmethod
    def from_pyscf_cas(cls, mol, cas_chkfile, tol=1e-4,
                       max_det=None) -> "Wavefunction":
        """
        Read a CASSCF/CASCI expansion from a PySCF checkpoint file. Always a
        `safiretools.PHMSDWavefunction`.

        See `safiretools.wavefunction.pyscf.from_pyscf_cas` for the full
        parameter documentation.
        """
        from safiretools.wavefunction.phmsd import PHMSDWavefunction
        from safiretools.wavefunction.pyscf import from_pyscf_cas

        _check_representation(cls, PHMSDWavefunction, 'from_pyscf_cas')

        return from_pyscf_cas(mol, cas_chkfile, tol=tol, max_det=max_det)

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
    def from_pbc_scf(cls, scf_data, ortho_ao=True, rediag=True, ndet_max=1,
                     low=0.1, high=0.95,
                     orthonormalize=True) -> "Wavefunction":
        """
        Build a trial wavefunction from a periodic PySCF SCF calculation.

        **Returns whichever representation the occupancies call for**: a
        `safiretools.PHMSDWavefunction` when bands are partially occupied and
        `ndet_max` allows more than one determinant, and a
        `safiretools.NOMSDWavefunction` otherwise. This is the factory that
        motivates dispatching from the base class at all — the representation
        cannot be known until the occupancies have been looked at.

        The subclass classmethods override this with the *narrowing* forms:
        `safiretools.NOMSDWavefunction.from_pbc_scf` forces ``ndet_max=1`` to
        guarantee a single determinant, and
        `safiretools.PHMSDWavefunction.from_pbc_scf` raises when a single
        determinant would describe the system exactly.

        See `safiretools.wavefunction.pbc.from_pbc_scf` for the full parameter
        documentation.
        """
        from safiretools.wavefunction.pbc import from_pbc_scf

        return from_pbc_scf(
            scf_data, ortho_ao=ortho_ao, rediag=rediag, ndet_max=ndet_max,
            low=low, high=high, orthonormalize=orthonormalize,
        )
