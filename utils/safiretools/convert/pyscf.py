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
Reading PySCF SCF objects for the Hamiltonian and wavefunction factories.

Every ``from_pyscf`` factory takes PySCF objects — an ``mf`` from ``scf``, an
``mc`` from ``mcscf``, a ``kmf`` from ``pbc.scf`` — and reads them by duck
typing: only their methods and attributes are used, so nothing here imports
PySCF.

Two roles are kept apart. The *source* object says what the physics is: its
``get_hcore()`` is the one-body Hamiltonian, so the relativistic or spin-orbit
treatment is chosen by how that object was set up (``sfx2c1e()``,
``x2c1e()``, a GHF with ``with_soc = True``). The *basis* says which orbitals
everything is expressed in; see `working_basis`. Pass the same basis to the
Hamiltonian and the wavefunction factory.

The spin symmetry is *determined here*, from the data, rather than passed in.
See `hamiltonian_spin_symm` and `determine_spin_symm`.
"""

import logging
from warnings import warn

import numpy as np

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

ORTHO_AO = 'ortho_ao'
"""The `basis` value selecting the canonically orthogonalized AO basis."""


def determine_spin_symm(mo_coeff, mo_occ, nao) -> SpinSymm:
    """
    Determine the spin symmetry of a PySCF SCF solution from the solution
    itself.

    Parameters
    ----------
    mo_coeff : array_like
        Orbital coefficients. A leading spin axis marks a spin-resolved (UHF)
        solution.
    mo_occ : array_like
        Orbital occupancies.
    nao : int
        Number of spatial basis functions, needed to recognize a spinor
        (GHF) solution.

    Returns
    -------
    SpinSymm

    Notes
    -----
    A spinor basis is noncollinear, spin-resolved orbitals are collinear,
    fractional or singly occupied orbitals are collinear, and a solution whose
    occupancies are all 0 or 2 is closed shell. That ordering matters — a GHF
    solution has one ``mo_coeff`` matrix like an RHF one, and only the basis
    size tells them apart.
    """
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)

    if mo_coeff.ndim == 2 and mo_coeff.shape[0] == 2 * nao:
        return SpinSymm.NONCOLLINEAR

    if mo_coeff.ndim == 3 or mo_occ.ndim == 2:
        return SpinSymm.COLLINEAR

    # an ROHF solution records both channels in one occupancy vector, so a
    #   singly occupied orbital is what distinguishes it from RHF
    if np.any((mo_occ > 1e-8) & (mo_occ < 2 - 1e-8)):
        return SpinSymm.COLLINEAR

    return SpinSymm.CLOSED


def hamiltonian_spin_symm(hcore, nao) -> SpinSymm:
    """
    The spin symmetry of a molecular one-body Hamiltonian, from its shape.

    A spinor ``(2 nao, 2 nao)`` matrix is noncollinear and a spatial
    ``(nao, nao)`` one closed: a spin-independent operator gains nothing from
    two identical spin sectors, whatever the reference state is.

    Raises
    ------
    ValueError
        If `hcore` has neither shape.
    """
    shape = np.shape(hcore)
    if shape == (nao, nao):
        return SpinSymm.CLOSED
    if shape == (2 * nao, 2 * nao):
        return SpinSymm.NONCOLLINEAR
    raise ValueError(
        f"hcore has shape {shape}, which is neither ({nao}, {nao}) nor the "
        f"spinor ({2 * nao}, {2 * nao}) for a molecule with {nao} basis functions"
    )


def one_body(mf):
    """
    The one-body Hamiltonian of the source object `mf`, with its spin symmetry.

    Warns when the molecule carries a spin-orbit ECP that the Hamiltonian does
    not include, which is what a GHF object without ``with_soc = True`` gives.

    Returns
    -------
    hcore : numpy.ndarray
        ``mf.get_hcore()`` in the AO (or spin-orbital AO) basis.
    spin_symm : SpinSymm
        See `hamiltonian_spin_symm`.
    """
    mol = mf.mol
    nao = mol.nao_nr()
    hcore = np.asarray(mf.get_hcore())
    spin_symm = hamiltonian_spin_symm(hcore, nao)

    if mol.has_ecp_soc() and _is_spin_free(hcore, nao):
        warn(
            "the molecule carries a spin-orbit ECP, but hcore has no spin-orbit "
            "term; if that is not intended, pass a GHF object with "
            "with_soc = True"
        )

    return hcore, spin_symm


def _is_spin_free(hcore, nao) -> bool:
    """Whether `hcore` acts identically on both spins and never flips one."""
    if hcore.shape == (nao, nao):
        return True
    return (np.array_equal(hcore[:nao, :nao], hcore[nao:, nao:])
            and not np.any(hcore[:nao, nao:]) and not np.any(hcore[nao:, :nao]))


def orbitals(mf):
    """
    ``(mo_coeff, mo_occ)`` of a converged SCF object.

    Raises
    ------
    ValueError
        If `mf` holds no orbitals, i.e. was never run.
    """
    if getattr(mf, 'mo_coeff', None) is None or getattr(mf, 'mo_occ', None) is None:
        raise ValueError(
            f"{type(mf).__name__} holds no orbitals; run the calculation first"
        )
    return np.asarray(mf.mo_coeff), np.asarray(mf.mo_occ)


def canonical_orthogonalization(overlap, lindep_cutoff=0.0):
    """
    The canonical orthogonalization transformation of an overlap matrix.

    Parameters
    ----------
    overlap : numpy.ndarray
        Overlap matrix.
    lindep_cutoff : float, optional
        Basis functions whose overlap eigenvalue falls below this are dropped.
        Default 0.0; set it in step with PySCF's
        ``scf.addons.remove_linear_dep``.

    Returns
    -------
    numpy.ndarray
        Transformation ``X`` into the orthogonalized basis.
    """
    eigenvalues, vectors = np.linalg.eigh(overlap)
    kept = eigenvalues > lindep_cutoff
    return vectors[:, kept] / np.sqrt(eigenvalues[kept])


# ----------------------------------------------------------------------
# molecular
# ----------------------------------------------------------------------

def working_basis(mf, basis=None, active_space=None):
    """
    Resolve a factory's `basis` argument into the transformation into the
    working basis, and the frozen-orbital counts.

    Parameters
    ----------
    mf
        The source SCF object; its ``mol`` gives the AO overlap and the
        electron count.
    basis : None or 'ortho_ao' or object or numpy.ndarray, optional
        ``None`` uses `mf`'s own orbitals, ``'ortho_ao'`` the canonically
        orthogonalized AO basis, an object its ``mo_coeff`` (an ROHF or CASSCF
        solution, say), and an array is used as the ``(nao, nmo)``
        transformation itself.
    active_space : tuple(int, int), optional
        ``(nelecas, ncas)`` active space. ``ncas == -1`` takes every orbital
        above the frozen core.

    Returns
    -------
    C : numpy.ndarray
        Transformation into the working basis, ``(nao, nmo)``.
    (nfzc, nfzv) : tuple(int, int)
        Numbers of frozen core and virtual orbitals.

    Raises
    ------
    ValueError
        If `basis` is an unknown string, if `active_space` is combined with
        ``'ortho_ao'``, if the basis orbitals are spin resolved (UHF/GHF), or if
        they do not span `mf`'s AO basis.
    """
    mol = mf.mol
    nao = mol.nao_nr()

    if isinstance(basis, str):
        if basis != ORTHO_AO:
            raise ValueError(
                f"unknown basis '{basis}': pass '{ORTHO_AO}', an object with "
                "mo_coeff, or an (nao, nmo) array"
            )
        if active_space is not None:
            raise ValueError(
                f"active_space and basis='{ORTHO_AO}' cannot be combined")
        return canonical_orthogonalization(mol.intor('int1e_ovlp')), (0, 0)

    if isinstance(basis, np.ndarray):
        C = basis
    else:
        source = mf if basis is None else basis
        if getattr(source, 'mo_coeff', None) is None:
            raise ValueError(
                f"{type(source).__name__} holds no orbitals to use as the basis; "
                "run the calculation first"
            )
        C = np.asarray(source.mo_coeff)

    if C.ndim == 3 or C.shape[0] == 2 * nao:
        raise ValueError(
            "UHF or GHF orbital bases are not supported; use "
            f"basis='{ORTHO_AO}'"
        )
    if C.shape[0] != nao:
        raise ValueError(
            f"the basis orbitals have {C.shape[0]} rows, but the molecule has "
            f"{nao} basis functions"
        )

    if active_space is None:
        return C, (0, 0)

    nelecas, ncas = active_space
    nfzc = (sum(mol.nelec) - nelecas) // 2
    nmo = C.shape[-1]
    if ncas == -1:
        ncas = nmo - nfzc

    return C, (nfzc, nmo - ncas - nfzc)


def cholesky_df(mf):
    """
    The density-fitting vectors of a density-fitted SCF object, in the AO basis.

    Returns
    -------
    numpy.ndarray
        ``(naux, nao*nao)``, unpacked from PySCF's lower-triangular storage.

    Raises
    ------
    ValueError
        If `mf` is not density fitted.
    """
    if getattr(mf, 'with_df', None) is None:
        raise ValueError(
            "df=True needs a density-fitted SCF object, e.g. mf.density_fit()"
        )

    nao = mf.mol.nao_nr()
    lower = np.tril_indices(nao)
    blocks = []
    for packed in mf.with_df.loop():
        block = np.zeros((packed.shape[0], nao, nao), dtype=packed.dtype)
        block[:, lower[0], lower[1]] = packed
        block[:, lower[1], lower[0]] = packed
        blocks.append(block.reshape(packed.shape[0], nao * nao))

    return np.concatenate(blocks)


# ----------------------------------------------------------------------
# periodic
# ----------------------------------------------------------------------

def periodic_solution(kmf, basis=None, lindep_cutoff=0.0) -> dict:
    """
    Read a periodic PySCF SCF object.

    Parameters
    ----------
    kmf
        A ``pbc.scf`` object, at one k-point or on a mesh, restricted or
        unrestricted.
    basis : None or 'ortho_ao', optional
        ``None`` works in the solution's own orbitals, which requires a
        closed-shell solution; ``'ortho_ao'`` in the canonically orthogonalized
        AO basis at each k-point.
    lindep_cutoff : float, optional
        Overlap eigenvalue below which an orthogonalized orbital is dropped;
        see `canonical_orthogonalization`. Default 0.0.

    Returns
    -------
    dict
        Keys ``cell``, ``kpts``, ``Xocc``, ``hcore``, ``X``, ``nmo_pk``,
        ``mo_coeff``, ``nao``, ``fock``, ``mo_energy`` and ``walker_type``.

    Raises
    ------
    ValueError
        If `basis` is anything else, if the solution is spin resolved and the
        basis is not ``'ortho_ao'``, if the cell is spin polarized without a
        spin-resolved solution, or if a per-k-point quantity matches no layout
        PySCF uses.
    """
    if basis is not None and not (isinstance(basis, str) and basis == ORTHO_AO):
        raise ValueError(
            f"a periodic basis is either None or '{ORTHO_AO}', not {basis!r}"
        )
    ortho_ao = basis is not None

    cell = kmf.cell
    nao = cell.nao_nr()
    kpts = np.reshape(kmf.kpts, (-1, 3))
    nkpts = len(kpts)

    if getattr(kmf, 'mo_coeff', None) is None:
        raise ValueError(
            f"{type(kmf).__name__} holds no orbitals; run the calculation first"
        )

    def per_kpoint(values, name, entry_ndim):
        return _per_kpoint(values, nkpts, entry_ndim, name)

    mo_occ, spin_resolved = per_kpoint(kmf.mo_occ, 'mo_occ', 1)
    mo_energy, _ = per_kpoint(kmf.mo_energy, 'mo_energy', 1)
    mo_coeff, _ = per_kpoint(kmf.mo_coeff, 'mo_coeff', 2)
    fock, _ = per_kpoint(kmf.get_fock(), 'fock', 2)

    # the Fock matrix is indexed [spin][kpt] downstream, with a spin axis of
    #   length one for a closed-shell solution
    fock = np.asarray(fock)
    if not spin_resolved:
        fock = fock[np.newaxis]

    hcore = np.reshape(kmf.get_hcore(), (-1, nao, nao))

    if cell.spin != 0 and not spin_resolved:
        raise ValueError(
            "a spin-polarized cell needs a spin-resolved (UHF) solution"
        )
    if spin_resolved and not ortho_ao:
        raise ValueError(
            "a spin-resolved (UHF) solution requires the orthogonalized AO "
            f"basis; pass basis='{ORTHO_AO}'"
        )

    if ortho_ao:
        overlaps = np.reshape(kmf.get_ovlp(), (-1, nao, nao))
        X = [canonical_orthogonalization(s1e, lindep_cutoff) for s1e in overlaps]
    else:
        # a closed-shell solution's own orbitals are the working basis
        X = [np.asarray(block) for block in mo_coeff]
    nmo_pk = np.array([Xk.shape[1] for Xk in X], dtype=np.int32)

    spin_symm = SpinSymm.COLLINEAR if spin_resolved else SpinSymm.CLOSED
    logger.info("read a %s periodic solution: %d k-point(s), nao=%d",
                spin_symm.label, nkpts, nao)

    return {
        'cell': cell,
        'kpts': kpts,
        'Xocc': mo_occ,
        'hcore': hcore,
        'X': X,
        'nmo_pk': nmo_pk,
        'mo_coeff': mo_coeff,
        'nao': nao,
        'fock': fock,
        'mo_energy': mo_energy,
        'walker_type': spin_symm,
    }


def _nesting_depth(values) -> int:
    """
    How many axes `values` has, counting a nested sequence as an axis.

    An array reports its own rank; a list of arrays reports one more than
    theirs.
    """
    if isinstance(values, np.ndarray):
        return values.ndim
    if isinstance(values, (list, tuple)):
        return 1 + _nesting_depth(values[0])
    return 0


def _per_kpoint(values, nkpts: int, entry_ndim: int, name: str):
    """
    Normalize a per-orbital quantity to one entry per k-point, and say whether
    it is spin resolved.

    Parameters
    ----------
    values : array_like
        The quantity as the SCF object holds it.
    nkpts : int
        Number of k-points.
    entry_ndim : int
        Rank of a single k-point's entry: 1 for occupancies and orbital
        energies, 2 for orbital coefficients and Fock matrices.
    name : str
        Quantity name, for the error messages.

    Returns
    -------
    values : list
        One entry per k-point, or ``[alpha, beta]`` of those when spin resolved.
    spin_resolved : bool
        Whether `values` carried a spin axis.

    Raises
    ------
    ValueError
        If the shape matches no layout PySCF uses.

    Notes
    -----
    PySCF's layout depends on how the calculation was set up: a single-k-point
    solution holds one entry with no k-point axis, a k-point mesh a sequence of
    them, and a spin-resolved solution prefixes a spin axis. The container
    varies independently — k-points sharing an orbital count come back as one
    array, ragged ones as a list of arrays — so what is tested here is the
    **nesting depth**, not whether the result happens to be a ``list``.

    The spin axis and the k-point axis are both length 2 for a two-k-point
    collinear solution, which is why the depth is what separates them; where a
    depth is genuinely shared, the k-point count decides.
    """
    depth = _nesting_depth(values)

    if depth == entry_ndim:
        if nkpts != 1:
            raise ValueError(
                f"'{name}' holds one entry with no k-point axis, but the "
                f"solution has {nkpts} k-points"
            )
        return [values], False

    if depth == entry_ndim + 1:
        if len(values) == nkpts:
            return list(values), False
        if nkpts == 1 and len(values) == 2:
            # spin resolved at a single k-point
            return [[values[0]], [values[1]]], True
        raise ValueError(
            f"'{name}' holds {len(values)} entries, which is neither the "
            f"{nkpts} k-points nor a spin pair at a single k-point"
        )

    if depth == entry_ndim + 2:
        if len(values) != 2 or any(len(spin) != nkpts for spin in values):
            raise ValueError(
                f"'{name}' is spin resolved, so it must hold 2 channels of "
                f"{nkpts} k-points each, not {len(values)} of "
                f"{[len(spin) for spin in values]}"
            )
        return [list(values[0]), list(values[1])], True

    raise ValueError(
        f"'{name}' has nesting depth {depth}, which matches no layout for a "
        f"quantity whose per-k-point entry has rank {entry_ndim}"
    )
