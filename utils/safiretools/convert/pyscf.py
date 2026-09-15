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
Reading a PySCF checkpoint file into the ``scf_data`` mapping the Hamiltonian
and wavefunction factories consume.

`load_pyscf_chk_mol` reads a molecular calculation and `load_pyscf_chk` a
periodic one. Both return a plain mapping, and every ``from_pyscf`` factory
accepts either a checkpoint path — which it loads through here — or an
already-loaded mapping, so that one load can serve several factories. That
matters when the basis and the wavefunction come from *different* SCF
calculations, which is the common case for a spin-orbit or
multi-reference workflow: the Hamiltonian is expressed in one solution's
orbitals while the trial wavefunction comes from another.

The spin symmetry is *determined here*, from what the checkpoint says about the
calculation, rather than inferred later from the shape of whatever was built.
See `determine_spin_symm`.
"""

import json
import logging
from pathlib import Path
from warnings import warn

import numpy as np
import h5py as h5

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

SOC_TYPES = (None, 'sfx2c', 'x2c', 'ecp')
"""Ways of including spin-orbit coupling in the one-body Hamiltonian."""


def as_scf_data(source, periodic=False):
    """
    Resolve a factory's `source` argument into an ``scf_data`` mapping.

    Parameters
    ----------
    source : str or pathlib.Path or dict
        A PySCF checkpoint file to load, or an already-loaded mapping to use as
        it stands.
    periodic : bool, optional
        Load a periodic calculation rather than a molecular one. Default False.
        Ignored when `source` is already a mapping.

    Returns
    -------
    dict
        The ``scf_data`` mapping.

    Raises
    ------
    ValueError
        If `source` is neither a path nor a mapping, or names a checkpoint of
        the wrong kind.
    """
    from collections.abc import Mapping

    if isinstance(source, Mapping):
        return source

    if not isinstance(source, (str, Path)):
        raise ValueError(
            "source must be a PySCF checkpoint path or an scf_data mapping, "
            f"not {type(source).__name__}"
        )

    if is_periodic_chk(source) != periodic:
        kind = 'periodic' if periodic else 'molecular'
        raise ValueError(
            f"'{source}' holds a {'molecular' if periodic else 'periodic'} "
            f"PySCF calculation, but a {kind} one is needed here"
        )

    return load_pyscf_chk(source) if periodic else load_pyscf_chk_mol(source)


def is_periodic_chk(chkfile) -> bool:
    """
    Whether the PySCF checkpoint at `chkfile` holds a periodic calculation.

    Decided by whether its serialized molecule carries lattice vectors, which
    PySCF stores under the key ``'a'`` for a ``Cell`` and not at all for a
    ``Mole``.
    """
    with h5.File(chkfile, 'r') as fh5:
        return 'a' in json.loads(fh5['mol'][()]).keys()


def determine_spin_symm(mo_coeff, mo_occ, soc_type=None, hcore=None,
                        nmo=None) -> SpinSymm:
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
    soc_type : {None, 'sfx2c', 'x2c', 'ecp'}, optional
        Spin-orbit treatment requested. ``'x2c'`` and ``'ecp'`` produce a
        spinor one-body Hamiltonian and so force a noncollinear symmetry.
    hcore : array_like, optional
        One-body Hamiltonian, checked for a spinor basis when given.
    nmo : int, optional
        Number of spatial orbitals, needed to recognize a spinor `hcore`.

    Returns
    -------
    SpinSymm

    Notes
    -----
    This reads the *calculation*, not the shape of a Slater matrix built from
    it: a spinor basis is noncollinear, spin-resolved orbitals are collinear,
    fractional or singly occupied orbitals are collinear, and a solution whose
    occupancies are all 0 or 2 is closed shell. That ordering matters — a GHF
    solution has one ``mo_coeff`` matrix like an RHF one, and only the basis
    size tells them apart.
    """
    mo_coeff = np.asarray(mo_coeff)
    mo_occ = np.asarray(mo_occ)

    if soc_type in ('x2c', 'ecp'):
        return SpinSymm.NONCOLLINEAR

    if nmo is not None and hcore is not None \
            and np.shape(hcore)[-1] == 2 * nmo:
        return SpinSymm.NONCOLLINEAR

    if nmo is not None and mo_coeff.ndim == 2 and mo_coeff.shape[0] == 2 * nmo:
        return SpinSymm.NONCOLLINEAR

    if mo_coeff.ndim == 3 or mo_occ.ndim == 2:
        return SpinSymm.COLLINEAR

    # an ROHF solution records both channels in one occupancy vector, so a
    #   singly occupied orbital is what distinguishes it from RHF
    if np.any((mo_occ > 1e-8) & (mo_occ < 2 - 1e-8)):
        return SpinSymm.COLLINEAR

    return SpinSymm.CLOSED


# ----------------------------------------------------------------------
# molecular
# ----------------------------------------------------------------------

def load_pyscf_chk_mol(chkfile, base='scf', soc_type=None) -> dict:
    """
    Read a molecular PySCF checkpoint file.

    Parameters
    ----------
    chkfile : str or pathlib.Path
        Checkpoint file to read.
    base : str, optional
        HDF5 group holding the solution. Default ``'scf'``; pass ``'mcscf'`` to
        read a CASSCF one.
    soc_type : {None, 'sfx2c', 'x2c', 'ecp'}, optional
        Include spin-orbit coupling in the one-body Hamiltonian, either
        spin-free exact two-component (``'sfx2c'``), full exact two-component
        (``'x2c'``), or through the ECP (``'ecp'``). The latter two make
        ``hcore`` a spinor matrix, which requires a noncollinear spin symmetry
        and the orthogonalized-AO basis.

    Returns
    -------
    dict
        Keys ``mol``, ``nelec``, ``mo_occ``, ``mo_coeff``, ``hcore``, ``norb``,
        ``X``, ``df_ints``, ``walker_type`` and ``soc_type``.

    Raises
    ------
    ValueError
        If `soc_type` is not recognized.
    """
    from pyscf import lib
    from pyscf.lib.chkfile import load_mol

    if soc_type not in SOC_TYPES:
        raise ValueError(
            f"unknown soc_type '{soc_type}': supported values are "
            f"{list(SOC_TYPES)}"
        )

    mol = load_mol(chkfile)
    nmo = mol.nao_nr()
    mo_occ = np.array(lib.chkfile.load(chkfile, f'{base}/mo_occ'))
    mo_coeff = np.array(lib.chkfile.load(chkfile, f'{base}/mo_coeff'))

    with h5.File(chkfile, 'r') as fh5:
        if '/scf/hcore' in fh5:
            if soc_type is not None:
                warn(
                    "reading hcore from the checkpoint file, so it is unclear "
                    f"whether the requested soc_type '{soc_type}' is included"
                )
            hcore = fh5['/scf/hcore'][:]
        else:
            hcore = _hcore_with_soc(mol, soc_type)

        if '/scf/orthoAORot' in fh5:
            X = fh5['/scf/orthoAORot'][:]
            # orbitals may have been dropped for linear dependence
            nmo = X.shape[-1]
        else:
            X = canonical_orthogonalization(mol.intor('int1e_ovlp_sph'))

        df_ints = fh5['j3c'][:] if 'j3c' in fh5 else None

    spin_symm = determine_spin_symm(mo_coeff, mo_occ, soc_type=soc_type,
                                    hcore=hcore, nmo=nmo)
    logger.info("read a %s molecular solution from %s: nelec=%s, norb=%d",
                spin_symm.label, chkfile, mol.nelec, nmo)

    return {
        'mol': mol,
        'nelec': mol.nelec,
        'mo_occ': mo_occ,
        'hcore': hcore,
        'norb': nmo,
        'X': X,
        'mo_coeff': mo_coeff,
        'df_ints': df_ints,
        'walker_type': spin_symm,
        'soc_type': soc_type,
    }


def _hcore_with_soc(mol, soc_type):
    """The one-body Hamiltonian for `mol`, with the requested SOC treatment."""
    from pyscf import scf

    if soc_type is None:
        return scf.hf.get_hcore(mol)
    if soc_type == 'sfx2c':
        return mol.RHF().sfx2c1e().get_hcore()
    if soc_type == 'x2c':
        return mol.GHF().x2c1e().get_hcore()
    return scf.GHF(mol).get_hcore() + ecp_soc(mol)


def ecp_soc(mol):
    """
    The ECP spin-orbit term for `mol`, as a spinor matrix.

    See PySCF's ``examples/gto/20-soc_ecp.py``.
    """
    from pyscf import lib

    spin = 0.5 * lib.PauliMatrices
    soc = -1j * lib.einsum('sxy,spq->xpyq', spin, mol.intor('ECPso'))
    return soc.reshape(soc.shape[0] * soc.shape[1],
                       soc.shape[2] * soc.shape[3])


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
# periodic
# ----------------------------------------------------------------------

def load_pyscf_chk(chkfile, hcore=None, ortho_ao=False) -> dict:
    """
    Read a periodic PySCF checkpoint file.

    Parameters
    ----------
    chkfile : str or pathlib.Path
        Checkpoint file to read.
    hcore : array_like, optional
        One-body Hamiltonian to use instead of the checkpoint's.
    ortho_ao : bool, optional
        Work in the orthogonalized AO basis, reading ``scf/orthoAORot`` and
        ``scf/nmo_per_kpt`` from the checkpoint. Default False, which uses the
        MO basis and requires a closed-shell solution.

    Returns
    -------
    dict
        Keys ``cell``, ``kpts``, ``Xocc``, ``hcore``, ``X``, ``nmo_pk``,
        ``mo_coeff``, ``nao``, ``fock``, ``mo_energy`` and ``walker_type``.

    Raises
    ------
    ValueError
        If the checkpoint holds a spin-resolved solution and `ortho_ao` is not
        set, if the cell is spin polarized without one, if a stored quantity
        matches no layout PySCF writes (see `_per_kpoint`), or if the stored
        quantities disagree about whether the solution is spin resolved.
    """
    from pyscf import lib
    from pyscf.pbc.lib.chkfile import load_cell

    cell = load_cell(chkfile)
    nao = cell.nao_nr()

    # a single-k-point calculation records 'scf/kpt', a k-point mesh 'scf/kpts'
    kpt = lib.chkfile.load(chkfile, 'scf/kpt')
    kpts = np.reshape(
        lib.chkfile.load(chkfile, 'scf/kpts') if kpt is None else kpt, (-1, 3))
    nkpts = len(kpts)

    def per_kpoint(name, entry_ndim):
        return _per_kpoint(lib.chkfile.load(chkfile, f'scf/{name}'), nkpts,
                           entry_ndim, name)

    mo_occ, spin_resolved = per_kpoint('mo_occ', 1)
    mo_energy, energy_spin = per_kpoint('mo_energy', 1)
    mo_coeff, coeff_spin = per_kpoint('mo_coeff', 2)
    fock, fock_spin = per_kpoint('fock', 2)

    if {energy_spin, coeff_spin, fock_spin} != {spin_resolved}:
        raise ValueError(
            "the checkpoint disagrees with itself about whether the solution "
            f"is spin resolved: mo_occ says {spin_resolved}, mo_energy "
            f"{energy_spin}, mo_coeff {coeff_spin}, fock {fock_spin}. "
            "'scf/fock' is usually written by hand after the SCF, so check "
            "that it carries the same spin structure as the solution"
        )

    # the Fock matrix is indexed [spin][kpt] downstream, with a spin axis of
    #   length one for a closed-shell solution
    fock = np.asarray(fock)
    if not spin_resolved:
        fock = fock[np.newaxis]

    if hcore is None:
        hcore = np.asarray(lib.chkfile.load(chkfile, 'scf/hcore'))
    hcore = np.reshape(hcore, (-1, nao, nao))

    if cell.spin != 0 and not spin_resolved:
        raise ValueError(
            "a spin-polarized cell needs a spin-resolved (UHF) solution"
        )
    if spin_resolved and not ortho_ao:
        raise ValueError(
            "a spin-resolved (UHF) solution requires the orthogonalized AO "
            "basis; pass ortho_ao=True"
        )

    if ortho_ao:
        rotation = np.asarray(
            lib.chkfile.load(chkfile, 'scf/orthoAORot')).reshape(nkpts, nao, -1)
        nmo_pk = np.atleast_1d(
            np.asarray(lib.chkfile.load(chkfile, 'scf/nmo_per_kpt')))
        X = [rotation[k][:, :nmo_pk[k]] for k in range(nkpts)]
    else:
        # a closed-shell solution's own orbitals are the working basis
        X = [np.asarray(block) for block in mo_coeff]
        nmo_pk = np.array([Xk.shape[1] for Xk in X], dtype=np.int32)

    spin_symm = SpinSymm.COLLINEAR if spin_resolved else SpinSymm.CLOSED
    logger.info("read a %s periodic solution from %s: %d k-point(s), nao=%d",
                spin_symm.label, chkfile, nkpts, nao)

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
        What ``pyscf.lib.chkfile.load`` returned.
    nkpts : int
        Number of k-points, taken from the checkpoint's k-point array.
    entry_ndim : int
        Rank of a single k-point's entry: 1 for occupancies and orbital
        energies, 2 for orbital coefficients and Fock matrices.
    name : str
        Dataset name, for the error messages.

    Returns
    -------
    values : list
        One entry per k-point, or ``[alpha, beta]`` of those when spin resolved.
    spin_resolved : bool
        Whether `values` carried a spin axis.

    Raises
    ------
    ValueError
        If the shape matches no layout PySCF writes.

    Notes
    -----
    PySCF's layout depends on how the calculation was set up: a single-k-point
    solution stores one entry with no k-point axis, a k-point mesh stores a
    sequence of them, and a spin-resolved solution prefixes a spin axis. The
    container varies independently — k-points sharing an orbital count come back
    as one array, ragged ones as a list of arrays — so what is tested here is
    the **nesting depth**, not whether the result happens to be a ``list``.

    The spin axis and the k-point axis are both length 2 for a two-k-point
    collinear solution, which is why the depth is what separates them; where a
    depth is genuinely shared, the k-point count decides.
    """
    depth = _nesting_depth(values)

    if depth == entry_ndim:
        if nkpts != 1:
            raise ValueError(
                f"'{name}' holds one entry with no k-point axis, but the "
                f"checkpoint has {nkpts} k-points"
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
