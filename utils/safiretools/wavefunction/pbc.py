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
Trial wavefunctions from periodic PySCF calculations.

The k-point orbitals are assembled into one supercell Slater matrix, block by
k-point. Which representation comes out depends on the occupancies: a system
with integer occupancies gives a single determinant, while partially occupied
degenerate bands — a metal, typically — can give a multi-determinant expansion
over the ways of distributing the leftover electrons among those bands.

Reached through `NOMSDWavefunction.from_pbc_scf` and
`PHMSDWavefunction.from_pbc_scf`.
"""

import itertools
import logging

import numpy as np
import scipy.linalg

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)


def from_pbc_scf(scf_data, ortho_ao=True, rediag=True, ndet_max=1, low=0.1,
                 high=0.95, orthonormalize=True):
    """
    Build a trial wavefunction from a periodic PySCF SCF calculation.

    Parameters
    ----------
    scf_data : dict
        Unpacked PySCF checkpoint, as produced by
        ``afqmctools.utils.pyscf_utils.load_from_pyscf_chk``. Uses the keys
        ``'X'``, ``'Xocc'``, ``'fock'``, ``'nmo_pk'``, ``'mo_energy'``,
        ``'kpts'`` and ``'walker_type'``.
    ortho_ao : bool, optional
        Whether the working basis is the orthogonalized AO basis. Must match the
        Hamiltonian. Default True; a collinear reference requires it.
    rediag : bool, optional
        Rediagonalize the Fock matrix to get MO coefficients in the
        orthogonalized AO basis. Default True.
    ndet_max : int or None, optional
        Largest number of determinants to keep. 1 (the default) forces a single
        determinant even when bands are partially occupied; None means "as many
        as the degeneracy allows".
    low, high : float, optional
        Occupancies strictly between these bounds count as partial, and their
        bands as the degenerate set to distribute the leftover electrons over.
        Defaults 0.1 and 0.95.
    orthonormalize : bool, optional
        Orthonormalize the resulting Slater matrices. Default True.

    Returns
    -------
    NOMSDWavefunction or PHMSDWavefunction
        A single determinant when the occupancies are integer or `ndet_max` is
        1; otherwise a particle-hole expansion over the degenerate bands, with
        the k-point orbital matrices as its reference.

    Raises
    ------
    ValueError
        If a collinear reference is combined with ``ortho_ao=False``, or a
        closed-shell reference has partially occupied bands (see Notes).

    Notes
    -----
    Partial occupancies are only handled for a collinear reference. The
    occupancy bookkeeping doubles the count of every filled band for a
    closed-shell one, which is only right for integer occupancies; afqmctools
    reached that case too and failed with a ``TypeError`` deeper in. Present
    such a calculation as a collinear reference.
    """
    from safiretools.wavefunction.nomsd import NOMSDWavefunction
    from safiretools.wavefunction.phmsd import PHMSDWavefunction

    spin_symm = SpinSymm.from_input(scf_data['walker_type'])
    collinear = spin_symm is SpinSymm.COLLINEAR

    mo_occ = np.array(scf_data['Xocc'])
    nmo_pk = scf_data['nmo_pk']
    nmo_tot = int(np.sum(nmo_pk))

    fock = scf_data['fock']
    if fock.ndim == 3:
        fock = fock.reshape((1,) + fock.shape)

    logger.info("generating a %s trial wavefunction over %d k-point(s)",
                spin_symm.label, len(scf_data['kpts']))
    for index, kpt in enumerate(scf_data['kpts']):
        logger.debug("  k-point %d: %s", index, kpt)

    eigenvalues, orbitals = _generate_orbitals(
        fock, scf_data['X'], nmo_pk, rediag=rediag, ortho_ao=ortho_ao,
        mo_energy=scf_data['mo_energy'], collinear=collinear)

    occupancies, expansion, ndeg, order = _reoccupy(
        mo_occ, eigenvalues, collinear=collinear, ndet_max=ndet_max, low=low,
        high=high)

    # mo_occ from pyscf is a list of arrays of potentially different length, so
    #   np.sum is not available
    nelec = tuple(int(round(sum(sum(occ) for occ in channel)))
                  for channel in occupancies)

    _log_eigenvalues(eigenvalues, order, nelec, collinear=collinear)

    if ndeg == 1 or ndet_max == 1:
        logger.info("writing a single Slater determinant trial wavefunction")
        wavefunction = NOMSDWavefunction(
            coeffs=np.array([1.0 + 0j]),
            dets=_supercell_slater(orbitals, occupancies, nmo_pk, nelec,
                                   collinear=collinear)[np.newaxis, ...],
            nelec=nelec,
            spin_symm=spin_symm,
            nmo=nmo_tot,
        )
    else:
        coeffs, occa, occb = expansion
        logger.info("writing a particle-hole trial wavefunction with %d "
                    "determinant(s)", len(coeffs))
        wavefunction = PHMSDWavefunction(
            coeffs=np.asarray(coeffs, dtype=np.complex128),
            occa=occa,
            occb=occb,
            nmo=nmo_tot,
            nelec=nelec,
            orbitals=[scipy.linalg.block_diag(*channel) for channel in orbitals
                      if len(channel)],
        )

    return wavefunction.orthonormalize() if orthonormalize else wavefunction


# ----------------------------------------------------------------------
# orbitals per k-point
# ----------------------------------------------------------------------

def rediag_fock(fock, X):
    """
    Rediagonalize one k-point's Fock matrix in the basis `X` maps into.

    Parameters
    ----------
    fock : numpy.ndarray
        Fock matrix for this k-point.
    X : numpy.ndarray
        Transformation into the working basis.

    Returns
    -------
    eigenvalues : numpy.ndarray
        MO eigenvalues.
    orbitals : numpy.ndarray
        MO coefficients, as columns.

    Notes
    -----
    The products are associated as ``X^H (F X)``, matching afqmctools. The other
    grouping differs in the last bits, which is enough to rotate the
    eigenvectors of a degenerate subspace by ~1e-6.
    """
    return np.linalg.eigh(X.conj().T @ (fock @ X))


def _generate_orbitals(fock, X, nmo_pk, rediag, ortho_ao, mo_energy, collinear):
    """
    Orbitals and eigenvalues for every k-point and spin channel.

    Returns
    -------
    eigenvalues : tuple(list, list)
        Per-spin eigenvalues, k-point by k-point.
    orbitals : tuple(list, list)
        Per-spin orbital matrices, one per k-point. The beta list is empty for a
        closed-shell reference.
    """
    if collinear and not ortho_ao:
        raise ValueError(
            "a collinear trial wavefunction requires the orthogonalized AO "
            "basis; pass ortho_ao=True"
        )

    eigenvalues = ([], [])
    orbitals = ([], [])

    for k in range(len(X)):
        logger.debug("generating trial orbitals for k-point %d", k)

        if not ortho_ao:
            eigenvalues[0].append(mo_energy[k])
            orbitals[0].append(np.eye(len(mo_energy[k]), dtype=np.complex128))
            continue

        for ispin in range(2 if collinear else 1):
            energies, orbs = rediag_fock(fock[ispin, k], X[k][:, :nmo_pk[k]])
            eigenvalues[ispin].extend(energies)
            orbitals[ispin].append(orbs)

    if not collinear:
        # the beta channel repeats alpha; `reoccupy` reads eigenvalues[0] for it
        eigenvalues[1].extend(eigenvalues[0])

    return eigenvalues, orbitals


def _supercell_slater(orbitals, occupancies, nmo_pk, nelec, collinear):
    """
    Assemble the per-k-point orbitals into one supercell Slater matrix.

    Each k-point contributes its occupied orbitals as a diagonal block: rows are
    that k-point's orbitals, columns its occupied ones.

    Returns
    -------
    numpy.ndarray
        ``(sum(nmo_pk), nup + ndown)`` when collinear, ``(sum(nmo_pk), nup)``
        otherwise — the column layout `safiretools.NOMSDWavefunction` takes.
    """
    nalpha, nbeta = nelec
    nmo_tot = int(sum(nmo_pk))
    ncols = nalpha + nbeta if collinear else nalpha

    logger.debug("supercell wavefunction shape (%d, %d) for nelec=%s",
                 nmo_tot, ncols, nelec)

    slater = np.zeros((nmo_tot, ncols), dtype=np.complex128)

    row = 0
    col_alpha, col_beta = 0, nalpha
    for k in range(len(nmo_pk)):
        nocca = int(round(sum(occupancies[0][k])))
        slater[row:row + nmo_pk[k], col_alpha:col_alpha + nocca] = \
            orbitals[0][k][:, :nocca]
        col_alpha += nocca

        if collinear:
            noccb = int(round(sum(occupancies[1][k])))
            slater[row:row + nmo_pk[k], col_beta:col_beta + noccb] = \
                orbitals[1][k][:, :noccb]
            col_beta += noccb

        row += nmo_pk[k]

    return slater


# ----------------------------------------------------------------------
# occupancies
# ----------------------------------------------------------------------

def _reoccupy(mo_occ, mo_energy, collinear, low=0.1, high=0.95, ndet_max=1):
    """
    Resolve the occupancies, and the multi-determinant expansion when bands are
    partially occupied.

    Returns
    -------
    occupancies : list
        Per-spin occupancies, k-point by k-point.
    expansion : tuple or None
        ``(coeffs, occa, occb)`` when a multi-determinant expansion was built.
    ndeg : int
        Number of partially occupied bands found; 1 when the occupancies are
        integer.
    order : numpy.ndarray or tuple
        Energy-sorted index order, for logging.
    """
    if not collinear:
        occupancies, ndeg, _, order, _, _ = _determine_occupancies(
            mo_occ, mo_energy[0], closed=True, low=low, high=high)

        if ndeg != 1:
            raise ValueError(
                "partially occupied bands with a closed-shell reference are "
                "not supported: the occupancy bookkeeping doubles the count of "
                "every filled band, which only works for integer occupancies. "
                "Present the calculation as a collinear reference instead"
            )

        # a closed-shell reference splits its occupancies evenly
        return [occupancies / 2.0, occupancies / 2.0], None, ndeg, order

    logger.debug("determining occupancies for the alpha electrons")
    occ_a, ndeg_a, msd_a, order_a, _, p_a = _determine_occupancies(
        mo_occ[0], mo_energy[0], closed=False, low=low, high=high)

    logger.debug("determining occupancies for the beta electrons")
    occ_b, ndeg_b, msd_b, order_b, _, p_b = _determine_occupancies(
        mo_occ[1], mo_energy[1], closed=False, low=low, high=high)

    ndeg = max(ndeg_a, ndeg_b)
    expansion = None

    if msd_a is not None and msd_b is not None:
        logger.info("maximum number of determinants: %d",
                    len(msd_a) * len(msd_b))

        if ndet_max == 1:
            expansion = (np.array([1.0]), np.array([msd_a[0]]),
                         np.array([msd_b[0]]))
        else:
            occs_a, occs_b = zip(*itertools.product(msd_a, msd_b))
            probabilities = np.outer(p_a, p_b).ravel()
            coeffs = (probabilities / sum(probabilities)) ** 0.5

            ndets = len(occs_a) if ndet_max is None \
                else min(len(occs_a), ndet_max)

            # NOTE: argsort is ascending, so this keeps the *least* probable
            #   determinants — and determinant 0 becomes the executable's
            #   reference configuration. Preserved from afqmctools rather than
            #   silently changed; see TASKS.md.
            keep = probabilities.argsort()[:ndets]
            expansion = (coeffs[keep], np.array(occs_a)[keep],
                         np.array(occs_b)[keep])

    return [occ_a, occ_b], expansion, ndeg, (order_a, order_b)


def _determine_occupancies(mo_occ, mo_energy, closed, low=0.1, high=0.95,
                           refdet=0):
    """
    Split the occupancies into a filled core and a partially occupied,
    degenerate set, and enumerate the ways of distributing the leftover
    electrons over the latter.

    Returns
    -------
    occupancies
        Per-k-point occupancies of the reference determinant.
    ndeg : int
        Number of partially occupied bands; 1 when there are none.
    determinants : list or None
        Occupied-orbital indices of each enumerated determinant, in
        (k-point, band) order. None when the occupancies are integer.
    order, inverse_order : numpy.ndarray
        Energy-sorted index order and its inverse.
    probabilities : numpy.ndarray or None
        Product of the partial occupancies of each enumerated determinant.

    Raises
    ------
    ValueError
        If electrons remain to be placed but no partially occupied band can be
        found to place them in.
    """
    nelec = sum(sum(occ) for occ in mo_occ)
    order = np.ravel(mo_energy).argsort()
    inverse_order = order.argsort()

    nocc = 0
    nmo_pk = []
    for occ in mo_occ:
        filled = occ > high
        nmo_pk.append(len(filled))
        nocc += sum(filled)

    if closed:
        nocc = 2 * nocc

    nleft = int(round(nelec - nocc))
    if nleft == 0:
        logger.debug("all occupancies are one or zero")
        return mo_occ, 1, None, order, inverse_order, None

    logger.info("found partially occupied bands: constructing a "
                "multi-determinant trial wavefunction from the degenerate "
                "orbitals")

    sorted_occ = np.array(mo_occ).ravel()[order]
    degenerate = (sorted_occ < high) & (sorted_occ > low)
    ndeg = int(sum(degenerate))

    if ndeg == 0:
        logger.warning("trying to occupy %d electrons in 0 orbitals", nleft)
        smallest = sorted_occ[(sorted_occ < low) & (sorted_occ > 1e-10)]
        if not len(smallest):
            raise ValueError(
                f"{nleft} electron(s) remain to be placed, but no partially "
                f"occupied band was found between {low} and {high}"
            )

        low = 0.5 * smallest[0]
        logger.warning("decreasing the 'low' parameter to %13.8e", low)
        degenerate = (sorted_occ < high) & (sorted_occ > low)
        ndeg = int(sum(degenerate))

        if ndeg == 0:
            raise ValueError(
                f"{nleft} electron(s) remain to be placed, but no partially "
                f"occupied band was found even with low={low:13.8e}; nonzero "
                f"MO occupancies: {smallest}"
            )

    logger.info("distributing %d electron(s) over %d orbital(s)", nleft, ndeg)

    # supercell-indexed
    degenerate_orbitals = np.where(degenerate)[0]
    partial = sorted_occ[degenerate]
    combinations = list(itertools.combinations(degenerate_orbitals, int(nleft)))
    probabilities = np.array(
        [np.prod(partial[np.array(c) - nocc]) for c in combinations])

    core = list(np.where(sorted_occ > high)[0])
    determinants = [core + list(extra) for extra in combinations]

    # remap to primitive-cell (kpoint, band) indexing
    reordered = []
    reference = None
    for index, determinant in enumerate(determinants):
        occupied = np.zeros(len(sorted_occ), dtype=np.int32)
        occupied[determinant] = 1
        if index == refdet:
            reference = occupied[inverse_order]
        reordered.append(np.where(occupied[inverse_order])[0])

    occupancies = []
    start, end = 0, nmo_pk[0]
    for k in range(len(mo_occ)):
        occupancies.append(reference[start:end])
        start += nmo_pk[k]
        end = start + nmo_pk[k + 1] if k + 1 < len(nmo_pk) else -1

    return occupancies, ndeg, reordered, order, inverse_order, probabilities


def _log_eigenvalues(eigenvalues, order, nelec, collinear) -> None:
    """Log the recomputed MO energies up to the highest occupied orbital."""
    if not logger.isEnabledFor(logging.DEBUG):
        return

    logger.debug("recomputed MO energies")

    if collinear:
        eigs_a = np.array(eigenvalues[0]).ravel()[order[0]]
        eigs_b = np.array(eigenvalues[1]).ravel()[order[1]]
        for index, (energy_a, energy_b) in enumerate(zip(eigs_a, eigs_b)):
            tag = ''
            if index == nelec[0] - 1:
                tag = ' <--- HOMO(alpha)'
            elif index == nelec[1] - 1:
                tag = ' <--- HOMO(beta)'
            logger.debug("  %5d  % 13.8e  % 13.8e%s", index, energy_a,
                         energy_b, tag)
            if index >= max(nelec[0], nelec[1]) - 1:
                break
        return

    eigs_a = np.array(eigenvalues[0]).ravel()[order]
    for index, energy in enumerate(eigs_a):
        tag = ' <--- HOMO(alpha)' if index == nelec[0] - 1 else ''
        logger.debug("  %5d  % 13.8e%s", index, energy, tag)
        if index >= nelec[0] - 1:
            break
