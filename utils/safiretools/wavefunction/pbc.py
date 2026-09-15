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
k-point, giving a single Slater determinant. Reached through
`NOMSDWavefunction.from_pbc_scf`.

.. note:: **CoQuí is the supported route for solids.** This path covers a PySCF
          mean-field reference only, and only as a single determinant; see
          DESIGN.md.
"""

import logging

import numpy as np

from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)


def from_pbc_scf(source, ortho_ao=True, rediag=True, low=0.1, high=0.95):
    """
    Build a single-determinant trial wavefunction from a periodic PySCF SCF
    calculation.

    Parameters
    ----------
    source : str or pathlib.Path or dict
        A PySCF checkpoint file, or an already-loaded ``scf_data`` mapping from
        `safiretools.convert.pyscf.load_pyscf_chk`. Uses the keys ``'X'``,
        ``'Xocc'``, ``'fock'``, ``'nmo_pk'``, ``'mo_energy'``, ``'kpts'`` and
        ``'walker_type'``.
    ortho_ao : bool, optional
        Whether the working basis is the orthogonalized AO basis. Must match the
        Hamiltonian. Default True; a collinear reference requires it.
    rediag : bool, optional
        Rediagonalize the Fock matrix to get MO coefficients in the
        orthogonalized AO basis. Default True.
    low, high : float, optional
        Occupancies strictly between these bounds count as partial. The leading
        configuration — the lowest-indexed of the partially occupied bands — is
        the one occupied. Defaults 0.1 and 0.95.

    Returns
    -------
    NOMSDWavefunction
        A single-determinant wavefunction.

    Raises
    ------
    ValueError
        If a collinear reference is combined with ``ortho_ao=False``, or a
        closed-shell reference has partially occupied bands (see Notes).

    Notes
    -----
    Partial occupancies are only handled for a collinear reference. The
    occupancy bookkeeping doubles the count of every filled band for a
    closed-shell one, which is only right for integer occupancies; present such
    a calculation as a collinear reference instead.

    The orbitals are eigenvectors of a Hermitian Fock matrix (or columns of the
    identity when ``ortho_ao=False``), so they are orthonormal by construction
    and nothing here orthonormalizes them.
    """
    from safiretools.convert.pyscf import as_scf_data
    from safiretools.wavefunction.nomsd import NOMSDWavefunction

    scf_data = as_scf_data(source, periodic=True)

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

    occupancies, order = _reoccupy(mo_occ, eigenvalues, collinear=collinear,
                                   low=low, high=high)

    # mo_occ from pyscf is a list of arrays of potentially different length, so
    #   np.sum is not available
    nelec = tuple(int(round(sum(sum(occ) for occ in channel)))
                  for channel in occupancies)

    _log_eigenvalues(eigenvalues, order, nelec, collinear=collinear)

    return NOMSDWavefunction(
        coeffs=np.array([1.0 + 0j]),
        dets=_supercell_slater(orbitals, occupancies, nmo_pk, nelec,
                               collinear=collinear)[np.newaxis, ...],
        nelec=nelec,
        spin_symm=spin_symm,
        nmo=nmo_tot,
    )


# ----------------------------------------------------------------------
# orbitals per k-point
# ----------------------------------------------------------------------

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
            Xk = X[k][:, :nmo_pk[k]]
            energies, orbs = np.linalg.eigh(Xk.conj().T @ (fock[ispin, k] @ Xk))
            eigenvalues[ispin].extend(energies)
            orbitals[ispin].append(orbs)

    if not collinear:
        # the beta channel repeats alpha; `_reoccupy` reads eigenvalues[0] for it
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

def _reoccupy(mo_occ, mo_energy, collinear, low=0.1, high=0.95):
    """
    Resolve the per-spin, per-k-point occupancies of the single determinant.

    Returns
    -------
    occupancies : list
        Per-spin occupancies, k-point by k-point.
    order : numpy.ndarray or tuple
        Energy-sorted index order, for logging.
    """
    if not collinear:
        occupancies, partial, order, _ = _determine_occupancies(
            mo_occ, mo_energy[0], closed=True, low=low, high=high)

        if partial:
            raise ValueError(
                "partially occupied bands with a closed-shell reference are "
                "not supported: the occupancy bookkeeping doubles the count of "
                "every filled band, which only works for integer occupancies. "
                "Present the calculation as a collinear reference instead"
            )

        # a closed-shell reference splits its occupancies evenly
        return [occupancies / 2.0, occupancies / 2.0], order

    logger.debug("determining occupancies for the alpha electrons")
    occ_a, _, order_a, _ = _determine_occupancies(
        mo_occ[0], mo_energy[0], closed=False, low=low, high=high)

    logger.debug("determining occupancies for the beta electrons")
    occ_b, _, order_b, _ = _determine_occupancies(
        mo_occ[1], mo_energy[1], closed=False, low=low, high=high)

    return [occ_a, occ_b], (order_a, order_b)


def _determine_occupancies(mo_occ, mo_energy, closed, low=0.1, high=0.95):
    """
    Split the occupancies into a filled core plus, when bands are partially
    occupied, the leading configuration over them.

    The leading configuration occupies the lowest-indexed of the partially
    occupied bands, which is the reference determinant of the expansion those
    bands span.

    Returns
    -------
    occupancies
        Per-k-point occupancies of the determinant to build.
    partial : bool
        Whether any band was partially occupied.
    order, inverse_order : numpy.ndarray
        Energy-sorted index order and its inverse.

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
        return mo_occ, False, order, inverse_order

    logger.info("found partially occupied bands: occupying the leading "
                "configuration over them")

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

    if nleft > ndeg:
        raise ValueError(
            f"{nleft} electron(s) remain to be placed over {ndeg} partially "
            f"occupied band(s) between {low} and {high}"
        )

    logger.info("distributing %d electron(s) over %d orbital(s)", nleft, ndeg)

    # supercell-indexed, energy-sorted; the leading configuration takes the
    #   first `nleft` of the partially occupied bands
    occupied = np.zeros(len(sorted_occ), dtype=np.int32)
    occupied[np.where(sorted_occ > high)[0]] = 1
    occupied[np.where(degenerate)[0][:nleft]] = 1

    # remap to primitive-cell (kpoint, band) indexing
    reference = occupied[inverse_order]

    occupancies = []
    start = 0
    for k in range(len(mo_occ)):
        occupancies.append(reference[start:start + nmo_pk[k]])
        start += nmo_pk[k]

    return occupancies, True, order, inverse_order


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
