# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

r"""
FCIDUMP read/write — the plain-text integral format shared with other quantum
chemistry codes.

FCIDUMP stores integrals in chemists' notation, one matrix element per line as
``value i k j l``, with :math:`(ik|jl) = \langle ij|kl \rangle`. Indices are
1-based, and ``0`` in the index columns marks a one-body element (``j == l ==
0``) or the constant (all zero).

This module is the implementation of that format and is dev-facing throughout.
A user reaches it through `~safiretools.MolecularHamiltonian.from_fcidump` /
`~safiretools.MolecularHamiltonian.to_fcidump` and
`~safiretools.PeriodicHamiltonian.to_fcidump`, which supply the Hamiltonian's
own integrals, electron count and dimensions.
"""

import ast
import logging
from itertools import product

import numpy as np
import scipy.linalg
import scipy.sparse

logger = logging.getLogger(__name__)

SUPPORTED_SYMMETRIES = (1, 4, 8)
"""Permutational symmetries of the two-electron integrals that can be read."""


def fcidump_header(nel: int, norb: int, spin: int) -> str:
    """The FCIDUMP header block for `nel` electrons, `norb` orbitals and ``MS2 = spin``."""
    return (
        "&FCI "
        f"NORB={norb:d}, "
        f"NELEC={nel:d}, "
        f"MS2={spin:d},\n"
        "ORBSYM=" + ",".join(["1"] * norb) + ",\n"
        "ISYM=1\n"
        "&END\n"
    )


def read_fcidump_header(filename, mline=24) -> dict:
    """
    Read the header of an FCIDUMP file.

    Parameters
    ----------
    filename : str or pathlib.Path
        File to read.
    mline : int, optional
        Give up after this many lines. Default 24.

    Returns
    -------
    dict
        Metadata found in the header: ``nbasis``, ``nelec``, ``ms2``, ``isym``.

    Raises
    ------
    ValueError
        If no end-of-header marker is found within `mline` lines.
    """
    fields = {'NORB': 'nbasis', 'NELEC': 'nelec', 'MS2': 'ms2', 'ISYM': 'isym'}

    meta = {}
    with open(filename) as f:
        for _ in range(mline):
            line = f.readline()
            if 'END' in line or '/' in line:
                return meta
            for entry in line.split(','):
                for token, key in fields.items():
                    if token in entry:
                        meta[key] = int(entry.split('=')[1])

    raise ValueError(f"FCIDUMP header in '{filename}' is longer than {mline} lines")


def read_fcidump(filename, symmetry=None, verbose=True):
    """
    Read integrals from an FCIDUMP file.

    Complex-valued integrals may be written either as ``(real,imag)`` or as two
    whitespace-separated columns; both are accepted.

    Parameters
    ----------
    filename : str or pathlib.Path
        File to read.
    symmetry : int, optional
        Permutational symmetry of the two-electron integrals. Read from the
        header when omitted, since some codes misreport it.
    verbose : bool, optional
        Log what was read.

    Returns
    -------
    h1e : numpy.ndarray
        One-body Hamiltonian, ``(nbasis, nbasis)``.
    h2e : numpy.ndarray
        Two-electron integrals, ``(nbasis,) * 4``.
    ecore : float
        Constant energy contribution.
    nelec : tuple(int, int)
        ``(nalpha, nbeta)``.

    Raises
    ------
    ValueError
        If `symmetry` is not one of 1, 4 or 8.
    """
    meta = read_fcidump_header(filename)
    nbasis, nelec, ms2 = meta['nbasis'], meta['nelec'], meta['ms2']

    if symmetry is None:
        symmetry = meta['isym']
    if symmetry not in SUPPORTED_SYMMETRIES:
        raise ValueError(
            f"Unsupported permutational symmetry {symmetry}: "
            f"expected one of {SUPPORTED_SYMMETRIES}"
        )

    if verbose:
        logger.info("reading integrals in plain text FCIDUMP format")
        logger.info("number of orbitals: %d", nbasis)
        logger.info("number of electrons: %d", nelec)

    h1e = np.zeros((nbasis, nbasis), dtype=np.complex128)
    h2e = np.zeros((nbasis, nbasis, nbasis, nbasis), dtype=np.complex128)
    ecore = 0.0 + 0.0j

    with open(filename) as f:
        while True:
            line = f.readline()
            if 'END' in line or '/' in line:
                break

        for line in f.readlines():
            integral, (i, k, j, l) = _parse_integral_line(line)

            if i == j == k == l == 0:
                ecore = integral
            elif j == 0 and l == 0:
                # <i|k> = <k|i>
                h1e[i - 1, k - 1] = integral
                h1e[k - 1, i - 1] = integral.conjugate()
            elif i > 0 and j > 0 and k > 0 and l > 0:
                _scatter_eri(h2e, integral, i - 1, k - 1, j - 1, l - 1, symmetry)

    if symmetry == 8:
        if np.any(np.abs(h1e.imag) > 1e-18):
            logger.warning("found complex numbers in the one-body Hamiltonian but "
                           "8-fold symmetry was specified")
        if np.any(np.abs(h2e.imag) > 1e-18):
            logger.warning("found complex numbers in the two-body Hamiltonian but "
                           "8-fold symmetry was specified")

    nalpha = (nelec + ms2) // 2
    if abs(ecore.imag) > 1e-8:
        logger.warning("found a complex core energy in the FCIDUMP; "
                       "ignoring the imaginary part")

    return h1e, h2e, ecore.real, (nalpha, nalpha - ms2)


def _parse_integral_line(line):
    """
    Split one FCIDUMP body line into its value and its four orbital indices.

    Handles the plain real form, the ``(real,imag)`` parenthesized form, and the
    Quantum Package form with real and imaginary parts in separate columns.
    """
    fields = line.split()

    if line.strip().startswith('('):
        left, right = line.split(')')
        fields = right.split()
        real, imag = left.split(',')
        integral = float(real.replace('(', '')) + 1j * float(imag)
    elif len(fields) == 6:
        # FCIDUMP from Quantum Package
        integral = float(fields[0]) + 1j * float(fields[1])
        fields = fields[1:]
    else:
        try:
            integral = complex(float(fields[0]))
        except ValueError:
            real, imag = ast.literal_eval(fields[0].strip())
            integral = real + 1j * imag

    return integral, tuple(int(x) for x in fields[-4:])


def _scatter_eri(h2e, integral, i, k, j, l, symmetry) -> None:
    r"""
    Place one stored integral and every element the requested permutational
    symmetry makes equal to it.

    With 8-fold symmetry that is
    :math:`\langle ij|kl \rangle = \langle ji|lk \rangle = \langle kl|ij
    \rangle = \langle lk|ji \rangle = \langle kj|il \rangle = \langle li|jk
    \rangle = \langle il|kj \rangle = \langle jk|li \rangle`; 4-fold drops the
    four that require real integrals; 1 stores only what was read.

    Indices are 0-based here.
    """
    h2e[i, k, j, l] = integral                    # (ik|jl)
    if symmetry == 1:
        return

    h2e[j, l, i, k] = integral                    # (jl|ik)
    h2e[k, i, l, j] = integral.conjugate()        # (ki|lj)
    h2e[l, j, k, i] = integral.conjugate()        # (lj|ki)
    if symmetry == 4:
        return

    h2e[k, i, j, l] = integral                    # (ki|jl)
    h2e[l, j, i, k] = integral                    # (lj|ik)
    h2e[i, k, l, j] = integral                    # (ik|lj)
    h2e[j, l, k, i] = integral                    # (jl|ki)


def check_sym(ikjl, nmo, sym) -> bool:
    """
    Whether an integral is the unique representative of its symmetry-equivalent
    set, and so should be written.

    Parameters
    ----------
    ikjl : tuple of int
        Orbital indices of the ERI.
    nmo : int
        Number of orbitals.
    sym : int
        Permutational symmetry to enforce.

    Returns
    -------
    bool
        True if the integral is the one to write.
    """
    if sym == 1:
        return True

    i, k, j, l = ikjl
    if sym == 4:
        return not (ikjl > (j, l, i, k) or ikjl > (k, i, l, j) or ikjl > (l, j, k, i))

    return (i >= k and j >= l) and (i + k * nmo) >= (j + l * nmo)


def h1_spat2spin(h1e):
    r"""
    `h1e` in the spinor basis a FCIDUMP file uses, which alternates up/down
    components: :math:`\{\phi_0 |\uparrow\rangle, \phi_0 |\downarrow\rangle,
    \phi_1 |\uparrow\rangle, \dots\}`.
    """
    nmo = h1e.shape[0]
    spinor = np.zeros((2 * nmo, 2 * nmo), dtype=np.complex128)
    spinor[0::2, 0::2] = h1e
    spinor[1::2, 1::2] = h1e
    return spinor


def h2_spat2spin(h2e):
    r"""
    The chemists' :math:`(ik|jl)` integrals `h2e` in the spinor basis of
    `h1_spat2spin`.

    .. warning:: There must be no matrix elements between opposite spins within
                 :math:`(ik|` or within :math:`|jl)`.
    """
    nmo = h2e.shape[0]
    spinor = np.zeros((2 * nmo,) * 4, dtype=np.complex128)
    spinor[0::2, 0::2, 0::2, 0::2] = h2e
    spinor[0::2, 1::2, 0::2, 1::2] = h2e
    spinor[1::2, 1::2, 1::2, 1::2] = h2e
    spinor[1::2, 0::2, 1::2, 0::2] = h2e
    return spinor


def fmt_integral(intg, i, k, j, l, cplx, paren=False) -> str:
    """
    Format one FCIDUMP integral line.

    Lines are formatted as:

    * real 2-body integrals ``(ij|kl)  i  k  j  l``
    * complex 2-body integrals ``Re Im i k j l`` -OR- ``(Re, Im) i k j l``
    * real 1-body integrals ``h_{ik} i k 0 0``
    * complex 1-body integrals ``Re Im i k 0 0`` -OR- ``(Re, Im) i k 0 0``
    * real constant ``C 0 0 0 0``
    * complex constant ``Re Im 0 0 0 0`` -OR- ``(Re, Im) 0 0 0 0``

    which variant of the complex form to use depends on which code will read the
    file.

    Parameters
    ----------
    intg : float or complex
        Integral value.
    i, k, j, l : int
        0-based orbital indices; written 1-based.
    cplx : bool
        Write the imaginary part too.
    paren : bool
        Write complex values parenthesized.

    Returns
    -------
    str
        One line, newline included.
    """
    if not cplx:
        return '  {: 13.8e}    {:4d}  {:4d}  {:4d}  {:4d}\n'.format(
            intg.real, i + 1, k + 1, j + 1, l + 1)

    fmt = ('  ({: 13.8e}, {: 13.8e}) {:4d}  {:4d}  {:4d}  {:4d}\n' if paren
           else '  {: 13.8e}    {: 13.8e}  {:4d}  {:4d}  {:4d}  {:4d}\n')
    return fmt.format(intg.real, intg.imag, i + 1, k + 1, j + 1, l + 1)


def _write_one_body_and_constant(f, hcore, enuc, tol, cplx, paren) -> None:
    """Write the lower triangle of `hcore` and then the constant `enuc`."""
    for i in range(hcore.shape[0]):
        for j in range(i + 1):
            if abs(hcore[i, j]) > tol:
                f.write(fmt_integral(hcore[i, j], i, j, -1, -1, cplx, paren=paren))

    f.write(fmt_integral(enuc + 0j, -1, -1, -1, -1, cplx, paren=paren))


def write_fcidump(filename, hcore, chol, enuc, nelec, tol=1e-8, ctol=1e-12,
                  sym=1, cplx=True, paren=False, use_spinor=False) -> None:
    """
    Write an FCIDUMP file from Cholesky-factorized integrals.

    Parameters
    ----------
    filename : str or pathlib.Path
        File to write.
    hcore : numpy.ndarray
        One-body Hamiltonian, ``(nmo, nmo)``.
    chol : numpy.ndarray or scipy.sparse.csr_array
        Cholesky matrix ``L[ik,n]``, ``(nmo*nmo, nchol)``.
    enuc : float
        Constant energy contribution.
    nelec : tuple(int, int)
        ``(nalpha, nbeta)``.
    tol : float, optional
        Only write integrals above this magnitude. Default 1e-8.
    ctol : float, optional
        Largest imaginary part tolerated when `cplx` is False. Default 1e-12.
    sym : int, optional
        Write only symmetry-inequivalent ERIs. Default 1, i.e. everything.
    cplx : bool, optional
        Write in complex format. Default True.
    paren : bool, optional
        Write complex numbers parenthesized.
    use_spinor : bool, optional
        Convert to a spinor basis before writing.

    Raises
    ------
    ValueError
        If `cplx` is False but the integrals have imaginary parts above `ctol`.
    """
    if use_spinor and not cplx:
        logger.warning("requested real-valued integrals for a spinor basis: "
                       "writing complex-valued integrals instead")
        cplx = True

    if use_spinor and sym > 1:
        # check_sym assumes a spatial orbital basis, not a spin orbital basis
        # TODO: add an option to check_sym to check symmetry for a spinor basis
        logger.warning("write_fcidump is not implemented for use_spinor with sym > 1: "
                       "using sym = 1")
        sym = 1

    if cplx and sym > 4:
        logger.warning("requested 8-fold permutational symmetry with complex "
                       "integrals: writing real integrals")
        cplx = False

    nmo = hcore.shape[0]

    # Generate M_{(ik),(lj)} = (ik|jl)
    if isinstance(chol, scipy.sparse.csr_array):
        eris = chol.dot(chol.conj().T).toarray().reshape((nmo, nmo, nmo, nmo))
    else:
        eris = chol.dot(chol.conj().T).reshape((nmo, nmo, nmo, nmo))

    if use_spinor:
        hcore = h1_spat2spin(hcore)
        eris = h2_spat2spin(eris)
        nmo = 2 * nmo

    with open(filename, 'w') as f:
        f.write(fcidump_header(sum(nelec), nmo, nelec[0] - nelec[1]))

        for i, k, l, j in product(range(nmo), repeat=4):
            if abs(eris[i, k, l, j]) <= tol:
                continue
            if not check_sym((i, k, j, l), nmo, sym):
                # Cholesky factorization can produce symmetry-forbidden entries
                logger.debug("%s not allowed by %d-fold symmetry", (i, j, k, l), sym)
                continue
            if not cplx and abs(eris[i, k, l, j].imag) > ctol:
                raise ValueError(
                    f"Found complex integrals with cplx=False at {(i, k, j, l)}"
                )
            f.write(fmt_integral(eris[i, k, l, j], i, k, j, l, cplx, paren=paren))

        _write_one_body_and_constant(f, hcore, enuc, tol, cplx, paren)


def write_fcidump_kpoint(filename, hcore, chol, enuc, nelec, qk_k2, tol=1e-8,
                         sym=1, paren=False, cplx=True, ctol=1e-12,
                         use_spinor=False) -> None:
    """
    Write an FCIDUMP file from a k-point Cholesky factorization.

    Parameters
    ----------
    filename : str or pathlib.Path
        File to write.
    hcore : numpy.ndarray
        One-body Hamiltonian, ``(nkpts, nmo, nmo)``.
    chol : sequence
        Cholesky matrices ``L[Q]``, each ``(nkpts, nmo**2 * nchol_Q)``.
    enuc : float
        Constant energy contribution.
    nelec : tuple(int, int)
        ``(nalpha, nbeta)``.
    qk_k2 : numpy.ndarray
        ``(q, k)`` to k-point map: ``Q = k_i - k_k + G``, ``qk_k2[iQ, ik_i] = ik_k``.
    tol : float, optional
        Only write integrals above this magnitude. Default 1e-8.
    sym : int, optional
        Write only symmetry-inequivalent ERIs. Default 1.
    paren : bool, optional
        Write complex numbers parenthesized.
    cplx : bool, optional
        Write in complex format. Default True.
    ctol : float, optional
        Largest imaginary part tolerated when `cplx` is False.
    use_spinor : bool, optional
        Not implemented here.

    Raises
    ------
    NotImplementedError
        If `use_spinor` is set.
    ValueError
        If `cplx` is False but the integrals have imaginary parts above `ctol`.

    Notes
    -----
    The combined orbital index is ``k * nmo + i``.
    """
    if use_spinor:
        raise NotImplementedError(
            "Conversion to a spinor basis is not implemented for k-point "
            "Hamiltonians. Write the FCIDUMP in the spatial-orbital basis "
            "instead, with use_spinor=False."
        )

    nkp, nmo = hcore.shape[:2]
    nmo_tot = nkp * nmo

    with open(filename, 'w') as f:
        f.write(fcidump_header(sum(nelec), nmo_tot, nelec[0] - nelec[1]))

        for iq, lq_vec in enumerate(chol):
            lq = np.asarray(lq_vec).reshape(nkp, nmo * nmo, -1)
            for ki in range(nkp):
                for kl in range(nkp):
                    # decompress Cholesky vectors to the physicists' v_ijkl =
                    #   <ij|kl>, i.e. c^\dag_i c^\dag_j c_l c_k, stored in
                    #   hermitian order (i,k)|(l,j)
                    eri = np.dot(lq[ki], lq[kl].conj().T)
                    if not cplx and abs(eri.imag).max() > ctol:
                        raise ValueError("Found complex integrals with cplx=False")

                    kk = qk_k2[iq, ki]
                    kj = qk_k2[iq, kl]
                    for i, k, l, j in product(range(nmo), repeat=4):
                        value = eri[i * nmo + k, l * nmo + j]
                        if abs(value) <= tol:
                            continue
                        I, K, L, J = ki * nmo + i, kk * nmo + k, kl * nmo + l, kj * nmo + j
                        if check_sym((I, K, J, L), nmo_tot, sym):
                            f.write(fmt_integral(value, I, K, J, L, cplx, paren=paren))
                        else:
                            # Cholesky can produce forbidden entries
                            logger.debug("%s not allowed by %d-fold symmetry",
                                         (I, J, K, L), sym)

        _write_one_body_and_constant(f, scipy.linalg.block_diag(*hcore), enuc, tol,
                                     cplx, paren)
