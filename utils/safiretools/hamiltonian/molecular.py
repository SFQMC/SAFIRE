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
`MolecularHamiltonian` — a Cholesky-factorized molecular Hamiltonian,

.. math:: \hat{H} = E_0 + \sum_{ij} h_{ij}\, c^\dagger_i c_j
          + \frac{1}{2}\sum_{\gamma} \Big(\sum_{ij} L^\gamma_{ij} c^\dagger_i c_j\Big)^2

stored densely as ``hcore`` plus the Cholesky matrix ``chol`` with elements
:math:`L_{(ij),\gamma}`, and written in the dense format the AFQMC executable's
``RealDenseHamiltonian`` reads.

The Cholesky decomposition itself is generated either from a PySCF ``mol``
object (`MolecularHamiltonian.from_pyscf`) or from integrals supplied directly
(`MolecularHamiltonian.from_integrals`).
"""

import logging
import time

import numpy as np
import scipy.linalg
import h5py as h5

from safiretools.hamiltonian.base import Hamiltonian, open_for_hamiltonian
from safiretools.hdf5 import from_complex, to_complex
from safiretools.types import SpinSymm

logger = logging.getLogger(__name__)

CHOLESKY_DATASET = 'Hamiltonian/DenseFactorized/L'
"""Where the dense Cholesky matrix lives; read by ``RealDenseHamiltonian``."""


class MolecularHamiltonian(Hamiltonian):
    r"""
    A dense, Cholesky-factorized molecular Hamiltonian.

    Parameters
    ----------
    hcore : numpy.ndarray
        One-body Hamiltonian in the working basis, ``(npol*nmo, npol*nmo)``
        with ``npol = 2`` for a noncollinear (GHF-like) Hamiltonian and 1
        otherwise.
    chol : numpy.ndarray
        Cholesky matrix :math:`L_{(ij),\gamma}`, shape ``(npol*nmo*npol*nmo,
        nchol)`` — note the *pair* index comes first.
    enuc : float, optional
        Constant (nuclear repulsion) energy. Default 0.0.
    nelec : tuple(int, int), optional
        ``(nup, ndown)``. Default ``(0, 0)``.
    spin_symm : SpinSymm or str or int, optional
        Spin symmetry. Default `SpinSymm.CLOSED`.
    ortho : numpy.ndarray, optional
        Transformation from the AO basis to the working basis, written as
        ``Hamiltonian/X`` when given.
    real_chol : bool, optional
        Force the on-disk dtype instead of taking it from the data. ``True``
        discards any imaginary part of the Cholesky matrix.
    """

    def __init__(self, hcore, chol, enuc=0.0, nelec=(0, 0),
                 spin_symm=SpinSymm.CLOSED, ortho=None, real_chol=None) -> None:
        super().__init__(spin_symm=spin_symm)

        self.hcore = np.asarray(hcore)
        self.chol = np.asarray(chol)
        self.enuc = float(np.real(enuc))
        self.nelec = tuple(nelec)
        self.ortho = ortho
        self.real_chol = real_chol

        if self.hcore.ndim != 2 or self.hcore.shape[0] != self.hcore.shape[1]:
            raise ValueError(f"hcore must be a square matrix, got shape {self.hcore.shape}")

        if self.hcore.shape[0] % self.npol:
            raise ValueError(
                f"hcore has shape {self.hcore.shape}, which is not divisible by the "
                f"{self.npol} spin polarizations of a {self.spin_symm.label} Hamiltonian"
            )
        self._nmo = self.hcore.shape[0] // self.npol

        # the Cholesky matrix may be stored in the spatial-orbital basis even when
        #   hcore is in the spin-orbital one
        valid_rows = {self._nmo**2, (self.npol * self._nmo)**2}
        if self.chol.shape[0] not in valid_rows:
            raise ValueError(
                f"Cholesky matrix has {self.chol.shape[0]} rows; for nmo={self._nmo} "
                f"and npol={self.npol} expected one of {sorted(valid_rows)}"
            )

    @property
    def npol(self) -> int:
        """2 for a noncollinear Hamiltonian (spin-orbital basis), else 1."""
        return 2 if self.spin_symm is SpinSymm.NONCOLLINEAR else 1

    @property
    def nmo(self) -> int:
        """Number of orbitals, i.e. spatial orbitals unless noncollinear."""
        return self._nmo

    @property
    def nchol(self) -> int:
        """Number of Cholesky vectors."""
        return self.chol.shape[-1]

    @property
    def complex_chol(self) -> bool:
        """
        Whether the Cholesky matrix is written as complex, which is what
        ``Hamiltonian/ComplexIntegrals`` records.

        .. note:: The AFQMC executable's ``RealDenseHamiltonian`` reads
                  ``DenseFactorized/L`` into a real array, so it cannot consume
                  a complex-Cholesky file. ``real_chol=False`` has always
                  produced one anyway; it is preserved here rather than
                  silently changed.
        """
        if self.real_chol is not None:
            return not self.real_chol
        return bool(np.any(np.iscomplex(self.chol)))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def from_integrals(cls, hcore, chol=None, eri=None, enuc=0.0, nelec=(0, 0),
                       spin_symm=SpinSymm.CLOSED, cholesky_tol=1e-6,
                       verbose=False) -> "MolecularHamiltonian":
        r"""
        Build a Hamiltonian from integrals supplied directly.

        Parameters
        ----------
        hcore : numpy.ndarray
            One-body Hamiltonian, ``(nmo, nmo)``.
        chol : numpy.ndarray, optional
            Cholesky vectors :math:`L_{\gamma,(ij)}`, shape ``(nchol, nmo**2)``.
            Exactly one of `chol` and `eri` is required.
        eri : numpy.ndarray, optional
            Coulomb repulsion tensor in chemists' notation, ``(ij|kl)``, shape
            ``(nmo, nmo, nmo, nmo)``. Decomposed here if given.
        enuc : float, optional
            Constant energy contribution. Default 0.0.
        nelec : tuple(int, int), optional
            ``(nup, ndown)``. Default ``(0, 0)``.
        spin_symm : SpinSymm or str or int, optional
            Spin symmetry. Default `SpinSymm.CLOSED`.
        cholesky_tol : float, optional
            Tolerance for decomposing `eri`. Default 1e-6.
        verbose : bool, optional
            Log the decomposition's convergence.

        Returns
        -------
        MolecularHamiltonian

        Raises
        ------
        ValueError
            If neither or both of `chol` and `eri` are given, or if either has
            the wrong shape.
        """
        if (chol is None) == (eri is None):
            raise ValueError("Provide exactly one of 'chol' or 'eri'.")

        nmo = hcore.shape[-1]

        if eri is not None:
            if eri.shape != (nmo, nmo, nmo, nmo):
                raise ValueError(
                    f"eri has shape {eri.shape}, expected ({nmo}, {nmo}, {nmo}, {nmo})"
                )
            logger.info("generating Cholesky vectors from the Coulomb repulsion tensor")
            chol = modified_cholesky_direct(
                eri.reshape(nmo**2, nmo**2), tol=cholesky_tol, verbose=verbose, cmax=4)
        elif chol.shape[1] != nmo**2:
            raise ValueError(
                f"chol has shape {chol.shape}, expected (nchol, {nmo**2}) — "
                "the orbital-pair index must come second"
            )

        # go from L_{gamma,(ij)} to L_{(ij),gamma}
        return cls(hcore=hcore, chol=chol.T, enuc=enuc, nelec=nelec,
                   spin_symm=spin_symm)

    @classmethod
    def from_pyscf(cls, scf_data, chol_cut=1e-5, cas=None, ortho_ao=False,
                   df=False, spin_symm=None, real_chol=None,
                   verbose=False) -> "MolecularHamiltonian":
        """
        Build a Hamiltonian from a PySCF molecular SCF calculation.

        Parameters
        ----------
        scf_data : dict
            Unpacked PySCF checkpoint, as produced by
            ``afqmctools.utils.pyscf_utils.load_from_pyscf_chk_mol``. Uses the
            keys ``'hcore'``, ``'mo_coeff'``, ``'X'``, ``'mol'``, ``'nelec'``,
            ``'norb'``, ``'walker_type'`` and (optionally) ``'df_ints'``.
        chol_cut : float, optional
            Cholesky decomposition accuracy. Default 1e-5.
        cas : tuple(int, int), optional
            ``(nelecas, ncas)`` active space; core orbitals are frozen into the
            one-body term and the constant. Incompatible with `ortho_ao`.
        ortho_ao : bool, optional
            Work in the orthogonalized AO basis rather than the MO basis.
            Required for UHF/GHF references.
        df : bool, optional
            Use density-fitted integrals from the checkpoint, if present.
        spin_symm : SpinSymm or str or int, optional
            Overrides ``scf_data['walker_type']``.
        real_chol : bool, optional
            Force the on-disk dtype. See the class docstring.
        verbose : bool, optional
            Log timing and convergence progress.

        Returns
        -------
        MolecularHamiltonian

        Raises
        ------
        ValueError
            If the reference is UHF/GHF and `ortho_ao` is not set, if `cas` and
            `ortho_ao` are combined, or if the Hamiltonian is noncollinear but
            `spin_symm` says otherwise.
        """
        if spin_symm is None:
            spin_symm = scf_data["walker_type"]
        spin_symm = SpinSymm.from_input(spin_symm)

        hcore = scf_data['hcore']
        mol = scf_data['mol']
        df_ints = scf_data.get('df_ints', None)

        X, (nfzc, nfzv) = _transform_from_scf_data(scf_data, ortho_ao, cas)
        nbasis = X.shape[-1]

        if hcore.shape == (2 * X.shape[0], 2 * X.shape[0]):
            if spin_symm is not SpinSymm.NONCOLLINEAR:
                raise ValueError(
                    f"Hamiltonian is noncollinear but spin_symm {spin_symm} is not"
                )
            Xspin = np.kron(np.eye(2), X)
            h1e = Xspin.conj().T @ hcore @ Xspin
        else:
            h1e = X.conj().T @ hcore @ X

        logger.info("number of basis functions: %d", nbasis)

        if df_ints is not None and df:
            logger.info("using DF integrals from the checkpoint file")
            chol_vecs = df_ints
            if chol_vecs.shape[1] != nbasis * nbasis:
                raise ValueError(
                    f"DF integrals have shape {chol_vecs.shape}, expected "
                    f"(nchol, {nbasis * nbasis})"
                )
        else:
            logger.info("performing modified Cholesky decomposition on the ERI tensor")
            chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)

        logger.info("orthogonalising Cholesky vectors")
        start = time.time()
        chol_trans = transform_cholesky(chol_vecs, X)
        logger.info("time to orthogonalise: %s s", time.time() - start)

        enuc = mol.energy_nuc()
        nelec = mol.nelec

        if (nfzc, nfzv) != (0, 0):
            h1e, chol_trans, enuc = freeze_core(
                h1e, chol_trans, enuc, nfzc, nbasis - nfzv - nfzc, verbose=verbose)
            h1e = h1e[0]

        return cls(
            hcore=h1e,
            chol=chol_trans.T,   # want L_{(ij),gamma}
            enuc=enuc,
            nelec=nelec,
            spin_symm=spin_symm,
            ortho=X[:, nfzc:nbasis - nfzv],
            real_chol=real_chol,
        )

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------

    def to_hdf5(self, path) -> None:
        """
        Write this Hamiltonian in the dense format ``RealDenseHamiltonian``
        reads.

        Parameters
        ----------
        path : str or pathlib.Path
            HDF5 file to write into. Created if it does not exist. A Hamiltonian
            already in the file is replaced; everything else — notably a
            ``Wavefunction`` — is left alone, so a Hamiltonian and a
            wavefunction can share one file in either order.
        """
        with open_for_hamiltonian(path) as fh5:
            write_dense_hamiltonian(
                fh5,
                hcore=self.hcore,
                chol=self.chol,
                nelec=self.nelec,
                nmo=self.nmo,
                enuc=self.enuc,
                complex_chol=self.complex_chol,
                ortho=self.ortho,
            )

    @classmethod
    def _read_hdf5(cls, path, fmt: str) -> "MolecularHamiltonian":
        """
        Read a dense Hamiltonian written by `to_hdf5`.

        Notes
        -----
        The dense format records the number of spin polarizations but not
        whether a single-polarization Hamiltonian is closed- or open-shell — the
        AFQMC executable takes that from the wavefunction, not the Hamiltonian.
        `spin_symm` therefore comes back `SpinSymm.NONCOLLINEAR` or
        `SpinSymm.CLOSED`; set it explicitly if the distinction matters.
        """
        enuc, hcore, chol, nelec, nmo = read_dense_hamiltonian(path)

        npol = hcore.shape[0] // nmo
        spin_symm = SpinSymm.NONCOLLINEAR if npol == 2 else SpinSymm.CLOSED

        return cls(hcore=hcore, chol=chol, enuc=enuc, nelec=nelec,
                   spin_symm=spin_symm)


# ----------------------------------------------------------------------
# the dense on-disk format, shared with the supercell periodic Hamiltonian
# ----------------------------------------------------------------------

def write_dense_hamiltonian(fh5, hcore, chol, nelec, nmo, enuc=0.0,
                            complex_chol=None, ortho=None) -> None:
    r"""
    Write a dense Cholesky-factorized Hamiltonian into the open HDF5 file
    `fh5`.

    Parameters
    ----------
    fh5 : h5py.File or h5py.Group
        Destination. Existing datasets are replaced.
    hcore : numpy.ndarray
        One-body Hamiltonian.
    chol : numpy.ndarray
        Cholesky matrix :math:`L_{(ij),\gamma}`.
    nelec : tuple(int, int)
        ``(nup, ndown)``.
    nmo : int
        Number of orbitals.
    enuc : float, optional
        Constant energy. Default 0.0.
    complex_chol : bool, optional
        On-disk dtype of the Cholesky matrix, and the value recorded in
        ``ComplexIntegrals``. Taken from the data when omitted; when False, only
        the real part of `chol` is written.
    ortho : numpy.ndarray, optional
        Basis transformation, written as ``Hamiltonian/X``.

    Notes
    -----
    ``hcore``'s dtype follows its own values rather than `complex_chol`.
    afqmctools decided it with ``numpy.all(numpy.iscomplex(hcore))``, which is
    false for any Hermitian matrix — its diagonal is real — so a genuinely
    complex ``hcore`` had its imaginary part silently discarded.
    """
    if complex_chol is None:
        complex_chol = bool(np.any(np.iscomplex(chol)))

    _write(fh5, 'Hamiltonian/DenseFactorized/L',
           to_complex(chol) if complex_chol else np.real(chol))

    complex_hcore = bool(np.any(np.iscomplex(hcore)))
    _write(fh5, 'Hamiltonian/hcore',
           to_complex(hcore) if complex_hcore else np.real(hcore))

    _write(fh5, 'Hamiltonian/Energies', np.array([enuc, 0.], dtype=np.float64))
    _write(fh5, 'Hamiltonian/dims',
           np.array([0, 0, 0, nmo, nelec[0], nelec[1], 0, chol.shape[-1]],
                    dtype=np.int32))
    _write(fh5, 'Hamiltonian/ComplexIntegrals',
           np.array([int(complex_chol)], dtype=np.int32))

    if ortho is not None:
        _write(fh5, 'Hamiltonian/X', np.asarray(ortho))


def read_dense_hamiltonian(path):
    r"""
    Read a dense Cholesky-factorized Hamiltonian written by
    `write_dense_hamiltonian`.

    Parameters
    ----------
    path : str or pathlib.Path
        HDF5 file to read.

    Returns
    -------
    enuc : float
        Constant energy contribution.
    hcore : numpy.ndarray
        One-body Hamiltonian, ``(npol*nmo, npol*nmo)``.
    chol : numpy.ndarray
        Cholesky matrix :math:`L_{(ij),\gamma}`.
    nelec : tuple(int, int)
        ``(nup, ndown)``.
    nmo : int
        Number of orbitals, as recorded in ``dims``.

    Raises
    ------
    ValueError
        If the file holds no Cholesky matrix, or if ``dims`` is malformed.
    """
    with h5.File(path, 'r') as fh5:

        dims = fh5['Hamiltonian/dims'][...]
        if len(dims) != 8:
            raise ValueError(
                f"Hamiltonian/dims in {path} has length {len(dims)}, expected 8"
            )

        nmo = int(dims[3])
        nelec = (int(dims[4]), int(dims[5]))
        nchol = int(dims[-1])
        enuc = float(fh5['Hamiltonian/Energies'][...][0])

        chol = from_complex(fh5[CHOLESKY_DATASET][...]).reshape(-1, nchol)
        hcore = from_complex(fh5['Hamiltonian/hcore'][...])

    return enuc, hcore, chol, nelec, nmo


def _write(fh5, name, data) -> None:
    """Create `name` in `fh5`, replacing it if it already exists."""
    if name in fh5:
        del fh5[name]
    fh5.create_dataset(name, data=data)


# ----------------------------------------------------------------------
# Cholesky decomposition and basis transformation
# ----------------------------------------------------------------------

def modified_cholesky_direct(M, tol=1e-5, verbose=False, cmax=20):
    """
    Modified Cholesky decomposition of a matrix.

    See, e.g. :cite:`motta_initio_2018`.

    Parameters
    ----------
    M : numpy.ndarray
        Positive semi-definite, symmetric matrix.
    tol : float, optional
        Accuracy desired. Default 1e-5.
    verbose : bool, optional
        Log convergence progress.
    cmax : int, optional
        Store at most ``cmax * sqrt(M.shape[0])`` Cholesky vectors.

    Returns
    -------
    numpy.ndarray
        Cholesky vectors, shape ``(nchol, M.shape[0])``.
    """
    delta = np.copy(M.diagonal())
    nchol_max = min(int(cmax * M.shape[0]**0.5), M.shape[1])

    nu = np.argmax(np.abs(delta))
    delta_max = delta[nu]

    if verbose:
        logger.info("performing Cholesky decomposition with tolerance %s", tol)
        logger.info("max number of cholesky vectors = %d", nchol_max)

    Mapprox = np.zeros(M.shape[0], dtype=M.dtype)
    chol_vecs = np.zeros((nchol_max, M.shape[0]), dtype=M.dtype)
    nchol = 0
    chol_vecs[0] = np.copy(M[:, nu]) / delta_max**0.5

    # -1 because we already have one vector
    while abs(delta_max) > tol and nchol < nchol_max - 1:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol].conj()
        delta = M.diagonal() - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        nchol += 1
        Munu0 = np.dot(chol_vecs[:nchol, nu].conj(), chol_vecs[:nchol, :])
        chol_vecs[nchol] = (M[:, nu] - Munu0) / delta_max**0.5
        if verbose:
            logger.info("cholesky iteration %d: max residual %13.8e", nchol, delta_max)

    return np.array(chol_vecs[:nchol])


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """
    Modified Cholesky decomposition of the PySCF ERI tensor, one shell chunk at
    a time.

    See, e.g. :cite:`motta_initio_2018`. Only works for molecular systems.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        PySCF mol object.
    max_error : float, optional
        Accuracy desired. Default 1e-6.
    verbose : bool, optional
        Log convergence progress.
    cmax : int, optional
        ``nchol_max = cmax * nao``; controls the Cholesky vector buffer size.

    Returns
    -------
    numpy.ndarray
        Cholesky vectors in the AO basis, shape ``(nchol, nao*nao)``.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    chol_vecs = np.zeros((nchol_max, nao * nao))

    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)

    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di = buf.shape[0]
        diag[ndiag:ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao

    nu = np.argmax(diag)
    delta_max = diag[nu]

    if verbose:
        logger.info("generating Cholesky decomposition of ERIs")
        logger.info("max number of cholesky vectors = %d", nchol_max)

    def shell_of(index):
        """The shell index the AO index `index` falls in."""
        shell = np.searchsorted(dims, index)
        if dims[shell] != index and index != 0:
            shell -= 1
        return shell

    j, l = nu // nao, nu % nao
    sj, sl = shell_of(j), shell_of(l)

    Mapprox = np.zeros(nao * nao)
    eri_col = mol.intor('int2e_sph',
                        shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # M'_ii = \sum_x L_i^x L_i^x, then D_ii = M_ii - M'_ii
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])

        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        j, l = nu // nao, nu % nao
        sj, sl = shell_of(j), shell_of(l)

        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)

        # updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[:nchol + 1, nu], chol_vecs[:nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / delta_max**0.5
        nchol += 1

        if verbose:
            logger.info("cholesky iteration %d: max residual %13.8e", nchol, delta_max)

    return chol_vecs[:nchol]


def transform_cholesky(chol, C):
    """
    Apply the basis rotation `C` to the Cholesky vectors `chol`.

    Parameters
    ----------
    chol : numpy.ndarray
        Cholesky vectors, shape ``(nchol, nao*nao)``.
    C : numpy.ndarray
        Transformation matrix, ``(nao, nmo)``.

    Returns
    -------
    numpy.ndarray
        Transformed vectors, shape ``(nchol, nmo*nmo)``.

    Notes
    -----
    Transforms in place through a flat view of `chol`, so that ``nao > nmo``
    needs no second buffer. `chol` is overwritten.
    """
    nao, nmo = C.shape
    nik = nmo * nmo
    nchol = chol.shape[0]

    chol_ = chol.ravel()
    for i in range(nchol):
        cv = chol[i].reshape(nao, nao)
        half = np.dot(cv, C)
        # if nao < nmo we overwrite the data
        chol_[i * nik:(i + 1) * nik] = np.dot(C.T, half).ravel()

    return chol_[:nchol * nik].reshape((nchol, nik))


def _transform_from_scf_data(scf_data, ortho_ao, cas=None):
    """
    Choose the working basis and the frozen-orbital counts from a PySCF
    checkpoint.

    Returns
    -------
    C : numpy.ndarray
        Transformation into the working basis.
    (nfzc, nfzv) : tuple(int, int)
        Numbers of frozen core and virtual orbitals.

    Raises
    ------
    ValueError
        If `cas` is combined with `ortho_ao`, or if the reference is UHF/GHF and
        `ortho_ao` is not set.
    """
    C = scf_data['mo_coeff']

    if ortho_ao:
        if cas is not None:
            raise ValueError("cas and ortho_ao cannot be used at the same time")
        return scf_data['X'], (0, 0)

    if C.ndim == 3 or C.shape[0] == 2 * scf_data["norb"]:
        raise ValueError(
            "UHF or GHF molecular orbital bases are not supported. Use ortho_ao."
        )

    if cas is None:
        return C, (0, 0)

    nfzc = (sum(scf_data["nelec"]) - cas[0]) // 2
    ncas = cas[1]
    nmo = C.shape[-1]
    if ncas == -1:
        ncas = nmo - nfzc

    return C, (nfzc, nmo - ncas - nfzc)


def freeze_core(h1e, chol, ecore, nc, ncas, verbose=True):
    """
    Freeze `nc` core orbitals into the one-body term and the constant energy,
    keeping `ncas` active orbitals.

    Parameters
    ----------
    h1e : numpy.ndarray
        One-body Hamiltonian, ``(nbasis, nbasis)``.
    chol : numpy.ndarray
        Cholesky vectors, ``(nchol, nbasis*nbasis)``.
    ecore : float
        Constant energy before freezing.
    nc : int
        Number of frozen core orbitals.
    ncas : int
        Number of active orbitals.
    verbose : bool, optional
        Log the frozen-core energy breakdown.

    Returns
    -------
    h1e : numpy.ndarray
        Active-space one-body Hamiltonian, ``(2, ncas, ncas)`` (up and down).
    chol : numpy.ndarray
        Active-space Cholesky vectors.
    efzc : float
        Total frozen-core energy.

    Raises
    ------
    ValueError
        If more orbitals would be frozen than the basis has.
    """
    nbasis = h1e.shape[-1]

    if nbasis - nc - ncas < 0:
        raise ValueError(
            f"freeze_core: ncore = {nc}, nactive = {ncas}, nbasis = {nbasis}:\n"
            "Can't freeze more orbitals than available basis set functions"
        )

    chol = chol.reshape((-1, nbasis, nbasis))
    psi = np.identity(nbasis)[:, :nc]
    Gcore = gab(psi, psi)
    efzc = local_energy_generic_cholesky(h1e, chol, [Gcore, Gcore], ecore)
    hc_a, hc_b = core_contribution_cholesky(chol, [Gcore, Gcore])

    h1e = np.array([h1e + 2 * hc_a, h1e + 2 * hc_b])
    h1e = h1e[:, nc:nc + ncas, nc:nc + ncas]

    nchol = chol.shape[0]
    chol = chol[:, nc:nc + ncas, nc:nc + ncas].reshape((nchol, -1))

    if verbose:
        logger.info("number of active orbitals: %d", ncas)
        logger.info("freezing %d core electrons and %d virtuals",
                    2 * nc, nbasis - nc - ncas)
        logger.info("total frozen core energy: %s", efzc[0])
        logger.info("E0 (input): %13.8e", ecore)
        logger.info("frozen 1-body contribution: %s", efzc[1] - ecore)
        logger.info("frozen 2-body contribution: %s", efzc[2])

    return h1e, chol, efzc[0]


def local_energy_generic_cholesky(h1e, chol_vecs, G, ecore):
    r"""
    Local energy for a generic two-body Hamiltonian, from Cholesky-decomposed
    two-electron integrals.

    Parameters
    ----------
    h1e : numpy.ndarray
        One-body Hamiltonian.
    chol_vecs : numpy.ndarray
        Cholesky vectors, ``(nchol, nbasis, nbasis)``.
    G : list of numpy.ndarray
        Up and down Green's functions.
    ecore : float
        Constant energy contribution.

    Returns
    -------
    (E, T, V) : tuple
        Total, one-body and two-body energies.
    """
    e1b = np.sum(h1e * G[0]) + np.sum(h1e * G[1])

    ecoul_uu = ecoul_dd = ecoul_ud = ecoul_du = 0
    exx_uu = exx_dd = 0
    for c in chol_vecs:
        ecoul_uu += np.sum(c * G[0]) * np.sum(c.conj().T * G[0])
        ecoul_dd += np.sum(c * G[1]) * np.sum(c.conj().T * G[1])
        ecoul_ud += np.sum(c * G[0]) * np.sum(c.conj().T * G[1])
        ecoul_du += np.sum(c * G[1]) * np.sum(c.conj().T * G[0])
        exx_uu += np.einsum('ij,ji->', np.dot(c.T, G[0]), np.dot(c.conj(), G[0]))
        exx_dd += np.einsum('ij,ji->', np.dot(c.T, G[1]), np.dot(c.conj(), G[1]))

    e2b = (0.5 * (ecoul_uu - exx_uu) + 0.5 * (ecoul_dd - exx_dd)
           + 0.5 * ecoul_ud + 0.5 * ecoul_du)

    return (e1b + e2b + ecore, e1b + ecore, e2b)


def core_contribution_cholesky(chol_vecs, G):
    """
    The frozen-core contribution to the one-body Hamiltonian, per spin sector.

    Parameters
    ----------
    chol_vecs : numpy.ndarray
        Cholesky vectors, ``(nchol, nbasis, nbasis)``.
    G : list of numpy.ndarray
        Up and down core Green's functions.

    Returns
    -------
    tuple(numpy.ndarray, numpy.ndarray)
        Up and down core contributions.
    """
    cv = chol_vecs

    hca_j = np.einsum('l,lij->ij', np.sum(cv * G[0], axis=(1, 2)), cv)
    hca_k = 0.5 * np.einsum('lrq,lsq->rs', np.einsum('lpr,pq->lrq', cv, G[0]), cv)

    hcb_j = np.einsum('l,lij->ij', np.sum(cv * G[1], axis=(1, 2)), cv)
    hcb_k = 0.5 * np.einsum('lrq,lsq->rs', np.einsum('lpr,pq->lrq', cv, G[1]), cv)

    return (hca_j - hca_k, hcb_j - hcb_k)


def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : numpy.ndarray
        Matrix representation of the bra used to construct G.
    B : numpy.ndarray
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : numpy.ndarray
        (One minus) the Green's function.
    """
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    return B.dot(inv_O.dot(A.conj().T))
