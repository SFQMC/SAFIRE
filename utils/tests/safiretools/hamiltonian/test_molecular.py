# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`MolecularHamiltonian`: the dense on-disk format and the Cholesky helpers."""

import h5py as h5
import numpy as np
import pytest

from safiretools import SpinSymm
from safiretools.hamiltonian.base import Hamiltonian, hamiltonian_format
from safiretools.hamiltonian.molecular import (
    MolecularHamiltonian,
    chunked_cholesky,
    freeze_core,
    modified_cholesky_direct,
    transform_cholesky,
)


@pytest.fixture
def random_hamiltonian():
    """A small positive-definite ERI tensor with a Hermitian one-body part."""
    rng = np.random.default_rng(7)
    nmo = 5

    hcore = rng.random((nmo, nmo))
    hcore = 0.5 * (hcore + hcore.T)

    chol = rng.random((9, nmo * nmo))
    eri = np.dot(chol.T, chol).reshape((nmo,) * 4)

    return nmo, hcore, chol, eri


class TestFromIntegrals:

    def test_cholesky_vectors_are_transposed_into_pair_major_order(self,
                                                                   random_hamiltonian):
        nmo, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian.from_integrals(hcore, chol=chol, enuc=1.5)

        assert hamiltonian.chol.shape == (nmo * nmo, chol.shape[0])
        assert np.allclose(hamiltonian.chol, chol.T)
        assert hamiltonian.nmo == nmo
        assert hamiltonian.nchol == chol.shape[0]
        assert hamiltonian.enuc == 1.5

    def test_an_eri_tensor_is_decomposed(self, random_hamiltonian):
        nmo, hcore, _, eri = random_hamiltonian
        hamiltonian = MolecularHamiltonian.from_integrals(hcore, eri=eri,
                                                          cholesky_tol=1e-10)

        reconstructed = (hamiltonian.chol @ hamiltonian.chol.conj().T)
        assert np.allclose(reconstructed.reshape((nmo,) * 4), eri, atol=1e-6)

    def test_exactly_one_of_chol_and_eri_is_required(self, random_hamiltonian):
        _, hcore, chol, eri = random_hamiltonian

        with pytest.raises(ValueError, match="exactly one"):
            MolecularHamiltonian.from_integrals(hcore)
        with pytest.raises(ValueError, match="exactly one"):
            MolecularHamiltonian.from_integrals(hcore, chol=chol, eri=eri)

    def test_a_misshaped_eri_tensor_is_rejected(self, random_hamiltonian):
        _, hcore, _, eri = random_hamiltonian

        with pytest.raises(ValueError, match="eri has shape"):
            MolecularHamiltonian.from_integrals(hcore, eri=eri[..., :2])

    def test_misshaped_cholesky_vectors_are_rejected(self, random_hamiltonian):
        _, hcore, chol, _ = random_hamiltonian

        with pytest.raises(ValueError, match="orbital-pair index must come second"):
            MolecularHamiltonian.from_integrals(hcore, chol=chol.T)


class TestConstruction:

    def test_hcore_must_be_square(self, random_hamiltonian):
        nmo, _, chol, _ = random_hamiltonian

        with pytest.raises(ValueError, match="square matrix"):
            MolecularHamiltonian(hcore=np.zeros((nmo, nmo + 1)), chol=chol.T)

    def test_cholesky_rows_must_match_the_basis(self, random_hamiltonian):
        nmo, hcore, _, _ = random_hamiltonian

        with pytest.raises(ValueError, match="Cholesky matrix has"):
            MolecularHamiltonian(hcore=hcore, chol=np.zeros((nmo * nmo + 1, 3)))

    def test_a_noncollinear_hcore_spans_two_polarizations(self, random_hamiltonian):
        nmo, _, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(
            hcore=np.zeros((2 * nmo, 2 * nmo)), chol=chol.T,
            spin_symm=SpinSymm.NONCOLLINEAR)

        assert hamiltonian.npol == 2
        assert hamiltonian.nmo == nmo


class TestHdf5:

    def test_round_trip(self, random_hamiltonian, tmp_path):
        _, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian.from_integrals(
            hcore, chol=chol, enuc=1.5, nelec=(3, 2))

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)
        assert hamiltonian_format(path) == 'dense'

        restored = Hamiltonian.from_hdf5(path)
        assert isinstance(restored, MolecularHamiltonian)
        assert np.allclose(restored.hcore, hamiltonian.hcore)
        assert np.allclose(restored.chol, hamiltonian.chol)
        assert restored.enuc == hamiltonian.enuc
        assert restored.nelec == (3, 2)

    def test_real_integrals_are_written_real(self, random_hamiltonian, tmp_path):
        _, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'ham.h5'
        MolecularHamiltonian.from_integrals(hcore, chol=chol).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/hcore'].ndim == 2
            assert fh5['Hamiltonian/DenseFactorized/L'].ndim == 2
            assert fh5['Hamiltonian/ComplexIntegrals'][0] == 0

    def test_a_complex_hcore_is_not_truncated(self, random_hamiltonian, tmp_path):
        """
        afqmctools decided hcore's dtype with ``all(iscomplex(hcore))``, which is
        false for any Hermitian matrix, so a complex hcore lost its imaginary
        part on write.
        """
        nmo, hcore, chol, _ = random_hamiltonian
        imag = np.triu(np.ones((nmo, nmo)), 1)
        complex_hcore = hcore + 1j * (imag - imag.T)

        hamiltonian = MolecularHamiltonian.from_integrals(complex_hcore, chol=chol)
        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/hcore'].shape[-1] == 2

        assert np.allclose(Hamiltonian.from_hdf5(path).hcore, complex_hcore)

    def test_the_ortho_matrix_is_written_when_given(self, random_hamiltonian, tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        ortho = np.eye(nmo)

        path = tmp_path / 'ham.h5'
        MolecularHamiltonian(hcore=hcore, chol=chol.T, ortho=ortho).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert np.allclose(fh5['Hamiltonian/X'][...], ortho)

    def test_a_file_with_no_cholesky_matrix_is_rejected(self, tmp_path):
        path = tmp_path / 'ham.h5'
        with h5.File(path, 'w') as fh5:
            fh5.create_dataset('Hamiltonian/DenseFactorized/L', data=np.zeros((4, 2)))
            fh5.create_dataset('Hamiltonian/dims', data=np.zeros(8, dtype=np.int32))

        with h5.File(path, 'a') as fh5:
            del fh5['Hamiltonian/DenseFactorized/L']

        with pytest.raises(ValueError, match="no Hamiltonian in a format"):
            Hamiltonian.from_hdf5(path)


class TestModifiedCholesky:

    def test_it_reproduces_a_positive_semidefinite_matrix(self):
        rng = np.random.default_rng(3)
        A = rng.random((12, 5))
        M = A @ A.T

        chol = modified_cholesky_direct(M, tol=1e-10, cmax=20)
        assert chol.shape[0] >= 5
        assert np.allclose(chol.T @ chol, M, atol=1e-6)

    def test_the_vector_count_is_capped(self):
        """
        The buffer holds ``min(cmax*sqrt(n), ncols)`` vectors and the loop stops
        one short of filling it, so a full-rank matrix is not reproduced exactly.
        """
        rng = np.random.default_rng(3)
        A = rng.random((12, 40))
        M = A @ A.T

        chol = modified_cholesky_direct(M, tol=1e-14, cmax=20)
        assert chol.shape[0] == 11


@pytest.mark.pyscf
class TestFromPyscf:
    """The PySCF path, checked against directly computed integrals."""

    def test_neon_hamiltonian_matches_the_scf_integrals(self, neon_atom, neon_rhf,
                                                        tmp_path):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        mf, _ = neon_rhf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)

        hamiltonian = MolecularHamiltonian.from_pyscf(scf_data, chol_cut=1e-5)

        C = mf.mo_coeff
        expected_hcore = C.conj().T @ mf.get_hcore() @ C
        assert np.allclose(hamiltonian.hcore, expected_hcore)
        assert np.isclose(hamiltonian.enuc, neon_atom.energy_nuc())
        assert hamiltonian.nmo == C.shape[-1]

        # the factorization must reproduce the MO-basis ERIs
        nmo = hamiltonian.nmo
        eri_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C,
                           neon_atom.intor('int2e', aosym='s1'), C, C)
        reconstructed = (hamiltonian.chol @ hamiltonian.chol.conj().T)
        assert np.allclose(reconstructed.reshape((nmo,) * 4), eri_mo, atol=1e-4)

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)
        assert hamiltonian_format(path) == 'dense'
        assert np.allclose(Hamiltonian.from_hdf5(path).hcore, hamiltonian.hcore)

    def test_chunked_cholesky_reproduces_the_eri_tensor(self, neon_atom):
        chol = chunked_cholesky(neon_atom, max_error=1e-6)
        eri = neon_atom.intor('int2e', aosym='s1').reshape(25, 25)

        assert np.allclose(chol.T @ chol, eri, atol=1e-5)

    def test_frozen_core_matches_pyscf_casscf(self, neon_atom, neon_rhf, neon_casscf):
        mf, _ = neon_rhf
        C = mf.mo_coeff

        chol = chunked_cholesky(neon_atom, max_error=1e-5)
        transform_cholesky(chol, C)
        h1e = C.T @ mf.get_hcore() @ C

        h1e, chol, efzc = freeze_core(h1e, chol, 0, 1, 4, verbose=False)

        h1eff, ecore = neon_casscf.get_h1eff()
        assert h1e.shape == (2, 4, 4)
        assert np.isclose(efzc, ecore)
        assert np.allclose(h1eff, h1e, atol=1e-8, rtol=1e-5)

    def test_freezing_more_orbitals_than_exist_is_rejected(self, neon_atom, neon_rhf):
        mf, _ = neon_rhf
        C = mf.mo_coeff
        chol = chunked_cholesky(neon_atom, max_error=1e-5)
        transform_cholesky(chol, C)
        h1e = C.T @ mf.get_hcore() @ C

        with pytest.raises(ValueError, match="Can't freeze more orbitals"):
            freeze_core(h1e, chol, 0, 3, 4, verbose=False)

    def test_a_uhf_reference_needs_ortho_ao(self, neon_rhf):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        mf, _ = neon_rhf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)
        scf_data['mo_coeff'] = np.array([mf.mo_coeff, mf.mo_coeff])

        with pytest.raises(ValueError, match="Use ortho_ao"):
            MolecularHamiltonian.from_pyscf(scf_data)

    def test_cas_and_ortho_ao_are_mutually_exclusive(self, neon_rhf):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        mf, _ = neon_rhf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)

        with pytest.raises(ValueError, match="cannot be used at the same time"):
            MolecularHamiltonian.from_pyscf(scf_data, cas=(4, 4), ortho_ao=True)


class TestRealAndComplexAreToldApartByRank:
    """
    `safiretools.hdf5.to_complex` appends a trailing axis of length 2, so the two
    on-disk layouts differ in *rank*. These cases check that end-to-end through a
    Hamiltonian round trip; `tests/safiretools/test_hdf5.py` checks the helper
    itself.
    """

    @staticmethod
    def _round_trip(tmp_path, hcore, chol, name):
        hamiltonian = MolecularHamiltonian.from_integrals(
            hcore, chol=chol, enuc=0.5, nelec=(1, 1))
        path = tmp_path / name
        hamiltonian.to_hdf5(path)

        restored = Hamiltonian.from_hdf5(path)
        assert np.allclose(restored.hcore, hamiltonian.hcore)
        assert np.allclose(restored.chol, hamiltonian.chol)
        return restored

    def test_two_orbitals(self, tmp_path):
        """`nmo == 2` makes a real hcore look interleaved by the old rule."""
        hcore = np.array([[-1.25, 0.0], [0.0, -0.48]])
        chol = np.arange(3 * 4, dtype=float).reshape(3, 4)   # (nchol=3, nmo**2)

        restored = self._round_trip(tmp_path, hcore, chol, 'nmo2.h5')
        assert restored.nmo == 2
        assert restored.hcore.shape == (2, 2)
        assert not np.iscomplexobj(restored.hcore)

    def test_two_cholesky_vectors(self, tmp_path):
        """`nchol == 2` makes a real Cholesky matrix look interleaved."""
        hcore = np.diag([-1.0, -0.5, -0.25])
        chol = np.arange(2 * 9, dtype=float).reshape(2, 9)   # (nchol=2, nmo**2)

        restored = self._round_trip(tmp_path, hcore, chol, 'nchol2.h5')
        assert restored.nchol == 2
        assert restored.chol.shape == (9, 2)
        assert not np.iscomplexobj(restored.chol)

    def test_complex_data_is_still_recognized(self, tmp_path):
        hcore = np.array([[-1.25 + 0.0j, 0.3j], [-0.3j, -0.48 + 0.0j]])
        chol = np.arange(3 * 4, dtype=complex).reshape(3, 4) + 1j

        restored = self._round_trip(tmp_path, hcore, chol, 'cplx.h5')
        assert np.iscomplexobj(restored.hcore)
        assert np.iscomplexobj(restored.chol)

    def test_a_malformed_rank_is_rejected(self, tmp_path):
        path = tmp_path / 'bad.h5'
        MolecularHamiltonian.from_integrals(
            np.diag([-1.0, -0.5]), chol=np.ones((2, 4))).to_hdf5(path)

        with h5.File(path, 'a') as fh5:
            del fh5['Hamiltonian/hcore']
            fh5.create_dataset('Hamiltonian/hcore', data=np.zeros((2, 2, 2, 2)))

        with pytest.raises(ValueError, match="rank 4"):
            Hamiltonian.from_hdf5(path)
