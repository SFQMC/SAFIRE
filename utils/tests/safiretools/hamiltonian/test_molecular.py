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
from safiretools.hamiltonian.fcidump import read_fcidump_header
from safiretools.hamiltonian.molecular import (
    MolecularHamiltonian,
    chunked_cholesky,
    freeze_core,
    modified_cholesky_direct,
    transform_cholesky,
)


def pair_matrix(hamiltonian):
    """The shared Cholesky vectors as the flat ``(npairs, nchol)`` matrix."""
    return hamiltonian.chol.reshape(-1, hamiltonian.nchol)


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

        assert hamiltonian.chol.shape == (1, 1, nmo, 1, nmo, chol.shape[0])
        assert np.allclose(pair_matrix(hamiltonian), chol.T)
        assert hamiltonian.nmo == nmo
        assert hamiltonian.nchol == chol.shape[0]
        assert hamiltonian.enuc == 1.5

    def test_an_eri_tensor_is_decomposed(self, random_hamiltonian):
        nmo, hcore, _, eri = random_hamiltonian
        hamiltonian = MolecularHamiltonian.from_integrals(hcore, eri=eri,
                                                          cholesky_tol=1e-10)

        L = pair_matrix(hamiltonian)
        reconstructed = L @ L.conj().T
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

    def test_hcore_must_match_its_spin_symmetry(self, random_hamiltonian):
        nmo, _, chol, _ = random_hamiltonian

        with pytest.raises(ValueError, match=r"expected \(nspin, npol\*nmo"):
            MolecularHamiltonian(hcore=np.zeros((nmo, nmo + 1)), chol=chol.T)

    def test_one_matrix_goes_into_both_collinear_spin_sectors(self,
                                                              random_hamiltonian):
        """The core Hamiltonian does not depend on spin; the two-body term does."""
        nmo, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(hcore=hcore, chol=chol.T,
                                           spin_symm=SpinSymm.COLLINEAR)

        assert hamiltonian.hcore.shape == (2, 1, nmo, 1, nmo)
        assert np.allclose(hamiltonian.hcore[0], hamiltonian.hcore[1])

    def test_a_collinear_hcore_may_give_each_spin_sector(self, random_hamiltonian):
        """``(nspin, npol*nmo, npol*nmo)``, one one-body matrix per sector."""
        nmo, hcore, chol, _ = random_hamiltonian
        beta = hcore + np.diag(np.arange(nmo, dtype=float))

        hamiltonian = MolecularHamiltonian(hcore=np.stack([hcore, beta]),
                                           chol=chol.T,
                                           spin_symm=SpinSymm.COLLINEAR)

        assert hamiltonian.hcore.shape == (2, 1, nmo, 1, nmo)
        assert np.allclose(hamiltonian.hcore[0].reshape(nmo, nmo), hcore)
        assert np.allclose(hamiltonian.hcore[1].reshape(nmo, nmo), beta)

    def test_the_stacked_collinear_layout_is_rejected(self, random_hamiltonian):
        """
        The dense format used to store a collinear ``hcore`` as one
        ``(2*nmo, nmo)`` matrix with the sectors stacked. The spin axis is its
        own dimension now, so that shape is no longer a layout anything accepts.
        """
        _, hcore, chol, _ = random_hamiltonian
        stacked = np.concatenate([hcore, hcore], axis=0)

        with pytest.raises(ValueError, match=r"expected \(nspin, npol\*nmo"):
            MolecularHamiltonian(hcore=stacked, chol=chol.T,
                                 spin_symm=SpinSymm.COLLINEAR)

    def test_a_blocked_hcore_is_taken_as_given(self, random_hamiltonian):
        nmo, _, chol, _ = random_hamiltonian
        blocked = np.zeros((1, 1, nmo, 1, nmo))

        assert MolecularHamiltonian(hcore=blocked, chol=chol.T).hcore.shape \
            == blocked.shape

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
            hcore, chol=chol, enuc=1.5)

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)
        assert hamiltonian_format(path) == 'dense'
        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian'].attrs['type'] == 'dense'

        restored = Hamiltonian.from_hdf5(path)
        assert isinstance(restored, MolecularHamiltonian)
        assert np.allclose(restored.hcore, hamiltonian.hcore)
        assert np.allclose(restored.chol, hamiltonian.chol)
        assert restored.enuc == hamiltonian.enuc

    @pytest.mark.parametrize("spin_symm", list(SpinSymm))
    def test_the_spin_symmetry_round_trips(self, random_hamiltonian, tmp_path,
                                           spin_symm):
        """``hcore``'s blocked shape records the spin symmetry it was written in."""
        _, hcore, chol, _ = random_hamiltonian
        if spin_symm is SpinSymm.NONCOLLINEAR:
            hcore = np.kron(np.eye(2), hcore)

        path = tmp_path / 'ham.h5'
        MolecularHamiltonian(hcore=hcore, chol=chol.T,
                             spin_symm=spin_symm).to_hdf5(path)

        assert Hamiltonian.from_hdf5(path).spin_symm is spin_symm

    def test_the_electron_count_is_not_recorded(self, random_hamiltonian, tmp_path):
        """A Hamiltonian carries no electron count; the wavefunction does."""
        _, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'ham.h5'
        MolecularHamiltonian.from_integrals(hcore, chol=chol).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            group = fh5['Hamiltonian']
            assert 'dims' not in group
            assert set(group.attrs) == {'format_version', 'type', 'nuclear_energy'}

    def test_real_integrals_are_written_real(self, random_hamiltonian, tmp_path):
        _, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'ham.h5'
        MolecularHamiltonian.from_integrals(hcore, chol=chol).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/hcore'].ndim == 5
            assert fh5['Hamiltonian/DenseFactorized/L'].ndim == 6
            assert fh5['Hamiltonian/DenseFactorized/L'].dtype == np.float64
            assert fh5['Hamiltonian/hcore'].dtype == np.float64

    def test_the_cholesky_vectors_are_spin_blocked_on_disk(self, random_hamiltonian,
                                                           tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'ham.h5'
        MolecularHamiltonian.from_integrals(hcore, chol=chol).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            L = fh5['Hamiltonian/DenseFactorized/L'][...]

        assert L.shape == (1, 1, nmo, 1, nmo, chol.shape[0])
        assert np.allclose(L.reshape(nmo * nmo, -1), chol.T)

    def test_collinear_vectors_shared_by_both_spins_round_trip(self, random_hamiltonian,
                                                               tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(hcore=hcore, chol=chol.T,
                                           spin_symm=SpinSymm.COLLINEAR)
        assert hamiltonian.chol.shape == (1, 1, nmo, 1, nmo, chol.shape[0])

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)
        restored = Hamiltonian.from_hdf5(path)

        assert restored.spin_symm is SpinSymm.COLLINEAR
        assert restored.hcore.shape == (2, 1, nmo, 1, nmo)
        assert np.array_equal(restored.chol, hamiltonian.chol)

    def test_spin_dependent_vectors_round_trip(self, random_hamiltonian, tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        per_spin = np.stack([chol.T, 2 * chol.T]).reshape(2, 1, nmo, 1, nmo, -1)
        hamiltonian = MolecularHamiltonian(hcore=hcore, chol=per_spin,
                                           spin_symm=SpinSymm.COLLINEAR)

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        assert np.array_equal(Hamiltonian.from_hdf5(path).chol, per_spin)

    def test_spin_dependent_vectors_need_a_collinear_hamiltonian(self,
                                                                 random_hamiltonian):
        nmo, hcore, chol, _ = random_hamiltonian
        per_spin = np.stack([chol.T, chol.T]).reshape(2, 1, nmo, 1, nmo, -1)

        with pytest.raises(ValueError, match="nspin in"):
            MolecularHamiltonian(hcore=hcore, chol=per_spin)

    def test_a_flat_cholesky_file_is_rejected(self, random_hamiltonian, tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'ham.h5'
        MolecularHamiltonian.from_integrals(hcore, chol=chol).to_hdf5(path)

        with h5.File(path, 'a') as fh5:
            del fh5['Hamiltonian/DenseFactorized/L']
            fh5.create_dataset('Hamiltonian/DenseFactorized/L', data=chol.T)

        with pytest.raises(ValueError, match="DenseFactorized/L"):
            Hamiltonian.from_hdf5(path)

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
            assert fh5['Hamiltonian/hcore'].dtype == np.complex128
            assert fh5['Hamiltonian/hcore'].shape == (1, 1, nmo, 1, nmo)

        assert np.allclose(Hamiltonian.from_hdf5(path).hcore.reshape(nmo, nmo),
                           complex_hcore)

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

        with h5.File(path, 'a') as fh5:
            del fh5['Hamiltonian/DenseFactorized/L']

        with pytest.raises(ValueError, match="no Hamiltonian in a format"):
            Hamiltonian.from_hdf5(path)


class TestFcidump:
    """The FCIDUMP external format, reached through the two class methods."""

    def test_round_trip(self, random_hamiltonian, tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian.from_integrals(
            hcore, chol=chol, enuc=1.5)

        path = tmp_path / 'FCIDUMP'
        hamiltonian.to_fcidump(path, nelec=(3, 2), tol=1e-12)
        restored = MolecularHamiltonian.from_fcidump(path, cholesky_tol=1e-10)

        assert restored.nmo == nmo
        assert restored.spin_symm is SpinSymm.COLLINEAR
        assert np.isclose(restored.enuc, 1.5)
        assert np.allclose(restored.hcore[0].reshape(nmo, nmo), hcore)

        L, L_restored = pair_matrix(hamiltonian), pair_matrix(restored)
        assert np.allclose(L_restored @ L_restored.conj().T, L @ L.conj().T,
                           atol=1e-6)

    def test_complex_integrals_round_trip(self, tmp_path):
        r"""
        Pins the orbital-pair order the decomposition needs. FCIDUMP holds the
        chemists' :math:`(ik|jl)`, whose hermitian pair matrix is
        :math:`\{(ik), (lj)\}` — the same tensor with its last two indices
        swapped. For real integrals the two coincide; for complex ones the
        wrong one is not even hermitian, so its decomposition is meaningless.
        """
        rng = np.random.default_rng(3)
        nmo = 3

        # a physically shaped Cholesky matrix: L_ij = conj(L_ji) per vector
        vectors = (rng.standard_normal((nmo, nmo, 4))
                   + 1j * rng.standard_normal((nmo, nmo, 4)))
        vectors = vectors + vectors.transpose((1, 0, 2)).conj()
        chol = vectors.reshape(nmo * nmo, 4)

        hcore = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
        hcore = hcore + hcore.conj().T

        path = tmp_path / 'FCIDUMP'
        MolecularHamiltonian(hcore=hcore, chol=chol).to_fcidump(
            path, nelec=(2, 2), tol=1e-12)
        restored = MolecularHamiltonian.from_fcidump(path, cholesky_tol=1e-10)

        assert np.allclose(restored.hcore.reshape(nmo, nmo), hcore)
        L_restored = pair_matrix(restored)
        assert np.allclose(L_restored @ L_restored.conj().T,
                           chol @ chol.conj().T, atol=1e-6)

    @pytest.mark.parametrize("nelec,expected", [((2, 2), SpinSymm.CLOSED),
                                                ((3, 1), SpinSymm.COLLINEAR)])
    def test_the_header_supplies_the_spin_symmetry(self, random_hamiltonian,
                                                   tmp_path, nelec, expected):
        _, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'FCIDUMP'
        MolecularHamiltonian(hcore=hcore, chol=chol.T).to_fcidump(path, nelec=nelec)

        assert MolecularHamiltonian.from_fcidump(path).spin_symm is expected

    def test_an_explicit_spin_symmetry_wins(self, random_hamiltonian, tmp_path):
        """A spinor-basis FCIDUMP is not self-describing, so it has to be named."""
        _, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'FCIDUMP'
        MolecularHamiltonian(hcore=hcore, chol=chol.T).to_fcidump(path, nelec=(3, 1))

        restored = MolecularHamiltonian.from_fcidump(path, spin_symm='closed')
        assert restored.spin_symm is SpinSymm.CLOSED

    def test_a_spinor_basis_doubles_the_orbital_count(self, random_hamiltonian,
                                                      tmp_path):
        nmo, hcore, chol, _ = random_hamiltonian
        path = tmp_path / 'FCIDUMP'
        MolecularHamiltonian(hcore=hcore, chol=chol.T).to_fcidump(
            path, use_spinor=True, tol=1e-12)

        assert read_fcidump_header(path)['nbasis'] == 2 * nmo

    def test_a_noncollinear_hamiltonian_cannot_be_converted_again(self,
                                                                  random_hamiltonian,
                                                                  tmp_path):
        nmo, _, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(
            hcore=np.zeros((2 * nmo, 2 * nmo)),
            chol=np.zeros(((2 * nmo)**2, 4)),
            spin_symm=SpinSymm.NONCOLLINEAR)

        with pytest.raises(ValueError, match="already in a spin-orbital one"):
            hamiltonian.to_fcidump(tmp_path / 'FCIDUMP', use_spinor=True)

    def test_a_cholesky_matrix_in_the_other_basis_is_rejected(self,
                                                              random_hamiltonian,
                                                              tmp_path):
        """
        ``__init__`` allows a spatial-basis Cholesky matrix beside a
        spin-orbital ``hcore``; a FCIDUMP holds one basis, so it cannot.
        """
        nmo, _, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(
            hcore=np.zeros((2 * nmo, 2 * nmo)), chol=chol.T,
            spin_symm=SpinSymm.NONCOLLINEAR)

        with pytest.raises(ValueError, match="have to be in the same one"):
            hamiltonian.to_fcidump(tmp_path / 'FCIDUMP')

    def test_a_collinear_hamiltonian_is_rejected(self, random_hamiltonian,
                                                 tmp_path):
        """FCIDUMP has room for one one-body matrix, and collinear has two."""
        _, hcore, chol, _ = random_hamiltonian
        hamiltonian = MolecularHamiltonian(hcore=hcore, chol=chol.T,
                                           spin_symm=SpinSymm.COLLINEAR)

        with pytest.raises(ValueError, match="independent spin sectors"):
            hamiltonian.to_fcidump(tmp_path / 'FCIDUMP')


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
        mf, _ = neon_rhf

        hamiltonian = MolecularHamiltonian.from_pyscf(mf, chol_cut=1e-5)

        C = mf.mo_coeff
        expected_hcore = C.conj().T @ mf.get_hcore() @ C
        assert np.allclose(hamiltonian.hcore[0, 0, :, 0, :], expected_hcore)
        assert np.isclose(hamiltonian.enuc, neon_atom.energy_nuc())
        assert hamiltonian.nmo == C.shape[-1]

        # the factorization must reproduce the MO-basis ERIs
        nmo = hamiltonian.nmo
        eri_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C,
                           neon_atom.intor('int2e', aosym='s1'), C, C)
        L = pair_matrix(hamiltonian)
        reconstructed = L @ L.conj().T
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
        assert h1e.shape == (4, 4)
        assert np.isclose(efzc, ecore)
        assert np.allclose(h1eff, h1e, atol=1e-8, rtol=1e-5)

    @pytest.fixture
    def phased(self, neon_rhf):
        """The RHF orbitals, each multiplied by an arbitrary phase: a complex basis."""
        mf, _ = neon_rhf
        nmo = mf.mo_coeff.shape[1]
        phases = np.exp(1j * np.linspace(0.3, 2.9, nmo))
        return mf.mo_coeff * phases, phases

    def test_a_complex_basis_reproduces_the_eri_tensor(self, neon_atom, neon_rhf,
                                                       phased):
        r"""
        In a complex basis the vectors are :math:`C^\dagger L^\gamma C`, which
        stay hermitian in the orbital pair, and :math:`\sum_\gamma L_{ij} L_{kl}`
        is the MO-basis :math:`(ij|kl)`.
        """
        mf, _ = neon_rhf
        C, _ = phased

        hamiltonian = MolecularHamiltonian.from_pyscf(mf, basis=C, chol_cut=1e-6)

        nmo = hamiltonian.nmo
        L = pair_matrix(hamiltonian).T.reshape(-1, nmo, nmo)
        assert hamiltonian.complex_chol
        assert np.allclose(L, L.conj().transpose(0, 2, 1))

        eri_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', C.conj(), C,
                           neon_atom.intor('int2e', aosym='s1'), C.conj(), C)
        reconstructed = np.einsum('gij,gkl->ijkl', L, L)
        assert np.allclose(reconstructed, eri_mo, atol=1e-5)

    def test_a_complex_basis_freezes_the_same_core(self, neon_rhf, phased):
        """
        Phases change no physics: the frozen-core constant is the same, and the
        active one-body term only picks up the phases of its orbitals.
        """
        mf, _ = neon_rhf
        C, phases = phased

        real = MolecularHamiltonian.from_pyscf(mf, active_space=(8, -1))
        complex_ = MolecularHamiltonian.from_pyscf(mf, basis=C,
                                                   active_space=(8, -1))

        active = phases[1:]
        expected = active.conj()[:, None] * real.hcore[0, 0, :, 0, :] * active
        assert np.isclose(complex_.enuc, real.enuc)
        assert np.allclose(complex_.hcore[0, 0, :, 0, :], expected)

    def test_complex_cholesky_vectors_warn_on_write(self, neon_rhf, phased,
                                                    tmp_path):
        mf, _ = neon_rhf
        C, _ = phased
        hamiltonian = MolecularHamiltonian.from_pyscf(mf, basis=C)

        with pytest.warns(UserWarning, match="RealDenseHamiltonian"):
            hamiltonian.to_hdf5(tmp_path / 'complex.h5')

    def test_freezing_more_orbitals_than_exist_is_rejected(self, neon_atom, neon_rhf):
        mf, _ = neon_rhf
        C = mf.mo_coeff
        chol = chunked_cholesky(neon_atom, max_error=1e-5)
        transform_cholesky(chol, C)
        h1e = C.T @ mf.get_hcore() @ C

        with pytest.raises(ValueError, match="Can't freeze more orbitals"):
            freeze_core(h1e, chol, 0, 3, 4, verbose=False)

    def test_a_uhf_basis_needs_ortho_ao(self, neon_rhf):
        mf, _ = neon_rhf

        with pytest.raises(ValueError, match="basis='ortho_ao'"):
            MolecularHamiltonian.from_pyscf(mf.to_uhf())

        hamiltonian = MolecularHamiltonian.from_pyscf(mf.to_uhf(),
                                                      basis='ortho_ao')
        assert hamiltonian.spin_symm is SpinSymm.CLOSED

    def test_an_active_space_and_ortho_ao_are_mutually_exclusive(self, neon_rhf):
        mf, _ = neon_rhf

        with pytest.raises(ValueError, match="cannot be combined"):
            MolecularHamiltonian.from_pyscf(mf, active_space=(4, 4),
                                            basis='ortho_ao')

    def test_a_ghf_source_gives_the_spin_free_noncollinear_hamiltonian(
            self, neon_rhf):
        """
        A GHF object's hcore is the scalar one on both spins, so the result is
        the closed Hamiltonian's one-body term promoted to the spinor basis.
        """
        mf, _ = neon_rhf

        closed = MolecularHamiltonian.from_pyscf(mf, chol_cut=1e-5)
        noncollinear = MolecularHamiltonian.from_pyscf(mf.to_ghf(), basis=mf,
                                                       chol_cut=1e-5)

        nmo = closed.nmo
        assert noncollinear.spin_symm is SpinSymm.NONCOLLINEAR
        assert np.array_equal(
            noncollinear.hcore.reshape(2 * nmo, 2 * nmo),
            np.kron(np.eye(2), closed.hcore.reshape(nmo, nmo)))
        assert np.array_equal(noncollinear.chol, closed.chol)

    def test_density_fitting_vectors_are_used_when_asked(self, neon_atom):
        from pyscf import scf

        mf = scf.RHF(neon_atom).density_fit().run()

        hamiltonian = MolecularHamiltonian.from_pyscf(mf, df=True)

        C = mf.mo_coeff
        assert hamiltonian.nchol == mf.with_df.get_naoaux()
        eri_mo = np.einsum('pi,qj,pqrs,rk,sl->ijkl', C, C,
                           neon_atom.intor('int2e', aosym='s1'), C, C)
        L = pair_matrix(hamiltonian)
        reconstructed = L @ L.conj().T
        assert np.allclose(reconstructed.reshape((hamiltonian.nmo,) * 4),
                           eri_mo, atol=1e-2)

    def test_df_needs_a_density_fitted_object(self, neon_rhf):
        mf, _ = neon_rhf

        with pytest.raises(ValueError, match="density-fitted"):
            MolecularHamiltonian.from_pyscf(mf, df=True)


class TestRealAndComplexAreToldApart:
    """
    Complex integrals are stored as complex datasets and real ones stay real, so the
    dtype on disk follows the data. These cases check that end-to-end through a
    Hamiltonian round trip; `tests/safiretools/test_hdf5.py` checks the read helper
    itself.

    The `nmo == 2` and `nchol == 2` shapes are kept because they are the ones an
    earlier rank-based decoder mistook for interleaved complex data.
    """

    @staticmethod
    def _round_trip(tmp_path, hcore, chol, name):
        hamiltonian = MolecularHamiltonian.from_integrals(
            hcore, chol=chol, enuc=0.5)
        path = tmp_path / name
        hamiltonian.to_hdf5(path)

        restored = Hamiltonian.from_hdf5(path)
        assert np.allclose(restored.hcore, hamiltonian.hcore)
        assert np.allclose(restored.chol, hamiltonian.chol)
        return restored

    def test_two_orbitals(self, tmp_path):
        """A real `(2, 2)` hcore is shaped like an interleaved complex vector."""
        hcore = np.array([[-1.25, 0.0], [0.0, -0.48]])
        chol = np.arange(3 * 4, dtype=float).reshape(3, 4)   # (nchol=3, nmo**2)

        restored = self._round_trip(tmp_path, hcore, chol, 'nmo2.h5')
        assert restored.nmo == 2
        assert restored.hcore.shape == (1, 1, 2, 1, 2)
        assert not np.iscomplexobj(restored.hcore)

    def test_two_cholesky_vectors(self, tmp_path):
        """A real Cholesky matrix with two vectors is shaped like interleaved data."""
        hcore = np.diag([-1.0, -0.5, -0.25])
        chol = np.arange(2 * 9, dtype=float).reshape(2, 9)   # (nchol=2, nmo**2)

        restored = self._round_trip(tmp_path, hcore, chol, 'nchol2.h5')
        assert restored.nchol == 2
        assert restored.chol.shape == (1, 1, 3, 1, 3, 2)
        assert not np.iscomplexobj(restored.chol)

    def test_complex_data_is_still_recognized(self, tmp_path):
        hcore = np.array([[-1.25 + 0.0j, 0.3j], [-0.3j, -0.48 + 0.0j]])
        chol = np.arange(3 * 4, dtype=complex).reshape(3, 4) + 1j

        with pytest.warns(UserWarning, match="cannot run this file"):
            restored = self._round_trip(tmp_path, hcore, chol, 'cplx.h5')
        assert np.iscomplexobj(restored.hcore)
        assert np.iscomplexobj(restored.chol)

    def test_a_malformed_hcore_is_rejected(self, tmp_path):
        path = tmp_path / 'bad.h5'
        MolecularHamiltonian.from_integrals(
            np.diag([-1.0, -0.5]), chol=np.ones((2, 4))).to_hdf5(path)

        with h5.File(path, 'a') as fh5:
            del fh5['Hamiltonian/hcore']
            fh5.create_dataset('Hamiltonian/hcore', data=np.zeros((2, 2, 2, 2)))

        with pytest.raises(ValueError, match="expected .nspin, npol"):
            Hamiltonian.from_hdf5(path)
