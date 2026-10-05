# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`PeriodicHamiltonian`: momentum bookkeeping, and the two representations the
one `kp_sym`-flagged solver produces."""

import h5py as h5
import numpy as np
import pytest
import scipy.linalg

from safiretools.convert.pyscf import canonical_orthogonalization
from safiretools.hamiltonian.base import Hamiltonian, hamiltonian_format
from safiretools.hamiltonian.fcidump import write_fcidump_kpoint
from safiretools.hamiltonian.periodic import (
    PeriodicHamiltonian,
    construct_qk_maps,
    generate_grid_shifts,
    setup_basis_map,
)


# ----------------------------------------------------------------------
# bookkeeping, no PySCF needed
# ----------------------------------------------------------------------

def test_setup_basis_map_numbers_orbitals_consecutively():
    ik2n, nmo_tot = setup_basis_map([3, 2])

    assert nmo_tot == 5
    assert list(ik2n[:, 0]) == [0, 1, 2]
    # the second k-point has one fewer orbital, so the last slot stays unset
    assert list(ik2n[:, 1]) == [3, 4, -1]


# ----------------------------------------------------------------------
# the in-memory Hamiltonian, no PySCF needed
# ----------------------------------------------------------------------

def _kpoint_hamiltonian(nkpts=2, nmo=3, nchol=4, madelung_constant=0.0):
    rng = np.random.default_rng(5)
    hcore = [rng.random((nmo, nmo)) + 1j * rng.random((nmo, nmo))
             for _ in range(nkpts)]
    chol = {Q: rng.random((nkpts, nmo * nmo * nchol))
               + 1j * rng.random((nkpts, nmo * nmo * nchol))
            for Q in range(nkpts)}

    return PeriodicHamiltonian(
        hcore=hcore, chol=chol, kpts=rng.random((nkpts, 3)),
        qk_to_k2=np.zeros((nkpts, nkpts), dtype=np.int32),
        minus_k=np.arange(nkpts, dtype=np.int32),
        enuc=-1.25,
        madelung_constant=madelung_constant,
    )


def test_the_orbital_count_is_read_off_hcore():
    assert _kpoint_hamiltonian(nkpts=2, nmo=3).nmo == 3


def test_uneven_orbital_counts_are_refused():
    with pytest.raises(ValueError):
        PeriodicHamiltonian(hcore=[np.zeros((3, 3)), np.zeros((2, 2))], chol={},
                            kpts=np.zeros((2, 3)),
                            qk_to_k2=np.zeros((2, 2), dtype=np.int32),
                            minus_k=np.zeros(2, dtype=np.int32))


@pytest.mark.parametrize("field,value,match", [
    ('hcore', np.zeros((3, 3, 3)), "hcore has shape"),
    ('hcore', np.zeros((2, 3, 2)), "hcore has shape"),
    ('qk_to_k2', np.zeros((3, 3), dtype=np.int32), "qk_to_k2 has shape"),
    ('minus_k', np.zeros(3, dtype=np.int32), "minus_k has shape"),
    ('chol', {0: np.zeros((2, 10))}, r"chol\[0\] has shape"),
    ('basis_rotation', np.zeros((3, 4, 3)), "basis_rotation has shape"),
    ('basis_rotation', np.zeros((2, 4, 2)), "basis_rotation has shape"),
])
def test_the_data_must_match_the_kpoint_count(field, value, match):
    kwargs = dict(hcore=np.zeros((2, 3, 3)), chol={}, kpts=np.zeros((2, 3)),
                  qk_to_k2=np.zeros((2, 2), dtype=np.int32),
                  minus_k=np.zeros(2, dtype=np.int32))
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        PeriodicHamiltonian(**kwargs)


class TestKpointFormat:

    def test_round_trip(self, tmp_path):
        hamiltonian = _kpoint_hamiltonian(madelung_constant=0.75)
        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        assert hamiltonian_format(path) == 'kpoint'
        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian'].attrs['type'] == 'kpoint'

        restored = Hamiltonian.from_hdf5(path)
        assert isinstance(restored, PeriodicHamiltonian)
        assert restored.nkpts == hamiltonian.nkpts
        assert restored.enuc == hamiltonian.enuc
        assert restored.madelung_constant == hamiltonian.madelung_constant
        assert np.allclose(restored.hcore, hamiltonian.hcore)
        assert set(restored.chol) == set(hamiltonian.chol)
        for Q, L in hamiltonian.chol.items():
            assert np.allclose(restored.chol[Q], L)

    def test_the_basis_rotation_is_the_identity_at_every_kpoint_by_default(
            self, tmp_path):
        hamiltonian = _kpoint_hamiltonian(nkpts=2, nmo=3)
        identity = np.stack([np.eye(3)] * 2)
        assert np.array_equal(hamiltonian.basis_rotation, identity)

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert 'BasisRotation' not in fh5['Hamiltonian']
        assert np.array_equal(Hamiltonian.from_hdf5(path).basis_rotation, identity)

    def test_the_basis_rotation_round_trips(self, tmp_path):
        reference = _kpoint_hamiltonian(nkpts=2, nmo=3)
        rng = np.random.default_rng(9)
        rotation = rng.random((2, 5, 3)) + 1j * rng.random((2, 5, 3))
        hamiltonian = PeriodicHamiltonian(
            hcore=reference.hcore, chol=reference.chol, kpts=reference.kpts,
            qk_to_k2=reference.qk_to_k2, minus_k=reference.minus_k,
            basis_rotation=rotation)

        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        assert np.array_equal(Hamiltonian.from_hdf5(path).basis_rotation, rotation)

    def test_the_shapes_carry_the_sizes(self, tmp_path):
        """
        The dense layouts with a leading k-point axis: the executable takes the
        k-point, orbital and Cholesky-vector counts from them.
        """
        hamiltonian = _kpoint_hamiltonian(nkpts=2, nmo=3, nchol=4)
        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            group = fh5['Hamiltonian']
            assert set(group.attrs) == {'format_version', 'type', 'nuclear_energy',
                                        'madelung_constant'}
            assert not {'NMOPerKP', 'NCholPerKP'} & set(group)
            assert group['hcore'].shape == (2, 1, 1, 3, 1, 3)
            for Q in hamiltonian.chol:
                assert group[f'KPFactorized/L{Q}'].shape == (2, 1, 1, 3, 1, 3, 4)


def _mirrored_hamiltonian(nmo=2, nchol=2):
    """
    A 3-k-point Hamiltonian whose ``Q = 2`` block is not stored, since
    ``minus_k[2] == 1 < 2`` — the layout every k-point factorization writes.
    """
    nkpts = 3
    rng = np.random.default_rng(11)
    hcore = [rng.random((nmo, nmo)) + 1j * rng.random((nmo, nmo))
             for _ in range(nkpts)]
    chol = {Q: rng.random((nkpts, nmo * nmo * nchol))
               + 1j * rng.random((nkpts, nmo * nmo * nchol))
            for Q in (0, 1)}

    return PeriodicHamiltonian(
        hcore=hcore, chol=chol, kpts=rng.random((nkpts, 3)),
        # k1 - k2 = Q on a 1D three-point mesh
        qk_to_k2=np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int32),
        minus_k=np.array([0, 2, 1], dtype=np.int32),
        enuc=0.5,
    )


class TestFcidump:
    """The FCIDUMP external format, over the combined basis of every k-point."""

    def test_it_matches_the_writer_on_a_full_momentum_set(self, tmp_path):
        """The Madelung correction is folded into the one constant FCIDUMP has."""
        hamiltonian = _kpoint_hamiltonian(madelung_constant=0.75)
        chol = [hamiltonian.chol[Q] for Q in range(hamiltonian.nkpts)]

        nelec = (2, 2)
        from_method = tmp_path / 'method'
        hamiltonian.to_fcidump(from_method, nelec=nelec, tol=1e-12)

        direct = tmp_path / 'direct'
        write_fcidump_kpoint(direct, hamiltonian.hcore, chol, -1.25 - 0.75 * 4,
                             nelec, hamiltonian.qk_to_k2, tol=1e-12)

        assert from_method.read_text() == direct.read_text()

    def test_the_unstored_momentum_transfer_is_reconstructed(self):
        r"""
        :math:`L^{-Q}_{k_1}[i, j, n] = (L^{Q}_{k_2}[j, i, n])^*` with
        :math:`k_2 = \mathrm{qk\_to\_k2}[-Q, k_1]`, and the vector count comes
        from the partner, where the factorization actually ran.
        """
        hamiltonian = _mirrored_hamiltonian()
        nkpts, nmo, nchol = hamiltonian.nkpts, hamiltonian.nmo, 2

        chol = hamiltonian._chol_all_momenta()

        assert [block.shape for block in chol] == [(nkpts, nmo * nmo * nchol)] * nkpts

        for Q in (0, 1):
            assert np.array_equal(chol[Q], hamiltonian.chol[Q])

        stored = hamiltonian.chol[1].reshape(nkpts, nmo, nmo, nchol)
        expected = np.array([stored[k2].conj().swapaxes(0, 1)
                             for k2 in hamiltonian.qk_to_k2[2]])
        assert np.allclose(chol[2], expected.reshape(nkpts, nmo * nmo * nchol))

    def test_a_momentum_transfer_with_no_partner_is_rejected(self, tmp_path):
        hamiltonian = _mirrored_hamiltonian()
        del hamiltonian.chol[1]

        with pytest.raises(ValueError, match="momentum transfer 1, nor for its -Q"):
            hamiltonian.to_fcidump(tmp_path / 'FCIDUMP')

    def test_a_madelung_constant_needs_the_electron_count(self, tmp_path):
        with pytest.raises(ValueError, match="pass nelec"):
            _kpoint_hamiltonian(madelung_constant=0.75).to_fcidump(tmp_path / 'FCIDUMP')

    def test_a_spinor_basis_is_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="spatial-orbital basis"):
            _kpoint_hamiltonian().to_fcidump(tmp_path / 'FCIDUMP', use_spinor=True)


# ----------------------------------------------------------------------
# the PySCF-backed generation path
# ----------------------------------------------------------------------

@pytest.mark.pyscf
class TestGeneration:

    def test_grid_shifts_cover_every_reciprocal_lattice_offset(self, diamond):
        gmap, Qi, ngs = generate_grid_shifts(diamond)

        assert gmap.shape == (27, ngs)
        assert Qi.shape == (27, 3)
        assert ngs == int(np.prod(diamond.mesh))
        # the identity shift is in the middle of the (-1, 0, 1)^3 sweep
        assert np.allclose(Qi[13], 0.0)
        assert np.array_equal(gmap[13], np.arange(ngs))
        # every shift is a permutation of the grid
        assert all(len(np.unique(row)) == ngs for row in gmap)

    def test_momentum_maps(self, diamond, diamond_lda):
        _, kpts = diamond_lda
        qk, km = construct_qk_maps(diamond, kpts)

        assert qk.shape == (2, 2)
        assert km.shape == (2,)
        assert qk[1, 1] == 0
        assert qk[0, 1] == 1
        assert km[1] == 1

    def test_kpoint_hamiltonian_round_trips(self, diamond_lda, tmp_path):
        kmf, _ = diamond_lda
        hamiltonian = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=True, chol_cut=1e-3,
            maxvecs=20)

        assert hamiltonian.nkpts == 2
        assert set(hamiltonian.chol) == {0, 1}
        assert all(L.shape[1] > 0 for L in hamiltonian.chol.values())
        # the Madelung term is left to the executable, which knows the electron count
        assert np.isclose(hamiltonian.enuc, 2 * kmf.cell.energy_nuc())
        assert hamiltonian.madelung_constant != 0.0

        path = tmp_path / 'kp.h5'
        hamiltonian.to_hdf5(path)
        assert hamiltonian_format(path) == 'kpoint'

        restored = Hamiltonian.from_hdf5(path)
        assert np.allclose(restored.chol[0], hamiltonian.chol[0])
        assert restored.madelung_constant == hamiltonian.madelung_constant

        overlaps = np.reshape(kmf.get_ovlp(), (2, kmf.cell.nao_nr(), -1))
        expected = np.array([canonical_orthogonalization(s) for s in overlaps])
        assert np.allclose(hamiltonian.basis_rotation, expected)
        assert np.array_equal(restored.basis_rotation, hamiltonian.basis_rotation)

    def test_no_exchange_divergence_treatment_records_no_madelung_constant(
            self, diamond_lda, monkeypatch):
        kmf, _ = diamond_lda
        monkeypatch.setattr(kmf, 'exxdiv', None)

        hamiltonian = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=False, chol_cut=1e-2,
            maxvecs=20)

        assert hamiltonian.madelung_constant == 0.0

    @pytest.mark.parametrize("exxdiv", ['vcut_sph', 'vcut_ws'])
    def test_a_truncated_coulomb_kernel_is_refused(self, diamond_lda, monkeypatch,
                                                   exxdiv):
        kmf, _ = diamond_lda
        monkeypatch.setattr(kmf, 'exxdiv', exxdiv)

        with pytest.raises(ValueError, match="only 'ewald' or None"):
            PeriodicHamiltonian.from_pyscf(kmf, basis='ortho_ao')

    def test_supercell_hamiltonian_is_a_gamma_point_kpoint_hamiltonian(
            self, diamond_lda, tmp_path):
        """
        A supercell Hamiltonian is the Γ point of the supercell, so it comes back
        in the ordinary k-point representation with a single k-point.
        """
        kmf, kpts = diamond_lda
        supercell = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=False, chol_cut=1e-3,
            maxvecs=20)
        original_nkpts = len(kpts)
        nao = kmf.cell.nao_nr()

        assert supercell.nkpts == 1
        # the combined basis really did absorb the original k-points
        nmo = original_nkpts * nao
        assert supercell.nmo == nmo
        assert np.allclose(supercell.kpts, 0.0)
        assert supercell.qk_to_k2.tolist() == [[0]]
        assert supercell.minus_k.tolist() == [0]
        assert set(supercell.chol) == {0}

        assert supercell.hcore.shape == (1, nmo, nmo)
        assert supercell.chol[0].shape[0] == 1
        assert supercell.chol[0].shape[1] % (nmo * nmo) == 0

        path = tmp_path / 'sc.h5'
        supercell.to_hdf5(path)
        assert hamiltonian_format(path) == 'kpoint'

        restored = Hamiltonian.from_hdf5(path)
        assert restored.nkpts == 1
        assert np.allclose(restored.chol[0], supercell.chol[0])

        # block diagonal: the Bloch AOs of each original k-point map onto that
        #   k-point's slice of the combined basis
        overlaps = np.reshape(kmf.get_ovlp(), (original_nkpts, nao, nao))
        expected = scipy.linalg.block_diag(
            *[canonical_orthogonalization(s) for s in overlaps])
        assert supercell.basis_rotation.shape == (1, original_nkpts * nao, nmo)
        assert np.allclose(supercell.basis_rotation[0], expected)
        assert np.array_equal(restored.basis_rotation, supercell.basis_rotation)

    def test_supercell_cholesky_vectors_stay_complex(self, diamond_lda, tmp_path):
        """
        The k-point format is complex throughout, which is what the executable's
        ``KPFactorizedHamiltonian`` reads — no real-valued special case.
        """
        kmf, _ = diamond_lda
        path = tmp_path / 'sc.h5'
        PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=False, chol_cut=1e-3,
            maxvecs=20).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/KPFactorized/L0'].dtype == np.complex128

    def test_the_one_body_hamiltonian_is_block_diagonal_in_k(self, diamond_lda):
        """
        The one-body operator conserves crystal momentum, so the combined
        supercell hcore couples no two different original k-points.
        """
        kmf, _ = diamond_lda
        supercell = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=False, chol_cut=1e-2,
            maxvecs=20)

        hcore = supercell.hcore[0]
        offset = kmf.cell.nao_nr()
        assert np.allclose(hcore[:offset, offset:], 0.0)
        assert np.allclose(hcore[offset:, :offset], 0.0)

    def test_the_two_representations_agree_on_the_two_body_energy(self,
                                                                  diamond_lda):
        r"""
        Both factorizations reconstruct the same two-body integrals, so
        :math:`\sum_\gamma |L^\gamma_{ii}|^2` — the diagonal Coulomb weight —
        must match between them, to the Cholesky tolerance.
        """
        kmf, _ = diamond_lda
        chol_cut = 1e-4
        supercell = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=False, chol_cut=chol_cut,
            maxvecs=20)
        kpoint = PeriodicHamiltonian.from_pyscf(
            kmf, basis='ortho_ao', kpoint_symmetry=True, chol_cut=chol_cut,
            maxvecs=20)

        nmo_tot = supercell.nmo
        L_sc = supercell.chol[0].reshape(nmo_tot * nmo_tot, -1)
        sc_trace = np.einsum('ig,ig->i', L_sc, L_sc.conj()).real.reshape(
            nmo_tot, nmo_tot).diagonal().sum()

        # the Q = 0 block holds the k-diagonal pair densities
        nmo = kpoint.nmo
        L0 = kpoint.chol[0].reshape(kpoint.nkpts, nmo * nmo, -1)
        kp_trace = sum(
            np.einsum('ig,ig->', L0[k, ::nmo + 1, :], L0[k, ::nmo + 1, :].conj()).real
            for k in range(kpoint.nkpts)
        )

        assert np.isclose(sc_trace, kp_trace, rtol=1e-2)
