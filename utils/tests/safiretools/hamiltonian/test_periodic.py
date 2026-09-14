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

from safiretools.hamiltonian.base import Hamiltonian, hamiltonian_format
from safiretools.hamiltonian.fcidump import write_fcidump_kpoint
from safiretools.hamiltonian.periodic import (
    PeriodicHamiltonian,
    construct_qk_maps,
    generate_grid_shifts,
    get_ortho_ao,
    setup_basis_map,
)


# ----------------------------------------------------------------------
# bookkeeping, no PySCF needed
# ----------------------------------------------------------------------

def test_setup_basis_map_numbers_orbitals_consecutively():
    ik2n, nmo_tot = setup_basis_map([3, 2], nkpts=2)

    assert nmo_tot == 5
    assert list(ik2n[:, 0]) == [0, 1, 2]
    # the second k-point has one fewer orbital, so the last slot stays unset
    assert list(ik2n[:, 1]) == [3, 4, -1]


# ----------------------------------------------------------------------
# the in-memory Hamiltonian, no PySCF needed
# ----------------------------------------------------------------------

def _kpoint_hamiltonian(nkpts=2, nmo=3, nchol=4):
    rng = np.random.default_rng(5)
    hcore = [rng.random((nmo, nmo)) + 1j * rng.random((nmo, nmo))
             for _ in range(nkpts)]
    chol = {Q: rng.random((nkpts, nmo * nmo * nchol))
               + 1j * rng.random((nkpts, nmo * nmo * nchol))
            for Q in range(nkpts)}

    return PeriodicHamiltonian(
        hcore=hcore, chol=chol, kpts=rng.random((nkpts, 3)),
        nmo_pk=[nmo] * nkpts,
        qk_to_k2=np.zeros((nkpts, nkpts), dtype=np.int32),
        minus_k=np.arange(nkpts, dtype=np.int32),
        nchol_pk=np.array([nchol] * nkpts, dtype=np.int32),
        enuc=-1.25, nelec=(2, 2),
    )


@pytest.mark.parametrize("field,value", [
    ('nmo_pk', [3, 3, 3]),
    ('qk_to_k2', np.zeros((3, 3), dtype=np.int32)),
    ('minus_k', np.zeros(3, dtype=np.int32)),
    ('nchol_pk', np.zeros(3, dtype=np.int32)),
])
def test_momentum_maps_must_match_the_kpoint_count(field, value):
    kwargs = dict(hcore=[], chol={}, kpts=np.zeros((2, 3)), nmo_pk=[3, 3],
                  qk_to_k2=np.zeros((2, 2), dtype=np.int32),
                  minus_k=np.zeros(2, dtype=np.int32),
                  nchol_pk=np.zeros(2, dtype=np.int32))
    kwargs[field] = value

    with pytest.raises(ValueError, match=f"{field} has shape"):
        PeriodicHamiltonian(**kwargs)


class TestKpointFormat:

    def test_round_trip(self, tmp_path):
        hamiltonian = _kpoint_hamiltonian()
        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        assert hamiltonian_format(path) == 'kpoint'
        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/type'].asstr()[()] == 'KPFactorized'

        restored = Hamiltonian.from_hdf5(path)
        assert isinstance(restored, PeriodicHamiltonian)
        assert restored.nkpts == hamiltonian.nkpts
        assert np.allclose(restored.nchol_pk, hamiltonian.nchol_pk)
        for ki in range(hamiltonian.nkpts):
            assert np.allclose(restored.hcore[ki], hamiltonian.hcore[ki])
        for Q, L in hamiltonian.chol.items():
            assert np.allclose(restored.chol[Q], L)

    def test_dims_records_the_kpoint_count(self, tmp_path):
        hamiltonian = _kpoint_hamiltonian(nkpts=2, nmo=3)
        path = tmp_path / 'ham.h5'
        hamiltonian.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            dims = fh5['Hamiltonian/dims'][...]
        assert dims[2] == 2
        assert dims[3] == hamiltonian.nmo_tot


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
        nmo_pk=[nmo] * nkpts,
        # k1 - k2 = Q on a 1D three-point mesh
        qk_to_k2=np.array([[0, 1, 2], [2, 0, 1], [1, 2, 0]], dtype=np.int32),
        minus_k=np.array([0, 2, 1], dtype=np.int32),
        nchol_pk=np.array([nchol, nchol, 0], dtype=np.int32),
        enuc=0.5, nelec=(2, 2),
    )


class TestFcidump:
    """The FCIDUMP external format, over the combined basis of every k-point."""

    def test_it_matches_the_writer_on_a_full_momentum_set(self, tmp_path):
        hamiltonian = _kpoint_hamiltonian()
        chol = [hamiltonian.chol[Q] for Q in range(hamiltonian.nkpts)]

        from_method = tmp_path / 'method'
        hamiltonian.to_fcidump(from_method, tol=1e-12)

        direct = tmp_path / 'direct'
        write_fcidump_kpoint(direct, hamiltonian.hcore, chol, hamiltonian.enuc,
                             hamiltonian.nmo_tot, hamiltonian.nelec,
                             hamiltonian.nmo_pk, hamiltonian.nchol_pk,
                             hamiltonian.qk_to_k2, tol=1e-12)

        assert from_method.read_text() == direct.read_text()

    def test_the_unstored_momentum_transfer_is_reconstructed(self):
        r"""
        :math:`L^{-Q}_{k_1}[i, j, n] = (L^{Q}_{k_2}[j, i, n])^*` with
        :math:`k_2 = \mathrm{qk\_to\_k2}[-Q, k_1]`, and the vector count comes
        from the partner, where the factorization actually ran.
        """
        hamiltonian = _mirrored_hamiltonian()
        nkpts, nmo, nchol = hamiltonian.nkpts, hamiltonian.nmo_max, 2

        chol, nchol_pk = hamiltonian._chol_all_momenta()

        assert len(chol) == nkpts
        assert np.array_equal(nchol_pk, [nchol, nchol, nchol])
        assert np.array_equal(hamiltonian.nchol_pk, [nchol, nchol, 0])

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

    def test_uneven_orbital_counts_are_rejected(self, tmp_path):
        hamiltonian = _kpoint_hamiltonian(nkpts=2, nmo=3)
        hamiltonian.nmo_pk = np.array([3, 2])

        with pytest.raises(ValueError, match="different orbital counts"):
            hamiltonian.to_fcidump(tmp_path / 'FCIDUMP')

    def test_a_spinor_basis_is_not_implemented(self, tmp_path):
        with pytest.raises(NotImplementedError, match="spatial-orbital basis"):
            _kpoint_hamiltonian().to_fcidump(tmp_path / 'FCIDUMP', use_spinor=True)


# ----------------------------------------------------------------------
# the PySCF-backed generation path
# ----------------------------------------------------------------------

@pytest.mark.pyscf
class TestGeneration:

    @pytest.fixture(scope='class')
    def scf_data(self, diamond, diamond_lda):
        mf, kpts = diamond_lda
        X, nmo_pk = get_ortho_ao(diamond, kpts)
        return {'cell': diamond, 'kpts': kpts, 'hcore': mf.get_hcore(),
                'X': X, 'nmo_pk': nmo_pk}

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

    def test_kpoint_hamiltonian_round_trips(self, scf_data, tmp_path):
        hamiltonian = PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=True, chol_cut=1e-3, maxvecs=20)

        assert hamiltonian.nkpts == 2
        assert set(hamiltonian.chol) == {0, 1}
        assert all(n > 0 for n in hamiltonian.nchol_pk)

        path = tmp_path / 'kp.h5'
        hamiltonian.to_hdf5(path)
        assert hamiltonian_format(path) == 'kpoint'

        restored = Hamiltonian.from_hdf5(path)
        assert np.allclose(restored.chol[0], hamiltonian.chol[0])

    def test_supercell_hamiltonian_is_a_gamma_point_kpoint_hamiltonian(self, scf_data,
                                                                       tmp_path):
        """
        A supercell Hamiltonian is the Γ point of the supercell, so it comes back
        in the ordinary k-point representation with a single k-point.
        """
        supercell = PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=False, chol_cut=1e-3, maxvecs=20)
        original_nkpts = len(scf_data['kpts'])

        assert supercell.nkpts == 1
        assert list(supercell.nmo_pk) == [int(np.sum(scf_data['nmo_pk']))]
        assert np.allclose(supercell.kpts, 0.0)
        assert supercell.qk_to_k2.tolist() == [[0]]
        assert supercell.minus_k.tolist() == [0]
        assert set(supercell.chol) == {0}

        nmo_tot = supercell.nmo_tot
        nchol = int(supercell.nchol_pk[0])
        assert supercell.hcore[0].shape == (nmo_tot, nmo_tot)
        assert supercell.chol[0].shape == (1, nmo_tot * nmo_tot * nchol)
        # the combined basis really did absorb the original k-points
        assert nmo_tot == original_nkpts * int(scf_data['nmo_pk'][0])

        path = tmp_path / 'sc.h5'
        supercell.to_hdf5(path)
        assert hamiltonian_format(path) == 'kpoint'

        restored = Hamiltonian.from_hdf5(path)
        assert restored.nkpts == 1
        assert np.allclose(restored.chol[0], supercell.chol[0])

    def test_supercell_cholesky_vectors_stay_complex(self, scf_data, tmp_path):
        """
        The k-point format is complex throughout, which is what the executable's
        ``KPFactorizedHamiltonian`` reads — no real-valued special case.
        """
        path = tmp_path / 'sc.h5'
        PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=False, chol_cut=1e-3, maxvecs=20).to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/KPFactorized/L0'].shape[-1] == 2
            assert fh5['Hamiltonian/ComplexIntegrals'][0] == 1

    def test_the_one_body_hamiltonian_is_block_diagonal_in_k(self, scf_data):
        """
        The one-body operator conserves crystal momentum, so the combined
        supercell hcore couples no two different original k-points.
        """
        supercell = PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=False, chol_cut=1e-2, maxvecs=20)

        hcore = supercell.hcore[0]
        offset = int(scf_data['nmo_pk'][0])
        assert np.allclose(hcore[:offset, offset:], 0.0)
        assert np.allclose(hcore[offset:, :offset], 0.0)

    def test_the_two_representations_agree_on_the_two_body_energy(self, scf_data):
        r"""
        Both factorizations reconstruct the same two-body integrals, so
        :math:`\sum_\gamma |L^\gamma_{ii}|^2` — the diagonal Coulomb weight —
        must match between them, to the Cholesky tolerance.
        """
        chol_cut = 1e-4
        supercell = PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=False, chol_cut=chol_cut, maxvecs=20)
        kpoint = PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=True, chol_cut=chol_cut, maxvecs=20)

        nmo_tot = supercell.nmo_tot
        nchol_sc = int(supercell.nchol_pk[0])
        L_sc = supercell.chol[0].reshape(nmo_tot * nmo_tot, nchol_sc)
        sc_trace = np.einsum('ig,ig->i', L_sc, L_sc.conj()).real.reshape(
            nmo_tot, nmo_tot).diagonal().sum()

        # the Q = 0 block holds the k-diagonal pair densities
        nmo = int(kpoint.nmo_pk[0])
        nchol = int(kpoint.nchol_pk[0])
        L0 = kpoint.chol[0].reshape(kpoint.nkpts, nmo * nmo, nchol)
        kp_trace = sum(
            np.einsum('ig,ig->', L0[k, ::nmo + 1, :], L0[k, ::nmo + 1, :].conj()).real
            for k in range(kpoint.nkpts)
        )

        assert np.isclose(sc_trace, kp_trace, rtol=1e-2)
