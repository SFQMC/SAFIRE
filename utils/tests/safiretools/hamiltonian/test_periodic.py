# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`PeriodicHamiltonian`: work partitioning, momentum bookkeeping, and the two
representations the one `kp_sym`-flagged solver produces."""

import h5py as h5
import numpy as np
import pytest

from safiretools.hamiltonian.base import Hamiltonian, hamiltonian_format
from safiretools.hamiltonian.periodic import (
    FileHandler,
    Partition,
    PeriodicHamiltonian,
    _SerialComm,
    bisect,
    construct_qk_maps,
    fair_share,
    generate_grid_shifts,
    get_ortho_ao,
    rank_filename,
    setup_basis_map,
)


class FakeComm:
    def __init__(self, size=1, rank=0):
        self.size = size
        self.rank = rank


# ----------------------------------------------------------------------
# partitioning, no PySCF needed
# ----------------------------------------------------------------------

class TestFairShare:

    def test_it_covers_every_item_exactly_once(self):
        covered = []
        for rank in range(5):
            i0, iN = fair_share(23, 5, rank)
            covered.extend(range(i0, iN))
        assert covered == list(range(23))

    def test_the_remainder_goes_to_the_first_ranks(self):
        sizes = [iN - i0 for i0, iN in (fair_share(23, 5, rank) for rank in range(5))]
        assert sizes == [5, 5, 5, 4, 4]


class TestBisect:

    def test_it_inserts_after_equal_entries(self):
        assert bisect([1, 3, 3, 5], 3) == 3
        assert bisect([1, 3, 3, 5], 0) == 0
        assert bisect([1, 3, 3, 5], 9) == 4

    def test_a_negative_lower_bound_is_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            bisect([1, 2], 1, lo=-1)


class TestPartition:

    def test_serial_partition_owns_everything(self):
        part = Partition(FakeComm(1), 20, 16, 8, 2, kp_sym=False)

        assert (part.kk0, part.kkN, part.nkk) == (0, 4, 4)
        assert (part.ij0, part.ijN, part.nij) == (0, 64, 64)
        assert part.nproc_pk == 1
        assert list(part.n2k1) == [0, 0, 1, 1]
        assert list(part.n2k2) == [0, 1, 0, 1]

    def test_kp_sym_partitions_over_single_kpoints(self):
        part = Partition(FakeComm(1), 20, 16, 8, 2, kp_sym=True)

        assert (part.kk0, part.kkN, part.nkk) == (0, 2, 2)
        assert not hasattr(part, 'n2k1')

    def test_more_ranks_than_kpoints_splits_orbital_pairs(self):
        parts = [Partition(FakeComm(4, rank), 20, 16, 8, 2, kp_sym=True)
                 for rank in range(4)]

        assert [p.nproc_pk for p in parts] == [2] * 4
        assert [p.kk0 for p in parts] == [0, 0, 1, 1]
        assert [(p.ij0, p.ijN) for p in parts] == [(0, 32), (32, 64)] * 2

    def test_ranks_must_divide_the_kpoint_count(self):
        with pytest.raises(ValueError, match="evenly divide"):
            Partition(FakeComm(3), 20, 16, 8, 2, kp_sym=True)


def test_setup_basis_map_numbers_orbitals_consecutively():
    ik2n, nmo_tot = setup_basis_map([3, 2], nkpts=2)

    assert nmo_tot == 5
    assert list(ik2n[:, 0]) == [0, 1, 2]
    # the second k-point has one fewer orbital, so the last slot stays unset
    assert list(ik2n[:, 1]) == [3, 4, -1]


def test_rank_filename_prefixes_only_the_basename(tmp_path):
    """
    afqmctools built this as ``"rank1_" + filename``, which produced an
    unopenable path whenever the file was not in the working directory.
    """
    name = rank_filename(1, tmp_path / 'ham.h5')

    assert name.endswith('rank1_ham.h5')
    assert name.startswith(str(tmp_path))


def test_file_handler_gives_each_rank_its_own_file(tmp_path):
    path = tmp_path / 'ham.h5'

    with FileHandler(FakeComm(2, rank=1), path) as fh5:
        fh5.create_dataset('x', data=1)

    assert (tmp_path / 'rank1_ham.h5').exists()
    assert not path.exists()


class TestSerialComm:

    def test_allgather_copies_the_send_buffer(self):
        comm = _SerialComm()
        recvbuf = np.zeros(5)
        comm.Allgather(np.arange(5, dtype=float), recvbuf)

        assert np.allclose(recvbuf, np.arange(5))


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

    def test_write_from_pyscf_matches_from_pyscf(self, scf_data, tmp_path):
        """The streaming writer and the in-memory factory must agree."""
        streamed = tmp_path / 'streamed.h5'
        PeriodicHamiltonian.write_from_pyscf(
            _SerialComm(), scf_data, streamed, kpoint_symmetry=True,
            chol_cut=1e-3, maxvecs=20)

        # the streaming writer tags the file after closing it, from one rank;
        # the comparison below reads names out of this file, so it alone would
        # not notice the tag going missing here
        with h5.File(streamed, 'r') as fh5:
            assert fh5['Hamiltonian/type'].asstr()[()] == 'KPFactorized'

        in_memory = tmp_path / 'in_memory.h5'
        PeriodicHamiltonian.from_pyscf(
            scf_data, kpoint_symmetry=True, chol_cut=1e-3,
            maxvecs=20).to_hdf5(in_memory)

        with h5.File(streamed) as a, h5.File(in_memory) as b:
            names = []
            a.visititems(lambda n, o: names.append(n) if isinstance(o, h5.Dataset) else None)
            assert names
            for name in names:
                assert name in b, name
                if h5.check_string_dtype(a[name].dtype):
                    assert a[name][()] == b[name][()], name
                else:
                    assert np.allclose(a[name][...], b[name][...]), name

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

    def test_from_pyscf_refuses_to_run_in_parallel(self, scf_data):
        with pytest.raises(ValueError, match="serial-only"):
            PeriodicHamiltonian.from_pyscf(scf_data, comm=FakeComm(4))

    def test_write_from_pyscf_refuses_a_distributed_supercell(self, scf_data, tmp_path):
        with pytest.raises(NotImplementedError, match="in parallel is not implemented"):
            PeriodicHamiltonian.write_from_pyscf(
                FakeComm(4), scf_data, tmp_path / 'sc.h5', kpoint_symmetry=False)
