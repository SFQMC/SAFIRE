# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""The native wavefunction HDF5 schema, exercised directly."""

import warnings

import h5py as h5
import numpy as np
import pytest

from safiretools import SpinSymm
from safiretools.wavefunction import io
from safiretools.wavefunction.slater import (
    CONDITION_MAX,
    overlap_condition_number,
)


@pytest.fixture
def group(tmp_path):
    with h5.File(tmp_path / 'schema.h5', 'w') as fh5:
        yield fh5.create_group('Wavefunction/NOMSD')


class TestHeader:

    def test_dims_records_the_shape_the_executable_reads(self, group):
        io.write_header(group, spin_symm=SpinSymm.COLLINEAR,
                        coeffs=np.array([1.0 + 0j, 0.5 + 0j]),
                        psi0=(np.eye(6, 3) + 0j, np.eye(6, 2) + 0j))

        dims = group['dims'][...]
        assert list(dims) == [6, 3, 2, int(SpinSymm.COLLINEAR), 2]
        assert dims.dtype == np.int32

    def test_it_round_trips(self, group, rng):
        coeffs = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi0 = (rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3)),
                rng.normal(size=(6, 2)) + 1j * rng.normal(size=(6, 2)))

        io.write_header(group, spin_symm=SpinSymm.COLLINEAR, coeffs=coeffs,
                        psi0=psi0)
        header = io.read_header(group)

        assert header['nmo'] == 6
        assert header['nelec'] == (3, 2)
        assert header['spin_symm'] is SpinSymm.COLLINEAR
        assert header['ndets'] == 3
        assert np.allclose(header['coeffs'], coeffs)
        assert np.allclose(header['psi0'][0], psi0[0])
        assert np.allclose(header['psi0'][1], psi0[1])

    def test_a_single_channel_symmetry_writes_no_beta_block(self, group):
        io.write_header(group, spin_symm=SpinSymm.CLOSED,
                        coeffs=np.array([1.0 + 0j]),
                        psi0=(np.eye(6, 3) + 0j,))

        assert 'Psi0_beta' not in group
        assert len(io.read_header(group)['psi0']) == 1

    def test_an_empty_beta_channel_still_gets_its_block(self, group):
        # a wavefunction with no beta electrons is collinear with ndown == 0,
        #   and the executable's reader opens Psi0_beta for any collinear file
        io.write_header(group, spin_symm=SpinSymm.COLLINEAR,
                        coeffs=np.array([1.0 + 0j]),
                        psi0=(np.eye(6, 3) + 0j,
                              np.zeros((6, 0), dtype=complex)))

        assert group['Psi0_beta'].shape == (6, 0, 2)
        assert io.read_header(group)['psi0'][1].shape == (6, 0)


class TestOrbitals:

    def test_they_are_stored_conjugate_transposed(self, group, rng):
        orbitals = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        io.write_orbitals(group, 'PsiT_0', orbitals)

        # the executable reads an (nelec, npol*nmo) CSR matrix
        assert list(group['PsiT_0/dims'][...])[:2] == [3, 6]
        assert np.allclose(io.read_orbitals(group, 'PsiT_0'), orbitals)

    def test_the_index_datasets_are_int32(self, group, rng):
        io.write_orbitals(group, 'PsiT_0', rng.normal(size=(4, 2)) + 0j)

        for name in ('dims', 'jdata_', 'pointers_begin_', 'pointers_end_'):
            assert group[f'PsiT_0/{name}'].dtype == np.int32

    def test_an_empty_block_round_trips(self, group):
        io.write_orbitals(group, 'PsiT_1', np.zeros((6, 0), dtype=complex))

        assert list(group['PsiT_1/dims'][...]) == [0, 6, 0]
        assert group['PsiT_1/pointers_begin_'].shape == (0,)
        assert io.read_orbitals(group, 'PsiT_1').shape == (6, 0)

    def test_zeros_are_not_stored(self, group):
        orbitals = np.zeros((6, 2), dtype=complex)
        orbitals[0, 0] = 1.0
        orbitals[3, 1] = 1.0
        io.write_orbitals(group, 'PsiT_0', orbitals)

        assert int(group['PsiT_0/dims'][2]) == 2


class TestNomsdPayload:

    @pytest.mark.parametrize('nelec_per_spin, expected', [
        ((3,), ['PsiT_0', 'PsiT_1']),
        ((3, 2), ['PsiT_0', 'PsiT_1', 'PsiT_2', 'PsiT_3']),
    ])
    def test_collinear_determinants_interleave_the_spin_channels(
            self, group, rng, nelec_per_spin, expected):
        ncols = sum(nelec_per_spin)
        dets = rng.normal(size=(2, 6, ncols)) + 0j
        io.write_nomsd(group, dets, nelec_per_spin)

        assert sorted(name for name in group if name.startswith('PsiT')) \
            == expected
        assert np.allclose(io.read_nomsd(group, 2, nelec_per_spin), dets)

    def test_the_orbital_index_is_determinant_major(self):
        assert io.nomsd_orbital_index(0, 0, 2) == 0
        assert io.nomsd_orbital_index(0, 1, 2) == 1
        assert io.nomsd_orbital_index(1, 0, 2) == 2
        assert io.nomsd_orbital_index(2, 0, 1) == 2

    def test_small_coefficients_are_thresholded(self, group):
        dets = np.ones((1, 4, 2), dtype=complex)
        dets[0, 0, 0] = 1e-12
        io.write_nomsd(group, dets, (2,))

        assert int(group['PsiT_0/dims'][2]) == 7

    def test_a_finite_temperature_group_is_reported_clearly(self, group):
        group['UL_0/dims'] = np.array([1, 1, 1], dtype=np.int32)

        with pytest.raises(ValueError, match="no PsiT_0 group"):
            io.read_nomsd(group, 1, (2,))


class TestHeaderDims:
    """
    ``nmo`` and the electron counts are not inputs: `psi0`'s shape and the spin
    symmetry fix both.
    """

    @pytest.mark.parametrize('spin_symm, widths, nmo, nelec', [
        (SpinSymm.CLOSED, (3,), 6, (3, 3)),
        (SpinSymm.COLLINEAR, (3, 2), 6, (3, 2)),
        (SpinSymm.COLLINEAR, (3, 0), 6, (3, 0)),
        (SpinSymm.NONCOLLINEAR, (5,), 6, (5, 0)),
    ])
    def test_it_reads_them_off_psi0(self, spin_symm, widths, nmo, nelec):
        npol = 2 if spin_symm is SpinSymm.NONCOLLINEAR else 1
        psi0 = tuple(np.zeros((npol * nmo, width)) for width in widths)

        assert io.header_dims(spin_symm, psi0) == (nmo, nelec)

    def test_a_wrong_block_count_is_rejected(self):
        with pytest.raises(ValueError, match="1 spin channel"):
            io.header_dims(SpinSymm.CLOSED, (np.zeros((6, 3)),
                                             np.zeros((6, 3))))

        with pytest.raises(ValueError, match="2 spin channel"):
            io.header_dims(SpinSymm.COLLINEAR, (np.zeros((6, 3)),))

    def test_a_noncollinear_block_needs_an_even_row_count(self):
        with pytest.raises(ValueError, match="even number"):
            io.header_dims(SpinSymm.NONCOLLINEAR, (np.zeros((7, 3)),))


class TestConditionNumberOnDisk:
    """
    Every Slater matrix that reaches disk has its overlap's condition number
    checked, because AFQMC inverts that overlap. The check sits *after*
    sparsifying, which is the state actually written — screening small values
    can only make the conditioning worse.
    """

    def test_a_well_conditioned_block_writes_silently(self, group, rng):
        block = rng.normal(size=(8, 2))
        assert overlap_condition_number(block) < CONDITION_MAX

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            io.write_nomsd(group, block[np.newaxis], (2,))

    def test_an_ill_conditioned_block_is_reported(self, group):
        block = np.eye(8, 2)
        block[:, 1] = block[:, 0] + 1e-12 * block[:, 1]

        with pytest.warns(UserWarning,
                          match=r"ill-conditioned overlap matrix: PsiT_0"):
            io.write_nomsd(group, block[np.newaxis], (2,))

    def test_an_all_zero_block_is_reported(self, group):
        """
        The degenerate case: a zero overlap matrix is singular, so its condition
        number is infinite rather than undefined.
        """
        assert overlap_condition_number(np.zeros((8, 2))) == np.inf

        with pytest.warns(UserWarning, match="cond inf"):
            io.write_nomsd(group, np.zeros((1, 8, 2)), (2,))

    def test_an_empty_block_is_fine(self, group):
        """
        A wavefunction with no beta electrons writes a zero-width beta block,
        whose overlap is the empty identity.
        """
        assert overlap_condition_number(np.zeros((8, 0))) == 1.0

        det = np.zeros((8, 2))
        det[0, 0] = det[1, 1] = 1.0
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            io.write_nomsd(group, det[np.newaxis], (2, 0))

    def test_sparsifying_can_be_what_breaks_it(self, group):
        """
        A block whose smallest singular value is carried entirely by entries
        below the threshold becomes exactly singular once they are screened.
        """
        block = np.zeros((8, 2))
        block[0, 0] = 1.0
        block[1, 1] = 1e-10
        assert np.min(np.abs(block[block != 0])) < io.DEFAULT_THRESHOLD

        with pytest.warns(UserWarning, match="cond inf"):
            io.write_nomsd(group, block[np.newaxis], (2,))

        # the screened column is gone entirely, leaving a rank-1 block
        written = io.read_nomsd(group, 1, (2,))[0]
        assert np.count_nonzero(written) == 1
        assert written[0, 0] == 1.0

    def test_the_written_block_is_not_repaired(self, group):
        block = np.eye(8, 2)
        block[:, 1] = block[:, 0] + 1e-12 * block[:, 1]

        with pytest.warns(UserWarning):
            io.write_nomsd(group, block[np.newaxis], (2,))

        assert np.allclose(io.read_nomsd(group, 1, (2,))[0], block)

    def test_the_spin_channels_are_checked_separately(self, group):
        """
        Alpha and beta columns need not be orthogonal to each other; only the
        columns *within* a channel matter. Identical alpha and beta orbitals are
        a perfectly ordinary collinear determinant.
        """
        det = np.zeros((8, 2))
        det[0, 0] = 1.0
        det[0, 1] = 1.0

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            io.write_nomsd(group, det[np.newaxis], (1, 1))

    def test_every_offending_block_is_named(self, group):
        dets = np.zeros((2, 8, 4))      # every block singular

        with pytest.warns(UserWarning) as record:
            io.write_nomsd(group, dets, (2, 2))

        message = str(record[0].message)
        for name in ('PsiT_0', 'PsiT_1', 'PsiT_2', 'PsiT_3'):
            assert name in message

    def test_phmsd_orbital_references_are_checked_too(self, group):
        occa, occb = np.array([[0, 1]]), np.array([[0, 1]])
        reference = np.zeros((6, 6))

        with pytest.warns(UserWarning,
                          match=r"ill-conditioned overlap matrix: PsiT_0"):
            io.write_phmsd(group, occa, occb, nmo=6, orbitals=[reference])


class TestPhmsdPayload:

    @pytest.fixture
    def occupations(self):
        return (np.array([[0, 1, 2], [0, 1, 3]]), np.array([[0, 1], [0, 2]]))

    def test_beta_indices_are_offset_by_nmo_on_disk(self, group, occupations):
        occa, occb = occupations
        io.write_phmsd(group, occa, occb, nmo=6)

        occs = group['occs'][...].reshape(2, 5)
        assert np.array_equal(occs[:, :3], occa)
        assert np.array_equal(occs[:, 3:], occb + 6)
        assert group['occs'].dtype == np.int32

    def test_it_round_trips(self, group, occupations):
        occa, occb = occupations
        io.write_phmsd(group, occa, occb, nmo=6)

        read_a, read_b, orbitals = io.read_phmsd(group, 2, (3, 2), 6)
        assert np.array_equal(read_a, occa)
        assert np.array_equal(read_b, occb)
        assert orbitals is None

    @pytest.mark.parametrize('nreferences', [0, 1, 2])
    def test_type_counts_the_references_actually_written(self, group,
                                                         occupations, rng,
                                                         nreferences):
        occa, occb = occupations
        references = [rng.normal(size=(6, 6)) + 0j
                      for _ in range(nreferences)] or None

        io.write_phmsd(group, occa, occb, nmo=6, orbitals=references)

        assert int(group['type'][()]) == nreferences
        assert sorted(name for name in group if name.startswith('PsiT')) \
            == [f'PsiT_{i}' for i in range(nreferences)]

    def test_a_none_reference_does_not_count(self, group, occupations, rng):
        # afqmctools crashed here: it wrote type=1 for any orbmat and then
        #   dereferenced a None beta matrix
        occa, occb = occupations
        io.write_phmsd(group, occa, occb, nmo=6,
                       orbitals=[rng.normal(size=(6, 6)) + 0j, None])

        assert int(group['type'][()]) == 1

    def test_references_round_trip(self, group, occupations, rng):
        occa, occb = occupations
        references = [rng.normal(size=(6, 6)) + 0j, rng.normal(size=(6, 6)) + 0j]
        io.write_phmsd(group, occa, occb, nmo=6, orbitals=references)

        _, _, read_back = io.read_phmsd(group, 2, (3, 2), 6)
        assert len(read_back) == 2
        assert np.allclose(read_back[0], references[0])
        assert np.allclose(read_back[1], references[1])

    def test_mismatched_determinant_counts_are_rejected(self, group):
        with pytest.raises(ValueError, match="different numbers of determinants"):
            io.write_phmsd(group, np.array([[0, 1], [0, 2]]),
                           np.array([[0]]), nmo=6)
