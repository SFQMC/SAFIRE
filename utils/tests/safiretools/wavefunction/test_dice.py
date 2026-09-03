# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`PHMSDWavefunction.from_dice`: reading Dice's HDF5 and text output."""

import h5py as h5
import numpy as np
import pytest

from safiretools import PHMSDWavefunction, SpinSymm, Wavefunction
from safiretools.wavefunction.dice import (
    from_dice,
    has_real_coefficients,
    parse_coefficient,
)

REAL_LOG = """Some Dice preamble
Printing most important determinants
 State :   0
0    -0.5032009288     2 2 2 2 0   0 0 0 0 0   0
1     0.0883320042     2 2 2 0 0   0 2 0 0 0   0
2    -0.0928684363     2 2 0 2 2   0 0 0 0 0   0
3     0.0804092922     2 2 0 2 a   b 0 0 0 0   0
trailing junk
"""

COMPLEX_LOG = """Printing most important determinants
 State :   0
0    -0.5032009288     0.7487247473 2 2 2 2 0   0 0 0 0 0   0
1     0.0883320042     0.0439272635 2 2 2 0 0   0 2 0 0 0   0
"""

TWO_STATE_LOG = """Printing most important determinants
 State :   0
0    -0.9000000000     2 2 0 0
1     0.1000000000     2 0 2 0
 State :   1
0     0.8000000000     2 0 0 2
1    -0.2000000000     0 2 2 0
"""

POLARIZED_LOG = """Printing most important determinants
 State :   0
0    -0.9000000000     a a 0 0
1     0.1000000000     a 0 a 0
"""


@pytest.fixture
def real_log(tmp_path):
    path = tmp_path / 'dice.out'
    path.write_text(REAL_LOG)
    return path


@pytest.fixture
def dice_hdf5(tmp_path):
    path = tmp_path / 'dice.h5'
    configurations = np.array(
        [[ord(character) for character in row] for row in
         ('22220000000', '22200200000', '2202ab00000')], dtype=np.int8)

    with h5.File(path, 'w') as fh5:
        fh5['/nroots'] = np.array([1])
        fh5['/ndets'] = np.array([3])
        fh5['/norbs'] = np.array([11])
        fh5['/coeff_r0'] = np.array([-0.5032009288, 0.0883320042,
                                     0.0804092922])
        fh5['/confg_r0'] = configurations

    return path


class TestAsciiOutput:

    def test_it_reads_the_occupations(self, real_log):
        wavefunction = from_dice(real_log, ndets=4)

        assert isinstance(wavefunction, PHMSDWavefunction)
        assert wavefunction.nmo == 11
        assert wavefunction.nelec == (4, 4)
        assert wavefunction.ndets == 4
        assert np.array_equal(wavefunction.occa[0], [0, 1, 2, 3])
        assert np.array_equal(wavefunction.occb[0], [0, 1, 2, 3])
        # '2 2 0 2 a   b' puts alpha on orbital 4 and beta on orbital 5
        assert np.array_equal(wavefunction.occa[3], [0, 1, 3, 4])
        assert np.array_equal(wavefunction.occb[3], [0, 1, 3, 5])

    def test_the_coefficients_are_read_in_file_order(self, real_log):
        wavefunction = from_dice(real_log, ndets=4)

        assert np.allclose(wavefunction.coeffs,
                           [-0.5032009288, 0.0883320042, -0.0928684363,
                            0.0804092922])

    def test_fewer_determinants_can_be_requested(self, real_log):
        assert from_dice(real_log, ndets=2).ndets == 2

    def test_a_complex_coefficient_column_is_recognized(self, tmp_path):
        path = tmp_path / 'complex.out'
        path.write_text(COMPLEX_LOG)

        wavefunction = from_dice(path, ndets=2)

        # afqmctools counted the imaginary column as an orbital and reported 12
        assert wavefunction.nmo == 11
        assert np.allclose(wavefunction.coeffs[0],
                           -0.5032009288 + 0.7487247473j)

    def test_the_requested_state_is_selected(self, tmp_path):
        path = tmp_path / 'two_states.out'
        path.write_text(TWO_STATE_LOG)

        first = from_dice(path, ndets=2, state=0)
        second = from_dice(path, ndets=2, state=1)

        assert np.allclose(first.coeffs, [-0.9, 0.1])
        assert np.allclose(second.coeffs, [0.8, -0.2])
        assert np.array_equal(second.occa[0], [0, 3])

    def test_a_polarized_expansion_is_collinear_with_no_beta_electrons(
            self, tmp_path):
        # this was afqmctools' walker_type 4, which no longer exists
        path = tmp_path / 'polarized.out'
        path.write_text(POLARIZED_LOG)

        wavefunction = from_dice(path, ndets=2)

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (2, 0)
        assert wavefunction.occb.shape == (2, 0)

    def test_a_missing_state_is_reported(self, real_log):
        with pytest.raises(ValueError, match="end of"):
            from_dice(real_log, ndets=4, state=7)

    def test_an_unknown_occupation_character_is_reported(self, tmp_path):
        path = tmp_path / 'bad.out'
        path.write_text(REAL_LOG.replace('2 2 2 2 0', '2 2 2 2 x'))

        with pytest.raises(ValueError, match="unknown occupation character"):
            from_dice(path, ndets=1)


class TestHdf5Output:

    def test_it_reads_the_occupations(self, dice_hdf5):
        wavefunction = from_dice(dice_hdf5, ndets=3)

        assert wavefunction.nmo == 11
        assert wavefunction.nelec == (4, 4)
        assert wavefunction.ndets == 3

    def test_determinants_come_back_largest_coefficient_first(self, dice_hdf5):
        wavefunction = from_dice(dice_hdf5, ndets=3)
        magnitudes = np.abs(wavefunction.coeffs)

        assert np.all(np.diff(magnitudes) <= 0)

    def test_an_out_of_range_state_is_reported(self, dice_hdf5):
        with pytest.raises(ValueError, match="state 5 does not exist"):
            from_dice(dice_hdf5, ndets=3, state=5)

    def test_asking_for_too_many_determinants_is_reported(self, dice_hdf5):
        with pytest.raises(ValueError, match="fewer than the 99 requested"):
            from_dice(dice_hdf5, ndets=99)

    def test_it_round_trips(self, dice_hdf5, tmp_path):
        wavefunction = from_dice(dice_hdf5, ndets=3)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, PHMSDWavefunction)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert np.array_equal(read_back.occb, wavefunction.occb)
        assert np.allclose(read_back.coeffs, wavefunction.coeffs)
        assert read_back.nmo == 11


class TestArgumentValidation:

    @pytest.mark.parametrize('ndets', [0, -1, 1.5, None])
    def test_ndets_must_be_a_positive_integer(self, real_log, ndets):
        with pytest.raises(ValueError, match="positive integer"):
            from_dice(real_log, ndets=ndets)


class TestHelpers:

    @pytest.mark.parametrize('text, expected', [
        ('-0.5032009288', -0.5032009288 + 0j),
        ('0.5', 0.5 + 0j),
        ('(0.5,0.25)', 0.5 + 0.25j),
    ])
    def test_a_coefficient_is_parsed(self, text, expected):
        assert parse_coefficient(text) == expected

    def test_an_unparsable_coefficient_is_reported(self):
        with pytest.raises(ValueError, match="Dice CI coefficient"):
            parse_coefficient('not a number')

    @pytest.mark.parametrize('fields, expected', [
        (['0', '-0.503', '2', '2'], True),
        (['0', '-0.503', 'a', '2'], True),
        (['0', '-0.503', '0', '2'], True),
        (['0', '-0.503', '0.748', '2'], False),
        (['0', '-0.503', '-0.748', '2'], False),
    ])
    def test_the_coefficient_column_count_is_detected(self, fields, expected):
        assert has_real_coefficients(fields) is expected

    def test_an_unrecognizable_third_column_is_reported(self):
        with pytest.raises(ValueError, match="Dice determinant line"):
            has_real_coefficients(['0', '-0.503', 'zzz'])


class TestEquivalenceWithAfqmctools:

    def test_the_ascii_reader_agrees(self, real_log):
        from afqmctools.wavefunction.converter import (
            read_dice_ascii_wavefunction)

        (coeffs, occa, occb), nmo, nup, ndown, _ = \
            read_dice_ascii_wavefunction(str(real_log), 4, 0)
        wavefunction = from_dice(real_log, ndets=4)

        assert nmo == wavefunction.nmo
        assert (nup, ndown) == wavefunction.nelec
        assert np.allclose(coeffs, wavefunction.coeffs)
        assert np.array_equal(occa, wavefunction.occa)
        assert np.array_equal(occb, wavefunction.occb)

    def test_the_hdf5_reader_agrees(self, dice_hdf5):
        from afqmctools.wavefunction.converter import read_dice_h5_wavefunction

        (coeffs, occa, occb), nmo, nup, ndown, _ = \
            read_dice_h5_wavefunction(str(dice_hdf5), 3, 0)
        wavefunction = from_dice(dice_hdf5, ndets=3)

        assert nmo == wavefunction.nmo
        assert (nup, ndown) == wavefunction.nelec
        assert np.allclose(coeffs, wavefunction.coeffs)
        assert np.array_equal(occa, wavefunction.occa)
        assert np.array_equal(occb, wavefunction.occb)
