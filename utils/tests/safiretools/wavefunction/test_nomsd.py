# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`NOMSDWavefunction`: construction, round trips, and spin-symmetry inference."""

import warnings

import h5py as h5
import numpy as np
import pytest

from safiretools import NOMSDWavefunction, SpinSymm, Wavefunction
from safiretools.wavefunction.nomsd import infer_spin_symm


class TestConstruction:

    @pytest.mark.parametrize('spin_symm, nelec, ncols', [
        ('closed', (3, 3), 3),
        ('collinear', (3, 2), 5),
        ('collinear', (3, 0), 3),
        ('noncollinear', (3, 2), 5),
    ])
    def test_the_column_layout_follows_the_spin_symmetry(self, make_nomsd,
                                                         spin_symm, nelec,
                                                         ncols):
        wavefunction = make_nomsd(spin_symm, nelec=nelec, nmo=6)

        assert wavefunction.dets.shape == (1, wavefunction.nrows, ncols)

    def test_nmo_is_inferred_from_the_determinants(self, orthonormal):
        wavefunction = NOMSDWavefunction(
            coeffs=[1.0], dets=orthonormal(12, 5)[np.newaxis], nelec=(3, 2),
            spin_symm='noncollinear')

        assert wavefunction.nmo == 6
        assert wavefunction.nrows == 12

    def test_a_wrong_column_count_is_rejected(self, orthonormal):
        with pytest.raises(ValueError, match="dets has shape"):
            NOMSDWavefunction(coeffs=[1.0], dets=orthonormal(6, 4)[np.newaxis],
                              nelec=(3, 2), spin_symm='collinear', nmo=6)

    def test_a_wrong_determinant_count_is_rejected(self, orthonormal):
        dets = np.array([orthonormal(6, 3)])

        with pytest.raises(ValueError, match="dets has shape"):
            NOMSDWavefunction(coeffs=[1.0, 0.5], dets=dets, nelec=(3, 3),
                              spin_symm='closed', nmo=6)

    def test_spin_blocks_views_the_channels(self, make_nomsd):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), nmo=6)
        alpha, beta = wavefunction.spin_blocks(0)

        assert alpha.shape == (6, 3)
        assert beta.shape == (6, 2)


class TestRoundTrip:

    @pytest.mark.parametrize('spin_symm, nelec, ndets', [
        ('closed', (3, 3), 1),
        ('closed', (4, 4), 3),
        ('collinear', (3, 2), 1),
        ('collinear', (4, 2), 2),
        ('collinear', (3, 0), 1),
        ('noncollinear', (3, 2), 1),
        ('noncollinear', (4, 4), 2),
    ])
    def test_it_round_trips(self, make_nomsd, tmp_path, spin_symm, nelec, ndets):
        wavefunction = make_nomsd(spin_symm, nelec=nelec, nmo=6, ndets=ndets)
        path = tmp_path / 'wfn.h5'

        # the collinear default-psi0 warning is covered in test_base.py
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wavefunction.to_hdf5(path)
        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nmo == wavefunction.nmo
        assert read_back.spin_symm is wavefunction.spin_symm
        assert read_back.nelec == wavefunction.nelec_on_disk
        assert read_back.ndets == ndets
        assert np.allclose(read_back.coeffs, wavefunction.coeffs)
        assert np.allclose(read_back.dets, wavefunction.dets)
        for read, original in zip(read_back.psi0, wavefunction.psi0):
            assert np.allclose(read, original)

    def test_a_polarized_wavefunction_writes_zero_width_beta_blocks(
            self, make_nomsd, tmp_path):
        wavefunction = make_nomsd('collinear', nelec=(3, 0), nmo=6)
        path = tmp_path / 'polarized.h5'

        with pytest.warns(UserWarning):
            wavefunction.to_hdf5(path)

        with h5.File(path, 'r') as fh5:
            group = fh5['Wavefunction/NOMSD']
            assert list(group['dims'][...]) == [6, 3, 0,
                                                int(SpinSymm.COLLINEAR), 1]
            assert group['Psi0_beta'].shape == (6, 0, 2)
            assert list(group['PsiT_1/dims'][...]) == [0, 6, 0]

    def test_an_explicit_psi0_survives(self, make_nomsd, orthonormal, tmp_path):
        wavefunction = make_nomsd('collinear', nelec=(3, 2), nmo=6)
        wavefunction.psi0 = (orthonormal(6, 3), orthonormal(6, 2))
        path = tmp_path / 'wfn.h5'

        wavefunction.to_hdf5(path)
        read_back = Wavefunction.from_hdf5(path)

        assert np.allclose(read_back.psi0[0], wavefunction.psi0[0])
        assert np.allclose(read_back.psi0[1], wavefunction.psi0[1])

    def test_only_the_requested_determinants_come_back(self, make_nomsd,
                                                       tmp_path):
        wavefunction = make_nomsd('closed', nelec=(3, 3), ndets=4)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        assert Wavefunction.from_hdf5(path).ndets == 4


class TestInferSpinSymm:

    @pytest.mark.parametrize('shape, nelec, expected', [
        ((6, 3), (3, 3), SpinSymm.CLOSED),
        ((6, 5), (3, 2), SpinSymm.COLLINEAR),
        ((6, 6), (3, 3), SpinSymm.COLLINEAR),
        ((6, 3), (3, 0), SpinSymm.COLLINEAR),
        ((12, 5), (3, 2), SpinSymm.NONCOLLINEAR),
    ])
    def test_it_reads_the_symmetry_off_the_shape(self, shape, nelec, expected):
        assert infer_spin_symm(np.zeros(shape), nelec, nmo=6) is expected

    def test_an_equal_population_collinear_matrix_is_not_closed_shell(self):
        # afqmctools tested nup == ndown alone and read this as closed shell,
        #   silently halving it
        assert infer_spin_symm(np.zeros((6, 6)), (3, 3), nmo=6) \
            is SpinSymm.COLLINEAR

    def test_a_polarized_matrix_is_collinear(self):
        # no beta electrons: one nup-wide block over nmo rows, ndown == 0
        assert infer_spin_symm(np.zeros((6, 3)), (3, 0), nmo=6) \
            is SpinSymm.COLLINEAR

    def test_a_wrong_row_count_is_rejected(self):
        with pytest.raises(ValueError, match="rows, expected 6"):
            infer_spin_symm(np.zeros((7, 3)), (3, 3), nmo=6)

    def test_a_wrong_column_count_is_rejected(self):
        with pytest.raises(ValueError, match="columns, which matches neither"):
            infer_spin_symm(np.zeros((6, 4)), (3, 3), nmo=6)

    def test_a_non_matrix_is_rejected(self):
        with pytest.raises(ValueError, match="2-dimensional orbital matrix"):
            infer_spin_symm(np.zeros((1, 6, 3)), (3, 3), nmo=6)
