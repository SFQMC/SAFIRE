# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`PHMSDWavefunction`: construction, validation and round trips."""

import h5py as h5
import numpy as np
import pytest

from safiretools import PHMSDWavefunction, SpinSymm, Wavefunction


class TestConstruction:

    def test_nelec_comes_from_the_occupation_widths(self, make_phmsd):
        assert make_phmsd().nelec == (3, 2)

    def test_it_is_always_collinear(self, make_phmsd):
        assert make_phmsd().spin_symm is SpinSymm.COLLINEAR
        assert make_phmsd().nspin == 2
        assert make_phmsd().npol == 1

    @pytest.mark.parametrize('spin_symm', ['closed', 'noncollinear'])
    def test_another_spin_symmetry_is_rejected(self, spin_symm):
        with pytest.raises(ValueError, match="always collinear"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1]]),
                              occb=np.array([[0, 1]]), nmo=4,
                              spin_symm=spin_symm)

    def test_an_empty_beta_channel_is_allowed(self):
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1, 2]]),
            occb=np.zeros((1, 0), dtype=int), nmo=6)

        assert wavefunction.nelec == (3, 0)
        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.psi0[1].shape == (6, 0)

    def test_mismatched_determinant_counts_are_rejected(self):
        with pytest.raises(ValueError, match="different numbers of determinants"):
            PHMSDWavefunction(coeffs=[1.0, 0.5],
                              occa=np.array([[0, 1], [0, 2]]),
                              occb=np.array([[0]]), nmo=6)

    def test_occupations_must_match_the_coefficients(self):
        with pytest.raises(ValueError, match=r"occa has shape"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1], [0, 2]]),
                              occb=np.array([[0], [1]]), nmo=6)

    def test_an_out_of_range_orbital_index_is_rejected(self):
        with pytest.raises(ValueError, match=r"outside \[0, 4\)"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 9]]),
                              occb=np.array([[0, 1]]), nmo=4)

    def test_more_than_two_references_are_rejected(self, rng):
        with pytest.raises(ValueError, match="at most two orbital references"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1]]),
                              occb=np.array([[0, 1]]), nmo=4,
                              orbitals=[np.eye(4) + 0j] * 3)

    def test_nreferences_counts_what_will_be_written(self, make_phmsd, rng):
        assert make_phmsd().nreferences == 0
        assert make_phmsd(orbitals=[np.eye(6) + 0j]).nreferences == 1
        assert make_phmsd(orbitals=[np.eye(6) + 0j, np.eye(6) + 0j]) \
            .nreferences == 2
        # a None entry does not count, and neither does an all-None list
        assert make_phmsd(orbitals=[np.eye(6) + 0j, None]).nreferences == 1
        assert make_phmsd(orbitals=[None]).nreferences == 0


class TestPsi0:

    def test_it_defaults_to_the_leading_determinant_occupations(self,
                                                               make_phmsd):
        alpha, beta = make_phmsd().psi0
        identity = np.eye(6)

        assert np.allclose(alpha, identity[:, [0, 1, 2]])
        assert np.allclose(beta, identity[:, [0, 1]])

    def test_the_default_is_orthonormal_so_writing_is_quiet(self, make_phmsd,
                                                            tmp_path,
                                                            recwarn):
        make_phmsd().to_hdf5(tmp_path / 'wfn.h5')

        assert [str(record.message) for record in recwarn] == []


class TestRoundTrip:

    def test_occupations_round_trip(self, make_phmsd, tmp_path):
        wavefunction = make_phmsd()
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, PHMSDWavefunction)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert np.array_equal(read_back.occb, wavefunction.occb)
        assert np.allclose(read_back.coeffs, wavefunction.coeffs)
        assert read_back.nelec == (3, 2)
        assert read_back.nmo == 6
        assert read_back.nreferences == 0

    @pytest.mark.parametrize('nreferences', [1, 2])
    def test_orbital_references_round_trip(self, make_phmsd, orthonormal,
                                           tmp_path, nreferences):
        references = [orthonormal(6, 6) for _ in range(nreferences)]
        wavefunction = make_phmsd(orbitals=references)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert read_back.nreferences == nreferences
        for read, original in zip(read_back.orbitals, references):
            assert np.allclose(read, original)

        with h5.File(path, 'r') as fh5:
            assert int(fh5['Wavefunction/PHMSD/type'][()]) == nreferences

    def test_a_polarized_expansion_round_trips(self, tmp_path):
        wavefunction = PHMSDWavefunction(
            coeffs=[0.9 + 0j, 0.1 + 0j],
            occa=np.array([[0, 1, 2], [0, 1, 3]]),
            occb=np.zeros((2, 0), dtype=int), nmo=6)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert read_back.nelec == (3, 0)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert read_back.occb.shape == (2, 0)


class TestOrthonormalize:

    def test_it_fixes_the_references(self, make_phmsd, rng):
        references = [rng.normal(size=(6, 6)) + 0j]
        fixed = make_phmsd(orbitals=references).orthonormalize()

        overlap = fixed.orbitals[0].conj().T @ fixed.orbitals[0]
        assert np.allclose(overlap, np.eye(6))

    def test_it_does_not_touch_the_original(self, make_phmsd, rng):
        references = [rng.normal(size=(6, 6)) + 0j]
        wavefunction = make_phmsd(orbitals=references)
        before = wavefunction.orbitals[0].copy()

        wavefunction.orthonormalize()

        assert np.array_equal(wavefunction.orbitals[0], before)

    def test_it_warns_on_write_when_a_reference_is_not_orthonormal(
            self, make_phmsd, rng, tmp_path):
        wavefunction = make_phmsd(orbitals=[rng.normal(size=(6, 6)) + 0j])

        with pytest.warns(UserWarning, match="not orthonormal"):
            wavefunction.to_hdf5(tmp_path / 'wfn.h5')
