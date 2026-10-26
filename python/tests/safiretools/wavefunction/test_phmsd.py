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

    def test_it_defaults_to_collinear(self, make_phmsd):
        assert make_phmsd().spin_symm is SpinSymm.COLLINEAR
        assert make_phmsd().nspin == 2
        assert make_phmsd().npol == 1

    def test_a_closed_shell_expansion_stores_alpha_alone(self):
        """The beta channel repeats alpha, so only one channel is stored."""
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1, 2]]),
            occb=np.zeros((1, 0), dtype=int), nmo=6, nelec=(3, 3),
            spin_symm='closed')

        assert wavefunction.nspin == 1
        assert (wavefunction.occa.shape, wavefunction.occb.shape) == ((1, 3), (1, 0))

    def test_a_noncollinear_expansion_indexes_spinors(self):
        """
        Both polarizations share one channel, so `occa` is ``nup + ndown`` wide
        and spans ``2*nmo`` — it already carries the beta offset.
        """
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1, 6, 7]]),
            occb=np.zeros((1, 0), dtype=int), nmo=6, nelec=(2, 2),
            spin_symm='noncollinear')

        assert (wavefunction.nspin, wavefunction.npol) == (1, 2)
        assert wavefunction.occa.shape == (1, 4)

    def test_a_spinor_index_outside_the_basis_is_rejected(self):
        with pytest.raises(ValueError, match=r"outside \[0, 12\)"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1, 6, 12]]),
                              occb=np.zeros((1, 0), dtype=int), nmo=6,
                              nelec=(2, 2), spin_symm='noncollinear')

    def test_an_empty_beta_channel_is_allowed(self):
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1, 2]]),
            occb=np.zeros((1, 0), dtype=int), nmo=6)

        assert wavefunction.nelec == (3, 0)
        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.occb.shape == (1, 0)

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

    def test_nreferences_counts_what_will_be_written(self, make_phmsd):
        # one array is a reference shared by both spins, a tuple a
        #   spin-resolved one
        assert make_phmsd().nreferences == 0
        assert make_phmsd(orbitals=np.eye(6)).nreferences == 1
        assert make_phmsd(orbitals=(np.eye(6), np.eye(6))).nreferences == 2

    def test_the_references_come_back_in_the_layout_they_were_given(
            self, make_phmsd, orthonormal, layouts_close):
        shared = orthonormal(6, 6)
        resolved = (orthonormal(6, 6), orthonormal(6, 6))

        assert layouts_close(make_phmsd(orbitals=shared).orbitals, shared)
        assert layouts_close(make_phmsd(orbitals=resolved).orbitals, resolved)

    def test_a_list_of_references_is_rejected(self, make_phmsd):
        with pytest.raises(TypeError, match="orbitals must be"):
            make_phmsd(orbitals=[np.eye(6), np.eye(6)])

    def test_a_spin_resolved_reference_needs_a_collinear_wavefunction(self):
        with pytest.raises(ValueError, match="needs a collinear wavefunction"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1, 2]]),
                              occb=np.zeros((1, 0), dtype=int), nmo=6,
                              nelec=(3, 3), spin_symm='closed',
                              orbitals=(np.eye(6), np.eye(6)))

    def test_a_noncollinear_reference_is_split_by_polarization(self,
                                                               orthonormal):
        reference = orthonormal(12, 12).reshape(2, 6, 12)
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1, 6, 7]]),
            occb=np.zeros((1, 0), dtype=int), nmo=6, nelec=(2, 2),
            spin_symm='noncollinear', orbitals=reference)

        assert wavefunction.nreferences == 1
        assert np.allclose(wavefunction.orbitals, reference)

    def test_a_noncollinear_wavefunction_needs_a_spinor_reference(self):
        with pytest.raises(ValueError, match="cannot take a closed orbital"):
            PHMSDWavefunction(coeffs=[1.0], occa=np.array([[0, 1, 6, 7]]),
                              occb=np.zeros((1, 0), dtype=int), nmo=6,
                              nelec=(2, 2), spin_symm='noncollinear',
                              orbitals=np.eye(12))

    def test_a_reference_must_span_the_basis(self, make_phmsd):
        with pytest.raises(ValueError, match="does not span the 6 orbitals"):
            make_phmsd(orbitals=np.eye(5))


class TestRoundTrip:

    def test_writing_is_quiet(self, make_phmsd, tmp_path, recwarn):
        make_phmsd().to_hdf5(tmp_path / 'wfn.h5')

        assert [str(record.message) for record in recwarn] == []

    def test_the_orbital_count_survives_unoccupied_orbitals(self, tmp_path):
        # no occupation number reaches the top orbitals, so only the recorded
        #   count can bring them back
        wavefunction = PHMSDWavefunction(
            coeffs=[1.0], occa=np.array([[0, 1]]), occb=np.array([[0]]),
            nmo=10)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        assert Wavefunction.from_hdf5(path).nmo == 10
        with h5.File(path, 'r') as fh5:
            assert int(fh5['Wavefunction/PHMSD'].attrs['number_of_orbitals']) == 10

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
                                           layouts_close, tmp_path,
                                           nreferences):
        references = orthonormal(6, 6) if nreferences == 1 \
            else (orthonormal(6, 6), orthonormal(6, 6))
        wavefunction = make_phmsd(orbitals=references)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert read_back.nreferences == nreferences
        assert layouts_close(read_back.orbitals, references)

        with h5.File(path, 'r') as fh5:
            group = fh5['Wavefunction/PHMSD']
            assert sorted(name for name in group if name.startswith('PsiT')) \
                == [f'PsiT_{i}' for i in range(nreferences)]

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
        fixed = make_phmsd(orbitals=rng.normal(size=(6, 6)) + 0j).orthonormalize()

        overlap = fixed.orbitals.conj().T @ fixed.orbitals
        assert np.allclose(overlap, np.eye(6))

    def test_it_keeps_a_spin_resolved_reference_spin_resolved(self, make_phmsd,
                                                              rng):
        references = (rng.normal(size=(6, 6)) + 0j, rng.normal(size=(6, 6)) + 0j)
        fixed = make_phmsd(orbitals=references).orthonormalize()

        assert isinstance(fixed.orbitals, tuple)
        assert fixed.nreferences == 2

    def test_it_does_not_touch_the_original(self, make_phmsd, rng):
        wavefunction = make_phmsd(orbitals=rng.normal(size=(6, 6)) + 0j)
        before = wavefunction.orbitals.copy()

        wavefunction.orthonormalize()

        assert np.array_equal(wavefunction.orbitals, before)

    def test_it_warns_on_write_when_a_reference_is_ill_conditioned(
            self, make_phmsd, tmp_path):
        reference = np.eye(6) + 0j
        reference[:, -1] = reference[:, 0] + 1e-12 * reference[:, -1]
        wavefunction = make_phmsd(orbitals=reference)

        with pytest.warns(UserWarning, match="ill-conditioned overlap"):
            wavefunction.to_hdf5(tmp_path / 'wfn.h5')

    def test_a_well_conditioned_reference_writes_silently(self, make_phmsd,
                                                          rng, tmp_path,
                                                          recwarn):
        wavefunction = make_phmsd(orbitals=rng.normal(size=(6, 6)) + 0j)

        wavefunction.to_hdf5(tmp_path / 'wfn.h5')

        assert [str(record.message) for record in recwarn] == []
