# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`NOMSDWavefunction.from_free_electron`: shell filling and the source forms."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sps
import toml

from safiretools import (
    Lattice,
    LatticeHamiltonian,
    NOMSDWavefunction,
    SpinSymm,
    Wavefunction,
)
from safiretools.wavefunction.free_electron import (
    DEFAULT_TWIST,
    fill_shells,
    from_free_electron,
    group_by_shell,
)


@pytest.fixture
def hubbard_params():
    return {
        'lattice': dict(L1=4, L2=4, boundary1='pbc', boundary2='pbc'),
        'hamiltonian': dict(t=1.0, U=4.0, spin_symm='collinear'),
    }


@pytest.fixture
def quiet():
    """Silence the collinear default-psi0 warning, covered in test_base.py."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        yield


class TestGroupByShell:

    def test_degenerate_eigenvalues_form_one_shell(self):
        eigenvalues = np.array([-2.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0])
        shells = group_by_shell(eigenvalues, np.eye(8, dtype=complex))

        assert [shell['degeneracy'] for shell in shells] == [2, 1, 3, 2]
        assert [shell['indices'] for shell in shells] == [
            [0, 1], [2], [3, 4, 5], [6, 7]]

    def test_near_degeneracy_is_governed_by_the_tolerance(self):
        eigenvalues = np.array([0.0, 5e-9, 1.2e-8, 1.0])

        assert len(group_by_shell(eigenvalues, np.eye(4, dtype=complex),
                                 tol=1e-6)) == 2
        assert len(group_by_shell(eigenvalues, np.eye(4, dtype=complex),
                                 tol=1e-10)) == 4

    def test_a_shell_boundary_is_measured_from_its_first_eigenvalue(self):
        # consecutive gaps below tol still split when the total drift exceeds it
        eigenvalues = np.array([0.0, 0.9e-6, 1.8e-6])
        shells = group_by_shell(eigenvalues, np.eye(3, dtype=complex),
                                tol=1e-6)

        assert [shell['degeneracy'] for shell in shells] == [2, 1]


class TestFillShells:

    @pytest.fixture
    def shells(self):
        eigenvalues = np.array([-1.0] + [0.0] * 6 + [1.0])
        return group_by_shell(eigenvalues, np.eye(8, dtype=complex))

    def test_aufbau_takes_the_orbitals_in_order(self, shells):
        _, indices = fill_shells(shells, 4, strategy='aufbau')

        assert indices == [0, 1, 2, 3]

    def test_balanced_spaces_them_evenly(self, shells):
        _, indices = fill_shells(shells, 4, strategy='balanced')

        assert indices == [0, 1, 3, 5]

    def test_alternating_works_from_the_edges_inward(self, shells):
        _, indices = fill_shells(shells, 4, strategy='alternating')

        assert indices == [0, 1, 6, 2]

    def test_hund_matches_aufbau_in_one_spin_channel(self, shells):
        assert fill_shells(shells, 4, strategy='hund')[1] \
            == fill_shells(shells, 4, strategy='aufbau')[1]

    def test_a_completely_filled_shell_ignores_the_strategy(self, shells):
        for strategy in ('aufbau', 'balanced', 'alternating', 'hund'):
            assert fill_shells(shells, 7, strategy=strategy)[1] \
                == list(range(7))

    def test_zero_electrons_gives_an_empty_block(self, shells):
        orbitals, indices = fill_shells(shells, 0)

        assert orbitals.shape == (8, 0)
        assert indices == []

    def test_too_few_orbitals_is_reported(self, shells):
        with pytest.raises(ValueError, match="Not enough orbitals"):
            fill_shells(shells, 99)

    def test_an_unknown_strategy_is_rejected(self, shells):
        with pytest.raises(ValueError, match="unknown filling strategy"):
            fill_shells(shells, 3, strategy='nonsense')


class TestFromFreeElectron:

    def test_it_builds_a_single_collinear_determinant(self, hubbard_params,
                                                      quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear')

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.ndets == 1
        assert wavefunction.nelec == (8, 8)
        assert wavefunction.dets.shape == (1, 16, 16)
        assert np.allclose(wavefunction.coeffs, [1.0])

    def test_a_polarized_system_is_collinear_with_no_beta_electrons(
            self, hubbard_params, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(5, 0), spin_symm='collinear')

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (5, 0)
        assert wavefunction.dets.shape == (1, 16, 5)

    def test_a_noncollinear_determinant_spans_the_spinor_basis(self, quiet):
        params = {
            'lattice': dict(L1=4, L2=1, boundary1='pbc', boundary2='open'),
            'hamiltonian': dict(t=1.0, U=2.0, nbands=2,
                                spin_symm='noncollinear'),
        }
        wavefunction = NOMSDWavefunction.from_free_electron(params,
                                                            nelec=(4, 4))

        assert wavefunction.spin_symm is SpinSymm.NONCOLLINEAR
        assert wavefunction.nmo == 8
        assert wavefunction.dets.shape == (1, 16, 8)

    def test_the_result_is_orthonormal(self, hubbard_params, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear')

        for block in wavefunction.spin_blocks(0):
            assert np.allclose(block.conj().T @ block, np.eye(block.shape[1]))

    def test_closed_shell_is_not_implemented(self, hubbard_params):
        hubbard_params['hamiltonian']['spin_symm'] = 'closed'

        with pytest.raises(NotImplementedError, match="closed spin symmetry"):
            NOMSDWavefunction.from_free_electron(hubbard_params, nelec=(8, 8))

    def test_the_spin_symmetry_defaults_to_the_hamiltonian(self,
                                                           hubbard_params,
                                                           quiet):
        hamiltonian = LatticeHamiltonian.from_dict(hubbard_params)
        wavefunction = NOMSDWavefunction.from_free_electron(hamiltonian,
                                                            nelec=(8, 8))

        assert wavefunction.spin_symm is hamiltonian.spin_symm

    def test_an_unsupported_source_is_rejected(self):
        with pytest.raises(ValueError, match="source must be a Hamiltonian"):
            NOMSDWavefunction.from_free_electron(42, nelec=(1, 1))


class TestSourceForms:

    def test_it_does_not_mutate_the_parameter_dict(self, hubbard_params, quiet):
        before = dict(hubbard_params['lattice'])

        NOMSDWavefunction.from_free_electron(hubbard_params, nelec=(8, 8),
                                            spin_symm='collinear')

        # afqmctools injected the twist into the caller's dict in place
        assert hubbard_params['lattice'] == before

    def test_a_toml_file_and_a_dict_agree(self, hubbard_params, tmp_path,
                                          quiet):
        path = tmp_path / 'input.toml'
        path.write_text(toml.dumps(hubbard_params))

        from_file = NOMSDWavefunction.from_free_electron(
            path, nelec=(8, 8), spin_symm='collinear')
        from_dict = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear')

        assert np.allclose(from_file.dets, from_dict.dets)

    def test_a_hamiltonian_hdf5_file_reproduces_the_same_determinant(
            self, hubbard_params, tmp_path, quiet):
        # the twist has to be spelled out, since only the dict path defaults it
        params = dict(hubbard_params)
        params['lattice'] = dict(hubbard_params['lattice'],
                                 twist=list(DEFAULT_TWIST))
        path = tmp_path / 'ham.h5'
        LatticeHamiltonian.from_dict(params).to_hdf5(path)

        from_file = NOMSDWavefunction.from_free_electron(path, nelec=(8, 8))
        from_dict = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear')

        assert np.allclose(from_file.dets, from_dict.dets, atol=1e-10)

    def test_a_supplied_lattice_is_used_as_given(self, hubbard_params, quiet):
        lattice = Lattice.from_dict(dict(hubbard_params['lattice'],
                                         twist=[0.05, 0.05]))
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear',
            lattice=lattice)

        assert wavefunction.dets.shape == (1, 16, 16)

    def test_a_twist_that_contradicts_the_hamiltonian_warns(self,
                                                            hubbard_params):
        hamiltonian = LatticeHamiltonian.from_dict(hubbard_params)

        with pytest.warns(UserWarning, match="differs from the Hamiltonian"):
            NOMSDWavefunction.from_free_electron(hamiltonian, nelec=(8, 8),
                                                 twist=[0.3, 0.3])

    def test_a_twist_alongside_a_lattice_warns(self, hubbard_params):
        lattice = Lattice.from_dict(hubbard_params['lattice'])

        with pytest.warns(UserWarning, match="twist angle is ignored"):
            NOMSDWavefunction.from_free_electron(
                hubbard_params, nelec=(8, 8), spin_symm='collinear',
                lattice=lattice, twist=[0.3, 0.3])


class TestOneBodyShapes:
    """
    A spin-independent one-body term is used for both channels. afqmctools
    sliced ``(nmo, nmo)`` unconditionally and produced an empty beta block.
    """

    @pytest.fixture
    def spin_independent(self):
        class FakeHamiltonian(LatticeHamiltonian):
            def get_one_body(self):
                return sps.csr_array(np.diag(np.arange(6.0)))

        hamiltonian = FakeHamiltonian(nsites=6, spin_symm='collinear')
        return hamiltonian

    def test_a_spin_independent_term_serves_both_channels(self,
                                                          spin_independent,
                                                          quiet):
        wavefunction = from_free_electron(spin_independent, nelec=(2, 1))

        assert wavefunction.dets.shape == (1, 6, 3)

    def test_a_misshaped_term_is_reported(self, quiet):
        class FakeHamiltonian(LatticeHamiltonian):
            def get_one_body(self):
                return sps.csr_array(np.zeros((5, 6)))

        with pytest.raises(ValueError, match="neither"):
            from_free_electron(FakeHamiltonian(nsites=6, spin_symm='collinear'),
                               nelec=(2, 1))


class TestRoundTrip:

    def test_a_free_electron_wavefunction_round_trips(self, hubbard_params,
                                                      tmp_path, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard_params, nelec=(8, 8), spin_symm='collinear')
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (8, 8)
        assert read_back.spin_symm is SpinSymm.COLLINEAR
        assert np.allclose(read_back.dets, wavefunction.dets)
