# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""`NOMSDWavefunction.from_free_electron`: shell filling and degeneracy."""

import warnings

import numpy as np
import pytest
import scipy.sparse as sps

from safiretools import (
    Lattice,
    LatticeHamiltonian,
    NOMSDWavefunction,
    SpinSymm,
    Wavefunction,
)
from safiretools.wavefunction.free_electron import (
    DEFAULT_TWIST,
    FILLING_STRATEGIES,
    SHELL_TOL,
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
def hubbard(hubbard_params):
    """A 4x4 Hubbard model. Untwisted, so its shells stay degenerate."""
    return LatticeHamiltonian.from_dict(hubbard_params)


@pytest.fixture
def quiet():
    """
    Silence the collinear default-psi0 warning (covered in test_base.py) and
    the open-shell warning (covered in `TestOpenShell`); 4x4 at half filling
    fills 3 of a 6-fold degenerate shell.
    """
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

    def test_alternating_works_from_the_edges_inward(self, shells):
        _, indices = fill_shells(shells, 4, strategy='alternating')

        assert indices == [0, 1, 6, 2]

    def test_a_completely_filled_shell_ignores_the_strategy(self, shells):
        for strategy in FILLING_STRATEGIES:
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

    def test_it_builds_a_single_collinear_determinant(self, hubbard, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard, nelec=(8, 8), spin_symm='collinear')

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.ndets == 1
        assert wavefunction.nelec == (8, 8)
        assert wavefunction.dets.shape == (1, 16, 16)
        assert np.allclose(wavefunction.coeffs, [1.0])

    def test_a_polarized_system_is_collinear_with_no_beta_electrons(
            self, hubbard, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard, nelec=(5, 0), spin_symm='collinear')

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (5, 0)
        assert wavefunction.dets.shape == (1, 16, 5)

    def test_a_noncollinear_determinant_spans_the_spinor_basis(self, quiet):
        hamiltonian = LatticeHamiltonian.from_dict({
            'lattice': dict(L1=4, L2=1, boundary1='pbc', boundary2='open'),
            'hamiltonian': dict(t=1.0, U=2.0, nbands=2,
                                spin_symm='noncollinear'),
        })
        wavefunction = NOMSDWavefunction.from_free_electron(hamiltonian,
                                                            nelec=(4, 4))

        assert wavefunction.spin_symm is SpinSymm.NONCOLLINEAR
        assert wavefunction.nmo == 8
        assert wavefunction.dets.shape == (1, 16, 8)

    def test_the_result_is_orthonormal(self, hubbard, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard, nelec=(8, 8), spin_symm='collinear')

        for block in wavefunction.spin_blocks(0):
            assert np.allclose(block.conj().T @ block, np.eye(block.shape[1]))

    def test_closed_shell_is_not_implemented(self, hubbard_params):
        hubbard_params['hamiltonian']['spin_symm'] = 'closed'
        hamiltonian = LatticeHamiltonian.from_dict(hubbard_params)

        with pytest.raises(NotImplementedError, match="closed spin symmetry"):
            NOMSDWavefunction.from_free_electron(hamiltonian, nelec=(8, 8))

    def test_the_spin_symmetry_defaults_to_the_hamiltonian(self, hubbard,
                                                           quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(hubbard,
                                                            nelec=(8, 8))

        assert wavefunction.spin_symm is hubbard.spin_symm

    def test_anything_but_a_lattice_hamiltonian_is_rejected(self):
        with pytest.raises(ValueError, match="LatticeHamiltonian instance"):
            NOMSDWavefunction.from_free_electron(42, nelec=(1, 1))


class TestOpenShell:
    """
    A determinant that stops part-way through a degenerate shell is not
    uniquely defined. A small irrational twist on the *lattice* lifts the
    degeneracy; the Hamiltonian the AFQMC run uses normally carries no twist,
    so the two are built separately.
    """

    def test_a_partly_filled_degenerate_shell_warns(self, hubbard):
        # 4x4 at half filling takes 3 of a 6-fold degenerate shell
        with pytest.warns(UserWarning, match="6-fold degenerate"):
            NOMSDWavefunction.from_free_electron(hubbard, nelec=(8, 8),
                                                 spin_symm='collinear')

    @staticmethod
    def _degeneracy_warnings(hamiltonian, nelec):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            NOMSDWavefunction.from_free_electron(hamiltonian, nelec=nelec,
                                                 spin_symm='collinear')
        return [w for w in caught if 'degenerate' in str(w.message)]

    def test_a_closed_shell_is_silent(self, hubbard):
        # 1 + 4 = 5 exactly closes the second shell
        assert not self._degeneracy_warnings(hubbard, (5, 5))

    def test_a_twisted_lattice_silences_it(self, hubbard_params):
        params = dict(hubbard_params)
        params['lattice'] = dict(hubbard_params['lattice'],
                                 twist=[0.01, 0.02])
        hamiltonian = LatticeHamiltonian.from_dict(params)

        assert not self._degeneracy_warnings(hamiltonian, (8, 8))

    @pytest.mark.parametrize('L1,L2,nelec', [(4, 4, (8, 8)), (4, 8, (16, 16)),
                                             (6, 6, (18, 18))])
    def test_the_default_twist_clears_the_default_tolerance(self, L1, L2,
                                                            nelec):
        """
        The warning tells callers to reach for `DEFAULT_TWIST`, so the twist
        has to split shells by more than `SHELL_TOL`. The splitting goes as the
        square of the twist and shrinks with lattice size, so check a few.
        """
        hamiltonian = LatticeHamiltonian.from_dict({
            'lattice': dict(L1=L1, L2=L2, boundary1='pbc', boundary2='pbc',
                            twist=list(DEFAULT_TWIST)),
            'hamiltonian': dict(t=1.0, U=4.0, spin_symm='collinear'),
        })

        assert not self._degeneracy_warnings(hamiltonian, nelec)

    def test_the_tolerance_stays_above_the_eigensolver_noise_floor(self):
        # a one-body term of bandwidth ~8t resolves eigenvalues to ~1e-13
        assert SHELL_TOL > 1e-12


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

    def test_a_free_electron_wavefunction_round_trips(self, hubbard,
                                                      tmp_path, quiet):
        wavefunction = NOMSDWavefunction.from_free_electron(
            hubbard, nelec=(8, 8), spin_symm='collinear')
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (8, 8)
        assert read_back.spin_symm is SpinSymm.COLLINEAR
        assert np.allclose(read_back.dets, wavefunction.dets)
