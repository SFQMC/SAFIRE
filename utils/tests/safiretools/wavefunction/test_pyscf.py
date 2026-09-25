# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
`NOMSDWavefunction.from_pyscf` and `PHMSDWavefunction.from_pyscf_cas`: the
molecular construction paths.
"""

import copy
import warnings

import numpy as np
import pytest

from safiretools import NOMSDWavefunction, PHMSDWavefunction, SpinSymm, Wavefunction
from safiretools.wavefunction.slater import spin_layout_shape

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


@pytest.fixture(scope='module')
def neon_rhf():
    from pyscf import gto, scf

    mol = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope='module')
def oxygen_rhf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope='module')
def oxygen_rohf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    return scf.ROHF(mol).run()


@pytest.fixture(scope='module')
def oxygen_uhf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    return scf.UHF(mol).run()


@pytest.fixture(scope='module')
def neon_rhf_631g():
    from pyscf import gto, scf

    # a basis with room for both a frozen core and virtual orbitals, so that a
    #   CAS expansion is non-trivial
    mol = gto.M(atom='Ne 0 0 0', basis='6-31g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope='module')
def lithium_rohf():
    from pyscf import gto, scf

    mol = gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
    return scf.ROHF(mol).run()


class TestFromPyscf:

    def test_a_closed_shell_reference(self, neon_rhf):
        wavefunction = NOMSDWavefunction.from_pyscf(neon_rhf)

        assert wavefunction.spin_symm is SpinSymm.CLOSED
        assert wavefunction.nelec == (5, 5)
        assert wavefunction.nmo == 5
        assert wavefunction.dets.shape == (1, 5, 5)

    def test_an_open_shell_reference_is_collinear(self, oxygen_rohf):
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_rohf)

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (5, 3)
        assert spin_layout_shape(wavefunction.dets) == ((1, 5, 5), (1, 5, 3))

    def test_a_ghf_reference_is_noncollinear(self, oxygen_rohf):
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_rohf.to_ghf(),
                                                    basis=oxygen_rohf)

        assert wavefunction.spin_symm is SpinSymm.NONCOLLINEAR
        # (ndets, npol, nmo, nelec): both polarizations over the 5 spatial orbitals
        assert wavefunction.dets.shape == (1, 2, 5, 8)

    def test_an_active_space_trims_and_reindexes_the_orbitals(self,
                                                              oxygen_rohf):
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_rohf,
                                                    active_space=(4, 3))

        # (8 - 4) // 2 = 2 frozen core orbitals, leaving 3 active ones
        assert wavefunction.nelec == (3, 1)
        assert wavefunction.nmo == 3

    def test_a_frozen_core_can_empty_the_beta_channel(self, lithium_rohf):
        # a reference with no beta electrons is collinear with ndown == 0
        wavefunction = NOMSDWavefunction.from_pyscf(lithium_rohf,
                                                    active_space=(1, 4))

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (1, 0)
        assert spin_layout_shape(wavefunction.dets) == ((1, 4, 1), (1, 4, 0))

    def test_a_wrong_occupancy_count_is_reported(self, neon_rhf):
        emptied = copy.copy(neon_rhf)
        emptied.mo_occ = np.zeros_like(np.asarray(neon_rhf.mo_occ))

        with pytest.raises(ValueError, match="alpha occupied orbitals"):
            NOMSDWavefunction.from_pyscf(emptied)

    def test_an_unconverged_reference_is_rejected(self, neon_rhf):
        from pyscf import scf

        with pytest.raises(ValueError, match="run the calculation"):
            NOMSDWavefunction.from_pyscf(scf.RHF(neon_rhf.mol))

    def test_an_active_space_and_ortho_ao_cannot_be_combined(self, neon_rhf):
        with pytest.raises(ValueError,
                           match="active_space and basis='ortho_ao'"):
            NOMSDWavefunction.from_pyscf(neon_rhf, basis='ortho_ao',
                                         active_space=(4, 2))

    def test_the_result_is_orthonormal(self, oxygen_rohf):
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_rohf)

        for block in wavefunction.determinant(0):
            assert np.allclose(block.conj().T @ block, np.eye(block.shape[1]))

    def test_it_round_trips(self, oxygen_rohf, layouts_close, tmp_path):
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_rohf)
        path = tmp_path / 'wfn.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wavefunction.to_hdf5(path)
        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (5, 3)
        assert read_back.spin_symm is SpinSymm.COLLINEAR
        assert layouts_close(read_back.dets, wavefunction.dets)


class TestBasisForms:
    """
    The *basis* is given separately from the solution the wavefunction is built
    from, as another SCF object, an orbital array or ``'ortho_ao'``.
    """

    def test_the_basis_defaults_to_the_solution_itself(self, oxygen_rohf,
                                                        layouts_close):
        explicit = NOMSDWavefunction.from_pyscf(oxygen_rohf, basis=oxygen_rohf)
        implicit = NOMSDWavefunction.from_pyscf(oxygen_rohf)

        assert layouts_close(explicit.dets, implicit.dets)

    def test_an_orbital_array_is_a_basis(self, oxygen_rohf, oxygen_rhf,
                                         layouts_close):
        by_object = NOMSDWavefunction.from_pyscf(oxygen_rohf, basis=oxygen_rhf)
        by_array = NOMSDWavefunction.from_pyscf(oxygen_rohf,
                                                basis=oxygen_rhf.mo_coeff)

        assert layouts_close(by_object.dets, by_array.dets)

    def test_the_basis_can_come_from_a_different_solution(self, oxygen_rohf,
                                                          oxygen_rhf,
                                                          layouts_close):
        """
        The Hamiltonian's basis and the trial wavefunction need not come from
        one calculation: here an open-shell solution is expressed in a
        closed-shell solution's molecular orbitals.
        """
        own_basis = NOMSDWavefunction.from_pyscf(oxygen_rohf)
        rhf_basis = NOMSDWavefunction.from_pyscf(oxygen_rohf, basis=oxygen_rhf)

        assert own_basis.nmo == rhf_basis.nmo
        assert own_basis.nelec == rhf_basis.nelec
        # a different basis is a different Slater matrix
        assert not layouts_close(own_basis.dets, rhf_basis.dets)

    def test_the_spin_symmetry_comes_from_the_solution_not_the_basis(
            self, oxygen_uhf):
        """
        The orthogonalized AO basis says nothing about spin: the symmetry is
        that of the *solution*.
        """
        wavefunction = NOMSDWavefunction.from_pyscf(oxygen_uhf, basis='ortho_ao')

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR


class TestFromPyscfCas:

    @pytest.fixture(scope='class')
    def casscf(self, neon_rhf_631g):
        from pyscf import mcscf

        return mcscf.CASSCF(neon_rhf_631g, 4, 4).run()

    def test_it_reads_the_expansion(self, casscf):
        mol = casscf.mol
        wavefunction = PHMSDWavefunction.from_pyscf_cas(casscf, tol=1e-6)

        assert isinstance(wavefunction, PHMSDWavefunction)
        assert wavefunction.nelec == mol.nelec
        assert wavefunction.nmo == mol.nao_nr()
        assert wavefunction.ndets > 1
        # the frozen core is reinserted into every determinant
        assert np.all(wavefunction.occa[:, :3] == np.arange(3))

    def test_the_coefficients_are_sorted_by_magnitude(self, casscf):
        coeffs = PHMSDWavefunction.from_pyscf_cas(casscf, tol=1e-6).coeffs
        magnitudes = np.abs(coeffs)

        assert np.all(np.diff(magnitudes) <= 1e-15)

    def test_max_det_truncates(self, casscf):
        wavefunction = PHMSDWavefunction.from_pyscf_cas(casscf, tol=1e-8,
                                                        max_det=3)

        assert wavefunction.ndets == 3

    def test_it_round_trips(self, casscf, tmp_path):
        wavefunction = PHMSDWavefunction.from_pyscf_cas(casscf, tol=1e-6)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, PHMSDWavefunction)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert np.array_equal(read_back.occb, wavefunction.occb)
        assert np.allclose(read_back.coeffs, wavefunction.coeffs)
