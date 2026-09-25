# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Reading PySCF SCF objects for the factories."""

import warnings

import numpy as np
import pytest

from safiretools import SpinSymm
from safiretools.convert.pyscf import (
    _nesting_depth,
    _per_kpoint,
    canonical_orthogonalization,
    cholesky_df,
    determine_spin_symm,
    hamiltonian_spin_symm,
    one_body,
    periodic_solution,
    working_basis,
)

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


@pytest.fixture(scope='module')
def rhf():
    from pyscf import gto, scf

    mol = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    return scf.RHF(mol).run()


@pytest.fixture(scope='module')
def rohf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    return scf.ROHF(mol).run()


@pytest.fixture(scope='module')
def uhf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    return scf.UHF(mol).run()


@pytest.fixture(scope='module')
def soc_ecp_mol():
    """An iodine atom, whose built-in CRENBL ECP carries a spin-orbit part."""
    from pyscf import gto

    return gto.M(atom='I 0 0 0', basis='crenbl', ecp='crenbl', spin=1, verbose=0)


def _diamond_cell():
    from pyscf.pbc import gto

    cell = gto.Cell()
    alat = 3.6
    cell.a = (np.ones((3, 3)) - np.eye(3)) * alat / 2.0
    cell.atom = (('C', 0, 0, 0), ('C', np.array([0.25, 0.25, 0.25]) * alat))
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [12] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)
    return cell


@pytest.fixture(scope='module')
def krks():
    """A 2x1x1 diamond KRKS calculation."""
    from pyscf.pbc import dft

    cell = _diamond_cell()
    return dft.KRKS(cell, kpts=cell.make_kpts([2, 1, 1])).run()


@pytest.fixture(scope='module')
def single_kpt_rhf():
    """
    A one-k-point calculation, which holds every per-orbital quantity with no
    k-point axis — the layout that needs a k-point axis put back.
    """
    from pyscf.pbc import scf

    cell = _diamond_cell()
    return scf.RHF(cell, kpt=cell.make_kpts([1, 1, 1])[0]).run()


class TestPerKpoint:
    """
    PySCF's periodic layout varies in two independent ways — whether there is a
    k-point axis, and whether the container is one array or a list of them — so
    `_per_kpoint` keys on nesting depth rather than on the container type.
    """

    @pytest.mark.parametrize('values, depth', [
        (np.zeros(8), 1),
        (np.zeros((2, 8)), 2),
        ([np.zeros(8), np.zeros(8)], 2),
        ([[np.zeros(8)] * 2] * 2, 3),
        ([np.zeros((2, 8))] * 2, 3),
    ])
    def test_nesting_depth_counts_sequences_as_axes(self, values, depth):
        assert _nesting_depth(values) == depth

    def test_one_entry_per_kpoint(self):
        values, spin_resolved = _per_kpoint([np.zeros(8)] * 2, 2, 1, 'mo_occ')

        assert spin_resolved is False
        assert len(values) == 2

    def test_a_ragged_orbital_count_is_kept_per_kpoint(self):
        """
        k-points with different orbital counts come back as a list of arrays
        rather than one array, which is why the container type says nothing.
        """
        values, spin_resolved = _per_kpoint([np.zeros(8), np.zeros(7)], 2, 1,
                                            'mo_occ')

        assert spin_resolved is False
        assert [v.shape for v in values] == [(8,), (7,)]

    def test_an_array_is_accepted_as_readily_as_a_list(self):
        values, spin_resolved = _per_kpoint(np.zeros((2, 8)), 2, 1, 'mo_occ')

        assert spin_resolved is False
        assert len(values) == 2

    def test_a_missing_kpoint_axis_is_put_back(self):
        values, spin_resolved = _per_kpoint(np.zeros(8), 1, 1, 'mo_occ')

        assert spin_resolved is False
        assert np.shape(values) == (1, 8)

    def test_a_spin_pair_at_one_kpoint_is_spin_resolved(self):
        values, spin_resolved = _per_kpoint([np.zeros(8)] * 2, 1, 1, 'mo_occ')

        assert spin_resolved is True
        assert np.shape(values) == (2, 1, 8)

    def test_spin_major_per_kpoint_is_spin_resolved(self):
        values, spin_resolved = _per_kpoint([[np.zeros(8)] * 2] * 2, 2, 1,
                                            'mo_occ')

        assert spin_resolved is True
        assert np.shape(values) == (2, 2, 8)

    def test_two_kpoints_are_not_mistaken_for_a_spin_pair(self):
        """
        The spin axis and a two-k-point axis are both length 2, so the depth is
        what separates them; at equal depth the k-point count decides.
        """
        _, spin_resolved = _per_kpoint([np.zeros(8)] * 2, 2, 1, 'mo_occ')

        assert spin_resolved is False

    def test_a_matrix_valued_entry_shifts_every_depth(self):
        """Fock matrices and orbital coefficients are rank 2 per k-point."""
        values, spin_resolved = _per_kpoint(np.zeros((2, 8, 8)), 2, 2, 'fock')
        assert (spin_resolved, np.shape(values)) == (False, (2, 8, 8))

        values, spin_resolved = _per_kpoint(np.zeros((2, 2, 8, 8)), 2, 2, 'fock')
        assert (spin_resolved, np.shape(values)) == (True, (2, 2, 8, 8))

    def test_a_missing_kpoint_axis_with_several_kpoints_is_rejected(self):
        with pytest.raises(ValueError, match="no k-point axis"):
            _per_kpoint(np.zeros(8), 2, 1, 'mo_occ')

    def test_a_length_matching_neither_axis_is_rejected(self):
        with pytest.raises(ValueError, match="neither the 4 k-points"):
            _per_kpoint([np.zeros(8)] * 3, 4, 1, 'mo_occ')

    def test_a_misshaped_spin_axis_is_rejected(self):
        with pytest.raises(ValueError, match="must hold 2 channels"):
            _per_kpoint([[np.zeros(8)] * 2] * 3, 2, 1, 'mo_occ')


class TestDetermineSpinSymm:
    """
    The spin symmetry of a solution is read off the *calculation*, not off the
    shape of a Slater matrix built from it later.
    """

    def test_an_rhf_solution_is_closed_shell(self, rhf):
        assert determine_spin_symm(rhf.mo_coeff, rhf.mo_occ,
                                   rhf.mol.nao_nr()) is SpinSymm.CLOSED

    def test_an_rohf_solution_is_collinear(self, rohf):
        assert determine_spin_symm(rohf.mo_coeff, rohf.mo_occ,
                                   rohf.mol.nao_nr()) is SpinSymm.COLLINEAR

    def test_a_uhf_solution_is_collinear(self, uhf):
        assert determine_spin_symm(uhf.mo_coeff, uhf.mo_occ,
                                   uhf.mol.nao_nr()) is SpinSymm.COLLINEAR

    def test_spin_resolved_orbitals_are_collinear(self):
        assert determine_spin_symm(np.zeros((2, 6, 6)), np.zeros((2, 6)), 6) \
            is SpinSymm.COLLINEAR

    def test_a_singly_occupied_orbital_is_collinear(self):
        """An ROHF solution records both channels in one occupancy vector."""
        occupancies = np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0])

        assert determine_spin_symm(np.zeros((6, 6)), occupancies, 6) \
            is SpinSymm.COLLINEAR

    def test_integer_double_occupancies_are_closed_shell(self):
        occupancies = np.array([2.0, 2.0, 0.0, 0.0])

        assert determine_spin_symm(np.zeros((4, 4)), occupancies, 4) \
            is SpinSymm.CLOSED

    def test_a_spinor_orbital_basis_is_noncollinear(self):
        """
        A GHF solution has one ``mo_coeff`` matrix like an RHF one; only the
        basis size separates them.
        """
        assert determine_spin_symm(np.zeros((12, 12)), np.zeros(12), 6) \
            is SpinSymm.NONCOLLINEAR


class TestHamiltonianSpinSymm:
    """The one-body operator, not the reference state, sets the symmetry."""

    def test_a_spatial_hcore_is_closed(self):
        assert hamiltonian_spin_symm(np.zeros((6, 6)), 6) is SpinSymm.CLOSED

    def test_a_spinor_hcore_is_noncollinear(self):
        assert hamiltonian_spin_symm(np.zeros((12, 12)), 6) \
            is SpinSymm.NONCOLLINEAR

    def test_any_other_shape_is_rejected(self):
        with pytest.raises(ValueError, match="neither"):
            hamiltonian_spin_symm(np.zeros((7, 7)), 6)

    def test_an_open_shell_source_still_gives_a_closed_hamiltonian(self, rohf):
        _, spin_symm = one_body(rohf)

        assert spin_symm is SpinSymm.CLOSED

    def test_a_ghf_source_gives_a_noncollinear_hamiltonian(self, rhf):
        hcore, spin_symm = one_body(rhf.to_ghf())

        assert spin_symm is SpinSymm.NONCOLLINEAR
        assert np.array_equal(hcore, np.kron(np.eye(2), rhf.get_hcore()))


class TestSpinOrbitEcp:
    """The spin-orbit term comes from how the source object was set up."""

    def test_with_soc_folds_the_ecp_term_into_hcore(self, soc_ecp_mol):
        from pyscf import scf

        nao = soc_ecp_mol.nao_nr()
        ghf = scf.GHF(soc_ecp_mol)
        ghf.with_soc = True

        with warnings.catch_warnings():
            warnings.simplefilter('error')
            hcore, spin_symm = one_body(ghf)

        assert spin_symm is SpinSymm.NONCOLLINEAR
        assert np.any(hcore[:nao, nao:])

    def test_a_forgotten_with_soc_warns(self, soc_ecp_mol):
        from pyscf import scf

        with pytest.warns(UserWarning, match="spin-orbit ECP"):
            one_body(scf.GHF(soc_ecp_mol))

    def test_a_spatial_hcore_warns_too(self, soc_ecp_mol):
        from pyscf import scf

        with pytest.warns(UserWarning, match="spin-orbit ECP"):
            one_body(scf.ROHF(soc_ecp_mol))


class TestWorkingBasis:

    def test_the_default_is_the_source_orbitals(self, rhf):
        C, frozen = working_basis(rhf)

        assert np.array_equal(C, rhf.mo_coeff)
        assert frozen == (0, 0)

    def test_ortho_ao_orthogonalizes_the_overlap(self, rhf):
        X, _ = working_basis(rhf, 'ortho_ao')
        overlap = rhf.mol.intor('int1e_ovlp')

        assert np.allclose(X.conj().T @ overlap @ X, np.eye(X.shape[1]))

    def test_another_object_supplies_its_orbitals(self, uhf, rohf):
        C, _ = working_basis(uhf, rohf)

        assert np.array_equal(C, rohf.mo_coeff)

    def test_an_array_is_used_as_it_stands(self, rhf):
        basis = np.eye(rhf.mol.nao_nr())[:, :3]

        C, _ = working_basis(rhf, basis)

        assert C is basis

    def test_an_active_space_freezes_the_core(self, rhf):
        _, frozen = working_basis(rhf, active_space=(6, 3))

        assert frozen == (2, 0)

    def test_an_unknown_string_is_rejected(self, rhf):
        with pytest.raises(ValueError, match="unknown basis 'mo'"):
            working_basis(rhf, 'mo')

    def test_an_active_space_and_ortho_ao_cannot_be_combined(self, rhf):
        with pytest.raises(ValueError, match="cannot be combined"):
            working_basis(rhf, 'ortho_ao', active_space=(6, 3))

    def test_spin_resolved_orbitals_are_rejected(self, uhf):
        with pytest.raises(ValueError, match="UHF or GHF"):
            working_basis(uhf)

    def test_orbitals_of_another_molecule_are_rejected(self, rhf, rohf):
        with pytest.raises(ValueError, match="basis functions"):
            working_basis(rhf, np.eye(rhf.mol.nao_nr() + 1))

    def test_an_unconverged_basis_is_rejected(self, rhf):
        from pyscf import scf

        with pytest.raises(ValueError, match="run the calculation"):
            working_basis(rhf, scf.RHF(rhf.mol))


class TestCholeskyDf:

    def test_the_vectors_are_the_unpacked_ones(self):
        from pyscf import gto, lib, scf

        mol = gto.M(atom='H 0 0 0; F 0 0 0.9', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).density_fit().run()
        nao = mol.nao_nr()

        expected = lib.unpack_tril(mf.with_df._cderi).reshape(-1, nao * nao)

        assert np.array_equal(cholesky_df(mf), expected)

    def test_it_needs_a_density_fitted_object(self, rhf):
        with pytest.raises(ValueError, match="density-fitted"):
            cholesky_df(rhf)


class TestCanonicalOrthogonalization:

    def test_it_orthogonalizes_the_overlap(self):
        matrix = np.random.default_rng(3).normal(size=(6, 6))
        overlap = matrix @ matrix.T + 6 * np.eye(6)

        X = canonical_orthogonalization(overlap)

        assert np.allclose(X.conj().T @ overlap @ X, np.eye(6))

    def test_linearly_dependent_functions_are_dropped(self):
        overlap = np.diag([1.0, 1.0, 1e-12])

        assert canonical_orthogonalization(overlap, 1e-8).shape == (3, 2)


class TestPeriodic:

    def test_the_mo_basis_is_the_default(self, krks):
        scf_data = periodic_solution(krks)

        assert set(scf_data) >= {'cell', 'kpts', 'Xocc', 'hcore', 'X',
                                 'nmo_pk', 'mo_coeff', 'nao', 'fock',
                                 'mo_energy', 'walker_type'}
        assert len(scf_data['kpts']) == 2
        assert scf_data['walker_type'] is SpinSymm.CLOSED
        assert list(scf_data['nmo_pk']) == [8, 8]
        assert all(np.array_equal(X, C)
                   for X, C in zip(scf_data['X'], krks.mo_coeff, strict=True))

    def test_the_orthogonalized_basis_orthogonalizes_each_overlap(self, krks):
        scf_data = periodic_solution(krks, 'ortho_ao')

        for X, overlap in zip(scf_data['X'], krks.get_ovlp(), strict=True):
            assert np.allclose(X.conj().T @ overlap @ X, np.eye(X.shape[1]))

    def test_a_single_kpoint_solution_gets_its_kpoint_axis_back(self,
                                                                single_kpt_rhf):
        """
        A one-k-point calculation holds everything without a k-point axis.
        afqmctools could not read this at all — it inferred the spin symmetry
        from ``mo_coeff[0]``'s shape, which is a *row* when the axis is absent,
        and raised "Unable to determine a valid Slater determinant type".
        """
        scf_data = periodic_solution(single_kpt_rhf)

        assert len(scf_data['kpts']) == 1
        assert scf_data['walker_type'] is SpinSymm.CLOSED
        assert np.shape(scf_data['fock']) == (1, 1, 8, 8)
        assert [np.shape(occ) for occ in scf_data['Xocc']] == [(8,)]
        assert [X.shape for X in scf_data['X']] == [(8, 8)]
        assert list(scf_data['nmo_pk']) == [8]

    def test_a_spin_resolved_solution_needs_the_orthogonalized_basis(self, krks):
        kuhf = krks.to_uhf()

        with pytest.raises(ValueError, match="basis='ortho_ao'"):
            periodic_solution(kuhf)

        assert periodic_solution(kuhf, 'ortho_ao')['walker_type'] \
            is SpinSymm.COLLINEAR

    def test_any_other_basis_is_rejected(self, krks):
        with pytest.raises(ValueError, match="either None or 'ortho_ao'"):
            periodic_solution(krks, np.eye(8))
