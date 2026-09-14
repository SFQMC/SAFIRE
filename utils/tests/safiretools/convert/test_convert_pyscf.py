# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Reading a PySCF checkpoint into the ``scf_data`` mapping."""

import numpy as np
import pytest

from safiretools import SpinSymm
from safiretools.convert.pyscf import (
    _nesting_depth,
    _per_kpoint,
    as_scf_data,
    canonical_orthogonalization,
    determine_spin_symm,
    is_periodic_chk,
    load_pyscf_chk,
    load_pyscf_chk_mol,
)

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


@pytest.fixture(scope='module')
def rhf_chk(tmp_path_factory):
    from pyscf import gto, scf

    mol = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.chkfile = str(tmp_path_factory.mktemp('rhf') / 'rhf.chk')
    mf.kernel()
    return mf.chkfile


@pytest.fixture(scope='module')
def rohf_chk(tmp_path_factory):
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.chkfile = str(tmp_path_factory.mktemp('rohf') / 'rohf.chk')
    mf.kernel()
    return mf.chkfile


@pytest.fixture(scope='module')
def uhf_chk(tmp_path_factory):
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf = scf.UHF(mol)
    mf.chkfile = str(tmp_path_factory.mktemp('uhf') / 'uhf.chk')
    mf.kernel()
    return mf.chkfile


@pytest.fixture(scope='module')
def krks_chk(tmp_path_factory):
    """A 2x1x1 diamond KRKS checkpoint, with the orthogonalized-AO basis
    added the way the workflow scripts add it."""
    import h5py as h5
    from pyscf.pbc import dft, gto

    from safiretools.hamiltonian.periodic import get_ortho_ao

    cell = gto.Cell()
    alat = 3.6
    cell.a = (np.ones((3, 3)) - np.eye(3)) * alat / 2.0
    cell.atom = (('C', 0, 0, 0), ('C', np.array([0.25, 0.25, 0.25]) * alat))
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [12] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kpts = cell.make_kpts([2, 1, 1])
    mf = dft.KRKS(cell, kpts=kpts)
    mf.chkfile = str(tmp_path_factory.mktemp('krks') / 'krks.chk')
    mf.kernel()

    X, nmo_per_kpt = get_ortho_ao(cell, kpts)
    with h5.File(mf.chkfile, 'a') as fh5:
        fh5['scf/orthoAORot'] = X
        fh5['scf/nmo_per_kpt'] = nmo_per_kpt
        fh5['scf/hcore'] = mf.get_hcore()
        fh5['scf/fock'] = mf.get_hcore() + mf.get_veff()

    return mf.chkfile


@pytest.fixture(scope='module')
def single_kpt_chk(tmp_path_factory):
    """
    A one-k-point calculation. PySCF records it under ``scf/kpt`` and stores
    every per-orbital quantity with no k-point axis, which is the layout that
    needs a k-point axis put back.
    """
    import h5py as h5
    from pyscf.pbc import gto, scf

    from safiretools.hamiltonian.periodic import get_ortho_ao

    cell = gto.Cell()
    alat = 3.6
    cell.a = (np.ones((3, 3)) - np.eye(3)) * alat / 2.0
    cell.atom = (('C', 0, 0, 0), ('C', np.array([0.25, 0.25, 0.25]) * alat))
    cell.basis = 'gth-szv'
    cell.pseudo = 'gth-pade'
    cell.mesh = [12] * 3
    cell.verbose = 0
    cell.build(parse_arg=False)

    kpts = cell.make_kpts([1, 1, 1])
    mf = scf.RHF(cell, kpt=kpts[0])
    mf.chkfile = str(tmp_path_factory.mktemp('single') / 'rhf.chk')
    mf.kernel()

    X, nmo_per_kpt = get_ortho_ao(cell, kpts)
    with h5.File(mf.chkfile, 'a') as fh5:
        fh5['scf/orthoAORot'] = X
        fh5['scf/nmo_per_kpt'] = nmo_per_kpt
        fh5['scf/hcore'] = mf.get_hcore()
        fh5['scf/fock'] = mf.get_hcore() + mf.get_veff()

    return mf.chkfile


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


class TestKindDetection:

    def test_a_molecule_has_no_lattice_vectors(self, rhf_chk):
        assert is_periodic_chk(rhf_chk) is False

    def test_a_cell_does(self, krks_chk):
        assert is_periodic_chk(krks_chk) is True

    def test_the_wrong_kind_is_reported(self, rhf_chk):
        with pytest.raises(ValueError, match="but a periodic one is needed"):
            as_scf_data(rhf_chk, periodic=True)

    def test_a_mapping_passes_straight_through(self):
        scf_data = {'anything': 1}

        assert as_scf_data(scf_data) is scf_data

    def test_anything_else_is_rejected(self):
        with pytest.raises(ValueError, match="checkpoint path or an scf_data"):
            as_scf_data(42)


class TestDetermineSpinSymm:
    """
    The spin symmetry is read off the *calculation*, not off the shape of a
    Slater matrix built from it later.
    """

    def test_an_rhf_solution_is_closed_shell(self, rhf_chk):
        assert load_pyscf_chk_mol(rhf_chk)['walker_type'] is SpinSymm.CLOSED

    def test_an_rohf_solution_is_collinear(self, rohf_chk):
        assert load_pyscf_chk_mol(rohf_chk)['walker_type'] is SpinSymm.COLLINEAR

    def test_a_uhf_solution_is_collinear(self, uhf_chk):
        assert load_pyscf_chk_mol(uhf_chk)['walker_type'] is SpinSymm.COLLINEAR

    def test_spin_resolved_orbitals_are_collinear(self):
        assert determine_spin_symm(np.zeros((2, 6, 6)), np.zeros((2, 6))) \
            is SpinSymm.COLLINEAR

    def test_a_singly_occupied_orbital_is_collinear(self):
        """An ROHF solution records both channels in one occupancy vector."""
        occupancies = np.array([2.0, 2.0, 1.0, 1.0, 0.0, 0.0])

        assert determine_spin_symm(np.zeros((6, 6)), occupancies) \
            is SpinSymm.COLLINEAR

    def test_integer_double_occupancies_are_closed_shell(self):
        occupancies = np.array([2.0, 2.0, 0.0, 0.0])

        assert determine_spin_symm(np.zeros((4, 4)), occupancies) \
            is SpinSymm.CLOSED

    @pytest.mark.parametrize('soc_type', ['x2c', 'ecp'])
    def test_a_spin_orbit_treatment_forces_noncollinear(self, soc_type):
        assert determine_spin_symm(np.zeros((6, 6)), np.zeros(6),
                                   soc_type=soc_type) is SpinSymm.NONCOLLINEAR

    def test_a_spin_free_two_component_treatment_does_not(self):
        """``sfx2c`` leaves hcore in the spatial-orbital basis."""
        assert determine_spin_symm(np.zeros((6, 6)), np.zeros(6),
                                   soc_type='sfx2c') is SpinSymm.CLOSED

    def test_a_spinor_hcore_is_noncollinear(self):
        assert determine_spin_symm(np.zeros((6, 6)), np.zeros(6),
                                   hcore=np.zeros((12, 12)),
                                   nmo=6) is SpinSymm.NONCOLLINEAR

    def test_a_spinor_orbital_basis_is_noncollinear(self):
        """
        A GHF solution has one ``mo_coeff`` matrix like an RHF one; only the
        basis size separates them.
        """
        assert determine_spin_symm(np.zeros((12, 12)), np.zeros(12), nmo=6) \
            is SpinSymm.NONCOLLINEAR


class TestMolecular:

    def test_it_reads_what_the_factories_need(self, rohf_chk):
        scf_data = load_pyscf_chk_mol(rohf_chk)

        assert set(scf_data) >= {'mol', 'nelec', 'mo_occ', 'mo_coeff', 'hcore',
                                 'norb', 'X', 'df_ints', 'walker_type',
                                 'soc_type'}
        assert scf_data['nelec'] == (5, 3)
        assert scf_data['norb'] == 5
        assert scf_data['hcore'].shape == (5, 5)
        assert scf_data['df_ints'] is None

    def test_an_unknown_soc_type_is_rejected(self, rohf_chk):
        with pytest.raises(ValueError, match="unknown soc_type"):
            load_pyscf_chk_mol(rohf_chk, soc_type='nonsense')

    def test_a_stored_hcore_warns_when_soc_was_requested(self, rohf_chk,
                                                         tmp_path):
        import shutil

        import h5py as h5

        copied = tmp_path / 'with_hcore.chk'
        shutil.copy(rohf_chk, copied)
        with h5.File(copied, 'a') as fh5:
            fh5['/scf/hcore'] = np.zeros((5, 5))

        with pytest.warns(UserWarning, match="unclear whether"):
            load_pyscf_chk_mol(copied, soc_type='sfx2c')

    def test_it_matches_afqmctools(self, rohf_chk):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        old = load_from_pyscf_chk_mol(rohf_chk)
        new = load_pyscf_chk_mol(rohf_chk)

        for key in ('nelec', 'norb'):
            assert old[key] == new[key]
        for key in ('mo_occ', 'mo_coeff', 'hcore', 'X'):
            assert np.allclose(old[key], new[key]), key


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

    def test_the_mo_basis_is_read_without_ortho_ao(self, krks_chk):
        scf_data = load_pyscf_chk(krks_chk, ortho_ao=False)

        assert set(scf_data) >= {'cell', 'kpts', 'Xocc', 'hcore', 'X',
                                 'nmo_pk', 'mo_coeff', 'nao', 'fock',
                                 'mo_energy', 'walker_type'}
        assert len(scf_data['kpts']) == 2
        assert scf_data['walker_type'] is SpinSymm.CLOSED
        assert list(scf_data['nmo_pk']) == [8, 8]

    def test_the_orthogonalized_basis_is_read_with_it(self, krks_chk):
        scf_data = load_pyscf_chk(krks_chk, ortho_ao=True)

        assert len(scf_data['X']) == 2
        assert all(Xk.shape[1] == n
                   for Xk, n in zip(scf_data['X'], scf_data['nmo_pk']))

    def test_a_single_kpoint_solution_gets_its_kpoint_axis_back(self,
                                                                single_kpt_chk):
        """
        A one-k-point calculation stores everything without a k-point axis.
        afqmctools could not read this at all — it inferred the spin symmetry
        from ``mo_coeff[0]``'s shape, which is a *row* when the axis is absent,
        and raised "Unable to determine a valid Slater determinant type".
        """
        scf_data = load_pyscf_chk(single_kpt_chk, ortho_ao=False)

        assert len(scf_data['kpts']) == 1
        assert scf_data['walker_type'] is SpinSymm.CLOSED
        assert np.shape(scf_data['fock']) == (1, 1, 8, 8)
        assert [np.shape(occ) for occ in scf_data['Xocc']] == [(8,)]
        assert [X.shape for X in scf_data['X']] == [(8, 8)]
        assert list(scf_data['nmo_pk']) == [8]

    def test_afqmctools_could_not_read_that_layout(self, single_kpt_chk):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk

        with pytest.raises(ValueError, match="Slater determinant type"):
            load_from_pyscf_chk(single_kpt_chk, orthoAO=False)

    def test_a_self_contradictory_checkpoint_is_reported(self, krks_chk,
                                                          tmp_path):
        """
        ``scf/fock`` is written by hand after the SCF in these workflows, so it
        can end up with a spin structure the solution does not have. Caught
        here rather than as an IndexError deep in the wavefunction builder.
        """
        import shutil

        import h5py as h5

        copied = tmp_path / 'contradictory.chk'
        shutil.copy(krks_chk, copied)
        with h5.File(copied, 'a') as fh5:
            fock = fh5['scf/fock'][...]
            del fh5['scf/fock']
            fh5['scf/fock'] = np.array([fock, fock])      # a spin axis

        with pytest.raises(ValueError, match="disagrees with itself"):
            load_pyscf_chk(copied, ortho_ao=True)

    def test_a_supplied_hcore_overrides_the_checkpoint(self, krks_chk):
        hcore = np.zeros((2, 8, 8))
        scf_data = load_pyscf_chk(krks_chk, hcore=hcore)

        assert np.allclose(scf_data['hcore'], 0.0)
