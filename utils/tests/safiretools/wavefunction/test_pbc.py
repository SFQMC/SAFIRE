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
`NOMSDWavefunction.from_pbc_scf` and `PHMSDWavefunction.from_pbc_scf`: the
periodic construction paths, and their equivalence with afqmctools.
"""

import warnings

import h5py as h5
import numpy as np
import pytest

from safiretools import NOMSDWavefunction, PHMSDWavefunction, SpinSymm, Wavefunction

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


@pytest.fixture(scope='module')
def diamond_krks():
    """A 2x1x1 diamond KRKS calculation, small enough to run in-line."""
    from pyscf.pbc import dft, gto

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
    mf.kernel()
    return cell, mf, kpts


@pytest.fixture
def closed_scf_data(diamond_krks):
    cell, mf, kpts = diamond_krks
    hcore = mf.get_hcore()

    return {
        'cell': cell,
        'mo_coeff': mf.mo_coeff,
        'Xocc': mf.mo_occ,
        'X': mf.mo_coeff,
        'fock': hcore + mf.get_veff(),
        'walker_type': 'closed',
        'hcore': hcore,
        'nmo_pk': np.array([C.shape[-1] for C in mf.mo_coeff]),
        'mo_energy': mf.mo_energy,
        'kpts': kpts,
    }


@pytest.fixture
def collinear_scf_data(diamond_krks):
    """
    The same calculation presented as a collinear reference, by duplicating the
    closed-shell data across the two spin channels.
    """
    cell, mf, kpts = diamond_krks
    hcore = mf.get_hcore()
    fock = hcore + mf.get_veff()
    occ = np.array([np.array(o) / 2.0 for o in mf.mo_occ])

    return {
        'cell': cell,
        'mo_coeff': mf.mo_coeff,
        'Xocc': np.array([occ, occ]),
        'X': mf.mo_coeff,
        'fock': np.array([fock, fock]),
        'walker_type': 'collinear',
        'hcore': hcore,
        'nmo_pk': np.array([C.shape[-1] for C in mf.mo_coeff]),
        'mo_energy': np.array([mf.mo_energy, mf.mo_energy]),
        'kpts': kpts,
    }


@pytest.fixture
def degenerate_scf_data(collinear_scf_data):
    """A collinear reference with two partially occupied bands at one k-point."""
    occ = np.array(collinear_scf_data['Xocc'])
    occ[:, 0, 3] = 0.5
    occ[:, 0, 4] = 0.5

    return dict(collinear_scf_data, Xocc=occ)


def datasets(path):
    found = {}
    with h5.File(path, 'r') as fh5:
        fh5.visititems(lambda name, obj: found.__setitem__(name, obj[...])
                       if isinstance(obj, h5.Dataset) else None)
    return found


class TestSingleDeterminant:

    def test_a_closed_shell_reference(self, closed_scf_data):
        wavefunction = NOMSDWavefunction.from_pbc_scf(closed_scf_data)

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.spin_symm is SpinSymm.CLOSED
        assert wavefunction.nelec == (8, 8)
        assert wavefunction.nmo == 16
        # a closed-shell determinant carries one spin block
        assert wavefunction.dets.shape == (1, 16, 8)

    def test_a_collinear_reference(self, collinear_scf_data):
        wavefunction = NOMSDWavefunction.from_pbc_scf(collinear_scf_data)

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (8, 8)
        assert wavefunction.dets.shape == (1, 16, 16)

    def test_partial_occupancies_still_give_one_determinant(self,
                                                            degenerate_scf_data):
        wavefunction = NOMSDWavefunction.from_pbc_scf(degenerate_scf_data)

        assert isinstance(wavefunction, NOMSDWavefunction)
        assert wavefunction.ndets == 1

    def test_a_collinear_reference_requires_the_orthogonalized_basis(
            self, collinear_scf_data):
        with pytest.raises(ValueError, match="orthogonalized AO basis"):
            NOMSDWavefunction.from_pbc_scf(collinear_scf_data, ortho_ao=False)

    def test_it_round_trips(self, closed_scf_data, tmp_path):
        wavefunction = NOMSDWavefunction.from_pbc_scf(closed_scf_data)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (8, 8)
        assert read_back.spin_symm is SpinSymm.CLOSED
        assert np.allclose(read_back.dets, wavefunction.dets)


class TestMultiDeterminant:

    def test_degenerate_bands_give_a_particle_hole_expansion(
            self, degenerate_scf_data):
        wavefunction = PHMSDWavefunction.from_pbc_scf(degenerate_scf_data,
                                                      ndet_max=4)

        assert isinstance(wavefunction, PHMSDWavefunction)
        assert wavefunction.ndets == 4
        assert wavefunction.nelec == (8, 8)
        # the k-point orbital matrices become the expansion's reference
        assert wavefunction.nreferences == 2

    def test_every_determinant_is_kept_by_default(self, degenerate_scf_data):
        # afqmctools raised TypeError on ndet_max=None, which is what its own
        #   pyscf_to_afqmc CLI passed when -n was not given
        wavefunction = PHMSDWavefunction.from_pbc_scf(degenerate_scf_data)
        capped = PHMSDWavefunction.from_pbc_scf(degenerate_scf_data,
                                                ndet_max=2)

        assert wavefunction.ndets == 4
        assert capped.ndets == 2

    def test_it_refuses_an_integer_occupancy_system(self, collinear_scf_data):
        with pytest.raises(ValueError,
                           match="no partially occupied degenerate bands"):
            PHMSDWavefunction.from_pbc_scf(collinear_scf_data)

    def test_a_partially_occupied_closed_shell_reference_is_rejected(
            self, closed_scf_data):
        # afqmctools reached this case too and failed with a TypeError deeper
        #   in, dividing a list of occupancy arrays by two
        occ = np.full_like(np.array(closed_scf_data['Xocc'], dtype=float), 0.9)
        scf_data = dict(closed_scf_data, Xocc=occ)

        with pytest.raises(ValueError, match="closed-shell reference are"):
            PHMSDWavefunction.from_pbc_scf(scf_data, ndet_max=4)

    def test_afqmctools_failed_on_that_case(self, closed_scf_data):
        from afqmctools.wavefunction.pbc import write_wfn_pbc

        occ = np.full_like(np.array(closed_scf_data['Xocc'], dtype=float), 0.9)
        scf_data = dict(closed_scf_data, Xocc=occ)

        with pytest.raises(TypeError):
            write_wfn_pbc(scf_data, True, '/dev/null', ndet_max=4)

    def test_it_round_trips(self, degenerate_scf_data, tmp_path):
        wavefunction = PHMSDWavefunction.from_pbc_scf(degenerate_scf_data,
                                                      ndet_max=4)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, PHMSDWavefunction)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert np.array_equal(read_back.occb, wavefunction.occb)
        assert read_back.nreferences == 2


class TestEquivalenceWithAfqmctools:

    @pytest.mark.parametrize('fixture, ortho_ao', [
        ('closed_scf_data', True),
        ('collinear_scf_data', True),
        ('degenerate_scf_data', True),
        ('closed_scf_data', False),
    ])
    def test_the_single_determinant_file_matches_write_wfn_pbc(self, request,
                                                               tmp_path,
                                                               fixture,
                                                               ortho_ao):
        from afqmctools.wavefunction.pbc import write_wfn_pbc

        scf_data = request.getfixturevalue(fixture)
        old = tmp_path / 'old.h5'
        new = tmp_path / 'new.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_wfn_pbc(scf_data, ortho_ao, old, rediag=True)
            NOMSDWavefunction.from_pbc_scf(scf_data,
                                           ortho_ao=ortho_ao).to_hdf5(new)

        a, b = datasets(old), datasets(new)
        assert set(a) == set(b)
        for key in a:
            # afqmctools also applied the 1e-8 sparsification threshold to
            #   Psi0, which is stored dense; safiretools does not
            assert np.allclose(a[key], b[key], atol=1e-8), key

    def test_the_expansion_matches_apart_from_the_reference_count(
            self, degenerate_scf_data, tmp_path):
        from afqmctools.wavefunction.pbc import write_wfn_pbc

        old = tmp_path / 'old.h5'
        new = tmp_path / 'new.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_wfn_pbc(degenerate_scf_data, True, old, rediag=True,
                          ndet_max=4)
            PHMSDWavefunction.from_pbc_scf(degenerate_scf_data,
                                           ndet_max=4).to_hdf5(new)

        a, b = datasets(old), datasets(new)
        assert set(a) == set(b)

        differing = [key for key in a
                     if not np.allclose(a[key], b[key], atol=1e-8)]
        assert differing == ['Wavefunction/PHMSD/type']

        # afqmctools declared one reference while writing two, so the
        #   executable read only the alpha one
        assert int(a['Wavefunction/PHMSD/type']) == 1
        assert int(b['Wavefunction/PHMSD/type']) == 2
