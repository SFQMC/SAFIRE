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

import warnings

import h5py as h5
import numpy as np
import pytest

from safiretools import NOMSDWavefunction, PHMSDWavefunction, SpinSymm, Wavefunction

pyscf = pytest.importorskip("pyscf")

pytestmark = pytest.mark.pyscf


def scf_data_from(mol, mf, spin_symm):
    """The subset of a PySCF checkpoint the wavefunction construction reads."""
    return {
        'mol': mol,
        'mo_coeff': mf.mo_coeff,
        'mo_occ': mf.mo_occ,
        'nelec': mol.nelec,
        'X': mf.mo_coeff,
        'norb': np.asarray(mf.mo_coeff).shape[-1],
        'walker_type': spin_symm,
        'mo_energy': mf.mo_energy,
    }


@pytest.fixture(scope='module')
def neon_rhf():
    from pyscf import gto, scf

    mol = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


@pytest.fixture(scope='module')
def neon_rhf_631g():
    from pyscf import gto, scf

    # a basis with room for both a frozen core and virtual orbitals, so that a
    #   CAS expansion is non-trivial
    mol = gto.M(atom='Ne 0 0 0', basis='6-31g', verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    return mol, mf


@pytest.fixture(scope='module')
def oxygen_rohf():
    from pyscf import gto, scf

    mol = gto.M(atom='O 0 0 0', basis='sto-3g', spin=2, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    return mol, mf


@pytest.fixture(scope='module')
def lithium_rohf():
    from pyscf import gto, scf

    mol = gto.M(atom='Li 0 0 0', basis='sto-3g', spin=1, verbose=0)
    mf = scf.ROHF(mol)
    mf.kernel()
    return mol, mf


class TestFromPyscf:

    def test_a_closed_shell_reference(self, neon_rhf):
        mol, mf = neon_rhf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'closed'))

        assert wavefunction.spin_symm is SpinSymm.CLOSED
        assert wavefunction.nelec == (5, 5)
        assert wavefunction.nmo == 5
        assert wavefunction.dets.shape == (1, 5, 5)

    def test_an_open_shell_reference_is_collinear(self, oxygen_rohf):
        mol, mf = oxygen_rohf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'collinear'))

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (5, 3)
        assert wavefunction.dets.shape == (1, 5, 8)

    def test_an_active_space_trims_and_reindexes_the_orbitals(self,
                                                              oxygen_rohf):
        mol, mf = oxygen_rohf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'collinear'), cas=(4, 3))

        # (8 - 4) // 2 = 2 frozen core orbitals, leaving 3 active ones
        assert wavefunction.nelec == (3, 1)
        assert wavefunction.nmo == 3

    def test_a_frozen_core_can_empty_the_beta_channel(self, lithium_rohf):
        # this was afqmctools' fully-polarized case; it is collinear with
        #   ndown == 0 now, and afqmctools itself crashed here
        mol, mf = lithium_rohf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'collinear'), cas=(1, 4))

        assert wavefunction.spin_symm is SpinSymm.COLLINEAR
        assert wavefunction.nelec == (1, 0)
        assert wavefunction.dets.shape == (1, 4, 1)

    def test_afqmctools_could_not_build_that_case(self, lithium_rohf):
        from afqmctools.wavefunction.mol import generate_wavefunction

        mol, mf = lithium_rohf
        with pytest.raises(IndexError):
            generate_wavefunction(scf_data_from(mol, mf, 'collinear'),
                                  cas=(1, 4))

    def test_the_spin_symmetry_can_be_overridden(self, neon_rhf):
        mol, mf = neon_rhf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'closed'), spin_symm='closed')

        assert wavefunction.spin_symm is SpinSymm.CLOSED

    def test_a_wrong_occupancy_count_is_reported(self, neon_rhf):
        mol, mf = neon_rhf
        scf_data = scf_data_from(mol, mf, 'closed')
        scf_data['mo_occ'] = np.zeros_like(np.asarray(mf.mo_occ))

        with pytest.raises(ValueError, match="alpha occupied orbitals"):
            NOMSDWavefunction.from_pyscf(scf_data)

    def test_cas_and_ortho_ao_cannot_be_combined(self, neon_rhf):
        mol, mf = neon_rhf

        with pytest.raises(ValueError, match="cas and ortho_ao"):
            NOMSDWavefunction.from_pyscf(scf_data_from(mol, mf, 'closed'),
                                         ortho_ao=True, cas=(4, 2))

    def test_the_result_is_orthonormal(self, oxygen_rohf):
        mol, mf = oxygen_rohf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'collinear'))

        for block in wavefunction.spin_blocks(0):
            assert np.allclose(block.conj().T @ block, np.eye(block.shape[1]))

    def test_it_round_trips(self, oxygen_rohf, tmp_path):
        mol, mf = oxygen_rohf
        wavefunction = NOMSDWavefunction.from_pyscf(
            scf_data_from(mol, mf, 'collinear'))
        path = tmp_path / 'wfn.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            wavefunction.to_hdf5(path)
        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, NOMSDWavefunction)
        assert read_back.nelec == (5, 3)
        assert read_back.spin_symm is SpinSymm.COLLINEAR
        assert np.allclose(read_back.dets, wavefunction.dets)


class TestEquivalenceWithAfqmctools:

    @pytest.mark.parametrize('fixture, spin_symm', [
        ('neon_rhf', 'closed'),
        ('oxygen_rohf', 'collinear'),
    ])
    def test_the_written_file_matches_write_wfn_mol(self, request, tmp_path,
                                                    fixture, spin_symm):
        from afqmctools.wavefunction.mol import write_wfn_mol

        mol, mf = request.getfixturevalue(fixture)
        scf_data = scf_data_from(mol, mf, spin_symm)

        old = tmp_path / 'old.h5'
        new = tmp_path / 'new.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_wfn_mol(scf_data, old)
            NOMSDWavefunction.from_pyscf(scf_data).to_hdf5(new)

        def datasets(path):
            found = {}
            with h5.File(path, 'r') as fh5:
                fh5.visititems(
                    lambda name, obj: found.__setitem__(name, obj[...])
                    if isinstance(obj, h5.Dataset) else None)
            return found

        a, b = datasets(old), datasets(new)
        assert set(a) == set(b)
        for key in a:
            assert np.allclose(a[key], b[key]), key


class TestFromPyscfCas:

    @pytest.fixture
    def casci_chkfile(self, neon_rhf_631g, tmp_path):
        from pyscf import mcscf

        mol, mf = neon_rhf_631g
        mc = mcscf.CASSCF(mf, 4, 4)
        mc.chkfile = str(tmp_path / 'cas.chk')
        mc.kernel()

        # pyscf dumps ncore/ncas but not the CI vector itself
        with h5.File(mc.chkfile, 'a') as fh5:
            if 'mcscf/ci' in fh5:
                del fh5['mcscf/ci']
            fh5['mcscf/ci'] = mc.ci

        return mc.chkfile

    def test_it_reads_the_expansion(self, neon_rhf_631g, casci_chkfile):
        mol, _ = neon_rhf_631g
        wavefunction = PHMSDWavefunction.from_pyscf_cas(mol, casci_chkfile,
                                                        tol=1e-6)

        assert isinstance(wavefunction, PHMSDWavefunction)
        assert wavefunction.nelec == mol.nelec
        assert wavefunction.nmo == mol.nao_nr()
        assert wavefunction.ndets > 1
        # the frozen core is reinserted into every determinant
        assert np.all(wavefunction.occa[:, :3] == np.arange(3))

    def test_the_coefficients_are_sorted_by_magnitude(self, neon_rhf_631g,
                                                      casci_chkfile):
        mol, _ = neon_rhf_631g
        coeffs = PHMSDWavefunction.from_pyscf_cas(mol, casci_chkfile,
                                                  tol=1e-6).coeffs
        magnitudes = np.abs(coeffs)

        assert np.all(np.diff(magnitudes) <= 1e-15)

    def test_max_det_truncates(self, neon_rhf_631g, casci_chkfile):
        mol, _ = neon_rhf_631g
        wavefunction = PHMSDWavefunction.from_pyscf_cas(mol, casci_chkfile,
                                                        tol=1e-8, max_det=3)

        assert wavefunction.ndets == 3

    def test_it_round_trips(self, neon_rhf_631g, casci_chkfile, tmp_path):
        mol, _ = neon_rhf_631g
        wavefunction = PHMSDWavefunction.from_pyscf_cas(mol, casci_chkfile,
                                                        tol=1e-6)
        path = tmp_path / 'wfn.h5'
        wavefunction.to_hdf5(path)

        read_back = Wavefunction.from_hdf5(path)

        assert isinstance(read_back, PHMSDWavefunction)
        assert np.array_equal(read_back.occa, wavefunction.occa)
        assert np.array_equal(read_back.occb, wavefunction.occb)
        assert np.allclose(read_back.coeffs, wavefunction.coeffs)

    def test_it_matches_afqmctools_write_cas_wfn(self, neon_rhf_631g,
                                                 casci_chkfile, tmp_path):
        from afqmctools.wavefunction.mol import write_cas_wfn

        mol, _ = neon_rhf_631g
        old = tmp_path / 'old.h5'
        new = tmp_path / 'new.h5'

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            write_cas_wfn(mol, casci_chkfile, outname=str(old),
                          tol_trunc=1e-6)
            PHMSDWavefunction.from_pyscf_cas(mol, casci_chkfile,
                                             tol=1e-6).to_hdf5(new)

        with h5.File(old, 'r') as fh5:
            old_group = {key: fh5[f'Wavefunction/PHMSD/{key}'][...]
                         for key in ('dims', 'occs', 'ci_coeffs')}
        with h5.File(new, 'r') as fh5:
            new_group = {key: fh5[f'Wavefunction/PHMSD/{key}'][...]
                         for key in ('dims', 'occs', 'ci_coeffs')}

        assert np.array_equal(old_group['dims'], new_group['dims'])
        assert np.array_equal(old_group['occs'], new_group['occs'])
        assert np.allclose(old_group['ci_coeffs'], new_group['ci_coeffs'])
