# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Format detection and `Hamiltonian.from_hdf5` dispatch."""

import h5py as h5
import numpy as np
import pytest

from safiretools.hamiltonian.base import Hamiltonian, hamiltonian_format
from safiretools.hamiltonian.model.lattice_hamiltonian import LatticeHamiltonian
from safiretools.hamiltonian.molecular import MolecularHamiltonian
from safiretools.hamiltonian.periodic import PeriodicHamiltonian


def _make(tmp_path, name, datasets):
    path = tmp_path / name
    with h5.File(path, 'w') as fh5:
        for key, value in datasets.items():
            fh5.create_dataset(key, data=value)
    return path


@pytest.mark.parametrize(
    "datasets,expected",
    [
        ({'Hamiltonian/ModelHamiltonian/number_of_components': 3}, 'model'),
        ({'Hamiltonian/DenseFactorized/L': np.zeros((4, 2))}, 'dense'),
        ({'Hamiltonian/KPFactorized/L0': np.zeros((2, 2))}, 'kpoint'),
        ({'Hamiltonian/THC/Luv': np.zeros((2, 2))}, 'thc'),
        ({'Interaction/Vq0': np.zeros(2)}, 'kpoint_coqui'),
    ],
)
def test_format_detection(tmp_path, datasets, expected):
    assert hamiltonian_format(_make(tmp_path, 'ham.h5', datasets)) == expected


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="no Hamiltonian in a format"):
        hamiltonian_format(_make(tmp_path, 'ham.h5', {'RandomName': 0}))


@pytest.mark.parametrize(
    "datasets,expected",
    [
        ({'Hamiltonian/ModelHamiltonian/number_of_components': 3}, LatticeHamiltonian),
        ({'Hamiltonian/DenseFactorized/L': np.zeros((4, 2))}, MolecularHamiltonian),
        ({'Hamiltonian/KPFactorized/L0': np.zeros((2, 2))}, PeriodicHamiltonian),
    ],
)
def test_from_hdf5_dispatches_to_the_right_subclass(tmp_path, monkeypatch,
                                                    datasets, expected):
    """
    The dispatch picks the class from the file's format. The read itself is
    stubbed out so this covers dispatch alone; the per-format readers are
    exercised in each subclass's own tests.
    """
    seen = {}
    monkeypatch.setattr(expected, '_read_hdf5',
                        classmethod(lambda cls, path, fmt: seen.setdefault('cls', cls)))

    Hamiltonian.from_hdf5(_make(tmp_path, 'ham.h5', datasets))
    assert seen['cls'] is expected


def test_from_hdf5_on_the_wrong_subclass_raises(tmp_path):
    path = _make(tmp_path, 'ham.h5', {'Hamiltonian/ModelHamiltonian/number_of_components': 1})

    with pytest.raises(ValueError, match="MolecularHamiltonian does not read"):
        MolecularHamiltonian.from_hdf5(path)


def test_thc_has_no_reader_yet(tmp_path):
    path = _make(tmp_path, 'ham.h5', {'Hamiltonian/THC/Luv': np.zeros((2, 2))})

    with pytest.raises(NotImplementedError, match="format 'thc'"):
        Hamiltonian.from_hdf5(path)


def test_spin_symm_is_coerced():
    """`spin_symm` is a plain attribute, but strings and ints are normalized."""
    hamiltonian = LatticeHamiltonian(nsites=4, spin_symm='uhf')
    assert hamiltonian.spin_symm.name == 'COLLINEAR'

    hamiltonian.spin_symm = 3
    assert hamiltonian.spin_symm.name == 'NONCOLLINEAR'

    with pytest.raises(ValueError, match="Unknown spin symmetry"):
        hamiltonian.spin_symm = 'sideways'


class TestWritingPreservesTheRestOfTheFile:
    """
    A SAFIRE input file holds at most one Hamiltonian and at most one
    wavefunction, so `to_hdf5` replaces the Hamiltonian rather than the file.
    That is what lets the two be written in either order.
    """

    @staticmethod
    def _wavefunction(path):
        """Stand in for a wavefunction writer: a `Wavefunction` group."""
        with h5.File(path, 'a') as fh5:
            fh5.create_dataset('Wavefunction/NOMSD/ci_coeffs', data=np.ones(2))

    @staticmethod
    def _hamiltonian(**kwargs):
        return LatticeHamiltonian.from_dict({
            'lattice': dict(L1=2, L2=2, boundary1='pbc', boundary2='pbc'),
            'hamiltonian': dict(t=1.0, nelec=(2, 2), **kwargs),
        })

    def test_a_wavefunction_written_first_survives(self, tmp_path):
        path = tmp_path / 'afqmc.h5'
        self._wavefunction(path)
        self._hamiltonian(U=4.0).to_hdf5(path)

        with h5.File(path) as fh5:
            assert 'Wavefunction/NOMSD/ci_coeffs' in fh5
            assert 'Hamiltonian' in fh5

    def test_a_wavefunction_written_second_survives(self, tmp_path):
        path = tmp_path / 'afqmc.h5'
        self._hamiltonian(U=4.0).to_hdf5(path)
        self._wavefunction(path)

        with h5.File(path) as fh5:
            assert 'Wavefunction/NOMSD/ci_coeffs' in fh5
        assert Hamiltonian.from_hdf5(path).num_components == 2

    def test_to_hdf5_creates_a_file_that_does_not_exist(self, tmp_path):
        path = tmp_path / 'new.h5'
        self._hamiltonian(U=4.0).to_hdf5(path)
        assert hamiltonian_format(path) == 'model'

    def test_rewriting_replaces_the_hamiltonian_without_leaving_stale_terms(self, tmp_path):
        path = tmp_path / 'afqmc.h5'
        self._wavefunction(path)

        many = self._hamiltonian(nbands=2, U=4.0, U1=1.0, J=0.25)
        many.to_hdf5(path)
        few = self._hamiltonian(U=4.0)
        few.to_hdf5(path)

        assert many.num_components > few.num_components

        restored = Hamiltonian.from_hdf5(path)
        assert restored.num_components == few.num_components
        assert sorted(restored.keys()) == sorted(few.keys())

        with h5.File(path) as fh5:
            components = fh5['Hamiltonian/ModelHamiltonian']
            stale = [k for k in components if k.startswith('ModelComponent_')]
            assert len(stale) == few.num_components
            assert 'Wavefunction/NOMSD/ci_coeffs' in fh5


# ----------------------------------------------------------------------
# base-class factory dispatch (Phase 4c)
# ----------------------------------------------------------------------

class TestFactoryDispatch:
    """
    Every construction factory is reachable from `Hamiltonian`, which picks the
    concrete subclass. See DESIGN.md "Every factory dispatches from the base
    class".
    """

    def test_from_dict_builds_a_lattice_hamiltonian(self):
        params = {'lattice': {'L1': 2, 'L2': 2},
                  'hamiltonian': {'t': 1.0, 'U': 4.0, 'nelec': (2, 2)}}

        assert isinstance(Hamiltonian.from_dict(params), LatticeHamiltonian)

    def test_from_integrals_builds_a_molecular_hamiltonian(self):
        hcore = np.eye(3)
        eri = np.zeros((3, 3, 3, 3))

        assert isinstance(Hamiltonian.from_integrals(hcore, eri=eri),
                          MolecularHamiltonian)

    def test_from_dict_matches_the_subclass_factory(self):
        params = {'lattice': {'L1': 2, 'L2': 2},
                  'hamiltonian': {'t': 1.0, 'U': 4.0, 'nelec': (2, 2)}}

        viaBase = Hamiltonian.from_dict(params)
        viaSubclass = LatticeHamiltonian.from_dict(params)

        assert sorted(viaBase.keys()) == sorted(viaSubclass.keys())
        assert viaBase.nsites == viaSubclass.nsites
        assert viaBase.spin_symm is viaSubclass.spin_symm

    # -- the fixed-domain guard --------------------------------------

    @pytest.mark.parametrize('called_on, factory, nargs, builds', [
        (MolecularHamiltonian, 'from_dict', 1, 'LatticeHamiltonian'),
        (PeriodicHamiltonian, 'from_dict', 1, 'LatticeHamiltonian'),
        (LatticeHamiltonian, 'from_integrals', 1, 'MolecularHamiltonian'),
        (PeriodicHamiltonian, 'from_integrals', 1, 'MolecularHamiltonian'),
        (MolecularHamiltonian, 'write_from_pyscf', 3, 'PeriodicHamiltonian'),
        (LatticeHamiltonian, 'write_from_pyscf', 3, 'PeriodicHamiltonian'),
    ])
    def test_a_fixed_domain_factory_refuses_the_wrong_subclass(self, called_on,
                                                               factory, nargs,
                                                               builds):
        with pytest.raises(ValueError, match=builds):
            getattr(called_on, factory)(*[None] * nargs)

    # -- from_pyscf, the one real dispatcher -------------------------

    @pytest.mark.parametrize('scf_data, found', [
        ({}, 'neither'),
        ({'hcore': None}, 'neither'),
        ({'mol': None, 'cell': None}, r"\['cell', 'mol'\]"),
    ])
    def test_from_pyscf_needs_exactly_one_domain_key(self, scf_data, found):
        with pytest.raises(ValueError, match=found):
            Hamiltonian.from_pyscf(scf_data)

    def test_from_pyscf_dispatches_on_the_key_not_the_type(self):
        """
        ``pyscf.pbc.gto.Cell`` subclasses ``pyscf.gto.Mole``, so an isinstance
        test would read a periodic cell as molecular. Dispatch keys off which
        key is present instead, which a sentinel object is enough to show.
        """
        sentinel = object()

        with pytest.raises(ValueError, match='PeriodicHamiltonian'):
            LatticeHamiltonian.from_pyscf({'cell': sentinel})

        with pytest.raises(ValueError, match='MolecularHamiltonian'):
            LatticeHamiltonian.from_pyscf({'mol': sentinel})


@pytest.mark.pyscf
class TestFactoryDispatchPyscf:
    """`Hamiltonian.from_pyscf` against real PySCF checkpoints."""

    def test_a_molecular_checkpoint_reaches_molecular(self, neon_atom, neon_rhf,
                                                      tmp_path):
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        mf, _ = neon_rhf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)

        viaBase = Hamiltonian.from_pyscf(scf_data, chol_cut=1e-5)
        viaSubclass = MolecularHamiltonian.from_pyscf(scf_data, chol_cut=1e-5)

        assert isinstance(viaBase, MolecularHamiltonian)
        assert np.allclose(viaBase.hcore, viaSubclass.hcore)
        assert np.allclose(viaBase.chol, viaSubclass.chol)
        assert np.isclose(viaBase.enuc, viaSubclass.enuc)

    def test_a_periodic_checkpoint_reaches_periodic(self, diamond, diamond_lda):
        from safiretools.hamiltonian.periodic import get_ortho_ao

        mf, kpts = diamond_lda
        X, nmo_pk = get_ortho_ao(diamond, kpts)
        scf_data = {'cell': diamond, 'kpts': kpts, 'hcore': mf.get_hcore(),
                    'X': X, 'nmo_pk': nmo_pk}

        viaBase = Hamiltonian.from_pyscf(scf_data, chol_cut=1e-3, maxvecs=20)
        viaSubclass = PeriodicHamiltonian.from_pyscf(scf_data, chol_cut=1e-3,
                                                     maxvecs=20)

        assert isinstance(viaBase, PeriodicHamiltonian)
        assert viaBase.nkpts == viaSubclass.nkpts
        assert viaBase.nmo_tot == viaSubclass.nmo_tot

    def test_a_domain_specific_keyword_names_the_real_parameter(self, neon_rhf):
        """
        `**kwargs` is forwarded rather than merged, so a periodic keyword on a
        molecular calculation raises TypeError from the concrete classmethod
        instead of being silently ignored.
        """
        from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol

        mf, _ = neon_rhf
        scf_data = load_from_pyscf_chk_mol(mf.chkfile)

        with pytest.raises(TypeError, match='kpoint_symmetry'):
            Hamiltonian.from_pyscf(scf_data, kpoint_symmetry=False)
