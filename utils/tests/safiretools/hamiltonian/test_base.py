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

from safiretools.hamiltonian.base import (
    Hamiltonian,
    hamiltonian_format,
    write_hamiltonian_format,
)
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
    """
    The layout fallback, which is for files written before
    ``Hamiltonian/type`` existed. See `TestTheRecordedFormat`.
    """
    assert hamiltonian_format(_make(tmp_path, 'ham.h5', datasets)) == expected


def test_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="no Hamiltonian in a format"):
        hamiltonian_format(_make(tmp_path, 'ham.h5', {'RandomName': 0}))


class TestTheRecordedFormat:
    """
    A writer records its format in ``Hamiltonian/type``, and
    `hamiltonian_format` reads that in preference to guessing from the layout.
    """

    def test_the_tag_is_read_in_preference_to_the_layout(self, tmp_path):
        """
        A tag that contradicts the layout still wins: it is what the writer
        said, and the layout heuristic is only a fallback for files that predate
        it. Nothing writes such a file — this pins which of the two is trusted.
        """
        path = _make(tmp_path, 'ham.h5', {
            'Hamiltonian/DenseFactorized/L': np.zeros((4, 2)),
            'Hamiltonian/type': 'ModelHamiltonian',
        })

        assert hamiltonian_format(path) == 'model'

    @pytest.mark.parametrize('tag,expected', [
        ('ModelHamiltonian', 'model'),
        ('RealDenseFactorized', 'dense'),
        ('KPFactorized', 'kpoint'),
        ('THC', 'thc'),
    ])
    def test_every_tag_maps_to_a_format(self, tmp_path, tag, expected):
        """The stored value is the executable's ``HamiltonianTypes`` name."""
        path = _make(tmp_path, 'ham.h5', {'Hamiltonian/type': tag})

        assert hamiltonian_format(path) == expected

    def test_an_unrecognized_tag_raises(self, tmp_path):
        path = _make(tmp_path, 'ham.h5', {'Hamiltonian/type': 'Sideways'})

        with pytest.raises(ValueError, match="unknown Hamiltonian type 'Sideways'"):
            hamiltonian_format(path)

    def test_writing_an_unknown_format_raises(self, tmp_path):
        with h5.File(tmp_path / 'ham.h5', 'w') as fh5:
            with pytest.raises(ValueError, match="not a valid HamiltonianFormat"):
                write_hamiltonian_format(fh5, 'sideways')

    def test_writing_an_unrecordable_format_raises(self, tmp_path):
        """A CoQuí file has no ``Hamiltonian`` group to record a type in."""
        with h5.File(tmp_path / 'ham.h5', 'w') as fh5:
            with pytest.raises(ValueError, match="is never recorded"):
                write_hamiltonian_format(fh5, 'kpoint_coqui')

    def test_rewriting_replaces_the_tag(self, tmp_path):
        path = tmp_path / 'ham.h5'
        with h5.File(path, 'w') as fh5:
            write_hamiltonian_format(fh5, 'dense')
            write_hamiltonian_format(fh5, 'model')

        assert hamiltonian_format(path) == 'model'


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
        with h5.File(path, 'r') as fh5:
            assert fh5['Hamiltonian/type'].asstr()[()] == 'ModelHamiltonian'

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
