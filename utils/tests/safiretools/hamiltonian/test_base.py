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
