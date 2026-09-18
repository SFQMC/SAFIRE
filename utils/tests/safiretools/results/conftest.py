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
Fixtures for the results tests: a synthetic results.h5 in the layout AFQMC writes,
and the bin series that went into it.
"""

import numpy as np
import h5py as h5
import pytest

NBINS = 120
NMO = 3


def nda_complex(group, name, array):
    """
    Write `array` the way nda's h5 layer does: the real and imaginary parts as a
    trailing axis of length 2, plus the ``__complex__`` marker.
    """
    array = np.asarray(array, dtype=np.complex128)
    interleaved = np.ascontiguousarray(array).view(np.float64)
    dataset = group.create_dataset(name, data=interleaved.reshape(array.shape + (2,)))
    dataset.attrs['__complex__'] = '1'
    return dataset


@pytest.fixture
def measured():
    """The bin series `results_file` holds, by the path they are measured under."""
    rng = np.random.default_rng(11)
    return {
        'BackPropEstimator/Steps=40/OneRDM': (
            rng.normal(0.5, 0.05, (NBINS, 2, NMO, NMO))
            + 1j * rng.normal(0.0, 0.01, (NBINS, 2, NMO, NMO))
        ),
        'Energy': rng.normal(-5.0, 0.1, NBINS) + 1j * rng.normal(0.0, 0.01, NBINS),
        'Norm': rng.normal(2.0, 0.02, NBINS) + 0j,
        # a series of its own length, to combine with one of the others
        'Short': rng.normal(0.0, 1.0, NBINS // 4),
        # a real dataset, which AFQMC writes without the __complex__ marker
        'Weight': rng.normal(1.0, 0.05, NBINS),
    }


@pytest.fixture
def results_file(tmp_path, measured):
    """A results.h5 carrying `measured` in Stage0, and a shorter Energy in Stage1."""
    path = tmp_path / 'afqmc.results.h5'

    with h5.File(path, 'w') as f:
        stage0 = f.create_group('Measurements/Stage0')
        for name, bins in measured.items():
            group = stage0.create_group(name)
            if np.iscomplexobj(bins):
                nda_complex(group, 'bins', bins)
            else:
                group.create_dataset('bins', data=bins)

        stage1 = f.create_group('Measurements/Stage1/Energy')
        nda_complex(stage1, 'bins', measured['Energy'][:7])

    return path
