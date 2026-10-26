# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Shared fixtures for the `safiretools.wavefunction` tests."""

import numpy as np
import pytest

from safiretools import NOMSDWavefunction, PHMSDWavefunction


@pytest.fixture
def rng():
    return np.random.default_rng(20260903)


@pytest.fixture
def orthonormal(rng):
    """A factory for random complex matrices with orthonormal columns."""
    def make(nrows, ncols):
        if ncols == 0:
            return np.zeros((nrows, 0), dtype=np.complex128)
        matrix = (rng.normal(size=(nrows, ncols))
                  + 1j * rng.normal(size=(nrows, ncols)))
        return np.linalg.qr(matrix)[0]
    return make


@pytest.fixture
def layouts_close():
    """
    `numpy.allclose` for values in the public spin layout, where a collinear
    value is a tuple of two differently shaped arrays.
    """
    def close(a, b):
        if isinstance(a, tuple) or isinstance(b, tuple):
            return isinstance(a, tuple) and isinstance(b, tuple) \
                and len(a) == len(b) \
                and all(x.shape == y.shape and np.allclose(x, y)
                        for x, y in zip(a, b))
        return a.shape == b.shape and np.allclose(a, b)
    return close


@pytest.fixture
def make_nomsd(orthonormal):
    """
    A factory for `NOMSDWavefunction`s of a given spin symmetry, with
    orthonormal spin blocks. A noncollinear one takes all of `nelec` in its
    single channel.
    """
    def make(spin_symm='collinear', nelec=(3, 2), nmo=6, ndets=1):
        def stack(nrows, width):
            return np.array([orthonormal(nrows, width) for _ in range(ndets)])

        if spin_symm == 'closed':
            dets = stack(nmo, nelec[0])
        elif spin_symm == 'collinear':
            dets = (stack(nmo, nelec[0]), stack(nmo, nelec[1]))
        else:
            dets = stack(2 * nmo, sum(nelec)).reshape(ndets, 2, nmo, sum(nelec))

        coeffs = np.array([1.0 + 0j] + [0.25 + 0j] * (ndets - 1))
        return NOMSDWavefunction(coeffs=coeffs, dets=dets)

    return make


@pytest.fixture
def make_phmsd():
    """A factory for small `PHMSDWavefunction`s."""
    def make(nmo=6, orbitals=None):
        return PHMSDWavefunction(
            coeffs=np.array([0.9 + 0j, 0.3 + 0j, 0.1 + 0j]),
            occa=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3]]),
            occb=np.array([[0, 1], [0, 2], [1, 2]]),
            nmo=nmo,
            orbitals=orbitals,
        )

    return make
