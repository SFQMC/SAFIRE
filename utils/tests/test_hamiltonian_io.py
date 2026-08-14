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
Tests for utils/afqmctools/hamiltonian/io.py that need no SCF backend.
"""

import h5py as h5
import numpy as np
import pytest

from afqmctools.hamiltonian.io import write_dense
from afqmctools.utils.io import from_complex


def _hermitian_cplx(nmo, seed=11):
    """A hermitian complex matrix -- i.e. one with a purely real diagonal."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((nmo, nmo)) + 1j * rng.standard_normal((nmo, nmo))
    return a + a.conj().T


def _real_chol(nmo, nchol, seed=12):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((nmo * nmo, nchol))


class TestWriteDenseHcore:
    """`write_dense` must decide hcore's storage from its dtype, not its values."""

    def test_hermitian_cplx_hcore_keeps_imaginary_part(self, tmp_path):
        # A hermitian hcore has a real diagonal, so an elementwise
        # `all(iscomplex(...))` test reports it as real even though it is not.
        nmo, nchol = 8, 5
        hcore = _hermitian_cplx(nmo)
        assert np.iscomplexobj(hcore)
        assert not np.all(np.iscomplex(hcore)), "fixture must have a real diagonal"

        fname = tmp_path / 'hamiltonian.h5'
        write_dense(hcore, _real_chol(nmo, nchol), (3, 3), nmo,
                    enuc=0.0, filename=fname)

        with h5.File(fname, 'r') as fh5:
            stored = fh5['Hamiltonian/hcore'][()]

        # internal complex storage is float64 pairs on a trailing axis
        assert stored.shape == (nmo, nmo, 2)
        got = from_complex(stored, shape=(nmo, nmo))
        assert np.array_equal(got, hcore)

    def test_real_hcore_stays_real(self, tmp_path):
        nmo, nchol = 8, 5
        rng = np.random.default_rng(13)
        hcore = rng.standard_normal((nmo, nmo))

        fname = tmp_path / 'hamiltonian.h5'
        write_dense(hcore, _real_chol(nmo, nchol), (3, 3), nmo,
                    enuc=0.0, filename=fname)

        with h5.File(fname, 'r') as fh5:
            stored = fh5['Hamiltonian/hcore'][()]

        assert stored.shape == (nmo, nmo)
        assert np.array_equal(stored, hcore)

    def test_fully_complex_hcore_unchanged(self, tmp_path):
        # The pre-existing behaviour for an hcore whose every entry is complex.
        nmo, nchol = 8, 5
        rng = np.random.default_rng(14)
        hcore = rng.standard_normal((nmo, nmo)) + 1j * (rng.standard_normal((nmo, nmo)) ** 2 + 1.0)
        assert np.all(np.iscomplex(hcore))

        fname = tmp_path / 'hamiltonian.h5'
        write_dense(hcore, _real_chol(nmo, nchol), (3, 3), nmo,
                    enuc=0.0, filename=fname)

        with h5.File(fname, 'r') as fh5:
            stored = fh5['Hamiltonian/hcore'][()]

        assert np.array_equal(from_complex(stored, shape=(nmo, nmo)), hcore)

    @pytest.mark.parametrize("real_chol", [True, False])
    def test_hcore_branch_is_independent_of_cholesky(self, tmp_path, real_chol):
        # hcore's storage must not be driven by the Cholesky vectors' dtype.
        nmo, nchol = 8, 5
        hcore = _hermitian_cplx(nmo)
        chol = _real_chol(nmo, nchol)
        if not real_chol:
            chol = chol + 1j * _real_chol(nmo, nchol, seed=15)

        fname = tmp_path / 'hamiltonian.h5'
        write_dense(hcore, chol, (3, 3), nmo, enuc=0.0,
                    filename=fname, real_chol=real_chol)

        with h5.File(fname, 'r') as fh5:
            stored = fh5['Hamiltonian/hcore'][()]

        assert np.array_equal(from_complex(stored, shape=(nmo, nmo)), hcore)
