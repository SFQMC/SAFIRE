# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Unit tests for `safiretools.wavefunction.slater`."""

import numpy as np
import pytest

from safiretools import SpinSymm
from safiretools.wavefunction.slater import (
    is_orthonormal,
    make_slater,
    modified_gram_schmidt,
    spin_blocks,
    transform_slater,
)


@pytest.fixture
def coeffs():
    """A 4-orbital coefficient matrix with distinguishable columns."""
    return np.arange(16.0).reshape(4, 4)


class TestMakeSlater:

    def test_closed_takes_one_block_of_alpha_columns(self, coeffs):
        slater = make_slater(SpinSymm.CLOSED, coeffs, ([0, 2], [0, 2]), (2, 2))

        assert np.array_equal(slater, coeffs[:, [0, 2]])
        assert slater.dtype == np.complex128

    def test_collinear_concatenates_the_two_channels(self, coeffs):
        # a 2-D mo_coeff is an ROHF reference: both channels share it
        slater = make_slater(SpinSymm.COLLINEAR, coeffs, ([0, 1], [0]), (2, 1))

        assert np.array_equal(slater, coeffs[:, [0, 1, 0]])

    def test_collinear_reads_one_matrix_per_spin_when_given_them(self, coeffs):
        # a 3-D mo_coeff is a UHF reference: one matrix per channel
        spin_coeffs = np.array([coeffs, -coeffs])
        slater = make_slater(SpinSymm.COLLINEAR, spin_coeffs, ([0, 1], [3]),
                             (2, 1))

        assert np.array_equal(slater[:, :2], coeffs[:, [0, 1]])
        assert np.array_equal(slater[:, 2:], -coeffs[:, [3]])

    def test_noncollinear_takes_every_electron_in_one_block(self):
        spinor_coeffs = np.arange(24.0).reshape(4, 6)
        slater = make_slater(SpinSymm.NONCOLLINEAR, spinor_coeffs,
                             ([0, 2, 5],), (2, 1))

        assert np.array_equal(slater, spinor_coeffs[:, [0, 2, 5]])

    def test_a_channel_count_mismatch_is_rejected(self, coeffs):
        with pytest.raises(ValueError, match="nocc describes 2 spin channels"):
            make_slater(SpinSymm.COLLINEAR, coeffs, ([0], [1]), (1,))


class TestTransformSlater:

    def test_it_applies_the_adjoint_transformation(self, rng):
        orbitals = rng.normal(size=(4, 2)) + 1j * rng.normal(size=(4, 2))
        transform = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))

        assert np.allclose(transform_slater(orbitals, transform),
                           transform.conj().T @ orbitals)

    def test_a_spatial_transform_is_promoted_for_spinor_orbitals(self, rng):
        orbitals = rng.normal(size=(8, 2)) + 1j * rng.normal(size=(8, 2))
        transform = rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3))

        promoted = np.kron(np.eye(2), transform)

        assert np.allclose(transform_slater(orbitals, transform),
                           promoted.conj().T @ orbitals)


class TestSpinBlocks:

    def test_it_splits_consecutive_column_blocks(self):
        orbitals = np.arange(6 * 5).reshape(6, 5)
        alpha, beta = spin_blocks(orbitals, (3, 2))

        assert np.array_equal(alpha, orbitals[:, :3])
        assert np.array_equal(beta, orbitals[:, 3:])

    def test_a_wrong_column_count_is_rejected(self):
        with pytest.raises(ValueError, match="expected 5 for electron counts"):
            list(spin_blocks(np.zeros((6, 4)), (3, 2)))


class TestOrthonormality:

    def test_gram_schmidt_orthonormalizes(self, rng):
        matrix = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        orthonormalized = modified_gram_schmidt(matrix)

        assert is_orthonormal(orthonormalized)
        # the column space is preserved
        assert np.linalg.matrix_rank(np.hstack([matrix, orthonormalized])) == 3

    def test_gram_schmidt_rejects_linearly_dependent_columns(self):
        matrix = np.ones((4, 2))

        with pytest.raises(ValueError, match="linearly dependent"):
            modified_gram_schmidt(matrix)

    def test_gram_schmidt_rejects_a_non_matrix(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            modified_gram_schmidt(np.zeros(4))

    def test_an_empty_block_counts_as_orthonormal(self):
        assert is_orthonormal(np.zeros((6, 0), dtype=complex))
