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
    format_spin_layout,
    is_orthonormal,
    make_slater,
    orthonormalize,
    parse_spin_layout,
    transform_slater,
)


@pytest.fixture
def coeffs():
    """A 4-orbital coefficient matrix with distinguishable columns."""
    return np.arange(16.0).reshape(4, 4)


class TestMakeSlater:

    def test_closed_takes_one_channel_of_alpha_columns(self, coeffs):
        (slater,) = make_slater(SpinSymm.CLOSED, coeffs, ([0, 2], [0, 2]),
                                (2, 2))

        assert np.array_equal(slater, coeffs[:, [0, 2]])
        assert slater.dtype == np.complex128

    def test_collinear_gives_one_matrix_per_channel(self, coeffs):
        # a 2-D mo_coeff is an ROHF reference: both channels share it
        alpha, beta = make_slater(SpinSymm.COLLINEAR, coeffs, ([0, 1], [0]),
                                  (2, 1))

        assert np.array_equal(alpha, coeffs[:, [0, 1]])
        assert np.array_equal(beta, coeffs[:, [0]])

    def test_collinear_reads_one_matrix_per_spin_when_given_them(self, coeffs):
        # a 3-D mo_coeff is a UHF reference: one matrix per channel
        spin_coeffs = np.array([coeffs, -coeffs])
        alpha, beta = make_slater(SpinSymm.COLLINEAR, spin_coeffs,
                                  ([0, 1], [3]), (2, 1))

        assert np.array_equal(alpha, coeffs[:, [0, 1]])
        assert np.array_equal(beta, -coeffs[:, [3]])

    def test_noncollinear_takes_every_electron_in_one_channel(self):
        spinor_coeffs = np.arange(24.0).reshape(4, 6)
        (slater,) = make_slater(SpinSymm.NONCOLLINEAR, spinor_coeffs,
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


class TestSpinLayout:

    @pytest.mark.parametrize('value, spin_symm, shapes', [
        (np.zeros((6, 3)), SpinSymm.CLOSED, [(6, 3)]),
        ((np.zeros((6, 3)), np.zeros((6, 2))), SpinSymm.COLLINEAR,
         [(6, 3), (6, 2)]),
        (np.zeros((2, 6, 5)), SpinSymm.NONCOLLINEAR, [(12, 5)]),
    ])
    def test_the_layout_decides_the_spin_symmetry(self, value, spin_symm,
                                                  shapes):
        found, blocks = parse_spin_layout(value)

        assert found is spin_symm
        assert [block.shape for block in blocks] == shapes
        assert all(block.dtype == np.complex128 for block in blocks)

    @pytest.mark.parametrize('value', [
        np.arange(18.0).reshape(6, 3),
        (np.arange(18.0).reshape(6, 3), np.arange(12.0).reshape(6, 2)),
        np.arange(60.0).reshape(2, 6, 5),
    ])
    def test_format_inverts_parse(self, value):
        spin_symm, blocks = parse_spin_layout(value)
        formatted = format_spin_layout(blocks, spin_symm)

        if isinstance(value, tuple):
            assert all(np.array_equal(a, b) for a, b in zip(formatted, value))
        else:
            assert np.array_equal(formatted, value)

    def test_noncollinear_puts_the_spin_up_rows_first(self):
        # the order transform_slater's kron(eye(2), X) promotion assumes
        value = np.array([np.ones((6, 5)), -np.ones((6, 5))])
        _, (block,) = parse_spin_layout(value)

        assert np.array_equal(block[:6], np.ones((6, 5)))
        assert np.array_equal(block[6:], -np.ones((6, 5)))

    def test_a_stacked_value_keeps_its_leading_axis(self):
        _, (block,) = parse_spin_layout(np.zeros((4, 2, 6, 5)), stacked=True)

        assert block.shape == (4, 12, 5)

    @pytest.mark.parametrize('value, got', [
        ([np.zeros((6, 3)), np.zeros((6, 2))], "got a list"),
        ((np.zeros((6, 3)),), "got a tuple of 1"),
        (np.zeros(6), r"got an array of shape \(6,\)"),
        (np.zeros((3, 6, 5)), r"got an array of shape \(3, 6, 5\)"),
    ])
    def test_anything_else_is_rejected(self, value, got):
        with pytest.raises(TypeError, match=got):
            parse_spin_layout(value)

    def test_collinear_channels_over_different_orbitals_are_rejected(self):
        with pytest.raises(ValueError, match="collinear channels of det"):
            parse_spin_layout((np.zeros((6, 3)), np.zeros((5, 2))), name='det')

    def test_a_spin_symmetry_mismatch_is_rejected(self):
        with pytest.raises(ValueError, match="psi0 must be in the collinear"):
            parse_spin_layout(np.zeros((6, 3)), 'collinear', name='psi0')


class TestOrthonormality:

    def test_orthonormalize_orthonormalizes(self, rng):
        matrix = rng.normal(size=(6, 3)) + 1j * rng.normal(size=(6, 3))
        orthonormalized = orthonormalize(matrix)

        assert is_orthonormal(orthonormalized)
        # the column space is preserved
        assert np.linalg.matrix_rank(np.hstack([matrix, orthonormalized])) == 3

    def test_an_orthonormal_matrix_is_returned_as_it_is(self):
        matrix = np.eye(6, 3)
        assert orthonormalize(matrix) is matrix

    def test_orthonormalize_rejects_linearly_dependent_columns(self):
        matrix = np.ones((4, 2))

        with pytest.raises(ValueError, match="linearly dependent"):
            orthonormalize(matrix)

    def test_orthonormalize_rejects_a_non_matrix(self):
        with pytest.raises(ValueError, match="2-dimensional"):
            orthonormalize(np.zeros(4))

    def test_an_empty_block_counts_as_orthonormal(self):
        assert is_orthonormal(np.zeros((6, 0), dtype=complex))
