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
Tests for the utils/afqmctools/analysis/transform.py module.

Tests cover all transform factory functions including:
- check_integer validation
- hermitize_factory for all walker types
- upper_triangle_factory for all walker types
- eval_hubbard_from_diag_two_rdm_factory
- eval_one_body_obs_factory
- eval_two_body_obs_factory
- Integration tests combining transforms
- Edge cases
"""

import numpy as np
import pytest

from afqmctools.analysis.transform import (
    check_integer,
    hermitize_factory,
    upper_triangle_factory,
    eval_hubbard_from_diag_two_rdm_factory,
    eval_one_body_obs_factory,
    eval_two_body_obs_factory,
)


class TestCheckInteger:
    """Tests for the check_integer validation function."""

    def test_valid_integers(self):
        """Test that valid integer values are correctly identified."""
        assert check_integer(3.0) == 3
        assert check_integer(5.0) == 5
        assert check_integer(100.0) == 100
        assert check_integer(0.0) == 0
        assert check_integer(3) == 3

    def test_invalid_non_integers(self):
        """Test that non-integer values raise ValueError."""
        with pytest.raises(ValueError, match="is not an integer"):
            check_integer(3.5)
        
        with pytest.raises(ValueError, match="is not an integer"):
            check_integer(2.1)

    def test_with_name_parameter(self):
        """Test that the name parameter appears in error messages."""
        with pytest.raises(ValueError, match="nbasis value.*is not an integer"):
            check_integer(3.7, name="nbasis")


class TestHermitizeFactory:
    """Tests for hermitize_factory with different walker types."""

    def test_closed_hermitize(self):
        """Test hermitize for closed shell walkers."""
        M = 3
        M_squared = M * M
        nsample = 2
        
        # Create non-hermitian data
        data = np.random.randn(nsample, M_squared) + 1j * np.random.randn(nsample, M_squared)
        
        hermitize = hermitize_factory('closed')
        result = hermitize(data)
        
        # Check shape
        assert result.shape == (nsample, M_squared)
        
        # Check hermiticity for each sample
        for k in range(nsample):
            mat = result[k].reshape(M, M, order='F')
            assert np.allclose(mat, mat.conj().T)

    def test_collinear_hermitize(self):
        """Test hermitize for collinear walkers."""
        M = 3
        M_squared = M * M
        nspin = 2
        nsample = 2
        
        # Create non-hermitian data
        data = np.random.randn(nsample, nspin * M_squared) + 1j * np.random.randn(nsample, nspin * M_squared)
        
        hermitize = hermitize_factory('collinear')
        result = hermitize(data)
        
        # Check shape
        assert result.shape == (nsample, nspin, M_squared)
        
        # Check hermiticity for each spin channel and sample
        for s in range(nspin):
            for k in range(nsample):
                mat = result[k, s].reshape(M, M, order='F')
                assert np.allclose(mat, mat.conj().T)

    def test_non_collinear_hermitize(self):
        """Test hermitize for non-collinear walkers."""
        M = 2
        M_spinor = 2 * M
        M_spinor_squared = M_spinor * M_spinor
        nsample = 2
        
        # Create non-hermitian spinor data
        data = np.random.randn(nsample, M_spinor_squared) + 1j * np.random.randn(nsample, M_spinor_squared)
        
        hermitize = hermitize_factory('non_collinear')
        result = hermitize(data)
        
        # Check shape
        assert result.shape == (nsample, M_spinor_squared)
        
        # Check hermiticity for each sample
        for k in range(nsample):
            mat = result[k].reshape(M_spinor, M_spinor, order='F')
            assert np.allclose(mat, mat.conj().T)

    def test_edge_case_single_basis(self):
        """Test hermitize with M=1 (minimum size)."""
        nsample = 2
        
        # Closed: M=1
        data_closed = np.random.randn(nsample, 1) + 1j * np.random.randn(nsample, 1)
        hermitize_closed = hermitize_factory('closed')
        result_closed = hermitize_closed(data_closed)
        assert result_closed.shape == (nsample, 1)
        # Single element is always hermitian and real
        assert np.allclose(result_closed, result_closed.conj())
        assert np.allclose(result_closed.imag, 0.0)

    def test_invalid_walker_type(self):
        """Test that invalid walker type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown walker type"):
            hermitize_factory('invalid_type')


class TestUpperTriangleFactory:
    """Tests for upper_triangle_factory with different walker types."""

    def test_closed_upper_triangle(self):
        """Test upper_triangle for closed shell walkers."""
        nbasis = 4
        nsample = 3
        dm_size = nbasis * (nbasis - 1) // 2
        
        # Create packed upper triangle data
        data = np.arange(nsample * dm_size).reshape(nsample, dm_size).astype(np.complex128)
        
        upper_triangle = upper_triangle_factory('closed')
        result = upper_triangle(data)
        
        # Check shape
        assert result.shape == (nsample, nbasis, nbasis)
        
        # Check symmetry
        for k in range(nsample):
            assert np.allclose(result[k], result[k].conj().T)
        
        # Check element mapping for first sample
        ij = 0
        for i in range(nbasis):
            for j in range(i + 1, nbasis):
                assert result[0, i, j] == data[0, ij]
                assert result[0, j, i] == data[0, ij].conj()
                ij += 1

    def test_collinear_upper_triangle(self):
        """Test upper_triangle for collinear walkers."""
        nbasis = 2  # spatial orbitals
        spin_nbasis = 2 * nbasis  # spin orbitals
        nsample = 3
        dm_size = spin_nbasis * (spin_nbasis - 1) // 2
        
        # Create packed upper triangle data
        data = np.arange(nsample * dm_size).reshape(nsample, dm_size).astype(np.complex128)
        
        upper_triangle = upper_triangle_factory('collinear')
        result = upper_triangle(data)
        
        # Check shape
        assert result.shape == (nsample, spin_nbasis, spin_nbasis)
        
        # Check symmetry
        for k in range(nsample):
            assert np.allclose(result[k], result[k].conj().T)

    def test_non_collinear_upper_triangle(self):
        """Test upper_triangle for non-collinear walkers."""
        nbasis = 2  # spatial orbitals
        spin_nbasis = 2 * nbasis  # spin orbitals
        nsample = 3
        dm_size = spin_nbasis * (spin_nbasis - 1) // 2
        
        # Create packed upper triangle data
        data = np.arange(nsample * dm_size).reshape(nsample, dm_size).astype(np.complex128)
        
        upper_triangle = upper_triangle_factory('non_collinear')
        result = upper_triangle(data)
        
        # Check shape
        assert result.shape == (nsample, spin_nbasis, spin_nbasis)
        
        # Check symmetry
        for k in range(nsample):
            assert np.allclose(result[k], result[k].conj().T)

    def test_edge_case_nbasis_2(self):
        """Test upper_triangle with nbasis=2 (minimal non-trivial case)."""
        nbasis = 2
        nsample = 2
        dm_size = nbasis * (nbasis - 1) // 2  # = 1
        
        data = np.array([[1.0 + 2.0j], [3.0 + 4.0j]])
        
        upper_triangle = upper_triangle_factory('closed')
        result = upper_triangle(data)
        
        assert result.shape == (nsample, nbasis, nbasis)
        assert result[0, 0, 1] == 1.0 + 2.0j
        assert result[0, 1, 0] == 1.0 - 2.0j
        assert result[1, 0, 1] == 3.0 + 4.0j
        assert result[1, 1, 0] == 3.0 - 4.0j

    def test_edge_case_nbasis_1(self):
        """Test upper_triangle with nbasis=1 (no off-diagonals)."""
        nbasis = 1
        nsample = 2
        dm_size = nbasis * (nbasis - 1) // 2  # = 0
        
        data = np.zeros((nsample, 0), dtype=np.complex128)
        
        upper_triangle = upper_triangle_factory('closed')
        result = upper_triangle(data)
        
        assert result.shape == (nsample, nbasis, nbasis)
        # All elements should be zero (only diagonal, which wasn't set)
        assert np.allclose(result, 0.0)

    def test_unsupported_walker_type(self):
        """Test that unsupported walker types raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            upper_triangle_factory('unknown_type')


class TestEvalHubbardFromDiagTwoRdmFactory:
    """Tests for eval_hubbard_from_diag_two_rdm_factory."""

    def test_collinear_hubbard(self):
        """Test Hubbard U energy calculation for collinear walkers."""
        U = 4.0
        nbasis = 3
        nsample = 5
        
        # Create diagonal 2-RDM: shape (nsample, 2*nbasis, 2*nbasis)
        diag2rdm = np.zeros((nsample, 2 * nbasis, 2 * nbasis), dtype=np.complex128)
        
        # Set up-down sector diagonal: diag2rdm[:, nbasis:, :nbasis]
        # For this test, set diagonal elements to known values
        diag_values = np.array([0.5, 0.3, 0.2])
        for k in range(nsample):
            for i in range(nbasis):
                diag2rdm[k, nbasis + i, i] = diag_values[i]
        
        transform = eval_hubbard_from_diag_two_rdm_factory(U, 'collinear')
        result = transform(diag2rdm)
        
        expected = U * np.sum(diag_values)
        assert result.shape == (nsample, 1)
        assert np.allclose(result, expected)

    def test_non_collinear_hubbard(self):
        """Test Hubbard U energy calculation for non-collinear walkers."""
        U = 2.5
        nbasis = 2
        nsample = 3
        
        # Create diagonal 2-RDM
        diag2rdm = np.zeros((nsample, 2 * nbasis, 2 * nbasis), dtype=np.complex128)
        
        diag_values = np.array([0.6, 0.4])
        for k in range(nsample):
            for i in range(nbasis):
                diag2rdm[k, nbasis + i, i] = diag_values[i]
        
        transform = eval_hubbard_from_diag_two_rdm_factory(U, 'non_collinear')
        result = transform(diag2rdm)
        
        expected = U * np.sum(diag_values)
        assert result.shape == (nsample, 1)
        assert np.allclose(result, expected)

    def test_closed_not_implemented(self):
        """Test that closed shell raises NotImplementedError."""
        transform = eval_hubbard_from_diag_two_rdm_factory(4.0, 'closed')
        dummy_data = np.zeros((2, 4, 4))
        
        with pytest.raises(NotImplementedError, match="Hubbard U energy.*not implemented for closed shell"):
            transform(dummy_data)

    def test_invalid_walker_type(self):
        """Test that invalid walker type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown walker type"):
            eval_hubbard_from_diag_two_rdm_factory(4.0, 'invalid')

    def test_edge_case_single_site(self):
        """Test Hubbard calculation with nbasis=1 (single site)."""
        U = 3.0
        nbasis = 1
        nsample = 2
        
        diag2rdm = np.zeros((nsample, 2 * nbasis, 2 * nbasis), dtype=np.complex128)
        diag2rdm[:, 1, 0] = 0.8
        
        transform = eval_hubbard_from_diag_two_rdm_factory(U, 'collinear')
        result = transform(diag2rdm)
        
        expected = U * 0.8
        assert np.allclose(result, expected)


class TestEvalOneBodyObsFactory:
    """Tests for eval_one_body_obs_factory."""

    def test_closed_one_body_obs(self):
        """Test one-body observable for closed shell walkers."""
        M = 3
        M_squared = M * M
        nsample = 4
        
        # Identity operator
        operator = np.eye(M, dtype=np.complex128)
        
        # Create density matrix with known trace
        density = np.zeros((nsample, M_squared), dtype=np.complex128)
        trace_value = 2.5
        for i in range(M):
            density[:, i * M + i] = trace_value / M
        
        transform = eval_one_body_obs_factory(operator, 'closed')
        result = transform(density)
        
        # Closed shell has factor of 2
        expected = 2 * trace_value
        assert result.shape == (nsample, 1)
        assert np.allclose(result, expected)

    def test_collinear_one_body_obs(self):
        """Test one-body observable for collinear walkers."""
        M = 3
        M_squared = M * M
        nsample = 4
        
        # Identity operator
        operator = np.eye(M, dtype=np.complex128)
        
        # Create density matrix with known traces per spin
        density = np.zeros((nsample, 2, M_squared), dtype=np.complex128)
        trace_up = 1.5
        trace_down = 1.0
        for i in range(M):
            density[:, 0, i * M + i] = trace_up / M
            density[:, 1, i * M + i] = trace_down / M
        
        transform = eval_one_body_obs_factory(operator, 'collinear')
        result = transform(density)
        
        expected = trace_up + trace_down
        assert result.shape == (nsample, 1)
        assert np.allclose(result, expected)

    def test_noncollinear_one_body_obs(self):
        """Test one-body observable for non-collinear walkers."""
        M = 2
        M_spinor = 2 * M
        M_spinor_squared = M_spinor * M_spinor
        nsample = 4
        
        # Identity operator (spinor size)
        operator = np.eye(M_spinor, dtype=np.complex128)
        
        # Create density matrix with known trace
        density = np.zeros((nsample, M_spinor_squared), dtype=np.complex128)
        trace_value = 3.0
        for i in range(M_spinor):
            density[:, i * M_spinor + i] = trace_value / M_spinor
        
        transform = eval_one_body_obs_factory(operator, 'noncollinear')
        result = transform(density)
        
        expected = trace_value
        assert result.shape == (nsample, 1)
        assert np.allclose(result, expected)

    def test_invalid_walker_type(self):
        """Test that invalid walker type raises ValueError."""
        operator = np.eye(2, dtype=np.complex128)
        with pytest.raises(ValueError, match="Unknown walker type"):
            eval_one_body_obs_factory(operator, 'invalid')

    def test_edge_case_single_orbital(self):
        """Test one-body observable with 1×1 operator."""
        operator = np.array([[2.0]], dtype=np.complex128)
        nsample = 3
        
        # Closed shell
        density = np.array([[0.5], [0.6], [0.7]], dtype=np.complex128)
        transform = eval_one_body_obs_factory(operator, 'closed')
        result = transform(density)
        
        expected = 2 * np.array([[0.5 * 2.0], [0.6 * 2.0], [0.7 * 2.0]])
        assert np.allclose(result, expected)


class TestEvalTwoBodyObsFactory:
    """Tests for eval_two_body_obs_factory."""

    def test_collinear_two_body_obs(self):
        """Test two-body observable for collinear walkers."""
        nbasis = 2
        nbasis_4 = nbasis ** 4
        num_spin_sectors = 3
        nsample = 3
        
        # Create simple two-body operator
        two_body_op = np.random.randn(nbasis_4).astype(np.complex128)
        
        # Create density data: (nsample, 3*nbasis^4) for 3 spin sectors
        density = np.random.randn(nsample, num_spin_sectors * nbasis_4).astype(np.complex128)
        
        transform = eval_two_body_obs_factory(two_body_operator=two_body_op, walker_type='collinear')
        result = transform(density)
        
        # Just check shape and that it runs without error
        assert result.shape == (nsample, 1)
        assert result.dtype == np.complex128

    def test_closed_not_implemented(self):
        """Test that closed shell raises NotImplementedError."""
        operator = np.random.randn(16).astype(np.complex128)
        with pytest.raises(NotImplementedError, match="Closed walker type not implemented"):
            eval_two_body_obs_factory(two_body_operator=operator, walker_type='closed')

    def test_noncollinear_not_implemented(self):
        """Test that non-collinear raises NotImplementedError."""
        operator = np.random.randn(16).astype(np.complex128)
        with pytest.raises(NotImplementedError, match="Non-collinear walker type not implemented"):
            eval_two_body_obs_factory(two_body_operator=operator, walker_type='noncollinear')

    def test_missing_operator_raises_error(self):
        """Test that missing two-body operator raises ValueError."""
        with pytest.raises(ValueError, match="Two-body integrals not provided"):
            eval_two_body_obs_factory(walker_type='collinear')

    def test_edge_case_nbasis_1(self):
        """Test two-body observable with nbasis=1."""
        nbasis = 1
        nbasis_4 = 1
        two_body_op = np.array([2.0], dtype=np.complex128)
        
        nsample = 2
        density = np.array([[0.5, 0.3, 0.2], [0.6, 0.4, 0.1]], dtype=np.complex128)
        
        transform = eval_two_body_obs_factory(two_body_operator=two_body_op, walker_type='collinear')
        result = transform(density)
        
        assert result.shape == (nsample, 1)


class TestIntegration:
    """Integration tests combining multiple transforms."""

    def test_hermitize_then_one_body_obs_closed(self):
        """Test hermitize followed by one-body observable for closed shell."""
        M = 3
        M_squared = M * M
        nsample = 2
        
        # Create non-hermitian density
        density_raw = np.random.randn(nsample, M_squared) + 1j * np.random.randn(nsample, M_squared)
        
        # Hermitize
        hermitize = hermitize_factory('closed')
        density_herm = hermitize(density_raw)
        
        # Compute observable
        operator = np.eye(M, dtype=np.complex128)
        transform = eval_one_body_obs_factory(operator, 'closed')
        result = transform(density_herm)
        
        # Verify result is real (trace of hermitian matrix with real operator)
        assert np.allclose(result.imag, 0.0)
        assert result.shape == (nsample, 1)

    def test_hermitize_then_one_body_obs_collinear(self):
        """Test hermitize followed by one-body observable for collinear."""
        M = 3
        M_squared = M * M
        nsample = 2
        
        # Create non-hermitian density
        density_raw = np.random.randn(nsample, 2 * M_squared) + 1j * np.random.randn(nsample, 2 * M_squared)
        
        # Hermitize
        hermitize = hermitize_factory('collinear')
        density_herm = hermitize(density_raw)
        
        # Compute observable
        operator = np.eye(M, dtype=np.complex128)
        transform = eval_one_body_obs_factory(operator, 'collinear')
        result = transform(density_herm)
        
        # Verify result is real
        assert np.allclose(result.imag, 0.0)
        assert result.shape == (nsample, 1)

    def test_upper_triangle_then_hubbard_collinear(self):
        """Test upper_triangle expansion followed by Hubbard U calculation."""
        nbasis = 3
        spin_nbasis = 2 * nbasis
        nsample = 4
        U = 5.0
        
        # Create packed upper triangle
        dm_size = spin_nbasis * (spin_nbasis - 1) // 2
        packed_data = np.random.randn(nsample, dm_size).astype(np.complex128)
        
        # Expand to full matrix
        upper_triangle = upper_triangle_factory('collinear')
        full_2rdm = upper_triangle(packed_data)
        
        # Compute Hubbard energy
        transform = eval_hubbard_from_diag_two_rdm_factory(U, 'collinear')
        result = transform(full_2rdm)
        
        assert result.shape == (nsample, 1)
        # Result should be real for real packed data
        assert result.dtype == np.complex128

    def test_upper_triangle_then_hubbard_non_collinear(self):
        """Test upper_triangle expansion followed by Hubbard U for non-collinear."""
        nbasis = 2
        spin_nbasis = 2 * nbasis
        nsample = 3
        U = 3.0
        
        # Create packed upper triangle
        dm_size = spin_nbasis * (spin_nbasis - 1) // 2
        packed_data = np.random.randn(nsample, dm_size).astype(np.complex128)
        
        # Expand to full matrix
        upper_triangle = upper_triangle_factory('non_collinear')
        full_2rdm = upper_triangle(packed_data)
        
        # Compute Hubbard energy
        transform = eval_hubbard_from_diag_two_rdm_factory(U, 'non_collinear')
        result = transform(full_2rdm)
        
        assert result.shape == (nsample, 1)


class TestEdgeCases:
    """Edge case tests for boundary conditions."""

    def test_zero_samples(self):
        """Test transforms with zero samples."""
        nsample = 0
        
        # Hermitize closed
        M = 3
        data = np.zeros((nsample, M * M), dtype=np.complex128)
        hermitize = hermitize_factory('closed')
        result = hermitize(data)
        assert result.shape == (nsample, M * M)
        
        # Upper triangle closed
        nbasis = 4
        dm_size = nbasis * (nbasis - 1) // 2
        data = np.zeros((nsample, dm_size), dtype=np.complex128)
        upper_triangle = upper_triangle_factory('closed')
        result = upper_triangle(data)
        assert result.shape == (nsample, nbasis, nbasis)

    def test_minimum_sizes(self):
        """Test transforms with minimum possible sizes."""
        nsample = 1
        
        # M=1 hermitize
        data = np.array([[1.0 + 0.5j]], dtype=np.complex128)
        hermitize = hermitize_factory('closed')
        result = hermitize(data)
        assert result.shape == (1, 1)
        # Single element: hermitize averages with conjugate, so imaginary part cancels
        assert np.allclose(result, [[1.0 + 0.0j]])
        
        # nbasis=1 upper triangle (no off-diagonals)
        data = np.zeros((nsample, 0), dtype=np.complex128)
        upper_triangle = upper_triangle_factory('closed')
        result = upper_triangle(data)
        assert result.shape == (nsample, 1, 1)

    def test_real_vs_complex_data(self):
        """Test that transforms work with both real and complex data."""
        nsample = 2
        M = 3
        
        # Real data
        data_real = np.random.randn(nsample, M * M)
        hermitize = hermitize_factory('closed')
        result_real = hermitize(data_real)
        assert result_real.shape == (nsample, M * M)
        
        # Complex data
        data_complex = np.random.randn(nsample, M * M) + 1j * np.random.randn(nsample, M * M)
        result_complex = hermitize(data_complex)
        assert result_complex.shape == (nsample, M * M)
