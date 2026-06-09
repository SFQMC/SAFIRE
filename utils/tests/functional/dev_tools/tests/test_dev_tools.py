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
Unit tests for the Results class.

These tests verify the functionality of the Results class including:
- Construction from user data
- HDF5 I/O operations
- Handling of various data types
- Edge cases
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path

from dev_tools.results import Results


class TestResultsFromData:
    """Test suite for Results.from_data() constructor."""
    
    def test_from_literature_data(self):
        """Test creating Results from literature values."""
        results = Results.from_data(
            return_code=0,
            run_time_seconds=0.0,
            energy=[-1.134, 0.003],
            weight=[1.02, 0.05],
            log_ovlp_factor=[0.1, 0.01],
            input_file="# Test data"
        )
        
        assert results.return_code == 0
        assert results.run_time_seconds == 0.0
        assert np.allclose(results.energy, [-1.134, 0.003])
        assert np.allclose(results.weight, [1.02, 0.05])
        assert np.allclose(results.log_ovlp_factor, [0.1, 0.01])
        assert results.input_file == "# Test data"
    
    def test_minimal_data_energy_only(self):
        """Test creating Results with only energy (single value)."""
        results = Results.from_data(energy=-1.134)
        
        assert results.energy is not None
        assert len(results.energy) == 2
        assert results.energy[0] == -1.134
        assert results.energy[1] == 0.0  # Auto-added zero uncertainty
    
    def test_minimal_data_energy_with_uncertainty(self):
        """Test creating Results with energy and uncertainty."""
        results = Results.from_data(energy=[-1.134, 0.003])
        
        assert np.allclose(results.energy, [-1.134, 0.003])
    
    def test_weight_data(self):
        """Test creating Results with weight data."""
        results = Results.from_data(weight=[1.05, 0.02])
        
        assert np.allclose(results.weight, [1.05, 0.02])
        assert results.energy is None  # Other fields should be None
    
    def test_log_ovlp_factor_data(self):
        """Test creating Results with log overlap factor."""
        results = Results.from_data(log_ovlp_factor=[0.15, 0.005])
        
        assert np.allclose(results.log_ovlp_factor, [0.15, 0.005])
    
    def test_single_value_arrays_auto_uncertainty(self):
        """Test that single values automatically get zero uncertainty."""
        results = Results.from_data(
            energy=[-1.5],
            weight=[1.0],
            log_ovlp_factor=[0.2]
        )
        
        assert results.energy[1] == 0.0
        assert results.weight[1] == 0.0
        assert results.log_ovlp_factor[1] == 0.0
    
    def test_with_1rdm_data(self):
        """Test creating Results with 1-RDM data."""
        n_orbitals = 4
        rdm = np.random.rand(n_orbitals, n_orbitals)
        rdm_error = np.random.rand(n_orbitals, n_orbitals) * 0.01
        
        results = Results.from_data(
            energy=[-1.234, 0.005],
            avg_1rdm=rdm,
            avg_1rdm_stoch_error=rdm_error
        )
        
        assert results.avg_1rdm.shape == (4, 4)
        assert results.avg_1rdm_stoch_error.shape == (4, 4)
        assert np.allclose(results.avg_1rdm, rdm)
        assert np.allclose(results.avg_1rdm_stoch_error, rdm_error)
    
    def test_kwargs_handling(self):
        """Test that additional kwargs are properly set."""
        results = Results.from_data(
            energy=[-1.0, 0.01],
            afqmc_raised_warning=True,
            afqmc_raised_error=False,
            afqmc_is_finite=True
        )
        
        assert results.afqmc_raised_warning is True
        assert results.afqmc_raised_error is False
        assert results.afqmc_is_finite is True
    
    def test_default_values(self):
        """Test that defaults are set correctly."""
        results = Results.from_data(energy=-1.0)
        
        assert results.return_code == 0
        assert results.run_time_seconds == -1.0
        assert results.afqmc_raised_warning is False
        assert results.afqmc_raised_error is False
        assert results.afqmc_is_finite is True


class TestResultsHDF5IO:
    """Test suite for HDF5 save/load operations."""
    
    def test_save_and_load_basic(self):
        """Test basic save and load functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_results.h5"
            
            # Create and save
            original = Results.from_data(
                energy=[-1.134, 0.003],
                weight=[1.02, 0.05]
            )
            original.to_hdf5(filepath)
            
            # Load
            loaded = Results.from_hdf5(filepath)
            
            # Compare
            assert np.allclose(original.energy, loaded.energy)
            assert np.allclose(original.weight, loaded.weight)
            assert original.return_code == loaded.return_code
    
    def test_save_and_load_full_data(self):
        """Test save/load with all data fields populated."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_full_results.h5"
            
            rdm = np.random.rand(3, 3)
            rdm_error = np.random.rand(3, 3) * 0.01
            
            original = Results.from_data(
                return_code=0,
                run_time_seconds=123.45,
                energy=[-1.234, 0.005],
                weight=[1.01, 0.03],
                log_ovlp_factor=[0.12, 0.008],
                input_file="Test input file",
                avg_1rdm=rdm,
                avg_1rdm_stoch_error=rdm_error,
                afqmc_raised_warning=True,
                afqmc_raised_error=False,
                afqmc_is_finite=True
            )
            
            # Add warning and error messages
            original.warning_messages = {"Warning 1", "Warning 2"}
            original.error_messages = {"Error 1"}
            
            original.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert loaded.return_code == 0
            assert loaded.run_time_seconds == 123.45
            assert np.allclose(loaded.energy, original.energy)
            assert np.allclose(loaded.weight, original.weight)
            assert np.allclose(loaded.log_ovlp_factor, original.log_ovlp_factor)
            assert loaded.input_file == "Test input file"
            assert np.allclose(loaded.avg_1rdm, rdm)
            assert np.allclose(loaded.avg_1rdm_stoch_error, rdm_error)
            assert loaded.afqmc_raised_warning is True
            assert loaded.afqmc_raised_error is False
            assert loaded.afqmc_is_finite is True
            assert loaded.warning_messages == {"Warning 1", "Warning 2"}
            assert loaded.error_messages == {"Error 1"}
    
    def test_save_and_load_minimal_data(self):
        """Test save/load with minimal data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_minimal.h5"
            
            original = Results.from_data(energy=-1.5)
            original.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert np.allclose(loaded.energy, [-1.5, 0.0])
            assert loaded.weight is None
            assert loaded.log_ovlp_factor is None
    
    def test_load_nonexistent_file(self):
        """Test that loading a nonexistent file raises an error."""
        with pytest.raises(FileNotFoundError):
            Results.from_hdf5("nonexistent_file.h5")
    
    def test_default_filename(self):
        """Test that default filename 'results.h5' works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_dir = os.getcwd()
            try:
                os.chdir(tmpdir)
                
                original = Results.from_data(energy=[-1.0, 0.01])
                original.to_hdf5()  # Should use default "results.h5"
                
                assert Path("results.h5").exists()
                
                loaded = Results.from_hdf5()  # Should load default "results.h5"
                assert np.allclose(loaded.energy, [-1.0, 0.01])
            finally:
                os.chdir(original_dir)


class TestResultsComparison:
    """Test suite for comparing results."""
    
    def test_simple_comparison(self):
        """Test comparing two Results objects."""
        my_results = Results.from_data(energy=[-1.136, 0.004])
        lit_results = Results.from_data(energy=[-1.134, 0.003])
        
        diff = abs(my_results.energy[0] - lit_results.energy[0])
        combined_error = np.sqrt(my_results.energy[1]**2 + lit_results.energy[1]**2)
        
        assert np.isclose(diff, 0.002)
        assert np.isclose(combined_error, 0.005)
        assert np.isclose(diff / combined_error, 0.4)
    
    def test_exact_match(self):
        """Test that identical results compare correctly."""
        results1 = Results.from_data(energy=[-1.0, 0.01], weight=[1.0, 0.02])
        results2 = Results.from_data(energy=[-1.0, 0.01], weight=[1.0, 0.02])
        
        assert np.allclose(results1.energy, results2.energy)
        assert np.allclose(results1.weight, results2.weight)


class TestResultsRepr:
    """Test suite for Results string representation."""
    
    def test_repr_basic(self):
        """Test __repr__ with basic data."""
        results = Results.from_data(energy=[-1.134, 0.003])
        repr_str = repr(results)
        
        assert "Results(return_code=0" in repr_str
        assert "energy=-1.134000" in repr_str
        assert "0.003000" in repr_str
    
    def test_repr_with_warnings(self):
        """Test __repr__ includes warning count."""
        results = Results.from_data(
            energy=[-1.0, 0.01],
            afqmc_raised_warning=True
        )
        results.warning_messages = {"Warning 1", "Warning 2", "Warning 3"}
        
        repr_str = repr(results)
        assert "warnings=3" in repr_str
    
    def test_repr_with_errors(self):
        """Test __repr__ includes error count."""
        results = Results.from_data(
            energy=[-1.0, 0.01],
            afqmc_raised_error=True
        )
        results.error_messages = {"Error 1"}
        
        repr_str = repr(results)
        assert "errors=1" in repr_str


class TestResultsEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_empty_results(self):
        """Test creating completely empty Results object."""
        results = Results()
        
        assert results.return_code is None
        assert results.energy is None
        assert results.weight is None
        assert results.log_ovlp_factor is None
    
    def test_numpy_array_input(self):
        """Test that numpy arrays are handled correctly."""
        energy_array = np.array([-1.5, 0.01])
        results = Results.from_data(energy=energy_array)
        
        assert np.allclose(results.energy, energy_array)
    
    def test_list_input(self):
        """Test that lists are handled correctly."""
        results = Results.from_data(energy=[-1.5, 0.01])
        
        assert isinstance(results.energy, np.ndarray)
        assert np.allclose(results.energy, [-1.5, 0.01])
    
    def test_tuple_input(self):
        """Test that tuples are handled correctly."""
        results = Results.from_data(energy=(-1.5, 0.01))
        
        assert isinstance(results.energy, np.ndarray)
        assert np.allclose(results.energy, [-1.5, 0.01])
    
    def test_negative_return_code(self):
        """Test handling of non-zero return codes."""
        results = Results.from_data(return_code=-1)
        
        assert results.return_code == -1
    
    def test_large_1rdm(self):
        """Test handling of large 1-RDM arrays."""
        n_orbitals = 100
        rdm = np.random.rand(n_orbitals, n_orbitals)
        
        results = Results.from_data(
            energy=[-10.5, 0.1],
            avg_1rdm=rdm
        )
        
        assert results.avg_1rdm.shape == (100, 100)
        
        # Test save/load with large array
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "large_rdm.h5"
            results.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert np.allclose(loaded.avg_1rdm, rdm)


class TestResultsWarningsAndErrors:
    """Test suite for warning and error message handling."""
    
    def test_warning_messages(self):
        """Test handling of warning messages."""
        results = Results.from_data(energy=[-1.0, 0.01])
        results.warning_messages = {"Warning 1", "Warning 2"}
        results.afqmc_raised_warning = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_warnings.h5"
            results.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert loaded.warning_messages == {"Warning 1", "Warning 2"}
            assert loaded.afqmc_raised_warning is True
    
    def test_error_messages(self):
        """Test handling of error messages."""
        results = Results.from_data(energy=[-1.0, 0.01])
        results.error_messages = {"Error A", "Error B", "Error C"}
        results.afqmc_raised_error = True
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_errors.h5"
            results.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert loaded.error_messages == {"Error A", "Error B", "Error C"}
            assert loaded.afqmc_raised_error is True
    
    def test_empty_messages(self):
        """Test that empty message sets are handled correctly."""
        results = Results.from_data(energy=[-1.0, 0.01])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_no_messages.h5"
            results.to_hdf5(filepath)
            loaded = Results.from_hdf5(filepath)
            
            assert loaded.warning_messages == set()
            assert loaded.error_messages == set()
    
    def test_groups_always_exist(self):
        """Test that warning_messages and error_messages groups always exist in HDF5, even when empty."""
        results = Results.from_data(energy=[-1.0, 0.01])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_groups_exist.h5"
            results.to_hdf5(filepath)
            
            # Verify groups exist in the HDF5 file
            import h5py as h5
            with h5.File(filepath, 'r') as f:
                assert "warning_messages" in f
                assert "error_messages" in f
                assert f["warning_messages"]["num_messages"][()] == 0
                assert f["error_messages"]["num_messages"][()] == 0
    
    def test_overwrite_existing_file(self):
        """Test that to_hdf5 properly overwrites existing files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_overwrite.h5"
            
            # Create initial results
            results1 = Results.from_data(energy=[-1.0, 0.01], weight=[1.0, 0.02])
            results1.to_hdf5(filepath)
            
            # Verify initial data
            loaded1 = Results.from_hdf5(filepath)
            assert np.allclose(loaded1.energy, [-1.0, 0.01])
            assert np.allclose(loaded1.weight, [1.0, 0.02])
            
            # Overwrite with different data
            results2 = Results.from_data(energy=[-2.0, 0.03], weight=[2.0, 0.04])
            results2.to_hdf5(filepath)
            
            # Verify overwritten data
            loaded2 = Results.from_hdf5(filepath)
            assert np.allclose(loaded2.energy, [-2.0, 0.03])
            assert np.allclose(loaded2.weight, [2.0, 0.04])
            
            # Ensure old data is completely gone
            assert not np.allclose(loaded2.energy, [-1.0, 0.01])
            assert not np.allclose(loaded2.weight, [1.0, 0.02])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
