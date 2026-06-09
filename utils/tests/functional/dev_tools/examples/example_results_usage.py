#!/usr/bin/env python3
"""
Example usage of the Results class for AFQMC calculations.

This demonstrates the three ways to create a Results object:
1. From AFQMC run output files
2. From user-provided data (e.g., from literature)
3. From an existing results.h5 file
"""

import numpy as np
from results import Results


def example_from_afqmc_run():
    """Example: Create Results from actual AFQMC run output files."""
    print("Example 1: Creating Results from AFQMC run files")
    print("-" * 60)
    
    # This assumes you've run AFQMC and have output files in the current directory
    results = Results.from_afqmc_run(
        return_code=0,
        run_time_secs=123.45,
        nequil=5.0,
        run_bp=False,
        output_file="afqmc.out",
        input_file="afqmc.json",
        scalar_file="qmc.s000.scalar.dat"
    )
    
    # Save to HDF5
    results.to_hdf5("results.h5")
    print(f"Saved results: {results}")
    print()


def example_from_literature():
    """Example: Create Results from literature/paper data."""
    print("Example 2: Creating Results from literature data")
    print("-" * 60)
    
    # Suppose you're reproducing results from a paper that reports:
    # Energy: -1.134 +/- 0.003 Ha
    # Weight: 1.02 +/- 0.05
    
    results = Results.from_data(
        return_code=0,
        run_time_seconds=0.0,  # Not applicable for literature data
        energy=[-1.134, 0.003],
        weight=[1.02, 0.05],
        log_ovlp_factor=[0.1, 0.01],
        input_file="# Parameters from Smith et al. (2024)\n# System: H2 molecule"
    )
    
    # Save to HDF5 for later comparison
    results.to_hdf5("literature_results.h5")
    print(f"Created results from literature: {results}")
    print(f"  Energy: {results.energy[0]:.6f} +/- {results.energy[1]:.6f}")
    print(f"  Weight: {results.weight[0]:.6f} +/- {results.weight[1]:.6f}")
    print()


def example_minimal_data():
    """Example: Create Results with minimal data (just energy)."""
    print("Example 3: Creating Results with minimal data")
    print("-" * 60)
    
    # You can provide just energy if that's all you have
    results = Results.from_data(
        energy=-1.134  # Will automatically add zero uncertainty
    )
    
    print(f"Minimal results: {results}")
    print(f"  Energy: {results.energy}")
    print()


def example_from_hdf5():
    """Example: Load Results from existing HDF5 file."""
    print("Example 4: Loading Results from HDF5 file")
    print("-" * 60)
    
    # First create and save a Results object
    original = Results.from_data(
        energy=[-1.134, 0.003],
        weight=[1.02, 0.05]
    )
    original.to_hdf5("temp_results.h5")
    
    # Now load it back
    loaded = Results.from_hdf5("temp_results.h5")
    
    print(f"Loaded results: {loaded}")
    print(f"  Energy matches: {np.allclose(original.energy, loaded.energy)}")
    print(f"  Weight matches: {np.allclose(original.weight, loaded.weight)}")
    print()


def example_with_1rdm():
    """Example: Create Results with 1-RDM data."""
    print("Example 5: Creating Results with 1-RDM data")
    print("-" * 60)
    
    # Example for a 4-site system
    n_orbitals = 4
    rdm = np.random.rand(n_orbitals, n_orbitals)
    rdm_error = np.random.rand(n_orbitals, n_orbitals) * 0.01
    
    results = Results.from_data(
        energy=[-1.234, 0.005],
        avg_1rdm=rdm,
        avg_1rdm_stoch_error=rdm_error
    )
    
    results.to_hdf5("results_with_rdm.h5")
    print(f"Created results with 1-RDM: {results}")
    print(f"  RDM shape: {results.avg_1rdm.shape}")
    print()


def example_comparison():
    """Example: Compare your results with literature."""
    print("Example 6: Comparing results")
    print("-" * 60)
    
    # Your calculation
    my_results = Results.from_data(energy=[-1.136, 0.004])
    
    # Literature value
    lit_results = Results.from_data(energy=[-1.134, 0.003])
    
    # Compare
    diff = abs(my_results.energy[0] - lit_results.energy[0])
    combined_error = np.sqrt(my_results.energy[1]**2 + lit_results.energy[1]**2)
    
    print(f"My result:     {my_results.energy[0]:.6f} +/- {my_results.energy[1]:.6f}")
    print(f"Literature:    {lit_results.energy[0]:.6f} +/- {lit_results.energy[1]:.6f}")
    print(f"Difference:    {diff:.6f}")
    print(f"Combined σ:    {combined_error:.6f}")
    print(f"Agreement:     {diff/combined_error:.2f}σ")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Results Class Usage Examples")
    print("=" * 60)
    print()
    
    # Run examples that don't require actual AFQMC output files
    example_from_literature()
    example_minimal_data()
    example_from_hdf5()
    example_with_1rdm()
    example_comparison()
    
    print("=" * 60)
    print("Note: example_from_afqmc_run() requires actual AFQMC output files")
    print("=" * 60)
