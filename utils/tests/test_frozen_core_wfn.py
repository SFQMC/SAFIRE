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
Test for frozen core wavefunction writing.
This test reproduces the issue from GitHub issue #100.
"""
import os
import numpy as np
import h5py as h5
import pytest

pytest.importorskip("pyscf")

from pyscf import gto, scf
from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn_mol


@pytest.mark.pyscf
class TestFrozenCoreWavefunction:
    """Test that frozen core wavefunctions can be written correctly."""

    def test_frozen_core_dimension_preservation(self, tmp_path, neon_atom, neon_rhf):
        """
        Test that write_hamil_mol with frozen core does not modify the input scf_data,
        and that write_wfn_mol can be called successfully afterwards.
        
        This test reproduces the issue where write_hamil_mol was modifying
        basis_scf_data['mo_coeff'] in place, causing a dimension mismatch
        when write_wfn_mol was called later.
        """
        mf, energy = neon_rhf
        atom = neon_atom
        
        # Create scf_data dictionary similar to what load_from_pyscf_chk_mol returns
        scf_data = {
            'mol': atom,
            'nelec': atom.nelec,
            'mo_occ': mf.mo_occ,
            'hcore': mf.get_hcore(),
            'norb': mf.mo_coeff.shape[-1],
            'X': mf.mo_coeff,
            'mo_coeff': mf.mo_coeff.copy(),  # Use a copy to ensure we test modification
            'isUHF': False,
            'df_ints': None,
            'rohf': False,
            'walker_type': 'closed',
            'with_x2c': False
        }
        
        # Make a copy of basis_scf_data to use for both hamil and wfn
        basis_scf_data = {
            'mol': atom,
            'nelec': atom.nelec,
            'mo_occ': mf.mo_occ,
            'hcore': mf.get_hcore(),
            'norb': mf.mo_coeff.shape[-1],
            'X': mf.mo_coeff,
            'mo_coeff': mf.mo_coeff.copy(),  # Use a copy
            'isUHF': False,
            'df_ints': None,
            'rohf': False,
            'walker_type': 'closed',
            'with_x2c': False
        }
        
        # Store original mo_coeff shape
        original_shape = basis_scf_data['mo_coeff'].shape
        
        # Write hamiltonian with frozen core
        # For Ne (10 electrons), cas=(8, -1) freezes 1 core orbital
        fout = tmp_path / 'afqmc.h5'
        write_hamil_mol(
            scf_data=basis_scf_data,
            hamil_file=fout,
            chol_cut=1e-6,
            verbose=False,
            cas=(8, -1),
            ortho_ao=False,
            nelec=None,
            real_chol=True,
            dense=True,
            df=False,
            walker_type="collinear",
            with_soc=False
        )
        
        # Check that basis_scf_data['mo_coeff'] was NOT modified
        assert basis_scf_data['mo_coeff'].shape == original_shape, \
            f"basis_scf_data['mo_coeff'] was modified! Original shape: {original_shape}, " \
            f"New shape: {basis_scf_data['mo_coeff'].shape}"
        
        # Now write wavefunction - this should not fail with dimension mismatch
        nelec_returned = write_wfn_mol(
            scf_data=scf_data,
            filename=fout,
            basis_scf_data=basis_scf_data,
            wfn=None,
            init=None,
            verbose=False
        )
        
        # Verify the file was created and contains expected data
        with h5.File(fout, 'r') as f:
            assert 'Wavefunction' in f
            assert 'Hamiltonian' in f
            
            # Check that frozen core dimensions match between Hamiltonian and Wavefunction
            ham_dims = f['Hamiltonian/dims'][:]
            wfn_dims = f['Wavefunction/NOMSD/dims'][:]
            
            # Verify dimensions match as requested by reviewer
            assert ham_dims[3] == wfn_dims[0], \
                f"nmo mismatch: ham_dims[3]={ham_dims[3]} != wfn_dims[0]={wfn_dims[0]}"
            assert ham_dims[4] == wfn_dims[1], \
                f"nelec_alpha mismatch: ham_dims[4]={ham_dims[4]} != wfn_dims[1]={wfn_dims[1]}"
            assert ham_dims[5] == wfn_dims[2], \
                f"nelec_beta mismatch: ham_dims[5]={ham_dims[5]} != wfn_dims[2]={wfn_dims[2]}"
    
    def test_frozen_core_active_space_correct(self, tmp_path, neon_atom, neon_rhf):
        """
        Test that the active space Hamiltonian has the correct dimensions
        when using frozen core, and that the wavefunction dimensions match.
        """
        mf, energy = neon_rhf
        atom = neon_atom
        
        scf_data = {
            'mol': atom,
            'nelec': atom.nelec,
            'mo_occ': mf.mo_occ,
            'hcore': mf.get_hcore(),
            'norb': mf.mo_coeff.shape[-1],
            'X': mf.mo_coeff,
            'mo_coeff': mf.mo_coeff.copy(),
            'isUHF': False,
            'df_ints': None,
            'rohf': False,
            'walker_type': 'closed',
            'with_x2c': False
        }
        
        basis_scf_data = scf_data.copy()
        basis_scf_data['mo_coeff'] = mf.mo_coeff.copy()
        
        fout = tmp_path / 'afqmc_cas.h5'
        
        # For Ne with sto-3g (5 orbitals, 10 electrons)
        # cas=(8, -1) means:
        #   - 8 active electrons
        #   - nfzc = (10 - 8) // 2 = 1 frozen core orbital
        #   - ncas = -1 means all remaining = 5 - 1 = 4 active orbitals
        write_hamil_mol(
            scf_data=basis_scf_data,
            hamil_file=fout,
            chol_cut=1e-6,
            verbose=False,
            cas=(8, -1),
            ortho_ao=False,
            nelec=None,
            real_chol=True,
            dense=True,
            df=False,
            walker_type="collinear",  # Use collinear (same as first test)
            with_soc=False
        )
        
        # Write wavefunction with frozen core
        write_wfn_mol(
            scf_data=scf_data,
            filename=fout,
            basis_scf_data=basis_scf_data,
            wfn=None,
            init=None,
            verbose=False
        )
        
        # Check the Hamiltonian and Wavefunction files have correct and matching dimensions
        with h5.File(fout, 'r') as f:
            hcore = f['Hamiltonian/hcore'][:]
            # Should be 4x4 for the active space
            assert hcore.shape == (4, 4), \
                f"Expected hcore shape (4, 4), got {hcore.shape}"
            
            # Check Hamiltonian dimensions
            # dims structure: [0, 0, 0, nmo, nelec[0], nelec[1], 0, nchol]
            ham_dims = f['Hamiltonian/dims'][:]
            nmo = ham_dims[3]
            nelec_alpha = ham_dims[4]
            nelec_beta = ham_dims[5]
            
            assert nmo == 4, f"Expected nmo 4, got {nmo}"
            assert nelec_alpha == 4, f"Expected nelec_alpha 4, got {nelec_alpha}"
            assert nelec_beta == 4, f"Expected nelec_beta 4, got {nelec_beta}"
            
            # Check Wavefunction dimensions match Hamiltonian
            wfn_dims = f['Wavefunction/NOMSD/dims'][:]
            
            assert ham_dims[3] == wfn_dims[0], \
                f"nmo mismatch: ham_dims[3]={ham_dims[3]} != wfn_dims[0]={wfn_dims[0]}"
            assert ham_dims[4] == wfn_dims[1], \
                f"nelec_alpha mismatch: ham_dims[4]={ham_dims[4]} != wfn_dims[1]={wfn_dims[1]}"
            assert ham_dims[5] == wfn_dims[2], \
                f"nelec_beta mismatch: ham_dims[5]={ham_dims[5]} != wfn_dims[2]={wfn_dims[2]}"
