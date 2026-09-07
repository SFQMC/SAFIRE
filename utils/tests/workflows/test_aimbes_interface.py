# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import h5py as h5
import numpy as np
import pytest

from afqmctools.utils.aimbes_utils import (
    check_aimbes_energy,
    read_aimbes_Hamiltonian,
    _infer_aimbes_hamil_type,
    AIMBESHamiltonian
)

TEST_ROOT = Path(__file__).resolve().parent

TEST_INPUT_DATA_PATH = TEST_ROOT / "data/aimbes"
TEST_REF_DATA_PATH = TEST_ROOT / "data/aimbes/ref"

@pytest.mark.dev
@pytest.mark.aimbes
@pytest.mark.parametrize("aimbes_file,hamil_type,expected_error,ref_hamil_file",
    [
        (
            TEST_INPUT_DATA_PATH/"mf_downfold.mbpt.h5",
            AIMBESHamiltonian.FROZEN_CORE,
            None,
            TEST_REF_DATA_PATH / "ref_H_nbnd6.h5"
        ),
        (
            TEST_INPUT_DATA_PATH/"mf_downfold.mbpt.h5",
            AIMBESHamiltonian.BARE,
            ValueError,
            None,
        ),
        (
            TEST_INPUT_DATA_PATH/"mf_downfold.mbpt.h5",
            AIMBESHamiltonian.CRPA,
            ValueError,
            None
        ),
        (
            TEST_INPUT_DATA_PATH/"aimbes.nb6.h5",
            AIMBESHamiltonian.FROZEN_CORE,
            ValueError,
            None
        ),
        (
            TEST_INPUT_DATA_PATH/"aimbes.nb6.h5",
            AIMBESHamiltonian.BARE,
            None,
            TEST_REF_DATA_PATH / "ref_H_nbnd6.h5"
        ),
        (
            TEST_INPUT_DATA_PATH/"aimbes.nb6.h5",
            AIMBESHamiltonian.CRPA,
            ValueError,
            None
        )
    ]
)
class TestAIMBEStoAFQMC:
    """
    Test the interface from AIMBES to AFQMC.

    The interface should be able to:
    1.a. infer the correct H to build based on the contents of the
         input AIMBES file
      b. should be able to TRY to build a user-specified type of H (and fail
         if it doesn't have what it needs)
    2. generate the correct Hamiltonian as determined by
      a. the actual matrix elements (to within the Cholesky tolerance for 2-body, machine prec. for 1-body)
      b. Variational energy.
    3. AFQMC should work: 
      a. running with the same seed / num. walkers / num. mpi tasks should give the same answer.
      b. running with the same total num. walkers and projection time should give same results to within stoch. uncertainty.

    Notes:
    The converter shouldn't need to know if the Hamiltonian was generated via `mf_downfold` or using explicit `downfold_2e` / `downfold_1e`, it should simply read what it gets, and try to build an AFQMC Hamiltonain based on what it's been given.
    """
    
    @pytest.mark.debug
    def test_infer_type(self,aimbes_file,hamil_type,expected_error,ref_hamil_file):
        """
        Test if the converter can correctly infer the Hamiltonian type
        """
        if expected_error is None:
            hamil_type = _infer_aimbes_hamil_type(aimbes_fname=aimbes_file,hamil_type=hamil_type)
        else:
            with pytest.raises(expected_error) as e:
                hamil_type = _infer_aimbes_hamil_type(aimbes_fname=aimbes_file,hamil_type=hamil_type)
    
    @pytest.mark.debug
    def test_read_hamiltonian(self,aimbes_file,hamil_type,expected_error,ref_hamil_file):
        """
        Test if the Hamiltonian can be read

        TODO: compare against a known correct Hamiltonian.
        """
        if expected_error is None:
            H1,H2,chol_delta = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type)
            assert H1 is not None and H2 is not None and chol_delta is not None
        else:
            with pytest.raises(expected_error) as e:
                H1,H2,_ = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type)

    @pytest.mark.debug
    def test_correct_hamiltonian(self,aimbes_file,hamil_type,expected_error,ref_hamil_file):
        """
        If failing, try running `$ pytest -m "aimbes and debug" -vvv -rP` to track
           down where things stop working correctly.
        """
        if expected_error is None:
            H1,L_test,delta_cholesky_test = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type,chol_tol=1.0E-4)
            with h5.File(ref_hamil_file) as f:
                H1_ref = f["H1_ref"][...]
                H1_ref = H1_ref[:,:,0] + 1j*H1_ref[:,:,1] 
                V_ref = f["V_abcd_ref"][...]
                delta_cholesky_ref = f["delta_cholesky"][...]

                # TODO: we may need to make V_abcd here from Cholesky vectors
                L_test_dagger = np.transpose(L_test,axes=(0,2,1)).conj()
                V_test = np.einsum("nab,ncd->abcd",L_test,L_test_dagger)

                comp_tol = max(delta_cholesky_test,delta_cholesky_ref)
                # TODO: we'll likely need to define a tolerance here!
                print(f"Comparing Coulomb interaction to absolute tolerance {comp_tol}")
                assert np.allclose(H1,H1_ref)

                if not np.allclose(V_test,V_ref,atol=comp_tol):

                    # analyze the differnce:
                    delta_V_r = (V_test - V_ref).real
                    delta_V_i = (V_test - V_ref).imag

                    print("=== comparing real part ======\n")
                    print(f" max deltaV = {np.max(delta_V_r)}")
                    print(f" max abs(deltaV) = {np.max(np.abs(delta_V_r))}")
                    print(f" mean deltaV = {np.mean(delta_V_r)}")
                    print(f" mean abs(deltaV) = {np.mean(np.abs(delta_V_r))}")
                    print(f" std deltaV = {np.std(delta_V_r)}")
                    print(f" std abs(deltaV) = {np.std(np.abs(delta_V_r))}")

                    print("=== comparing imaginary part ======\n")
                    print(f" max deltaV = {np.max(delta_V_i)}")
                    print(f" max abs(deltaV) = {np.max(np.abs(delta_V_i))}")
                    print(f" mean deltaV = {np.mean(delta_V_i)}")
                    print(f" mean abs(deltaV) = {np.mean(np.abs(delta_V_i))}")
                    print(f" std deltaV = {np.std(delta_V_i)}")
                    print(f" std abs(deltaV) = {np.std(np.abs(delta_V_i))}")

                    assert np.allclose(V_test,V_ref,atol=comp_tol)
        else:
            with pytest.raises(expected_error) as e:
                H1,L_test = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type)

    def test_correct_e_var(self,aimbes_file,hamil_type,expected_error,ref_hamil_file):
        """
        Test that the energy evaluated from the density matrix and Hamiltonian 
          in `aimbes_file` are correct when evaluated as :

          1. the variational "Hartree-Fock" energy (F[rho_n], and rho_n are used )
          2. the aimbes "Hartree-Fock" energy (F[rho_{n-1}], and rho_n are used )
        """
        if expected_error is None:
            H1,L_test,delta_cholesky_test = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type)            

            # TODO: we may need to make V_abcd here from Cholesky vectors
            L_test_dagger = np.transpose(L_test,axes=(0,2,1)).conj()
            V_test = np.einsum("nab,ncd->abcd",L_test,L_test_dagger)

            with h5.File(ref_hamil_file) as f:
                E_var_0 = f["E_var_0"][...]
                E_aimbes_hf_1 = f["E_aimbes_hf_1"][...]

                E_var_0_test = check_aimbes_energy(
                    aimbes_file=aimbes_file,
                    H1=H1,
                    V_abcd=V_test,
                    scf_iter=0,
                    mode="variational"
                )
                _E_var_0_is_correct = np.isclose(E_var_0_test,E_var_0)
                # this isn't always applicable
                if hamil_type == AIMBESHamiltonian.CRPA:
                    E_aimbes_hf_1_test = check_aimbes_energy(
                        aimbes_file=aimbes_file,
                        H1=H1,
                        V_abcd=V_test
                    )
                    _E_aimbes_1_is_correct = np.isclose(E_aimbes_hf_1_test,E_aimbes_hf_1)
                assert _E_var_0_is_correct and _E_aimbes_1_is_correct
        else:
            with pytest.raises(expected_error) as e:
                H1,L_test = read_aimbes_Hamiltonian(aimbes_file,hamil_type=hamil_type)


@pytest.mark.skip
class TestAIMBEStoDice:
    """
    Test the interface from AIMBES to Dice

    Test:
    - CAS-like workflow (maybe this should be a new
                test class that )
    """
    @pytest.mark.dev
    def test_run(self):
        """
        Dummy Test
        """
        raise NotImplementedError

