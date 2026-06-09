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
This submodule defines and runs functional tests for the
  entire SAFIRE ecosystem.
"""
from warnings import warn

import pytest
import h5py as h5
import numpy as np

from dev_tools.run_and_record import AFQMCHelper

class ResultChecker:
    """
    A helper class to compare the results of a current
      test against known reference results.

    Both the new results and the reference results must 
      exist within an HDF5 file with layout:

    ```log
    HDF5 "results.h5" {
    FILE_CONTENTS {
    group      /
    dataset    /LogOvlpFactor
    dataset    /afqmc_raised_error
    dataset    /afqmc_raised_warning
    [group      /error_messages]
    dataset    /avg_1rdm
    dataset    /avg_1rdm_stoch_error
    dataset    /energy
    dataset    /input_file
    dataset    /return_code
    dataset    /run_time_seconds
    dataset    /weight
    }
    }
    ```

    Explainations of datasets:

    - `/return_code` is an integer value corresponding to the return value from running the AFQMC executable
    - `/run_time_seconds` is the runtime of the AFQMC executable in seconds. It is include as information-only and is not compared between the current test and the reference value; however, both are printed in the test output.
    - `/afqmc_raised_error` is a boolean value (represented as H5T_ENUM in the HDF5 file) which is True if afqmc raised an error in this case and False if afqmc ran without errors
    - (group) `/error_messages` is a group containing a dataset for each unique error raised (named `/error_messages/error_[i]` where `i` is an index) by the AFQMC code. This group also has a dataset called `/error_messages/num_error_messages` which contains the number of unique error messages found.
    - `/afqmc_raised_warning` is a boolean value (represented as H5T_ENUM in the HDF5 file) which is True if afqmc raised an warning in this case and False if afqmc ran without warnings
    - (group) `/warning_messages` is a group containing each unique warning raised by the AFQMC code. It is internalyl structured similarly to `error_messages`.
    - `/avg_1rdm` is the averaged 1-rdm from AFQMC
    - `/avg_1rdm_stoch_error` is the stochastic uncertainty matrix for the 1-rdm
    - `/energy` is the average AFQMC energy along with the stochastic uncertainty. It is saves as a two-element array where the first element is the energy, and the second is the stochastic uncertainty.
    - `/weight` is the average AFQMC walker weight along with the stochastic uncertainty. It is saves as a two-element array where the first element is the weight, and the second is the stochastic uncertainty.
    - `/LogOvlpFactor` is the average AFQMC overlap between the walkers and the trial wavefunction along with the stochastic uncertainty. It is saves as a two-element array where the first element is the average, and the second is the stochastic uncertainty.
    - `/input_file` is a string containing the entire AFQMC input file used.
    """

    def __init__(self,check_bp=False) -> None:
        self.simple_datasets = [
            "afqmc_raised_warning",
            "afqmc_raised_error"
            ]
        self.message_groups = [
            "error_messages",
            "warning_messages"
        ]
        self.numeric_datasets = [
            "energy"
        ]
        if check_bp:
            self.numeric_array_datasets = [
                "avg_1rdm",
            ]
        else:
            self.numeric_array_datasets = []
        self.print_only_datasets = [
            "run_time_seconds",
            "weight",
            "LogOvlpFactor"
        ]

    def _same_messages(self,test_group:h5.Group,ref_group:h5.Group):
        """
        a helper function to check that the specific 'message' groups
           contain exactly the same messages.

        This is primarily used for checking that the same error messages
          or the same warning messages exist in both groups.
        """
        num_messages_g1 = test_group.get("num_messages")[...]
        num_messages_g2 = ref_group.get("num_messages")[...]

        if num_messages_g1 > 0:
            group_name = test_group.name.split('/')[-1] 
            prefix = group_name.split('_')[0]  # Extract prefix (e.g., 'error' or 'warning')
            messages1 = set(
                [str(test_group[f"{prefix}_{i}"][...]) for i in range(num_messages_g1)] 
            )
        else:
            messages1 = set()

        if num_messages_g2 > 0:
            group_name = ref_group.name.split('/')[-1] 
            prefix = group_name.split('_')[0]  # Extract prefix (e.g., 'error' or 'warning')
            messages2 = set(
                [str(ref_group[f"{prefix}_{i}"][...]) for i in range(num_messages_g2)] 
            )
        else:
            messages2 = set()

        if messages1 != messages2:
            print("results files have different messages")
            print("Test group has messages:", messages1 )
            print("Reference group has messages:", messages2 )
            return False
        else:
            print("results files have the same messages")
            print("Messages are: (ref)", messages1)
            print(" (this test): ", messages2)
            return True


    def _compare_stochastic_arrays(self,A:np.ndarray,A_stoch_uncertainty:np.ndarray,B:np.ndarray,B_stoch_uncertainty:np.ndarray,relative_tolerance=1e-4):
        """
        Compare stochastic arrays by checking the percentage of elements that agree / disagree
        with the reference values within a certain tolerance.
    
        explicitly handle the real and imaginary parts.

        Parameters
        ----------
        A : np.ndarray
            The first array to compare.
        A_stoch_uncertainty : np.ndarray
            The stochastic uncertainty for the first array.
        B : np.ndarray
            The second array to compare. This is the reference array.
        B_stoch_uncertainty : np.ndarray
            The stochastic uncertainty for the second array. This is the reference uncertainty.
        relative_tolerance : float, optional
            The relative tolerance for the comparison, by default 1e-4.
        """

        if A.shape != B.shape or A_stoch_uncertainty.shape != B_stoch_uncertainty.shape:
            raise ValueError("Arrays to compare must have the same shape.")
        
        A = np.asarray(A, dtype=np.complex128)
        B = np.asarray(B, dtype=np.complex128)

        joint_stoch_uncertainty = np.sqrt(np.power(A_stoch_uncertainty,2) + np.power(B_stoch_uncertainty,2)).real

        def _percentage_close(arr1:np.ndarray, arr2:np.ndarray, atol:np.ndarray, rtol:float) -> float:
            """
            Calculate the percentage of elements in arr1 that are close to arr2
            within the specified absolute and relative tolerances.
            """
            return np.mean(np.isclose(arr1, arr2, atol=atol, rtol=rtol)) * 100

        print("\n=======[ Comparing stochastic arrays ]=======\n")

        print(f"  ----- Comparing Re[A] and Re[B] -----")
        real_one_sigma = _percentage_close(A.real, B.real, joint_stoch_uncertainty, relative_tolerance)
        real_two_sigma = _percentage_close(A.real, B.real, 2*joint_stoch_uncertainty, relative_tolerance)
        real_three_sigma = _percentage_close(A.real, B.real, 3*joint_stoch_uncertainty, relative_tolerance)
        
        print(f"[+] Percentage of elements that agree to within one sigma (joint stochastic uncertainty): {real_one_sigma:.2f}%")
        print(f"[+] Percentage of elements that agree to within two sigma (joint stochastic uncertainty): {real_two_sigma:.2f}%")
        print(f"[+] Percentage of elements that agree to within three sigma (joint stochastic uncertainty): {real_three_sigma:.2f}%")
        
        # Check against expected percentages from normal distribution
        if real_one_sigma < 68.27:
            warn(f"Real part one sigma agreement ({real_one_sigma:.2f}%) below expected 68.27%")
        if real_two_sigma < 95.45:
            warn(f"Real part two sigma agreement ({real_two_sigma:.2f}%) below expected 95.45%")
        if real_three_sigma < 99.73:
            warn(f"Real part three sigma agreement ({real_three_sigma:.2f}%) below expected 99.73%")
        
        print(f"  ----- Comparing Im[A] and Im[B] -----")
        imag_one_sigma = _percentage_close(A.imag, B.imag, joint_stoch_uncertainty, relative_tolerance)
        imag_two_sigma = _percentage_close(A.imag, B.imag, 2*joint_stoch_uncertainty, relative_tolerance)
        imag_three_sigma = _percentage_close(A.imag, B.imag, 3*joint_stoch_uncertainty, relative_tolerance)
        
        print(f"[+] Percentage of elements that agree to within one sigma (joint stochastic uncertainty): {imag_one_sigma:.2f}%")
        print(f"[+] Percentage of elements that agree to within two sigma (joint stochastic uncertainty): {imag_two_sigma:.2f}%")
        print(f"[+] Percentage of elements that agree to within three sigma (joint stochastic uncertainty): {imag_three_sigma:.2f}%")
        
        # Check against expected percentages from normal distribution
        if imag_one_sigma < 68.27:
            warn(f"Imaginary part one sigma agreement ({imag_one_sigma:.2f}%) below expected 68.27%")
        if imag_two_sigma < 95.45:
            warn(f"Imaginary part two sigma agreement ({imag_two_sigma:.2f}%) below expected 95.45%")
        if imag_three_sigma < 99.73:
            warn(f"Imaginary part three sigma agreement ({imag_three_sigma:.2f}%) below expected 99.73%")

        # return True to indicate that comparisons finished, for now, we need to manually check the output.
        return True
    

    #TODO: it may be necessary to be more careful with accessing
    #        test data saved in hdf5 (i.e. if any data is missing)
    #        if so, use .get() to retrieve the data
    def results_are_same(self,fname_test,fname_ref) -> bool:
        """
        Compare the results in file 'fname_new' to the results in file 'fname_ref'.
        """
        
        are_same = list()

        with h5.File(fname_test,'r') as f_test:
            with h5.File(fname_ref,'r') as f_ref:
                
                if not f_test["afqmc_is_finite"][...]:
                    raise ValueError("AFQMC test results contain NaN or Inf values.")

                test_return_code = f_test["return_code"][...]
                if test_return_code is None:
                    raise ValueError("[for developers] Invalid test resuls.h5 file: does not contain a `return_code`.")
   
                # treat all positive, non-zero codes as an exit for now
                if test_return_code > 0:
                    test_return_code = 1
                else:
                    test_return_code = 0
                
                ref_return_code = f_ref["return_code"][...]
                if ref_return_code is None:
                    raise ValueError("[for developers] Invalid reference resuls.h5 file: does not contain a `return_code`.")

                # treat all positive, non-zero codes as an exit for now
                if ref_return_code > 0:
                    ref_return_code = 1
                else:
                    ref_return_code = 0
                
                assert test_return_code == ref_return_code

                for dataset in self.simple_datasets:
                    data1 = f_test[dataset][...]
                    data2 = f_ref[dataset][...]
                    
                    if data1 == data2:
                        are_same.append(True)
                    else:
                        are_same.append(False)

                    matches = are_same[-1]
                    print(f"Does dataset {dataset} match? {matches}")
                    #if not matches:
                    print(f"  - this test: {data1}\n  - ref. value: {data2}")

                for group in self.message_groups:
                    test_group = f_test.get(group,None)
                    ref_group = f_ref.get(group,None)
                    
                    if test_group is None and ref_group is None:
                        continue 
                    
                    if not self._same_messages(test_group,ref_group):
                        warn(f"Message group {group} does not match between test and reference files")
                        matches = False

                # numeric within tolerance
                if ref_return_code == 0:
                    for dataset in self.numeric_datasets:
                        data1 = f_test[dataset][...]
                        data2 = f_ref[dataset][...]                        
                        tolerance = data2[1]

                        if data1.shape != data2.shape:
                            are_same.append(False)
                            continue

                        if np.allclose(data1[0],data2[0],atol=tolerance):
                            are_same.append(True)
                        elif np.allclose(data1[0],data2[0],atol=2*tolerance):
                            warn(f"Results for {dataset} are within 2x tolerance.")
                            are_same.append(True)
                        else:
                            are_same.append(False)

                        matches = are_same[-1]
                        print(f"Does dataset {dataset} match? {matches}")
                        #if not matches:
                        print(f"  - this test: {data1}\n  - ref. value: {data2}")
                    
                    for dataset in self.numeric_array_datasets:
                        data1 = f_test[dataset][...]
                        data2 = f_ref[dataset][...]
                        stoch_error1 = f_test[f"{dataset}_stoch_error"][...]
                        stoch_error2 = f_ref[f"{dataset}_stoch_error"][...]

                        if data1.shape != data2.shape or stoch_error1.shape != stoch_error2.shape:
                            print(f"Dataset {dataset} has different shapes in test and reference files.")
                            are_same.append(False)
                            continue

                        print(f"Comparing dataset {dataset} as stochastic arrays")
                        if self._compare_stochastic_arrays(data1,stoch_error1,data2,stoch_error2):
                            # NOTE: for now, compare_stochastic_arrays only raises a warning
                            #       if the agreement is inconsistent with statistical expectations
                            are_same.append(True)
                        else:
                            are_same.append(False)

                print("Print-only test data:")
                for dataset in self.print_only_datasets:
                    data1 = f_test[dataset][...]
                    data2 = f_ref[dataset][...]
                    print(f"{dataset}:\n  - this test: {data1}\n  - ref. value: {data2}")

        return all(are_same)

    def _same_return_code(self,test_file:h5.File,ref_file:h5.File,must_match=None):
        """Checks that the test and reference return codes are the same
        
        Parameters
        ----------
        test_file : h5py.File
            The File instance corresponding to the file containing test results
        ref_file : h5py.File
            The File instance corresponding to the file containing reference results
        must_math : int
            if supplied, _same_return_code will check that both return codes match
            this value.
        """
        test_return_code = test_file["return_code"][...]
        if test_return_code is None:
            raise ValueError("[for developers] Invalid test resuls.h5 file: does not contain a `return_code`.")

        # treat all positive, non-zero codes as an exit for now
        if test_return_code > 0:
            test_return_code = 1
        else:
            test_return_code = 0
        
        ref_return_code = ref_file["return_code"][...]
        if ref_return_code is None:
            raise ValueError("[for developers] Invalid reference resuls.h5 file: does not contain a `return_code`.")

        # treat all positive, non-zero codes as an exit for now
        if ref_return_code > 0:
            ref_return_code = 1
        else:
            ref_return_code = 0
        
        if must_match is not None:
            # TODO: generalize exit codes
            if must_match > 0:
                must_match = 1
            else:
                must_match = 0
            
            if test_return_code == must_match and ref_return_code == must_match:
                print("Both return codes match the expected value.")
                return True
            else:
                print("Return codes do not match the expected value.")
                return False
        else:
            if test_return_code == ref_return_code:
                print("Return codes match.")
                return True
            else:
                print("Return codes do not match.")
                return False
        

    def same_error(self,fname_test,fname_ref):
        """Checks that SAFIRE raised an error in both the test and reference cases. Errors that
        do not originate from SAFIRE are intentionally not 
            Warns if the specific error messages do not match.
        
        Parameters
        ----------
        fname_test : str or Path
            The path to the results.h5 file containing the test results
        fname_ref : str or Path
            The path to the results.h5 file containing the reference results
        
        Raises
        ------
        ValueError
            when either the test or the reference case is missing a 'return_code'
        """

        with h5.File(fname_test,'r') as f_test:
            with h5.File(fname_ref,'r') as f_ref:
                num_messages_g1 = f_test["error_messages"].get("num_messages")[...]
                num_messages_g2 = f_ref["error_messages"].get("num_messages")[...]
                assert num_messages_g1 > 0, "Test case does not contain any error messages, but an error was expected."
                assert num_messages_g2 > 0, "Reference case does not contain any error messages, but an error was expected."

                # warn if error message are not the same
                if not self._same_messages(
                        test_group=f_test["error_messages"],
                        ref_group=f_ref["error_messages"]):
                   warn("Error messages do not match between test and reference files.") 
                return self._same_return_code(f_test,f_ref,must_match=1)

@pytest.fixture(scope="session")
def result_checker():
    """
    A wrapper fixture to return a session-scoped ResultChecker instance.

    session scope was chosen because there is currently no need to rebuild the checker
      for every test.
    """
    return ResultChecker()


@pytest.fixture(scope="session")
def result_checker_bp():
    """
    A wrapper fixture to return a session-scoped ResultChecker instance which
    checks Back Propagation results.

    session scope was chosen because there is currently no need to rebuild the checker
      for every test.
    """
    return ResultChecker(check_bp=True)



@pytest.fixture(scope="session")
def afqmc_helper(afqmc_exec,num_mpi_tasks,timeout_mins):
    """
    A wrapper fixture to return a session-scoped AFQMCHelper instance.

    session scope was chosen because there is currently no need to rebuild the AFQMCHelper
      for every run.
    """
    return AFQMCHelper(afqmc_exec,num_mpi_tasks,timeout_mins)
