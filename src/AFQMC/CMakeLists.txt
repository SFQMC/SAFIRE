
#-------------------------------------------------------------------
# Sources
#-------------------------------------------------------------------
set (AFQMC_SRCS
  Utilities/AFQMCTimer.cpp 
  Utilities/readWfn.cpp
  Utilities/probit.cpp
  AFQMCFactory.cpp
  Drivers/DriverFactory.cpp
  Drivers/AFQMCNewDriver.cpp
  Drivers/AFQMCDriver.cpp
  Drivers/CSAFQMCDriver.cpp
  Propagators/AFQMCBasePropagator.cpp
  Propagators/AFQMCModelPropagator.cpp
  Propagators/PropagatorFactory.cpp
  Wavefunctions/WavefunctionFactory.cpp
  Hamiltonians/KPFactorizedHamiltonian.cpp
  Hamiltonians/RealDenseHamiltonian.cpp
  Hamiltonians/RealDenseHamiltonian_v2.cpp
  Hamiltonians/ModelHamOpsGenerator.cpp
  Hamiltonians/HamiltonianFactory.cpp
  Hamiltonians/THCHamiltonian.cpp
  Hamiltonians/FactorizedSparseHamiltonian.cpp
  )

if(ENABLE_CUDA)
  foreach( X ${AFQMC_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${AFQMC_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

add_library(afqmc ${AFQMC_SRCS})
target_link_libraries(afqmc PUBLIC utils qmcio memory BLAS::BLAS LAPACK::LAPACK Boost::boost)

if(ENABLE_CUDA)
  target_link_libraries(afqmc PUBLIC kernels CUDA::cublas CUDA::cusparse CUDA::curand CUDA::cusolver)
elseif(ENABLE_HIP)
  target_link_libraries(afqmc PUBLIC kernels )
else(ENABLE_CUDA)
endif(ENABLE_CUDA)

if (BUILD_UNIT_TESTS)
  INCLUDE_DIRECTORIES(${PROJECT_SOURCE_DIR}/extern/catch)
  if (ENABLE_DEVICE)
    set(AFQMC_UNIT_TEST_INPUTS 
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_uhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_msd.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_chol_1e-5.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_thc_1e-6.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5     	
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/hubb4x4U1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/wfn_uhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard-kanamori/4x1square/hk_4x1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard-kanamori/4x1square/hk_4x1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Li_noncollinear/Li_ham.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Li_noncollinear/Li_ghf_wfn.h5
    )
  else()
    set(AFQMC_UNIT_TEST_INPUTS 
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_uhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/ham_chol_sc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_dzvp/wfn_msd.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_chol_1e-5.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/ham_thc_1e-6.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/C_1x1x1_ks_basis/wfn_mf_pbe.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/ham_chol_uc.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/He_2x2x2_dzv/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/ham_chol_dense.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Ne_cc-pvdz/wfn_rhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/hubb4x4U1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard/4x4squareU1/wfn_uhf.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard-kanamori/4x1square/hk_4x1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/hubbard-kanamori/4x1square/hk_4x1.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Li_noncollinear/Li_ham.h5
      ${PROJECT_SOURCE_DIR}/tests/unit_test_files/Li_noncollinear/Li_ghf_wfn.h5
    )
  endif()
  SUBDIRS(SlaterDeterminantOperations/tests)
  SUBDIRS(SlaterDeterminantOperations/time_propagation)
  SUBDIRS(Walkers/tests)
#  SUBDIRS(Hamiltonians/tests)
  SUBDIRS(HamiltonianOperations/tests)
  SUBDIRS(Wavefunctions/tests)
  SUBDIRS(Propagators/tests)
  SUBDIRS(Drivers/tests)
  SUBDIRS(Estimators/tests)
endif()

