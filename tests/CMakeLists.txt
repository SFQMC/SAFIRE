add_library(catch_main STATIC catch_main.cpp)
target_link_libraries(catch_main PUBLIC safire_lib Catch2::Catch2)
target_include_directories(catch_main PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
if(ENABLE_CPPTRACE)
  target_link_libraries(safire_lib PUBLIC cpptrace::cpptrace)
endif()
if(ENABLE_CUDA)
  target_link_libraries(catch_main PUBLIC CCCL::CCCL)
endif()

# Consolidated AFQMC test binary
add_executable(test_afqmc
  test_allocators.cpp
  test_driver_factory.cpp
  test_estimator_handler.cpp
  test_estimators.cpp
  test_polarized_consistency.cpp
  test_hamiltonian_factory.cpp
  test_hamiltonian_operations.cpp
  test_observables_reference.cpp
  test_one_body.cpp
  test_phmsd.cpp
  test_phmsd_components.cpp
  test_propagator_factory.cpp
  test_sdet_ops.cpp
  test_sharedwset.cpp
  test_tensor_ops.cpp
  test_wfn_factory.cpp
)
set_target_properties(test_afqmc PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
target_link_libraries(test_afqmc PRIVATE catch_main)

catch_discover_tests(test_afqmc
  WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR}
  PROPERTIES LABELS "afqmc"
)
