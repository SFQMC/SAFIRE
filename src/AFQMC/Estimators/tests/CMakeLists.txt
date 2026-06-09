
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID afqmc_estimators)
set( TEST_SRCS 
	test_estimators.cpp
        )

if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)
list(LENGTH AFQMC_UNIT_TEST_INPUTS NUM_PAIRS)
math(EXPR ENDP "${NUM_PAIRS}-1")
foreach(I RANGE 0 ${ENDP} 2)
    list(GET AFQMC_UNIT_TEST_INPUTS ${I} HAMIL_FILE)
    math(EXPR J "${I}+1")
    list(GET AFQMC_UNIT_TEST_INPUTS ${J} WFN_FILE)
    get_filename_component(HAMIL ${HAMIL_FILE} NAME_WE)
    get_filename_component(WFN ${WFN_FILE} NAME_WE)
    # back propagate not yet working with model hamiltonian 
    #if(NOT (${HAMIL} MATCHES "hubb4x4U1") )
      set(UTEST_NAME unit_test_${TEST_ID}_${HAMIL}_${WFN})
      add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                    "--hamil ${HAMIL_FILE}" "--wfn ${WFN_FILE}")
      set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
      set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
    #endif()
endforeach()

set(TEST_ID afqmc_estimator_handler)
set( TEST_SRCS 
  test_estimator_handler.cpp
        )

if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)
list(LENGTH AFQMC_UNIT_TEST_INPUTS NUM_PAIRS)
math(EXPR ENDP "${NUM_PAIRS}-1")
foreach(I RANGE 0 ${ENDP} 2)
    list(GET AFQMC_UNIT_TEST_INPUTS ${I} HAMIL_FILE)
    math(EXPR J "${I}+1")
    list(GET AFQMC_UNIT_TEST_INPUTS ${J} WFN_FILE)
    get_filename_component(HAMIL ${HAMIL_FILE} NAME_WE)
    get_filename_component(WFN ${WFN_FILE} NAME_WE)
    # chosen arbitrarily - the EstimatorHandler's behavior is independent of the hamiltonian
    if(${HAMIL} MATCHES "hubb4x4U1")
      set(UTEST_NAME unit_test_${TEST_ID}_${HAMIL}_${WFN})
      add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                    "--hamil ${HAMIL_FILE}" "--wfn ${WFN_FILE}")
      set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
      set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
    endif()
endforeach()

set(TEST_ID afqmc_observables)
set( TEST_SRCS
  test_observables.cpp
        )

if(ENABLE_CUDA)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE CUDA)
  endforeach()
elseif(ENABLE_HIP)
  foreach( X ${TEST_SRCS} )
    set_source_files_properties(${X} PROPERTIES LANGUAGE HIP)
  endforeach()
endif()

add_executable( test_${TEST_ID} ${TEST_SRCS} )
target_link_libraries(test_${TEST_ID} catch_main afqmc)
set(UTEST_NAME unit_test_${TEST_ID})
add_unit_test(${UTEST_NAME} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(${UTEST_NAME} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
set_property(TEST ${UTEST_NAME} APPEND PROPERTY LABELS "afqmc")
