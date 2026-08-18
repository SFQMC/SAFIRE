
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID const_shared_array)

add_executable(test_${TEST_ID} test_const_shared_array.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main)

# This test needs >1 rank to exercise the node-shared / internode broadcast
# paths, so it is launched under MPI explicitly rather than via add_unit_test.
set(CONST_SHARED_NRANKS 4)
add_test(NAME test_${TEST_ID}
         COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${CONST_SHARED_NRANKS}
                 ${MPIEXEC_PREFLAGS} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}"
                 ${MPIEXEC_POSTFLAGS})
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})
