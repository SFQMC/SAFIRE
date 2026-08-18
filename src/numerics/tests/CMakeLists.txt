#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID nda_functions)

add_executable(test_${TEST_ID} test_nda_functions.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main safire_lib)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})


set(TEST_ID math_product)

add_executable(test_${TEST_ID} test_math_product.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main safire_lib)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})


set(TEST_ID bisect)

add_executable(test_${TEST_ID} test_bisect.cpp)
target_link_libraries(test_${TEST_ID} PRIVATE catch_main safire_lib)

add_unit_test(test_${TEST_ID} "${PROJECT_UNIT_TEST_DIR}/test_${TEST_ID}")
set_tests_properties(test_${TEST_ID} PROPERTIES WORKING_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

