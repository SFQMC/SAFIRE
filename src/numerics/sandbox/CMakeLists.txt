#//////////////////////////////////////////////////////////////////////////////////////
#//////////////////////////////////////////////////////////////////////////////////////

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_UNIT_TEST_DIR})

set(TEST_ID sandbox)

set( TEST_SRCS
     contract_tests.cpp 
#     mpi_tests.cpp
   )

add_executable(${TEST_ID} ${TEST_SRCS} )
target_link_libraries(${TEST_ID} PRIVATE catch_main)

#//////////////////////////////////////////////////////////////////////////////////////

#set(TEST_ID h5_tests)

#set( TEST_SRCS
#     h5_tests.cpp
#   )


#add_executable(${TEST_ID} ${TEST_SRCS} )
#target_link_libraries(${TEST_ID} PRIVATE catch_main)

