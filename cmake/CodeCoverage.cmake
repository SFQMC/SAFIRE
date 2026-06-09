######################################################################
# Code Coverage Configuration Module
######################################################################
# This module provides functionality to enable code coverage reporting
# for C++ projects using gcov/lcov or gcovr.
#
# Usage:
#   1. Set ENABLE_COVERAGE=ON when configuring cmake
#   2. Build the project
#   3. Run tests with 'ctest' or 'make test'
#   4. Generate coverage report with 'make coverage' (lcov) or 'make coverage_gcovr' (gcovr)
#
# Requirements:
#   - GCC or Clang compiler
#   - lcov (for HTML reports via lcov)
#   - gcovr (alternative HTML/XML/JSON reports)
#   - genhtml (part of lcov package)
######################################################################

# Check for prerequisites
if(ENABLE_COVERAGE)
  if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(WARNING "Code coverage results with an optimized build may be misleading. "
                    "Recommended to use -DCMAKE_BUILD_TYPE=Debug with -DENABLE_COVERAGE=ON")
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(STATUS "Code coverage enabled for ${CMAKE_CXX_COMPILER_ID}")
  else()
    message(FATAL_ERROR "Code coverage requires GCC or Clang compiler. "
                        "Current compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()

  # Find coverage tools (optional, will give warnings if not found)
  find_program(LCOV_PATH lcov)
  find_program(GENHTML_PATH genhtml)
  find_program(GCOVR_PATH gcovr)

  if(NOT LCOV_PATH)
    message(WARNING "lcov not found. Install lcov to use 'make coverage' target. "
                    "On macOS: brew install lcov")
  endif()

  if(NOT GCOVR_PATH)
    message(WARNING "gcovr not found. Install gcovr to use 'make coverage_gcovr' target. "
                    "On macOS: pip install gcovr")
  endif()
endif()

# Function to add coverage compiler and linker flags
function(append_coverage_compiler_flags)
  if(ENABLE_COVERAGE)
    # Coverage compile flags
    set(COVERAGE_COMPILE_FLAGS "--coverage -fprofile-arcs -ftest-coverage -fno-inline -fno-inline-small-functions -fno-default-inline -O0 -g")
    # Coverage link flags
    set(COVERAGE_LINK_FLAGS "--coverage")

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
      # GCC specific flags
      set(COVERAGE_COMPILE_FLAGS "${COVERAGE_COMPILE_FLAGS} -fprofile-abs-path")
    endif()

    # Set the flags at parent scope
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COVERAGE_COMPILE_FLAGS}" PARENT_SCOPE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COVERAGE_COMPILE_FLAGS}" PARENT_SCOPE)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${COVERAGE_LINK_FLAGS}" PARENT_SCOPE)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${COVERAGE_LINK_FLAGS}" PARENT_SCOPE)

    message(STATUS "Code coverage compile flags: ${COVERAGE_COMPILE_FLAGS}")
    message(STATUS "Code coverage link flags: ${COVERAGE_LINK_FLAGS}")
  endif()
endfunction()

# Function to setup coverage targets
function(setup_coverage_targets)
  if(NOT ENABLE_COVERAGE)
    return()
  endif()

  set(COVERAGE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/coverage")
  set(COVERAGE_REPORT_DIR "${COVERAGE_OUTPUT_DIR}/html")

  # Create coverage output directory
  file(MAKE_DIRECTORY ${COVERAGE_OUTPUT_DIR})

  # Target: coverage (using lcov)
  if(LCOV_PATH AND GENHTML_PATH)
    add_custom_target(coverage
      COMMAND ${CMAKE_COMMAND} -E echo "Cleaning old coverage data..."
      COMMAND ${LCOV_PATH} --directory . --zerocounters
      
      COMMAND ${CMAKE_COMMAND} -E echo "Running tests..."
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      
      COMMAND ${CMAKE_COMMAND} -E echo "Collecting coverage data..."
      COMMAND ${LCOV_PATH} --directory . --capture --output-file ${COVERAGE_OUTPUT_DIR}/coverage.info
              --ignore-errors inconsistent,inconsistent
              --ignore-errors unsupported,unsupported
              --ignore-errors format,format
      
      COMMAND ${CMAKE_COMMAND} -E echo "Extracting coverage for src directory only..."
      COMMAND ${LCOV_PATH} --extract ${COVERAGE_OUTPUT_DIR}/coverage.info 
              "${CMAKE_SOURCE_DIR}/src/*"
              --output-file ${COVERAGE_OUTPUT_DIR}/coverage_filtered.info
              --ignore-errors inconsistent,inconsistent
              --ignore-errors format,format
              --ignore-errors unused,unused
      
      COMMAND ${CMAKE_COMMAND} -E echo "Generating HTML report..."
      COMMAND ${GENHTML_PATH} ${COVERAGE_OUTPUT_DIR}/coverage_filtered.info 
              --output-directory ${COVERAGE_REPORT_DIR}
              --title "SAFIRE Code Coverage Report"
              --legend
              --demangle-cpp
              --ignore-errors inconsistent,inconsistent
              --ignore-errors corrupt,corrupt
              --ignore-errors format,format
              --ignore-errors category,category
      
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated at: ${COVERAGE_REPORT_DIR}/index.html"
      COMMAND ${CMAKE_COMMAND} -E echo "Open with: open ${COVERAGE_REPORT_DIR}/index.html"
      
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating code coverage report with lcov"
      VERBATIM
    )

    message(STATUS "Added 'coverage' target (lcov). Run with: make coverage")
  endif()

  # Target: coverage_gcovr (using gcovr for alternative reporting)
  if(GCOVR_PATH)
    add_custom_target(coverage_gcovr
      COMMAND ${CMAKE_COMMAND} -E echo "Cleaning old coverage data..."
      COMMAND find ${CMAKE_BINARY_DIR} -type f -name '*.gcda' -delete
      
      COMMAND ${CMAKE_COMMAND} -E echo "Running tests..."
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      
      COMMAND ${CMAKE_COMMAND} -E echo "Generating coverage report with gcovr (src directory only)..."
      COMMAND ${GCOVR_PATH} ${CMAKE_BINARY_DIR}
              --root ${CMAKE_SOURCE_DIR}
              --exclude '.*/extern/.*'
              --exclude '.*/_deps/.*'
              --exclude '.*/CommandLineTools/.*'
              --exclude '.*/usr/.*'
              --exclude '.*/tests/.*'
              --gcov-ignore-parse-errors=all
              --html --html-details 
              --output ${COVERAGE_REPORT_DIR}/gcovr_index.html
              --print-summary
      
      COMMAND ${CMAKE_COMMAND} -E echo ""
      COMMAND ${CMAKE_COMMAND} -E echo "Coverage report generated at: ${COVERAGE_REPORT_DIR}/gcovr_index.html"
      COMMAND ${CMAKE_COMMAND} -E echo "Open with: open ${COVERAGE_REPORT_DIR}/gcovr_index.html"
      
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating code coverage report with gcovr"
      VERBATIM
    )

    # Also add XML and text coverage targets
    add_custom_target(coverage_xml
      COMMAND ${CMAKE_COMMAND} -E echo "Running tests..."
      COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
      
      COMMAND ${CMAKE_COMMAND} -E echo "Generating XML coverage report (src directory only)..."
      COMMAND ${GCOVR_PATH} ${CMAKE_BINARY_DIR}
              --root ${CMAKE_SOURCE_DIR}
              --exclude '.*/extern/.*'
              --exclude '.*/_deps/.*'
              --exclude '.*/CommandLineTools/.*'
              --exclude '.*/usr/.*'
              --exclude '.*/tests/.*'
              --gcov-ignore-parse-errors=all
              --xml --xml-pretty
              --output ${COVERAGE_OUTPUT_DIR}/coverage.xml
              --print-summary
      
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
      COMMENT "Generating XML code coverage report"
      VERBATIM
    )

    message(STATUS "Added 'coverage_gcovr' and 'coverage_xml' targets (gcovr). Run with: make coverage_gcovr")
  endif()

  # Target: coverage_clean - Clean all coverage data
  add_custom_target(coverage_clean
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning coverage data..."
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${COVERAGE_OUTPUT_DIR}
    COMMAND find ${CMAKE_BINARY_DIR} -type f -name '*.gcda' -delete || true
    COMMAND find ${CMAKE_BINARY_DIR} -type f -name '*.gcno' -delete || true
    COMMAND find ${CMAKE_BINARY_DIR} -type f -name '*.gcov' -delete || true
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all coverage data"
    VERBATIM
  )

  message(STATUS "Added 'coverage_clean' target to clean coverage data")
endfunction()
