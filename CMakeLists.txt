cmake_minimum_required(VERSION 3.25.0)

project(safire VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")
set(PROJECT_CMAKE ${CMAKE_SOURCE_DIR}/cmake)

set (LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib CACHE PATH "Library directory.")
set (EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin CACHE PATH "Binary directory.")

######################################################################
# CUDA/HIP build options. Read first, to be able to set policies! 
######################################################################
option(ENABLE_CUDA " Build device code with CUDA backend." OFF)
option(ENABLE_HIP " Build device code with HIP backend." OFF)

######################################################################
# CMake Policies 
######################################################################
# find_package uses <PackageName>_ROOT variables
cmake_policy(SET CMP0074 NEW)
# Include file check macros honor CMAKE_REQUIRED_LIBRARIES. 
cmake_policy(SET CMP0075 NEW)
# Initialize CMAKE_CUDA_ARCHITECTURES when CMAKE_CUDA_COMPILER_ID is NVIDIA. 
if(ENABLE_CUDA OR ENABLE_HIP)
 cmake_policy(SET CMP0104 NEW)
 if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
  cmake_policy(SET CMP0146 NEW)
 endif()
endif()

# Use both BOOST_ROOT and Boost_ROOT variables
if (CMAKE_VERSION VERSION_GREATER_EQUAL "3.27.0")
  cmake_policy(SET CMP0144 NEW)
endif()
# Do not use FindBoost.cmake
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30.0")
  cmake_policy(SET CMP0167 NEW)
endif()

include(CMakeDependentOption)
include(FetchContent)

######################################################################
# CTest
######################################################################
SET(PROJECT_UNIT_TEST_DIR ${PROJECT_BINARY_DIR}/tests/bin)
enable_testing()
include(CTest)

######################################################################
# Code Coverage
######################################################################
if(ENABLE_COVERAGE)
  include(CodeCoverage)
  append_coverage_compiler_flags()
endif()


######################################################################
# Build level
######################################################################
if ( NOT CMAKE_BUILD_TYPE ) 
  set( CMAKE_BUILD_TYPE Release )
endif()

######################################################################
# Check C++ standard and stdlib. Currently requires 20. 
######################################################################
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

######################################################################
# set options
######################################################################
option(ENABLE_ROCM " Build device code with ROCM backend" OFF)
option(HAVE_MKL "Assume MKL BLAS/LAPACK library." OFF)
option(BUILD_UNIT_TESTS "Build unit tests" ON)
option(BUILD_STATIC "Link to static libraries" OFF)
OPTION(BUILD_DOCUMENTATION "Enable compiling C++ Docs" OFF)
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)
option(ENABLE_SPDLOG "Build with spdlog" ON)
option(ENABLE_CPPTRACE "Build with cpptrace support" ON)
option(ENABLE_ASAN "Compile with AddressSanitizer for host C/C++ code" OFF)
option(ENABLE_UBSAN "Compile with UndefinedBehaviorSanitizer for host C/C++ code" OFF)

# Keep dependency-local sanitizer toggles disabled. SAFIRE controls sanitizer
# flags globally through ENABLE_ASAN/ENABLE_UBSAN to avoid toolchain conflicts.
if(ENABLE_ASAN OR ENABLE_UBSAN)
  set(ASAN OFF CACHE BOOL "Disable dependency-local ASAN option" FORCE)
  set(UBSAN OFF CACHE BOOL "Disable dependency-local UBSAN option" FORCE)
endif()

#option(ENABLE_FFTW "Build using FFTW3 library." ON)
option(COMPILE_NDA_TESTS "Build with nda tests" OFF)
option(NDA_PYTHON_SUPPORT "Build nda with python support" OFF)
option(FETCH_CCCL "Fetch CCCL and enable support. Only used with ENABLE_CUDA" ON)

if( ENABLE_CUDA OR ENABLE_HIP ) #OR ENABLE_OPENMP)
  set(ENABLE_DEVICE ON)
else()
  set(ENABLE_DEVICE OFF)
endif() 
set(ENABLE_TBLIS ON) 

######################################################################
# CUDA
######################################################################
if(ENABLE_CUDA)
  enable_language(CUDA)
  if(DEFINED CUDA_ARCH)
    set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCH})
  else()
    set(CMAKE_CUDA_ARCHITECTURES 80)
  endif()
  find_package(CUDAToolkit 12.0.0 REQUIRED)
  if(NOT FETCH_CCCL)
    find_package(CCCL REQUIRED)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
#  set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
  string(APPEND CMAKE_CUDA_FLAGS " -std=c++20 --expt-relaxed-constexpr --extended-lambda -Xcudafe \"--diag_suppress=implicit_return_from_non_void_function\" -Wdeprecated-declarations -w") 
  message( STATUS " CMAKE_CUDA_FLAGS: " ${CMAKE_CUDA_FLAGS})
endif()

######################################################################
# ROCM 
######################################################################
if(ENABLE_ROCM)
  message(STATUS "ROCM_ROOT: ${ROCM_ROOT}")
  add_library(ROCM::libraries INTERFACE IMPORTED)
  # temporarily put rocsolver rocrand here for convenience, should be moved to Platforms.
  set_target_properties(ROCM::libraries PROPERTIES INTERFACE_INCLUDE_DIRECTORIES  "${ROCM_ROOT}/include"
                                                   INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lrocsolver;-lrocrand")
endif(ENABLE_ROCM)

######################################################################
#  HIP 
######################################################################
if(ENABLE_HIP)
  if(NOT ENABLE_ROCM)
    message(FATAL_ERROR "ROCM is required to use HIP. Please set ENABLE_ROCM=ON.")
  endif()
  set(CMAKE_MODULE_PATH "${ROCM_ROOT}/hip/cmake" ${CMAKE_MODULE_PATH})
  find_package(HIP REQUIRED)

  add_library(HIP::HIP INTERFACE IMPORTED)
  # temporarily put hipsparse hipblas here for convenience, should be moved to Platforms.
  set_target_properties(HIP::HIP PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ROCM_ROOT}/include"
                                            INTERFACE_COMPILE_DEFINITIONS "ENABLE_HIP"
                                            INTERFACE_LINK_LIBRARIES "-L${ROCM_ROOT}/lib;-lhipsparse;-lhipblas;-lamdhip64")
endif(ENABLE_HIP)


######################################################################
# MPI
######################################################################
find_package(MPI REQUIRED)
message(STATUS "MPI Executable for running programs:" ${MPIEXEC_EXECUTABLE})
message(STATUS "MPI number of processors detected on the host system: " ${MPIEXEC_MAX_NUMPROCS})
include(CMakePrintHelpers)
cmake_print_properties(TARGETS MPI::MPI_CXX PROPERTIES INTERFACE_LINK_LIBRARIES INTERFACE_INCLUDE_DIRECTORIES)

######################################################################
# HDF5
# we still use some of the raw HDF5 API. Remove once we no longer do.
######################################################################
find_package(HDF5 REQUIRED C HL)

######################################################################
# nda: a C++ library providing multi-dimensional array classes
######################################################################
if(DEFINED NDA)
  set(FETCHCONTENT_SOURCE_DIR_NDA ${NDA})
endif()
# set nda options before FetchContent_Declare
block()
  set(CutensorSupport ${ENABLE_CUDA})
  set(Cutensor_USE_STATIC ON)
  set(CudaSupport ${ENABLE_CUDA})
  set(TblisSupport ON)
  set(Build_Tests ${COMPILE_NDA_TESTS})
  set(PythonSupport OFF)
  FetchContent_Declare(
          nda
          GIT_REPOSITORY "https://github.com/triqs/nda.git"
          GIT_TAG        tensor
          SYSTEM
  )
  FetchContent_MakeAvailable(nda)
endblock()

######################################################################
# CCCL from github (useful if CudaToolkit is old)
######################################################################
if(ENABLE_CUDA AND FETCH_CCCL)
  # set nevanlinna options before FetchContent_Declare
  FetchContent_Declare(
        cccl
        GIT_REPOSITORY "https://github.com/NVIDIA/cccl.git"
        GIT_TAG        v3.3.3
        # do not add SYSTEM here because then the CUDA-included headers may have precedence
  )
  FetchContent_MakeAvailable(cccl)
endif()


######################################################################
# Boost: Header-only
######################################################################
find_package(Boost 1.70.0 REQUIRED CONFIG)

######################################################################
# spdlog - disable warnings for external library
######################################################################
# Save current warning flags
if(ENABLE_SPDLOG)
  if(DEFINED SPDLOG) 
    set(FETCHCONTENT_SOURCE_DIR_SPDLOG ${SPDLOG})
  endif()

  set(SPDLOG_BUILD_SHARED OFF)

  FetchContent_Declare(
          spdlog
          GIT_REPOSITORY https://github.com/gabime/spdlog.git
          GIT_TAG        v1.8.3
          SYSTEM
  )
  FetchContent_MakeAvailable(spdlog)
endif()

######################################################################
# cpptrace 
######################################################################
if(ENABLE_CPPTRACE)
  FetchContent_Declare(
    cpptrace
    GIT_REPOSITORY https://github.com/jeremy-rifkin/cpptrace.git
    GIT_TAG        v0.8.2
    SYSTEM
  )
  FetchContent_MakeAvailable(cpptrace)
endif(ENABLE_CPPTRACE)

######################################################################
# cxxopts 
######################################################################
find_package(cxxopts 3.3.1 QUIET)
if(NOT cxxopts_FOUND)
  FetchContent_Declare(
    cxxopts
    GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
    GIT_TAG v3.3.1
    SYSTEM
  )
  FetchContent_MakeAvailable(cxxopts)
endif()

######################################################################
# FFTW: not using MKL's FFTW yet, figure out how if requested 
######################################################################
#if(ENABLE_FFTW)
#  find_package(FFTW REQUIRED DOUBLE_OPENMP_LIB) 
#  add_compile_definitions(ENABLE_FFTW)
#endif()


######################################################################
# catch2
######################################################################
find_package(Catch2 3.14.0 QUIET)
if(NOT Catch2_FOUND)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.14.0
  )
  FetchContent_MakeAvailable(Catch2)
endif()

list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/contrib)
include(Catch)

######################################################################
# mpi3
######################################################################

# We only need the headers, it seems.
add_library(mpi3 INTERFACE)
target_link_libraries(mpi3 INTERFACE MPI::MPI_CXX)
target_include_directories(mpi3 INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/extern/mpi_wrapper)


######################################################################
configure_file(${PROJECT_SOURCE_DIR}/src/safire.settings ${PROJECT_BINARY_DIR}/bin/safire.settings)
install(FILES ${PROJECT_BINARY_DIR}/bin/safire.settings DESTINATION bin)

#Create an Interface target for compiler warnings
add_library(${PROJECT_NAME}_warnings INTERFACE)
target_compile_options(${PROJECT_NAME}_warnings
  INTERFACE
   -Wall
   -Wextra
   -Wfloat-conversion
   -Wpedantic
   -Wno-sign-compare
   -Wno-unused-parameter
    $<$<CXX_COMPILER_ID:GNU>:-Wno-attributes>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-deprecated-comma-subscript>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-deprecated-declarations>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-unknown-warning-option>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-gcc-compat>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-c++20-extensions>
    $<$<CXX_COMPILER_ID:AppleClang,Clang,IntelLLVM>:-Wno-c++20-compat>
    $<$<CXX_COMPILER_ID:IntelLLVM>:-Wno-tautological-constant-compare>
    # CUDA NVCC compiler warnings
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
    $<$<COMPILE_LANGUAGE:CUDA>:--Werror all-warnings>
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=20208> # esa_on_defaulted_function_ignored
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-error=128> # unreachable_code
)

#######################################################################
# Sanitizers
########################################################################
if(ENABLE_ASAN OR ENABLE_UBSAN)
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM")
    message(FATAL_ERROR "Sanitizers require a GCC/Clang-compatible host compiler")
  endif()

  if(ENABLE_CUDA OR ENABLE_HIP)
    message(FATAL_ERROR "ENABLE_ASAN/ENABLE_UBSAN currently support host-only builds. Disable CUDA/HIP when using sanitizers.")
  endif()

  if(CMAKE_BUILD_TYPE STREQUAL "RELEASE")
    message(WARNING "Sanitizers are best used with Debug/RelWithDebInfo builds")
  endif()

  # Apply sanitizer flags only to host C/C++ sources.
  add_compile_options(
    $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fno-omit-frame-pointer>
    $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-g>
  )
  add_link_options(-fno-omit-frame-pointer)

  if(ENABLE_ASAN)
    add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=address>)
    add_link_options(-fsanitize=address)
    message(STATUS "AddressSanitizer enabled")
  endif()

  if(ENABLE_UBSAN)
    add_compile_options(
      $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=undefined>
      $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=float-divide-by-zero>
      $<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-fsanitize=float-cast-overflow>
    )
    add_link_options(-fsanitize=undefined -fsanitize=float-divide-by-zero -fsanitize=float-cast-overflow)
    message(STATUS "UndefinedBehaviorSanitizer enabled")
  endif()
endif()

#######################################################################
# Git info
########################################################################
find_package(Git)

if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
  set(IS_GIT_PROJECT 1)
endif()

add_subdirectory(src)

######################################################################
# Setup code coverage targets
######################################################################
if(ENABLE_COVERAGE)
  setup_coverage_targets()
endif()

#######################################################################
# C++ Docs
########################################################################
if(NOT IS_SUBPROJECT AND BUILD_DOCUMENTATION)
  add_subdirectory(${PROJECT_SOURCE_DIR}/docs_cpp)
endif()

