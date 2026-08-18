////////////////////////////////////////////////////////////////////////////////
// This file is distributed under the Apache License, Version 2.0 License.
// See LICENSE file in top directory for details.
//
// Copyright (c) 2021-2025 The Simons Foundation, Inc.
//
// You may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// This file includes portions derived from work licensed under the
// University of Illinois/NCSA Open Source License. See the NOTICE file
// and LICENSES/NCSA.txt for details.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <complex>

#if defined __INTEL_COMPILER
#pragma warning disable 2196
#endif

#define BOOST_NO_AUTO_PTR

#define ADD_TESTS_TIMERS

#define AFQMC_DEBUG 3
#define AFQMC_TIMER

#define MAXIMUM_EMPLACE_BUFFER_SIZE 102400

// maximum size in Bytes for a dataset with walker data on WalkerIO
#define WALKER_HDF_BLOCK_SIZE 100000000

// maximum size in Bytes for a block of data in CSR matrix HDF IO
#define CSR_HDF_BLOCK_SIZE 2000000

// careful here that RealType is consistent with this!!!
#define MKL_INT int
#define MKL_Complex8 std::complex<float>
#define MKL_Complex16 std::complex<double>

#define byRows 999
#define byCols 111

namespace sfqmc
{
namespace afqmc
{

typedef int IndexType;
typedef int OrbitalType;
typedef double RealType;
typedef float FloatType;

typedef std::complex<RealType> ComplexType;
typedef std::complex<FloatType> ComplexFloat;

} // namespace afqmc
} // namespace sfqmc


