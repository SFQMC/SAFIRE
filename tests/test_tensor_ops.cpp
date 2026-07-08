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

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "configuration.hpp"
#include "utilities/check.hpp"
#include "test_common.hpp"
#include "numerics/device_kernels/kernels.h"
#include "numerics/operations/tensor.hpp"

#include <complex>
#include <cmath>

namespace sfqmc
{
namespace afqmc
{

using namespace std::complex_literals;

TEST_CASE("tensor: hermitize", "[tensor]")
{
  using Type = ComplexType;
  auto all = nda::range::all;

  // 1) Known 3x3 block (rank-3 with a single block), hand-computed reference.
  {
    nda::array<Type, 3> A(1, 3, 3);
    A(0, nda::ellipsis{}) = nda::array<Type, 2>{
        {1.0 + 0.5i, 2.0 + 1.0i, 0.5 - 0.3i},
        {0.4 - 0.2i, 3.0 - 0.7i, 1.1 + 0.9i},
        {0.8 + 0.6i, 2.2 + 0.1i, 2.0 + 0.4i}};

    nda::array<Type, 3> expected(1, 3, 3);
    expected(0, nda::ellipsis{}) = nda::array<Type, 2>{
        {1.0 + 0.0i, 1.20 + 0.60i, 0.65 - 0.45i},
        {1.20 - 0.60i, 3.0 + 0.0i, 1.65 + 0.40i},
        {0.65 + 0.45i, 1.65 - 0.40i, 2.0 + 0.0i}};

    math::hermitize(A);
    CHECK_THAT(A, utils::Approx(expected));
  }

  // 2) Random batched rank-3 array: result must equal the Hermitian part, and
  //    every block diagonal must be real.
  {
    const int nb = 2;
    const int m  = 4;
    nda::array<Type, 3> A = utils::make_random<Type>(nb, m, m);
    auto M = A;  // keep the original

    math::hermitize(A);
    for(int i = 0; i < A.extent(0); i++) {
      auto ref = nda::make_regular(0.5 * A(i, all, all) + 0.5 * nda::conj(nda::transpose(A(i, all, all))));
      CHECK_THAT(A(i, all, all), utils::Approx(ref));
    }

    for(int k = 0; k < nb; k++) {
      for(int i = 0; i < m; i++) {
        CHECK(std::abs(A(k, i, i).imag()) < 1e-12);
      }
    }
  }

  // 3) Rank-4 array (2,2,3,3): check idempotency of hermitize
  {
    const int m = 3;
    nda::array<Type, 4> A4(2, 2, m, m);
    utils::fillRandomArray(A4);
    
    math::hermitize(A4);
    auto M4 = A4;
    math::hermitize(A4);
    
    CHECK_THAT(M4, utils::Approx(A4));
  }
}

} // namespace afqmc
} // namespace sfqmc
