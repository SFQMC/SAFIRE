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
#include "AFQMC/config.h"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "test_common.hpp"
#include "AFQMC/HamiltonianOperations/detail/one_body.hpp"

#include <complex>

namespace sfqmc
{
namespace afqmc
{

using namespace std::complex_literals;

// broadcast_one_body copies a 5D one-body operator of shape
// [nspin][npol][NMO][npol][NMO] into a destination of a lesser spin symmetry
// (CLOSED <= COLLINEAR <= NONCOLLINEAR), replicating/reshuffling the spin and
// polarization blocks. Walker types are inferred from dims:
//   CLOSED       -> nspin=1, npol=1
//   COLLINEAR    -> nspin=2, npol=1
//   NONCOLLINEAR -> nspin=1, npol=2
TEST_CASE("one_body: broadcast_one_body", "[one_body]")
{
  using Type = ComplexType;
  auto all = nda::range::all;
  const int NMO = 2;

  // Distinct NMO x NMO blocks used to build the source operators.
  nda::array<Type, 2> A{{1.0 + 0.5i, 2.0 - 1.0i}, {0.4 - 0.2i, 3.0 + 0.7i}};
  nda::array<Type, 2> B{{5.0 - 0.1i, 6.0 + 0.3i}, {7.0 + 0.8i, 8.0 - 0.6i}};
  nda::array<Type, 2> C{{9.0 + 0.2i, 1.5 - 0.4i}, {2.5 + 0.9i, 3.5 - 0.3i}};
  nda::array<Type, 2> D{{4.5 - 0.7i, 5.5 + 0.6i}, {6.5 - 0.5i, 7.5 + 0.1i}};

  // 1) CLOSED -> CLOSED: identity copy.
  {
    nda::array<Type, 5> src(1, 1, NMO, 1, NMO);
    src(0, 0, all, 0, all) = A;

    nda::array<Type, 5> dest(1, 1, NMO, 1, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(src));
  }

  // 2) CLOSED -> COLLINEAR: closed block replicated into both spin channels.
  {
    nda::array<Type, 5> src(1, 1, NMO, 1, NMO);
    src(0, 0, all, 0, all) = A;

    nda::array<Type, 5> expected(2, 1, NMO, 1, NMO);
    expected(0, 0, all, 0, all) = A;
    expected(1, 0, all, 0, all) = A;

    nda::array<Type, 5> dest(2, 1, NMO, 1, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(expected));
  }

  // 3) CLOSED -> NONCOLLINEAR: closed block on both pol-diagonal blocks,
  //    off-diagonal pol blocks zero.
  {
    nda::array<Type, 5> src(1, 1, NMO, 1, NMO);
    src(0, 0, all, 0, all) = A;

    nda::array<Type, 5> expected(1, 2, NMO, 2, NMO);
    expected() = 0;
    expected(0, 0, all, 0, all) = A;
    expected(0, 1, all, 1, all) = A;

    nda::array<Type, 5> dest(1, 2, NMO, 2, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(expected));
  }

  // 4) COLLINEAR -> NONCOLLINEAR: spin-up -> pol block (0,0), spin-down -> pol
  //    block (1,1), off-diagonal pol blocks zero.
  {
    nda::array<Type, 5> src(2, 1, NMO, 1, NMO);
    src(0, 0, all, 0, all) = A;  // spin up
    src(1, 0, all, 0, all) = B;  // spin down

    nda::array<Type, 5> expected(1, 2, NMO, 2, NMO);
    expected() = 0;
    expected(0, 0, all, 0, all) = A;
    expected(0, 1, all, 1, all) = B;

    nda::array<Type, 5> dest(1, 2, NMO, 2, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(expected));
  }

  // 5) COLLINEAR -> COLLINEAR: identity copy.
  {
    nda::array<Type, 5> src(2, 1, NMO, 1, NMO);
    src(0, 0, all, 0, all) = A;
    src(1, 0, all, 0, all) = B;

    nda::array<Type, 5> dest(2, 1, NMO, 1, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(src));
  }

  // 6) NONCOLLINEAR -> NONCOLLINEAR: identity copy, including off-diagonal pol
  //    blocks.
  {
    nda::array<Type, 5> src(1, 2, NMO, 2, NMO);
    src(0, 0, all, 0, all) = A;
    src(0, 0, all, 1, all) = B;
    src(0, 1, all, 0, all) = C;
    src(0, 1, all, 1, all) = D;

    nda::array<Type, 5> dest(1, 2, NMO, 2, NMO);
    broadcast_one_body(src, dest);
    CHECK_THAT(dest, utils::Approx(src));
  }
}

} // namespace afqmc
} // namespace sfqmc
