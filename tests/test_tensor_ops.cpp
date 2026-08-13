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
#include "numerics/operations/tensor.hpp"
#include "numerics/operations/add_diagonal.hpp"

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

namespace
{

/// Elementwise cast on the host, by logical index, so the result does not depend on the layout of
/// either side.
template<typename TB, nda::Array A_t>
auto host_cast(A_t const& A)
{
  nda::array<TB, nda::get_rank<A_t>> out(A.shape());
  out() = A();
  return out;
}

/**
 * Approx contracts a device operand with math::copy, which needs a same-type kernel for that value
 * type. cast_pairs.cuh has no float or complex<float> diagonal, so a float-valued result has to
 * come back to the host before it can be compared. On HOST_MEMORY this is a plain copy.
 */
template<nda::MemoryArray A_t>
auto readback(A_t const& A)
{
  return nda::to_host(A);
}

/// b += TB(alpha*a) by logical index: the product is formed in the source precision and rounded
/// once on the way into b, which is what the kernel does.
template<typename S, nda::Array A_t, nda::Array B_t>
void host_accumulate(S alpha, A_t const& a, B_t&& b)
{
  using TB = nda::get_value_t<std::decay_t<B_t>>;
  nda::for_each(a.shape(), [&](auto... i) { b(i...) += TB(alpha * a(i...)); });
}

/// A float destination carries ~7 digits, so it cannot meet Approx's 1e-8 default.
template<typename T>
constexpr RealType cast_tol()
{
  if constexpr(std::is_same_v<T, float> or std::is_same_v<T, std::complex<float>>) {
    return 1e-5;
  } else {
    return 1e-9;
  }
}

/**
 * math::copy picks its implementation from the value types, the stride orders and a runtime
 * contiguity test. Each block names the branch it exists to reach; under HOST_MEMORY they all
 * collapse onto B() = A(), which is what makes the expected values below independent of it.
 */
template<MEMORY_SPACE MEM>
void tensor_copy_paths()
{
  using nda::range;
  auto all = range::all;

  // flat memcpy: one value type, one stride order, both contiguous
  {
    nda::array<ComplexType, 3> A_h(3, 4, 5);
    utils::fillRandomArray(A_h);
    memory::array<MEM, ComplexType, 3> A(A_h), B(3, 4, 5);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(A_h));
  }

  // ... and a Fortran layout reaches it too, as long as both sides agree
  {
    nda::array<ComplexType, 2, nda::F_layout> A_h(4, 5);
    utils::fillRandomArray(A_h);
    memory::array<MEM, ComplexType, 2, nda::F_layout> A(A_h), B(4, 5);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(A_h));
  }

  // C into Fortran: the stride orders differ, so pairing by memory position would be wrong and
  // only the rank-R kernel, which pairs by index, can do it
  {
    nda::array<ComplexType, 2> A_h(4, 5);
    utils::fillRandomArray(A_h);
    memory::array<MEM, ComplexType, 2> A(A_h);
    memory::array<MEM, ComplexType, 2, nda::F_layout> B(4, 5);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(A_h));
  }

  // a cast still needs a kernel, but a contiguous pair is flattened so it is a rank-1 one
  {
    nda::array<double, 3> A_h(3, 4, 5);
    utils::fillRandomArray(A_h);
    memory::array<MEM, double, 3> A(A_h);
    memory::array<MEM, float, 3> B(3, 4, 5);
    math::copy(A, B);
    CHECK_THAT(nda::to_host(B), utils::Approx(host_cast<float>(A_h), cast_tol<float>(), cast_tol<float>()));
  }

  // the widening direction, and the real -> complex pair
  {
    nda::array<float, 2> A_h(4, 6);
    utils::fillRandomArray(A_h);
    memory::array<MEM, float, 2> A(A_h);
    memory::array<MEM, double, 2> B(4, 6);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(host_cast<double>(A_h), cast_tol<float>(), cast_tol<float>()));
  }
  {
    nda::array<double, 2> A_h(4, 6);
    utils::fillRandomArray(A_h);
    memory::array<MEM, double, 2> A(A_h);
    memory::array<MEM, ComplexType, 2> B(4, 6);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(host_cast<ComplexType>(A_h)));
  }
  {
    nda::array<ComplexType, 2> A_h(4, 6);
    utils::fillRandomArray(A_h);
    memory::array<MEM, ComplexType, 2> A(A_h);
    memory::array<MEM, std::complex<float>, 2> B(4, 6);
    math::copy(A, B);
    CHECK_THAT(nda::to_host(B), utils::Approx(host_cast<std::complex<float>>(A_h), cast_tol<float>(),
                                          cast_tol<float>()));
  }

  // strided views: contiguity fails at runtime, so this lands on the rank-R kernel. Checking the
  // whole backing array also catches a kernel that writes outside the slice.
  {
    nda::array<ComplexType, 3> Abig_h(6, 4, 5);
    utils::fillRandomArray(Abig_h);
    nda::array<ComplexType, 3> ref(6, 4, 5);
    ref() = 0.0;
    ref(range(0, 6, 2), all, all) = Abig_h(range(0, 6, 2), all, all);

    memory::array<MEM, ComplexType, 3> Abig(Abig_h), Bbig(6, 4, 5);
    Bbig() = 0.0;
    math::copy(Abig(range(0, 6, 2), all, all), Bbig(range(0, 6, 2), all, all));
    CHECK_THAT(Bbig, utils::Approx(ref));
  }

  // rank 1, where there is no layout to get wrong
  {
    nda::array<ComplexType, 1> A_h(17);
    utils::fillRandomArray(A_h);
    memory::array<MEM, ComplexType, 1> A(A_h), B(17);
    math::copy(A, B);
    CHECK_THAT(B, utils::Approx(A_h));
  }

  // exactly max_copy_rank, the last rank with an instantiated kernel. Strided, so it does not
  // shortcut to the memcpy.
  {
    nda::array<ComplexType, 6> Abig_h(4, 2, 2, 2, 2, 3);
    utils::fillRandomArray(Abig_h);
    nda::array<ComplexType, 6> ref(4, 2, 2, 2, 2, 3);
    ref() = 0.0;
    ref(range(0, 4, 2), all, all, all, all, all) = Abig_h(range(0, 4, 2), all, all, all, all, all);

    memory::array<MEM, ComplexType, 6> Abig(Abig_h), Bbig(4, 2, 2, 2, 2, 3);
    Bbig() = 0.0;
    math::copy(Abig(range(0, 4, 2), all, all, all, all, all),
               Bbig(range(0, 4, 2), all, all, all, all, all));
    CHECK_THAT(Bbig, utils::Approx(ref));
  }

  // one rank above it: no kernel exists, so the slowest dimension is peeled off and recursed on
  {
    nda::array<ComplexType, 7> Abig_h(4, 2, 2, 2, 2, 2, 3);
    utils::fillRandomArray(Abig_h);
    nda::array<ComplexType, 7> ref(4, 2, 2, 2, 2, 2, 3);
    ref() = 0.0;
    ref(range(0, 4, 2), all, all, all, all, all, all) =
        Abig_h(range(0, 4, 2), all, all, all, all, all, all);

    memory::array<MEM, ComplexType, 7> Abig(Abig_h), Bbig(4, 2, 2, 2, 2, 2, 3);
    Bbig() = 0.0;
    math::copy(Abig(range(0, 4, 2), all, all, all, all, all, all),
               Bbig(range(0, 4, 2), all, all, all, all, all, all));
    CHECK_THAT(Bbig, utils::Approx(ref));
  }

  // an empty array must be a no-op rather than a zero-length memcpy or launch
  {
    memory::array<MEM, ComplexType, 2> A(0, 5), B(0, 5);
    math::copy(A, B);
    CHECK(B.size() == 0);
  }
}

/**
 * math::accumulate has one more layer than copy: before the kernels it tries cublas axpy, which
 * applies whenever both operands are a single arithmetic sequence in memory.
 */
template<MEMORY_SPACE MEM>
void tensor_accumulate_paths()
{
  using nda::range;
  auto all = range::all;

  // axpy with unit increments: one value type, one stride order, both contiguous
  {
    nda::array<ComplexType, 2> A_h(4, 5), B_h(4, 5);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    ComplexType alpha(0.5, -0.25);
    nda::array<ComplexType, 2> ref(4, 5);
    ref() = B_h() + alpha * A_h();

    memory::array<MEM, ComplexType, 2> A(A_h), B(B_h);
    math::accumulate(alpha, A, B);
    CHECK_THAT(B, utils::Approx(ref));
  }

  // axpy with a non-unit increment: a column of a C-layout matrix is strided but still 1d, and
  // the two sides may sit at different offsets and strides
  {
    nda::array<ComplexType, 2> A_h(6, 5), B_h(6, 7);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    ComplexType alpha(1.5, 0.0);
    nda::array<ComplexType, 2> ref(B_h);
    ref(all, 3) = B_h(all, 3) + alpha * A_h(all, 2);

    memory::array<MEM, ComplexType, 2> A(A_h), B(B_h);
    math::accumulate(alpha, A(all, 2), B(all, 3));
    CHECK_THAT(B, utils::Approx(ref));
  }

  // a cast cannot go through axpy, but a contiguous pair still flattens to a rank-1 kernel
  {
    nda::array<double, 3> A_h(3, 4, 5);
    nda::array<float, 3> B_h(3, 4, 5);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    double alpha = 0.75;
    nda::array<float, 3> ref(B_h);
    host_accumulate(alpha, A_h, ref);

    memory::array<MEM, double, 3> A(A_h);
    memory::array<MEM, float, 3> B(B_h);
    math::accumulate(alpha, A, B);
    CHECK_THAT(readback(B), utils::Approx(ref, cast_tol<float>(), cast_tol<float>()));
  }

  // strided on both sides and not 1d: no axpy, no flatten, so the rank-R kernel
  {
    nda::array<ComplexType, 3> A_h(6, 4, 5), B_h(6, 4, 5);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    ComplexType alpha(-0.5, 0.75);
    nda::array<ComplexType, 3> ref(B_h);
    ref(range(0, 6, 2), all, all) =
        B_h(range(0, 6, 2), all, all) + alpha * A_h(range(0, 6, 2), all, all);

    memory::array<MEM, ComplexType, 3> A(A_h), B(B_h);
    math::accumulate(alpha, A(range(0, 6, 2), all, all), B(range(0, 6, 2), all, all));
    CHECK_THAT(B, utils::Approx(ref));
  }

  // exactly max_accumulate_rank
  {
    nda::array<ComplexType, 4> A_h(6, 3, 2, 2), B_h(6, 3, 2, 2);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    ComplexType alpha(1.0, 0.5);
    nda::array<ComplexType, 4> ref(B_h);
    ref(range(0, 6, 2), all, all, all) =
        B_h(range(0, 6, 2), all, all, all) + alpha * A_h(range(0, 6, 2), all, all, all);

    memory::array<MEM, ComplexType, 4> A(A_h), B(B_h);
    math::accumulate(alpha, A(range(0, 6, 2), all, all, all), B(range(0, 6, 2), all, all, all));
    CHECK_THAT(B, utils::Approx(ref));
  }

  // above it, with one value type: cutensor takes over
  {
    nda::array<ComplexType, 5> A_h(6, 3, 2, 2, 2), B_h(6, 3, 2, 2, 2);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    ComplexType alpha(0.25, -1.0);
    nda::array<ComplexType, 5> ref(B_h);
    ref(range(0, 6, 2), all, all, all, all) =
        B_h(range(0, 6, 2), all, all, all, all) + alpha * A_h(range(0, 6, 2), all, all, all, all);

    memory::array<MEM, ComplexType, 5> A(A_h), B(B_h);
    math::accumulate(alpha, A(range(0, 6, 2), all, all, all, all),
                     B(range(0, 6, 2), all, all, all, all));
    CHECK_THAT(B, utils::Approx(ref));
  }

  // above it with two value types, which cutensor cannot express: peel and recurse until the rank
  // fits a kernel
  {
    nda::array<double, 5> A_h(6, 3, 2, 2, 2);
    nda::array<float, 5> B_h(6, 3, 2, 2, 2);
    utils::fillRandomArray(A_h);
    utils::fillRandomArray(B_h);
    double alpha = -0.5;
    nda::array<float, 5> ref(B_h);
    host_accumulate(alpha, A_h(range(0, 6, 2), all, all, all, all),
                    ref(range(0, 6, 2), all, all, all, all));

    memory::array<MEM, double, 5> A(A_h);
    memory::array<MEM, float, 5> B(B_h);
    math::accumulate(alpha, A(range(0, 6, 2), all, all, all, all),
                     B(range(0, 6, 2), all, all, all, all));
    CHECK_THAT(readback(B), utils::Approx(ref, cast_tol<float>(), cast_tol<float>()));
  }

  // an empty array returns before any of it
  {
    memory::array<MEM, ComplexType, 2> A(0, 5), B(0, 5);
    math::accumulate(1.0, A, B);
    CHECK(B.size() == 0);
  }
}

/// a(...) = alpha + beta*a(...), with alpha given either as a value or in memory.
template<MEMORY_SPACE MEM>
void tensor_add_scalar()
{
  using nda::range;
  auto all = range::all;

  nda::array<ComplexType, 1> a_h(8), alpha_h(1);
  utils::fillRandomArray(a_h);
  utils::fillRandomArray(alpha_h);
  const ComplexType alpha_v(-0.75, 0.5);

  // alpha in memory, on a prefix: the tail has to come back untouched
  {
    const long n = 5;
    nda::array<ComplexType, 1> ref(a_h);
    for(long i = 0; i < n; ++i) {
      ref(i) += alpha_h(0);
    }

    memory::array<MEM, ComplexType, 1> a(a_h), alpha(alpha_h);
    math::add_scalar(alpha, 1.0, a(range(0, n)));
    CHECK_THAT(a, utils::Approx(ref));
  }

  // ... and on the whole vector
  {
    nda::array<ComplexType, 1> ref(a_h);
    ref() += alpha_h(0);

    memory::array<MEM, ComplexType, 1> a(a_h), alpha(alpha_h);
    math::add_scalar(alpha, 1.0, a);
    CHECK_THAT(a, utils::Approx(ref));
  }

  // an empty target touches nothing
  {
    memory::array<MEM, ComplexType, 1> a(a_h), alpha(alpha_h);
    math::add_scalar(alpha, 1.0, a(range(0, 0)));
    CHECK_THAT(a, utils::Approx(a_h));
  }

  // alpha as a value, with beta scaling what is already there
  {
    const ComplexType beta(0.25, -1.5);
    nda::array<ComplexType, 1> ref(8);
    for(long i = 0; i < 8; ++i) {
      ref(i) = alpha_v + beta * a_h(i);
    }

    memory::array<MEM, ComplexType, 1> a(a_h);
    math::add_scalar(alpha_v, beta, a);
    CHECK_THAT(a, utils::Approx(ref));
  }

  // beta == 0 is a fill: a is overwritten, not read
  {
    nda::array<ComplexType, 1> ref(8);
    ref() = alpha_v;

    memory::array<MEM, ComplexType, 1> a(a_h);
    math::add_scalar(alpha_v, 0.0, a);
    CHECK_THAT(a, utils::Approx(ref));
  }

  // ... including onto a strided view of rank > 1, which is the layout nda cannot fill itself
  {
    nda::array<ComplexType, 3> A_h(6, 4, 5);
    utils::fillRandomArray(A_h);
    nda::array<ComplexType, 3> ref(A_h);
    ref(range(0, 6, 2), all, all) = alpha_v;

    memory::array<MEM, ComplexType, 3> A(A_h);
    math::add_scalar(alpha_v, 0.0, A(range(0, 6, 2), all, all));
    CHECK_THAT(A, utils::Approx(ref));
  }

  // a real value type, the other instantiated one
  {
    nda::array<double, 2> A_h(3, 4);
    utils::fillRandomArray(A_h);
    nda::array<double, 2> ref(3, 4);
    for(long i = 0; i < 3; ++i) {
      for(long j = 0; j < 4; ++j) {
        ref(i, j) = 2.0 + 0.5 * A_h(i, j);
      }
    }

    memory::array<MEM, double, 2> A(A_h);
    math::add_scalar(2.0, 0.5, A);
    CHECK_THAT(A, utils::Approx(ref));
  }
}

template<MEMORY_SPACE MEM>
void tensor_zero_imag()
{
  using nda::range;
  auto all = range::all;

  // contiguous: reshaped to rank 1 and done by the rank-1 kernel
  {
    nda::array<ComplexType, 2> A_h(4, 5);
    utils::fillRandomArray(A_h);
    nda::array<ComplexType, 2> ref(4, 5);
    for(long i = 0; i < 4; ++i) {
      for(long j = 0; j < 5; ++j) {
        ref(i, j) = ComplexType(A_h(i, j).real(), 0.0);
      }
    }

    memory::array<MEM, ComplexType, 2> A(A_h);
    math::zero_imag(A);
    CHECK_THAT(A, utils::Approx(ref));
  }

  // strided: no reshape, so the rank-R kernel. The rows outside the slice keep their imaginary
  // part, which is what makes this different from the contiguous case.
  {
    nda::array<ComplexType, 2> A_h(6, 5);
    utils::fillRandomArray(A_h);
    nda::array<ComplexType, 2> ref(A_h);
    for(long i = 0; i < 6; i += 2) {
      for(long j = 0; j < 5; ++j) {
        ref(i, j) = ComplexType(A_h(i, j).real(), 0.0);
      }
    }

    memory::array<MEM, ComplexType, 2> A(A_h);
    math::zero_imag(A(range(0, 6, 2), all));
    CHECK_THAT(A, utils::Approx(ref));
  }

  // a real array has no imaginary part to clear, so this compiles to nothing
  {
    nda::array<double, 2> A_h(4, 5);
    utils::fillRandomArray(A_h);
    memory::array<MEM, double, 2> A(A_h);
    math::zero_imag(A);
    CHECK_THAT(A, utils::Approx(A_h));
  }
}

/// A(b,i,i) += alpha, i.e. the diagonal view handed to add_scalar with beta = 1.
template<MEMORY_SPACE MEM>
void tensor_add_diagonal()
{
  nda::array<ComplexType, 3> A_h(3, 4, 4);
  utils::fillRandomArray(A_h);
  const ComplexType alpha(1.5, -0.25);

  nda::array<ComplexType, 3> ref(A_h);
  for(long b = 0; b < 3; ++b) {
    for(long i = 0; i < 4; ++i) {
      ref(b, i, i) += alpha;
    }
  }

  memory::array<MEM, ComplexType, 3> A(A_h);
  math::add_diagonal(alpha, A);
  CHECK_THAT(A, utils::Approx(ref));
}

/// set_identity writes through a diagonal view, which for rank > 2 is the diagonal of each
/// trailing square block.
template<MEMORY_SPACE MEM>
void tensor_set_identity()
{
  {
    nda::array<ComplexType, 2> ref(4, 4);
    ref() = 0.0;
    for(long i = 0; i < 4; ++i) {
      ref(i, i) = 1.0;
    }

    memory::array<MEM, ComplexType, 2> A(4, 4);
    math::set_identity(A);
    CHECK_THAT(A, utils::Approx(ref));
  }

  {
    nda::array<ComplexType, 3> ref(3, 4, 4);
    ref() = 0.0;
    for(long b = 0; b < 3; ++b) {
      for(long i = 0; i < 4; ++i) {
        ref(b, i, i) = 1.0;
      }
    }

    memory::array<MEM, ComplexType, 3> A(3, 4, 4);
    math::set_identity(A);
    CHECK_THAT(A, utils::Approx(ref));
  }

  {
    nda::array<double, 4> ref(2, 3, 4, 4);
    ref() = 0.0;
    for(long b = 0; b < 2; ++b) {
      for(long c = 0; c < 3; ++c) {
        for(long i = 0; i < 4; ++i) {
          ref(b, c, i, i) = 1.0;
        }
      }
    }

    memory::array<MEM, double, 4> A(2, 3, 4, 4);
    math::set_identity(A);
    CHECK_THAT(A, utils::Approx(ref));
  }
}

} // namespace

TEST_CASE("tensor: copy", "[tensor]")
{
  tensor_copy_paths<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_copy_paths<DEVICE_MEMORY>();
  tensor_copy_paths<UNIFIED_MEMORY>();
#endif
}

TEST_CASE("tensor: accumulate", "[tensor]")
{
  tensor_accumulate_paths<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_accumulate_paths<DEVICE_MEMORY>();
  tensor_accumulate_paths<UNIFIED_MEMORY>();
#endif
}

TEST_CASE("tensor: add_scalar", "[tensor]")
{
  tensor_add_scalar<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_add_scalar<DEVICE_MEMORY>();
  tensor_add_scalar<UNIFIED_MEMORY>();
#endif
}

TEST_CASE("tensor: zero_imag", "[tensor]")
{
  tensor_zero_imag<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_zero_imag<DEVICE_MEMORY>();
  tensor_zero_imag<UNIFIED_MEMORY>();
#endif
}

TEST_CASE("tensor: add_diagonal", "[tensor]")
{
  tensor_add_diagonal<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_add_diagonal<DEVICE_MEMORY>();
  tensor_add_diagonal<UNIFIED_MEMORY>();
#endif
}

TEST_CASE("tensor: set_identity", "[tensor]")
{
  tensor_set_identity<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  tensor_set_identity<DEVICE_MEMORY>();
  tensor_set_identity<UNIFIED_MEMORY>();
#endif
}

} // namespace afqmc
} // namespace sfqmc
