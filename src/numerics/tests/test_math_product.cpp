/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#undef NDEBUG

#include <complex>

#include "catch2/catch_test_macros.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "test_common.hpp"
#include "numerics/operations/product.hpp"
#include "numerics/sparse/sparse.hpp"

using sfqmc::utils::make_random;

namespace sfqmc::afqmc {

template<typename T>
auto input_arrays() {
  nda::array<T, 2> A{{1.0, 2, 3.0}, {1.0, 4.0, 5}};
  nda::array<T, 2> B{{1, 0}, {2.0, 3}, {1, 1}};
  auto C = nda::array<T,2>::zeros(2,2);
  nda::array<T, 2> C_expected{{8, 9}, {14, 17}};

  return std::make_tuple(A, B, C, C_expected);
}


template<>
auto input_arrays<std::complex<double>>() {
  using namespace std::complex_literals;
  using T = std::complex<double>;

  nda::array<T, 2> A{{1.0 + 1i, 2, 3.0 - 1i}, {1.0 - 1i, 4.0 + 1i, 5}};
  nda::array<T, 2> B{{1, 1i}, {2.0-1i, 3}, {1, 1}};
  auto C = nda::array<T,2>::zeros(2,2);
  nda::array<T, 2> C_expected{{8.0 - 2i, 8}, {15.0 - 3i, 18.0 + 4i}};

  return std::make_tuple(A, B, C, C_expected);
}

template<char op, typename Mat>
decltype(auto) with_op(Mat&& a) {
  if constexpr (op == 'N') {
    return std::forward<Mat>(a);
  } else if constexpr (op == 'T') {
    return nda::transpose(a);
  } else {
    return nda::dagger(a);
  }
}

template<MEMORY_SPACE MEM, typename Mat>
auto with_device(Mat&& A) {
  if constexpr(MEM == HOST_MEMORY) {
    return std::forward<Mat>(A);
  } else {
    return nda::to_device(A);
  }
}

template<bool sparse, MEMORY_SPACE MEM, typename Mat>
auto with_sparse(Mat&& A) {
  if constexpr(sparse) {
    return math::sparse::to_csr<MEM, int, int>(A, 0.0);
  } else {
    return std::forward<Mat>(A);
  }
}

template<bool stack, typename Mat>
auto with_stack(Mat&& A) {
  if constexpr(stack) {
    return nda::concatenate(reshape(A, 1, 1, A.extent(0), A.extent(1)));
  } else {
    return std::forward<Mat>(A);
  }
}
 
template<char opA, char opB>
auto check_result(const auto& A, const auto& B, auto& C, auto& C_expected) {
  math::product<opA,opB>(A, B, C);
  CHECK_THAT(nda::to_host(C), utils::Approx(C_expected));
  math::product<opA,opB>(0.6, A, B, 0.4, C);
  CHECK_THAT(nda::to_host(C), utils::Approx(C_expected));
}

template<char opA, char opB, bool sparseA, bool sparseB, bool stackA, bool stackB, typename T, MEMORY_SPACE MEM>
void check() {
  auto [A, B, C, C_expected] = input_arrays<T>();

  nda::array<T,2> At, Bt;
  At = with_op<opA>(A);
  Bt = with_op<opB>(B);

  auto Ad = with_device<MEM>(At);
  auto Bd = with_device<MEM>(Bt);
  auto Cd = with_device<MEM>(C);
  
  auto As = with_sparse<sparseA,MEM>(Ad);
  auto Bs = with_sparse<sparseB,MEM>(Bd);

  static_assert(!sparseA || !stackA);
  static_assert(!sparseB || !stackB);

  auto Ast = with_stack<stackA>(As);
  auto Bst = with_stack<stackB>(Bs);
  auto Cst = with_stack<stackA || stackB>(Cd);
  auto Cst_expected = with_stack<stackA || stackB>(C_expected);
  
  check_result<opA, opB>(As, Bs, Cd, C_expected);
}

template<char opA, char opB, bool sparseA, bool sparseB, bool stackA, bool stackB>
void check_mem() {
  check<opA,opB,sparseA,sparseB,stackA,stackB,double,HOST_MEMORY>();
  check<opA,opB,sparseA,sparseB,stackA,stackB,std::complex<double>,HOST_MEMORY>();
  #if defined(ENABLE_DEVICE)
    check<opA,opB,sparseA,sparseB,stackA,stackB,double,DEVICE_MEMORY>();
    check<opA,opB,sparseA,sparseB,stackA,stackB,std::complex<double>,DEVICE_MEMORY>();
  #endif
}

template<bool sparseA, bool sparseB, bool stackA, bool stackB>
void check_ops() {
  if constexpr(sparseA) {
    check_mem<'N','N',sparseA,sparseB,stackA,stackB>();
  } else if constexpr(sparseB) {
    check_mem<'N','T',sparseA,sparseB,stackA,stackB>();
    // check_mem<'N','N',sparseA,sparseB,stackA,stackB>();
  } else { 
    check_mem<'N','N',sparseA,sparseB,stackA,stackB>();
    check_mem<'N','T',sparseA,sparseB,stackA,stackB>();
    check_mem<'T','N',sparseA,sparseB,stackA,stackB>();
    check_mem<'T','T',sparseA,sparseB,stackA,stackB>();
    check_mem<'N','H',sparseA,sparseB,stackA,stackB>();
    check_mem<'H','N',sparseA,sparseB,stackA,stackB>();
    check_mem<'H','H',sparseA,sparseB,stackA,stackB>();
    check_mem<'H','T',sparseA,sparseB,stackA,stackB>();
    check_mem<'T','H',sparseA,sparseB,stackA,stackB>();
  }
}


TEST_CASE("math::product dense x dense", "[math_product]")
{
  check_ops<false,false,false,false>();
  check_ops<false,false,false,true>();
  check_ops<false,false,true,false>();
  check_ops<false,false,false,true>();
}
TEST_CASE("math::product dense x sparse", "[math_product]")
{
  check_ops<false,true,false,false>();
  check_ops<false,true,true,false>();
}
TEST_CASE("math::product sparse x dense", "[math_product]")
{
  check_ops<true,false,false,false>();
  check_ops<true,false,false,true>();
}

}
