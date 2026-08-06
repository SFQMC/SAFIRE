/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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

#include "nda/nda.hpp"
#include "test_common.hpp"
#include "utilities/Timer.hpp"

#include "numerics/sparse/sparse.hpp"

namespace bdft_tests
{

TEST_CASE("csr_concepts", "[csr]")
{
  using math::sparse::CSRVector;
  using math::sparse::CSRMatrix;
  using math::sparse::csr_matrix;
  using math::sparse::csr_matrix_view;

  static_assert(CSRVector<csr_matrix<double>::reference> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix<double>> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix<double>::matrix_view> ,"CONCEPT TEST");
  static_assert(CSRMatrix<csr_matrix_view<double>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_host<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_host<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_host<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_device<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_device<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_device<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_unified<csr_matrix<double,HOST_MEMORY>> ,"CONCEPT TEST");
  static_assert(not nda::mem::on_unified<csr_matrix<double,DEVICE_MEMORY>> ,"CONCEPT TEST");
  static_assert(nda::mem::on_unified<csr_matrix<double,UNIFIED_MEMORY>> ,"CONCEPT TEST");
}

template<typename Type, typename IndxType, typename IntType, MEMORY_SPACE MEM>
void test_csr_matrix()
{
  using ucsr_matrix = math::sparse::ucsr_matrix<Type, MEM, IndxType, IntType>;
  using csr_matrix  = math::sparse::csr_matrix<Type, MEM, IndxType, IntType>;

  auto As = nda::array<Type,2>::zeros({4,4});
  As(0,1) = 10;
  As(0,2) = 9;
  As(2,1) = 3;
  As(3,3) = 1;
  nda::array<IntType,1> nnz      = {2, 0, 1, 1};
  nda::array<IntType,1> nnz_plus = {12, 10, 11, 11};

  auto check = [](auto && A_, auto && SpM) {
    auto vals = nda::to_host(SpM.values());  
    auto cols = nda::to_host(SpM.columns());  
    auto row_begin = nda::to_host(SpM.row_begin());
    auto row_end = nda::to_host(SpM.row_end());
    auto nr = SpM.shape(0);
    auto i0 = row_begin(0);
    for(long r=0; r<nr; r++) 
      for(long i=row_begin(r); i<row_end(r); ++i)
        CHECK_THAT(A_(r,cols(i-i0)), sfqmc::utils::Approx(vals(i-i0))); 
  };

  {
    ucsr_matrix small({4, 4}, 2);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    [[maybe_unused]] auto val = small.values();
    [[maybe_unused]] auto col = small.columns();
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  {
    ucsr_matrix small({4, 4});
    small.reserve(2);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    ucsr_matrix small({4, 4}, nnz);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  {
    ucsr_matrix small({4, 4});
    small.reserve(nnz);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    ucsr_matrix small({4, 4}, nnz_plus);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    csr_matrix csr(small);
    REQUIRE(csr.nnz() == 4);
    check(As,csr);
  }

  { 
    ucsr_matrix small({4, 4});
    small.reserve(nnz_plus);
    small[3][3] = 1;
    small[0][2] = 9;
    small[2][1] = 3;
    small[0][1] = 10;
    
    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

// in gpu, csr_matrix only constructible from ucsr_matrix
  if constexpr (MEM==HOST_MEMORY) 
  {
  {
    csr_matrix small({4, 4}, 2);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(2);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(nnz);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    small.remove_empty_spaces();
    REQUIRE(small.nnz() == 4);
    REQUIRE(small.row_end()(3)-small.row_begin()(0) == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4});
    small.reserve(nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] += 10;
    small[0][2] += 9;
    small[2][1] += 3;
    small[3][3] += 1;

    REQUIRE(small.nnz() == 4);
    check(As,small);

    auto B = As;
    B *= 2.0;
    small[0][1] *= 2.0;
    small[0][2] *= 2.0;
    small[2][1] *= 2.0;
    small[3][3] *= 2.0;
    
    REQUIRE(small.nnz() == 4);
    check(B,small);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    auto A_ = math::sparse::to_array<'N'>(small);
    check(A_,small);
  }

  {
    auto csr_ = math::sparse::to_csr(As);
    check(As,csr_);
  }

  {
    csr_matrix small({4, 4}, nnz_plus);
    small[0][1] = 10;
    small[0][2] = 9;
    small[2][1] = 3;
    small[3][3] = 1;

    csr_matrix Ac(small);
    check(As,Ac);

    csr_matrix Am(std::move(small));
    check(As,Am);

    Am = Ac;
    check(As,Am);

    Am = std::move(Ac);
    check(As,Am);
  }

  {
    csr_matrix small({8, 4}, 2);
    small[2][1] = 10;
    small[2][2] = 9;
    small[4][1] = 3;
    small[5][3] = 1;

    auto A = small(nda::range(2,6));
    check(As,A);
  }
  }

}

template<typename Type, typename IndxType, typename IntType, MEMORY_SPACE MEM>
void test_array_of_csr()
{
  using csr_matrix  = math::sparse::csr_matrix<Type, MEM, IndxType, IntType>;

  auto As = nda::array<Type,2>::zeros({4,4});
  As(0,1) = 10;
  As(0,2) = 9;
  As(2,1) = 3;
  As(3,3) = 1;
  nda::array<IntType,1> nnz      = {2, 0, 1, 1};
  nda::array<IntType,1> nnz_plus = {12, 10, 11, 11};

  auto check = [](auto && A_, auto && SpM) {
    auto vals = nda::to_host(SpM.values());
    auto cols = nda::to_host(SpM.columns());
    auto row_begin = nda::to_host(SpM.row_begin());
    auto row_end = nda::to_host(SpM.row_end());
    auto nr = SpM.shape(0);
    auto i0 = row_begin(0);
    for(long r=0; r<nr; r++)
      for(long i=row_begin(r); i<row_end(r); ++i)
        CHECK_THAT(A_(r,cols(i-i0)), sfqmc::utils::Approx(vals(i-i0)));
  };

  auto csr_host = math::sparse::to_csr<HOST_MEMORY,IndxType,IntType>(As,0.0); 
  check(As,csr_host);

  csr_matrix csr(csr_host);
  check(As,csr);

  {
    nda::array<csr_matrix, 1> csr_array(10);
    csr_array(0) = csr;
    check(As,csr_array(0));
    REQUIRE(csr_array(0).nnz() == 4);
    for(int i=1; i<csr_array.extent(0); ++i)
      REQUIRE(csr_array(i).nnz() == 0);
  }

  {
    nda::array<csr_matrix, 1> csr_array(10,csr);
    for(int i=0; i<csr_array.extent(0); ++i) {
      check(As,csr_array(i));
      REQUIRE(csr_array(i).nnz() == 4);
    }
  }

  { 
    nda::array<csr_matrix, 3> csr_array(2,1,3);
    csr_array() = csr;
    for(auto& v : csr_array) {
      check(As,v);
      REQUIRE(v.nnz() == 4);
    }
  }
}

template<typename Type, typename IndxType, typename IntType>
void test_combine_csr()
{
  using nda::range;

  auto to_csr = [](auto const& A) {
    return math::sparse::to_csr<HOST_MEMORY,IndxType,IntType>(A,0.0);
  };
  auto to_dense = [](auto const& csr) {
    return math::sparse::to_array<'N'>(csr);
  };

  auto up = nda::array<Type,2>::zeros({2,4});
  up(0,1) = 10;
  up(0,3) = 9;
  up(1,0) = 3;

  auto dn = nda::array<Type,2>::zeros({3,4});
  dn(0,2) = 1;
  dn(1,1) = -2;
  dn(2,3) = 5;

  auto empty = nda::array<Type,2>::zeros({0,4});
  long shift = up.extent(1);

  // both blocks present: rows stacked, B shifted into the second half of the columns
  {
    auto c = math::sparse::combine_csr(to_csr(up),to_csr(dn),shift);
    REQUIRE(c.shape() == std::array<long,2>{5,8});
    auto ref = nda::array<Type,2>::zeros({5,8});
    ref(range(2),range(4)) = up;
    ref(range(2,5),range(4,8)) = dn;
    CHECK_THAT(to_dense(c), sfqmc::utils::Approx(ref));
  }

  // empty B: the result must still be widened to B_col_shift + B.cols. A fully polarized
  // trial hits this in readWfn's COLLINEAR -> NONCOLLINEAR upgrade, where returning A
  // verbatim leaves the spinor matrix with only NMO columns.
  {
    auto c = math::sparse::combine_csr(to_csr(up),to_csr(empty),shift);
    REQUIRE(c.shape() == std::array<long,2>{2,8});
    auto ref = nda::array<Type,2>::zeros({2,8});
    ref(range(2),range(4)) = up;
    CHECK_THAT(to_dense(c), sfqmc::utils::Approx(ref));
  }

  // empty A: B must still be shifted into the second half of the columns
  {
    auto c = math::sparse::combine_csr(to_csr(empty),to_csr(dn),shift);
    REQUIRE(c.shape() == std::array<long,2>{3,8});
    auto ref = nda::array<Type,2>::zeros({3,8});
    ref(range(3),range(4,8)) = dn;
    CHECK_THAT(to_dense(c), sfqmc::utils::Approx(ref));
  }
}

TEST_CASE("csr_matrix", "[csr]")
{
  test_csr_matrix<double, int, long, HOST_MEMORY>();
  test_csr_matrix<double, int, int, HOST_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, HOST_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  test_csr_matrix<double, int, long, DEVICE_MEMORY>();
  test_csr_matrix<double, int, int, DEVICE_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, DEVICE_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, DEVICE_MEMORY>();
  test_csr_matrix<double, int, long, UNIFIED_MEMORY>();
  test_csr_matrix<double, int, int, UNIFIED_MEMORY>();
  test_csr_matrix<std::complex<double>, int, long, UNIFIED_MEMORY>();
  test_csr_matrix<std::complex<double>, int, int, UNIFIED_MEMORY>();
#endif
}

TEST_CASE("arrays_of_csr")
{
  test_array_of_csr<double, int, long, HOST_MEMORY>();
}

TEST_CASE("combine_csr", "[csr]")
{
  test_combine_csr<double, int, long>();
  test_combine_csr<double, int, int>();
  test_combine_csr<std::complex<double>, int, long>();
  test_combine_csr<std::complex<double>, int, int>();
}

} // namespace bdft 
