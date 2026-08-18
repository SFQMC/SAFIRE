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
#include <limits>

#include "catch2/catch_test_macros.hpp"

#include "nda/nda.hpp"
#include "arch/arch.h"
#include "test_common.hpp"
#include "utilities/Timer.hpp"

#include "numerics/sparse/sparse.hpp"

namespace bdft_tests
{

template<typename Type, typename IndxType, typename IntType, MEMORY_SPACE MEM>
void test_csr_blas()
{
  using math::sparse::to_csr;
  decltype(nda::range::all) all;
  long m = 22;
  long n = 9;
  long k = 17;

  nda::array<Type,2> Ah = sfqmc::utils::make_random<Type>(m,k);
  nda::array<Type,2> Bh = sfqmc::utils::make_random<Type>(k,n);
  nda::array<Type,2> B2h = sfqmc::utils::make_random<Type>(m,n);
  nda::array<Type,1> xh = sfqmc::utils::make_random<Type>(k);
  nda::array<Type,1> yh = sfqmc::utils::make_random<Type>(m);

  nda::array<Type,2> AB(m,n);  
  nda::array<Type,2> AtB2(k,n);
  nda::array<Type,2> AhB2(k,n);
  nda::array<Type,1> Ax(m);
  nda::array<Type,1> Aty(k);

  nda::blas::gemm(Type(1.0),Ah,Bh,Type(0.0),AB); 
  nda::blas::gemm(Type(1.0),nda::transpose(Ah),B2h,Type(0.0),AtB2); 
  nda::blas::gemm(Type(1.0),nda::dagger(Ah),B2h,Type(0.0),AhB2); 
  nda::blas::gemv(Type(1.0),Ah,xh,Type(0.0),Ax); 
  nda::blas::gemv(Type(1.0),nda::transpose(Ah),yh,Type(0.0),Aty); 

  // device
  memory::array<MEM,Type,2> Ad(Ah);
  memory::array<MEM,Type,2> Bd(Bh);
  memory::array<MEM,Type,2> B2d(B2h);
  memory::array<MEM,Type,1> xd(xh);
  memory::array<MEM,Type,1> yd(yh); 

  memory::array<MEM,Type,2> ABd(m,n);
  memory::array<MEM,Type,2> AtB2d(k,n);
  memory::array<MEM,Type,2> AhB2d(k,n);
  memory::array<MEM,Type,1> Axd(m);
  memory::array<MEM,Type,1> Atyd(k);

  // test A*B with A csr
  {
    auto a = to_csr<MEM,IndxType,IntType>(Ah,0.0);
    math::sparse::csrmv<'N'>(Type(1.0),a,xd,Type(0.0),Axd); 
    math::sparse::csrmv<'T'>(Type(1.0),a,yd,Type(0.0),Atyd); 
    math::sparse::csrmm<'N'>(Type(1.0),a,Bd,Type(0.0),ABd); 
    math::sparse::csrmm<'T'>(Type(1.0),a,B2d,Type(0.0),AtB2d);  
    math::sparse::csrmm<'H'>(Type(1.0),a,B2d,Type(0.0),AhB2d); 

    CHECK_THAT(Ax, sfqmc::utils::Approx(nda::to_host(Axd)));
    CHECK_THAT(Aty, sfqmc::utils::Approx(nda::to_host(Atyd)));
    CHECK_THAT(AB, sfqmc::utils::Approx(nda::to_host(ABd)));
    CHECK_THAT(AtB2, sfqmc::utils::Approx(nda::to_host(AtB2d)));
    CHECK_THAT(AhB2, sfqmc::utils::Approx(nda::to_host(AhB2d)));
  }

  { 
    auto Ah_r = Ah(nda::range(5,15),all);
    auto B2h_r = B2h(nda::range(5,15),all);
    nda::blas::gemv(Type(1.0),nda::transpose(Ah_r),yh(nda::range(5,15)),
                    Type(0.0),Aty); 
    nda::blas::gemm(Type(1.0),nda::transpose(Ah_r),B2h_r,Type(0.0),AtB2); 
    nda::blas::gemm(Type(1.0),nda::dagger(Ah_r),B2h_r,Type(0.0),AhB2); 

    auto a_full = to_csr<MEM,IndxType,IntType>(Ah,0.0);
    auto a = a_full(nda::range(5,15));
    auto B2d_r = B2d(nda::range(5,15),all);
    math::sparse::csrmv<'N'>(Type(1.0),a,xd,Type(0.0),Axd(nda::range(5,15))); 
    math::sparse::csrmv<'T'>(Type(1.0),a,yd(nda::range(5,15)),Type(0.0),Atyd);
    math::sparse::csrmm<'N'>(Type(1.0),a,Bd,Type(0.0),ABd(nda::range(5,15),all)); 
    math::sparse::csrmm<'T'>(Type(1.0),a,B2d_r,Type(0.0),AtB2d);
    math::sparse::csrmm<'H'>(Type(1.0),a,B2d_r,Type(0.0),AhB2d);

    CHECK_THAT(Ax(nda::range(5,15)), sfqmc::utils::Approx(nda::to_host(Axd(nda::range(5,15)))));
    CHECK_THAT(Aty, sfqmc::utils::Approx(nda::to_host(Atyd)));
    CHECK_THAT(AB(nda::range(5,15),all), sfqmc::utils::Approx(nda::to_host(ABd(nda::range(5,15),all))));
    CHECK_THAT(AtB2, sfqmc::utils::Approx(nda::to_host(AtB2d)));
    CHECK_THAT(AhB2, sfqmc::utils::Approx(nda::to_host(AhB2d)));
  }

  // now test A*B with B csr using B^T * T(A)  
  {    
    auto b = to_csr<MEM,IndxType,IntType>(Bh,0.0); 
    math::sparse::csrmm<'T'>(Type(1.0),b,nda::transpose(Ad),Type(0.0),nda::transpose(ABd));
    CHECK_THAT(AB, sfqmc::utils::Approx(nda::to_host(ABd)));

    math::sparse::csrmm<'N'>(Type(1.0),Ad,b,Type(0.0),ABd);
    CHECK_THAT(AB, sfqmc::utils::Approx(nda::to_host(ABd)));

    auto b_r = b(nda::range(5,15));
    math::sparse::csrmm<'N'>(Type(1.0),Ad(all,nda::range(5,15)),b_r,Type(0.0),ABd);
    nda::blas::gemm(Type(1.0),Ah(all,nda::range(5,15)),Bh(nda::range(5,15),all),Type(0.0),AB); 
    CHECK_THAT(AB, sfqmc::utils::Approx(nda::to_host(ABd)));
  }

}

template<typename Type>
Type nan_value()
{
  if constexpr (nda::is_complex_v<Type>) {
    using real_t = typename Type::value_type;
    return Type{std::numeric_limits<real_t>::quiet_NaN(),
                std::numeric_limits<real_t>::quiet_NaN()};
  } else {
    return std::numeric_limits<Type>::quiet_NaN();
  }
}

// beta==0 means the output is not read (blas convention). Filling it with NaN detects
// backends that scale the output instead of assigning it, since NaN*0 == NaN.
template<typename Type, typename... Ints>
auto poisoned(Ints... shape)
{
  nda::array<Type,sizeof...(Ints)> a(shape...);
  a() = nan_value<Type>();
  return a;
}

template<typename Type, int Rank>
auto scaled(nda::array<Type,Rank> a, Type s)
{
  for( auto& v: a ) { v *= s; }
  return a;
}

// beta==0 with a poisoned output, through the public interface and directly through
// backup_impl. The latter is the non-mkl backend, called explicitly so that it is
// exercised in mkl builds as well.
template<typename Type, typename IndxType, typename IntType>
void test_csr_blas_beta_zero()
{
  using math::sparse::to_csr;
  long m = 22;
  long n = 9;
  long k = 17;

  nda::array<Type,2> Ah = sfqmc::utils::make_random<Type>(m,k);
  nda::array<Type,2> Bh = sfqmc::utils::make_random<Type>(k,n);
  nda::array<Type,2> B2h = sfqmc::utils::make_random<Type>(m,n);
  nda::array<Type,1> xh = sfqmc::utils::make_random<Type>(k);
  nda::array<Type,1> yh = sfqmc::utils::make_random<Type>(m);

  nda::array<Type,2> Ahdag = nda::make_regular(nda::dagger(Ah));

  auto AB = nda::zeros<Type>(m,n);
  auto AtB2 = nda::zeros<Type>(k,n);
  auto AhB2 = nda::zeros<Type>(k,n);
  auto Ax = nda::zeros<Type>(m);
  auto Aty = nda::zeros<Type>(k);
  auto Ahy = nda::zeros<Type>(k);

  nda::blas::gemm(Type(1.0),Ah,Bh,Type(0.0),AB);
  nda::blas::gemm(Type(1.0),nda::transpose(Ah),B2h,Type(0.0),AtB2);
  nda::blas::gemm(Type(1.0),nda::dagger(Ah),B2h,Type(0.0),AhB2);
  nda::blas::gemv(Type(1.0),Ah,xh,Type(0.0),Ax);
  nda::blas::gemv(Type(1.0),nda::transpose(Ah),yh,Type(0.0),Aty);
  nda::blas::gemv(Type(1.0),Ahdag,yh,Type(0.0),Ahy);

  auto a = to_csr<HOST_MEMORY,IndxType,IntType>(Ah,0.0);

  // public interface: with mkl, IndxType==IntType==int dispatches to mkl_sparse_?_mv/_mm
  {
    auto y = poisoned<Type>(m);
    math::sparse::csrmv<'N'>(Type(1.0),a,xh,Type(0.0),y);
    CHECK_THAT(Ax, sfqmc::utils::Approx(y));
  }
  {
    auto y = poisoned<Type>(k);
    math::sparse::csrmv<'T'>(Type(1.0),a,yh,Type(0.0),y);
    CHECK_THAT(Aty, sfqmc::utils::Approx(y));
  }
  {
    auto y = poisoned<Type>(k);
    math::sparse::csrmv<'H'>(Type(1.0),a,yh,Type(0.0),y);
    CHECK_THAT(Ahy, sfqmc::utils::Approx(y));
  }
  {
    auto C = poisoned<Type>(m,n);
    math::sparse::csrmm<'N'>(Type(1.0),a,Bh,Type(0.0),C);
    CHECK_THAT(AB, sfqmc::utils::Approx(C));
  }
  {
    auto C = poisoned<Type>(k,n);
    math::sparse::csrmm<'T'>(Type(1.0),a,B2h,Type(0.0),C);
    CHECK_THAT(AtB2, sfqmc::utils::Approx(C));
  }
  {
    auto C = poisoned<Type>(k,n);
    math::sparse::csrmm<'H'>(Type(1.0),a,B2h,Type(0.0),C);
    CHECK_THAT(AhB2, sfqmc::utils::Approx(C));
  }

  // non-mkl backend
  auto const* av = a.values().data();
  auto const* ac = a.columns().data();
  auto const* rb = a.row_begin().data();
  auto const* re = a.row_end().data();
  using math::sparse::backup_impl::csrmv;
  using math::sparse::backup_impl::csrmm;
  {
    auto y = poisoned<Type>(m);
    csrmv('N',int(m),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,xh.data(),Type(0.0),y.data());
    CHECK_THAT(Ax, sfqmc::utils::Approx(y));
  }
  {
    auto y = poisoned<Type>(k);
    csrmv('T',int(m),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,yh.data(),Type(0.0),y.data());
    CHECK_THAT(Aty, sfqmc::utils::Approx(y));
  }
  {
    auto y = poisoned<Type>(k);
    csrmv('H',int(m),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,yh.data(),Type(0.0),y.data());
    CHECK_THAT(Ahy, sfqmc::utils::Approx(y));
  }
  {
    auto C = poisoned<Type>(m,n);
    csrmm('N',int(m),int(n),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,Bh.data(),int(n),
          Type(0.0),C.data(),int(n));
    CHECK_THAT(AB, sfqmc::utils::Approx(C));
  }
  {
    auto C = poisoned<Type>(k,n);
    csrmm('T',int(m),int(n),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,B2h.data(),int(n),
          Type(0.0),C.data(),int(n));
    CHECK_THAT(AtB2, sfqmc::utils::Approx(C));
  }
  {
    auto C = poisoned<Type>(k,n);
    csrmm('H',int(m),int(n),int(k),Type(1.0),"GxxCxx",av,ac,rb,re,B2h.data(),int(n),
          Type(0.0),C.data(),int(n));
    CHECK_THAT(AhB2, sfqmc::utils::Approx(C));
  }
}

// scale factor applied to the example matrix, complex to check conjugation on 'H'
template<typename Type>
Type example_scale()
{
  if constexpr (nda::is_complex_v<Type>) {
    return Type{1.0,1.0};
  } else {
    return Type(1.0);
  }
}

//         [ 2 0 0 1 0 ]
//  A = s* [ 0 3 0 0 0 ]
//         [ 1 0 4 0 2 ]
//         [ 0 0 0 0 5 ]
template<typename Type>
auto example_matrix()
{
  auto s = example_scale<Type>();
  auto A = nda::zeros<Type>(4,5);
  A(0,0) = s*2.0;
  A(0,3) = s*1.0;
  A(1,1) = s*3.0;
  A(2,0) = s*1.0;
  A(2,2) = s*4.0;
  A(2,4) = s*2.0;
  A(3,4) = s*5.0;
  return A;
}

// results for an explicit matrix, computed by hand rather than with dense blas, through
// the public interface: with mkl, IndxType==IntType==int dispatches to mkl_sparse_?_mv/_mm
template<typename Type, typename IndxType, typename IntType>
void test_csr_blas_example()
{
  auto s = example_scale<Type>();
  Type sc = nda::conj(s);
  auto a = math::sparse::to_csr<HOST_MEMORY,IndxType,IntType>(example_matrix<Type>(),0.0);
  CHECK(a.nnz() == 7);

  nda::array<Type,1> x = {1.0,2.0,3.0,4.0,5.0};
  nda::array<Type,1> y = {1.0,2.0,3.0,4.0};
  nda::array<Type,2> B = {{1.0,5.0},{2.0,4.0},{3.0,3.0},{4.0,2.0},{5.0,1.0}};
  nda::array<Type,2> B2 = {{1.0,4.0},{2.0,3.0},{3.0,2.0},{4.0,1.0}};

  {
    auto r = nda::zeros<Type>(4);
    math::sparse::csrmv<'N'>(Type(1.0),a,x,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,1>{6.0,6.0,23.0,25.0},s), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5);
    math::sparse::csrmv<'T'>(Type(1.0),a,y,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,1>{5.0,6.0,12.0,1.0,26.0},s), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5);
    math::sparse::csrmv<'H'>(Type(1.0),a,y,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,1>{5.0,6.0,12.0,1.0,26.0},sc), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(4,2);
    math::sparse::csrmm<'N'>(Type(1.0),a,B,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,2>{{6.0,12.0},{6.0,12.0},{23.0,19.0},{25.0,5.0}},s),
               sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5,2);
    math::sparse::csrmm<'T'>(Type(1.0),a,B2,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,2>{{5.0,10.0},{6.0,9.0},{12.0,8.0},{1.0,4.0},{26.0,9.0}},s),
               sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5,2);
    math::sparse::csrmm<'H'>(Type(1.0),a,B2,Type(0.0),r);
    CHECK_THAT(scaled(nda::array<Type,2>{{5.0,10.0},{6.0,9.0},{12.0,8.0},{1.0,4.0},{26.0,9.0}},sc),
               sfqmc::utils::Approx(r));
  }
}

// entries with a column index >= K must be ignored. Only reachable through the low level
// interface, since the public one always takes K from the shape of the matrix. Inputs are
// allocated at full length and outputs are padded, so that a missing bound check shows up
// as a wrong result rather than as an access out of bounds.
template<typename Type, typename IndxType, typename IntType>
void test_csr_blas_column_range()
{
  auto s = example_scale<Type>();
  Type sc = nda::conj(s);
  auto a = math::sparse::to_csr<HOST_MEMORY,IndxType,IntType>(example_matrix<Type>(),0.0);

  int const M = 4;
  int const K = 3;
  int const N = 2;

  auto const* av = a.values().data();
  auto const* ac = a.columns().data();
  auto const* rb = a.row_begin().data();
  auto const* re = a.row_end().data();
  using math::sparse::backup_impl::csrmv;
  using math::sparse::backup_impl::csrmm;

  nda::array<Type,1> x = {1.0,2.0,3.0,4.0,5.0};
  nda::array<Type,1> y = {1.0,2.0,3.0,4.0};
  nda::array<Type,2> B = {{1.0,5.0},{2.0,4.0},{3.0,3.0},{4.0,2.0},{5.0,1.0}};
  nda::array<Type,2> B2 = {{1.0,4.0},{2.0,3.0},{3.0,2.0},{4.0,1.0}};

  {
    auto r = nda::zeros<Type>(M);
    csrmv('N',M,K,Type(1.0),"GxxCxx",av,ac,rb,re,x.data(),Type(0.0),r.data());
    CHECK_THAT(scaled(nda::array<Type,1>{2.0,6.0,13.0,0.0},s), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5);
    csrmv('T',M,K,Type(1.0),"GxxCxx",av,ac,rb,re,y.data(),Type(0.0),r.data());
    CHECK_THAT(scaled(nda::array<Type,1>{5.0,6.0,12.0,0.0,0.0},s), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5);
    csrmv('H',M,K,Type(1.0),"GxxCxx",av,ac,rb,re,y.data(),Type(0.0),r.data());
    CHECK_THAT(scaled(nda::array<Type,1>{5.0,6.0,12.0,0.0,0.0},sc), sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(M,N);
    csrmm('N',M,N,K,Type(1.0),"GxxCxx",av,ac,rb,re,B.data(),N,Type(0.0),r.data(),N);
    CHECK_THAT(scaled(nda::array<Type,2>{{2.0,10.0},{6.0,12.0},{13.0,17.0},{0.0,0.0}},s),
               sfqmc::utils::Approx(r));
  }
  {
    auto r = nda::zeros<Type>(5,N);
    csrmm('H',M,N,K,Type(1.0),"GxxCxx",av,ac,rb,re,B2.data(),N,Type(0.0),r.data(),N);
    CHECK_THAT(scaled(nda::array<Type,2>{{5.0,10.0},{6.0,9.0},{12.0,8.0},{0.0,0.0},{0.0,0.0}},sc),
               sfqmc::utils::Approx(r));
  }
}

TEST_CASE("csr_blas", "[csr]")
{
  test_csr_blas<double, long, long, HOST_MEMORY>();
  test_csr_blas<double, int, int, HOST_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, HOST_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
  test_csr_blas<double, long, long, DEVICE_MEMORY>();
  test_csr_blas<double, int, int, DEVICE_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, DEVICE_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, DEVICE_MEMORY>();
  test_csr_blas<double, long, long, UNIFIED_MEMORY>();
  test_csr_blas<double, int, int, UNIFIED_MEMORY>();
  test_csr_blas<std::complex<double>, long, long, UNIFIED_MEMORY>();
  test_csr_blas<std::complex<double>, int, int, UNIFIED_MEMORY>();
#endif
}

TEST_CASE("csr_blas_beta_zero", "[csr]")
{
  test_csr_blas_beta_zero<double, long, long>();
  test_csr_blas_beta_zero<double, int, int>();
  test_csr_blas_beta_zero<std::complex<double>, long, long>();
  test_csr_blas_beta_zero<std::complex<double>, int, int>();
}

TEST_CASE("csr_blas_example", "[csr]")
{
  test_csr_blas_example<double, long, long>();
  test_csr_blas_example<double, int, int>();
  test_csr_blas_example<std::complex<double>, long, long>();
  test_csr_blas_example<std::complex<double>, int, int>();
}

TEST_CASE("csr_blas_column_range", "[csr]")
{
  test_csr_blas_column_range<double, long, long>();
  test_csr_blas_column_range<double, int, int>();
  test_csr_blas_column_range<std::complex<double>, long, long>();
  test_csr_blas_column_range<std::complex<double>, int, int>();
}

} // namespace bdft 
