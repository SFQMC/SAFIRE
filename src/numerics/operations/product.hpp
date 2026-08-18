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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/sparse/sparse.hpp"

#define AVOID_TBLIS_CONTRACT

namespace math
{

namespace detail
{

template<char op_A>
constexpr auto _decorate_gemm_(nda::MemoryMatrix auto && a)
requires(math::is_valid_op(op_A))
{
  if constexpr (op_A == 'N') return a();
  else if constexpr (op_A == 'T') return nda::transpose(a);
  else if constexpr (op_A == 'H') return nda::dagger(a);
}

// MAM: assumes s.size() == 2, avoiding runtime check, add if needed
template<char op_A>
constexpr auto _decorate_contract_(nda::MemoryArrayOfRank<2> auto && a, std::string_view const s)
requires(math::is_valid_op(op_A))
{
  if constexpr (op_A=='N') {
    std::string s_ = std::string(1,s[0]) + std::string(1,s[1]); 
    return std::make_tuple(std::string(s_),a());;
  } else if constexpr (op_A == 'T') {
    std::string s_ = std::string(1,s[1]) + std::string(1,s[0]);
    return std::make_tuple(std::string(s_),a());;
  } else if constexpr (op_A == 'H') {
    std::string s_ = std::string(1,s[1]) + std::string(1,s[0]);
    return std::make_tuple(std::string(s_),nda::conj(a));;
  }
}

// MAM: assumes s.size() == 3, avoiding runtime check, add if needed
template<char op_A>
constexpr auto _decorate_contract_(nda::MemoryArrayOfRank<3> auto && a, std::string_view const s)
requires(math::is_valid_op(op_A))
{
  if constexpr (op_A=='N') {
    std::string s_ = std::string(1,s[0]) + std::string(1,s[1]) + std::string(1,s[2]);
    return std::make_tuple(std::string(s_),a());;
  } else if constexpr (op_A == 'T') {
    std::string s_ = std::string(1,s[0]) + std::string(1,s[2]) + std::string(1,s[1]);
    return std::make_tuple(std::string(s_),a());;
  } else if constexpr (op_A == 'H') {
    std::string s_ = std::string(1,s[0]) + std::string(1,s[2]) + std::string(1,s[1]);
    return std::make_tuple(std::string(s_),nda::conj(a));;
  } 
}

}


/**
 * 
 * General wrapper routines for array multiplication.
 * Dispatches:
 *  C = alpha*A*B + beta*C, where:
 *    - alpha/beta: scalars (same type as A)
 *    - A: nda Array of rank 2/3 or csr_matrix.
 *    - B: nda Array of rank 2/3 or csr_matrix
 *    - C: nda Array of rank 2/3
 *
 * Only some combinations are allowed. This routine inspects the types at 
 * compile time and dispatches to the appropriate backend.  
 * For now, only arrays with C layouts are allowed. 
 * Transposition and hermitian conjugate operations are allowed through template params.
 */ 


// M * S = M
// T * S = T
template<char op_A = 'N', char op_B = 'N', typename T = double> 
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        ) 
void product(T alpha, nda::MemoryArray auto const& a, 
                      math::sparse::CSRMatrix auto const& b, 
             T beta, nda::MemoryArray auto && c) 
{
  using A_t = std::decay_t<decltype(a)>; 
  using B_t = std::decay_t<decltype(b)>; 
  using C_t = std::decay_t<decltype(c)>; 
  using v_t = nda::get_value_t<A_t>;  // do I need is_convertible<T,v_t>???
  static_assert(nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
                nda::have_same_value_type_v<A_t,C_t>,
                "Failed requirements");
  static_assert(op_A=='N', "Invalid matrix operation");
  math::sparse::csrmm<op_B>(v_t{alpha},a,b,v_t{beta},c); 
}

template<char op_A = 'N', char op_B = 'N'>
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        )
void product(nda::MemoryArray auto const& a,
             math::sparse::CSRMatrix auto const& b,
             nda::MemoryArray auto && c)
{
  using A_t = std::decay_t<decltype(a)>;
  using v_t = nda::get_value_t<A_t>;  // do I need is_convertible<T,v_t>???
  product<op_A,op_B,v_t>(v_t{1.0},a,b,v_t{0.0},c);
}



// S * M = M
// S * T = T
template<char op_A = 'N', char op_B = 'N', typename T = double>
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        )
void product(T alpha, math::sparse::CSRMatrix auto const& a,
                      nda::MemoryArray auto const& b,
             T beta, nda::MemoryArray auto && c)
{
  using A_t = std::decay_t<decltype(a)>;
  using B_t = std::decay_t<decltype(b)>;
  using C_t = std::decay_t<decltype(c)>;
  using v_t = nda::get_value_t<B_t>;  // do I need is_convertible<T,v_t>???
  static_assert(nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
                nda::have_same_value_type_v<B_t,C_t>,
                "Failed requirements");
  static_assert(op_B=='N', "Invalid matrix operation");
  math::sparse::csrmm<op_A>(v_t{alpha},a,b,v_t{beta},c); 
}

template<char op_A = 'N', char op_B = 'N'>
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        )
void product(math::sparse::CSRMatrix auto const& a,
             nda::MemoryArray auto const& b,
             nda::MemoryArray auto && c)
{
  using B_t = std::decay_t<decltype(b)>;
  using v_t = nda::get_value_t<B_t>;  // do I need is_convertible<T,v_t>???
  product<op_A,op_B,v_t>(v_t{1.0},a,b,v_t{0.0},c);
}


// Dense
// M * M = M
// M * T = T
// T * M = T
// T * T = T
template<char op_A = 'N', char op_B = 'N', typename T = double>
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        )
void product(T alpha, nda::MemoryArray auto const& a,
                      nda::MemoryArray auto const& b,
             T beta, nda::MemoryArray auto && c)
{
  auto _ = nda::ellipsis{};
  using A_t = std::decay_t<decltype(a)>;
  using B_t = std::decay_t<decltype(b)>;
  using C_t = std::decay_t<decltype(c)>;
  using v_t = nda::get_value_t<A_t>;  // do I need is_convertible<T,v_t>???
  static_assert(nda::mem::have_compatible_addr_space<A_t,B_t,C_t> and
                nda::have_same_value_type_v<A_t,B_t,C_t>,
                "Failed requirements");
  static_assert(std::decay_t<A_t>::is_stride_order_C() and
                std::decay_t<B_t>::is_stride_order_C() and
                std::decay_t<C_t>::is_stride_order_C(), "Stride mismatch"); 

  if constexpr (nda::get_rank<A_t> == 2 ) {
    static_assert(nda::get_rank<B_t> == nda::get_rank<C_t>, "Rank mismatch");


    if constexpr (nda::get_rank<B_t> == 2 ) {
      // M * M = M
      // a/b with op_A/B applied in nda notation
      auto a_d = detail::_decorate_gemm_<op_A>(a);
      auto b_d = detail::_decorate_gemm_<op_B>(b);
      nda::blas::gemm(v_t{alpha},a_d,b_d,v_t{beta},c);
    } else if constexpr (nda::get_rank<B_t> == 3 ) {
      sfqmc::utils::check(b.extent(0) == c.extent(0), "Size mismatch");
      // M * T = T
#if defined(AVOID_TBLIS_CONTRACT)
      if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t,C_t>) {
#else
      if constexpr (true) {
#endif
        auto [a_s,a_d] = detail::_decorate_contract_<op_A>(a,"ik");
        auto [b_s,b_d] = detail::_decorate_contract_<op_B>(b,"nkj");
        nda::tensor::contract(v_t{alpha},a_d,a_s,b_d,b_s,v_t{beta},c,"nij");
      } else {
        long n = b.extent(0);
        // a/b with op_A/B applied in nda notation
        auto a_d = detail::_decorate_gemm_<op_A>(a);
        for(long i=0; i<n; ++i) {
          auto b_d = detail::_decorate_gemm_<op_B>(b(i,_));
          nda::blas::gemm(v_t{alpha},a_d,b_d,v_t{beta},c(i,_));
        }
      } 
    } else {
      sfqmc::utils::check(false,"Invalid argument type combination");
    }

  } else if constexpr (nda::get_rank<A_t> == 3 ) {

    static_assert( nda::get_rank<C_t> == 3, "Rank mismatch");
    if constexpr (nda::get_rank<B_t> == 2 ) {
      sfqmc::utils::check(a.extent(0) == c.extent(0), "Size mismatch");
      // T * M = T
#if defined(AVOID_TBLIS_CONTRACT)
      if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t,C_t>) {
#else
      if constexpr (true) {
#endif
        auto [a_s,a_d] = detail::_decorate_contract_<op_A>(a,"nik");
        auto [b_s,b_d] = detail::_decorate_contract_<op_B>(b,"kj");
        nda::tensor::contract(v_t{alpha},a_d,a_s,b_d,b_s,v_t{beta},c,"nij");
      } else {
        long n = a.extent(0);
        // a/b with op_A/B applied in nda notation
        auto b_d = detail::_decorate_gemm_<op_B>(b);
        for(long i=0; i<n; ++i) {
          auto a_d = detail::_decorate_gemm_<op_A>(a(i,_));
          nda::blas::gemm(v_t{alpha},a_d,b_d,v_t{beta},c(i,_));
        }
      }
    } else if constexpr (nda::get_rank<B_t> == 3 ) {
      sfqmc::utils::check(a.extent(0) == c.extent(0) and a.extent(0) == b.extent(0), 
                   "Size mismatch");
      // T * T = T
#if defined(AVOID_TBLIS_CONTRACT)
      if constexpr (nda::mem::have_device_compatible_addr_space<A_t,B_t,C_t>) {
#else
      if constexpr (true) {
#endif
        auto [a_s,a_d] = detail::_decorate_contract_<op_A>(a,"nik");
        auto [b_s,b_d] = detail::_decorate_contract_<op_B>(b,"nkj");
        nda::tensor::contract(v_t{alpha},a_d,a_s,b_d,b_s,v_t{beta},c,"nij");
      } else {
        long n = a.extent(0);
        // a/b with op_A/B applied in nda notation
        for(long i=0; i<n; ++i) {
          auto a_d = detail::_decorate_gemm_<op_A>(a(i,_));
          auto b_d = detail::_decorate_gemm_<op_B>(b(i,_));
          nda::blas::gemm(v_t{alpha},a_d,b_d,v_t{beta},c(i,_));
        }
      }
    } else {
      sfqmc::utils::check(false,"Invalid argument type combination");
    }

  }
}

template<char op_A = 'N', char op_B = 'N'>
requires(
  math::is_valid_op(op_A) and math::is_valid_op(op_B)
        )
void product(nda::MemoryArray auto const& a,
             nda::MemoryArray auto const& b,
             nda::MemoryArray auto && c)
{
  using A_t = std::decay_t<decltype(a)>;
  using v_t = nda::get_value_t<A_t>;  // do I need is_convertible<T,v_t>???
  product<op_A,op_B,v_t>(v_t{1.0},a,b,v_t{0.0},c);
}

} // math
