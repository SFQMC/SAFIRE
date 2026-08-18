// Copyright (c) 2024--present, The Simons Foundation
// This file is part of TRIQS/nda and is licensed under the Apache License, Version 2.0.
// SPDX-License-Identifier: Apache-2.0
// See LICENSE in the root of this distribution for details.

/**
 * @file
 * @brief Provides a generic interface to the LAPACK `geqp3` routine.
 */

#pragma once

#include "nda/lapack/interface/cxx_interface.hpp"
#include "nda/basic_array.hpp"
#include "nda/concepts.hpp"
#include "nda/declarations.hpp"
#include "nda/exceptions.hpp"
#include "nda/macros.hpp"
#include "nda/mem/address_space.hpp"
#include "nda/traits.hpp"

#include "nda/basic_functions.hpp"
#include "nda/layout/policies.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <type_traits>
#include <utility>

namespace nda 
{

namespace lapack 
{

  /**
   * @ingroup linalg_lapack
   * @brief Interface to the LAPACK `geqp3` routine.
   *
   * @details Computes a QR factorization with column pivoting of a matrix \f$ \mathbf{A} \f$:
   * \f[
   *   \mathbf{A P} = \mathbf{Q R}
   * \f]
   * using Level 3 BLAS.
   *
   * @tparam A nda::MemoryMatrix type.
   * @tparam JPVT nda::MemoryVector type.
   * @tparam TAU nda::MemoryVector type.
   * @param a Input/output matrix. On entry, the m-by-n matrix \f$ \mathbf{A} \f$. On exit, the upper triangle of the
   * array contains the `min(m,n)`-by-n upper trapezoidal matrix \f$ \mathbf{R} \f$; the elements below the diagonal,
   * together with the array `tau`, represent the unitary matrix \f$ \mathbf{Q} \f$ as a product of `min(m,n)`
   * elementary reflectors.
   * @param jpvt Input/output vector. On entry, if `jpvt(j) != 0`, the j-th column of \f$ \mathbf{A} \f$ is permuted to
   * the front of \f$ \mathbf{A P} \f$ (a leading column); if `jpvt(j) == 0`, the j-th column of \f$ \mathbf{A} \f$ is a
   * free column. On exit, if `jpvt(j) == k`, then the j-th column of \f$ \mathbf{A P} \f$ was the the k-th column of
   * \f$ \mathbf{A} \f$.
   * @param tau Output vector. The scalar factors of the elementary reflectors.
   * @return Integer return code from the LAPACK call.
   */
  /*
  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU, MemoryVector W>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau, W &&work) { // NOLINT (temporary views are allowed here)
    static_assert(has_F_layout<A>, "Error in nda::lapack::geqp3: C order not supported");
    static_assert(std::is_same_v<get_value_t<JPVT>, int>, "Error in nda::lapack::geqp3: Pivoting array must have elements of type int");
    static_assert(mem::have_host_compatible_addr_space<A, JPVT, TAU>, "Error in nda::lapack::geqp3: Only CPU is supported");

    auto [m, n] = a.shape();
    EXPECTS(tau.size() >= std::min(m, n));

    // must be lapack compatible
    EXPECTS(a.indexmap().min_stride() == 1);
    EXPECTS(jpvt.indexmap().min_stride() == 1);
    EXPECTS(tau.indexmap().min_stride() == 1);

    // first call to get the optimal buffersize
    using value_type = get_value_t<A>;
    value_type bufferSize_T{};
    int info = 0;
    array<remove_complex_t<value_type>, 1> rwork(2 * n);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info);
    int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

    // allocate work buffer and perform actual library call
    if (work.size() < bufferSize) work.resize(bufferSize);
    EXPECTS(work.indexmap().min_stride() == 1);
    lapack::f77::geqp3(m, n, a.data(), get_ld(a), jpvt.data(), tau.data(), work.data(), bufferSize, rwork.data(), info);
    jpvt -= 1; // Shift to 0-based indexing

    if (info) NDA_RUNTIME_ERROR << "Error in nda::lapack::geqp3: info = " << info;
    return info;
  }

  template <MemoryMatrix A, MemoryVector JPVT, MemoryVector TAU>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU>
             and mem::have_compatible_addr_space<A, JPVT, TAU>)
  int geqp3(A &&a, JPVT &&jpvt, TAU &&tau) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return geqp3(std::forward<A>(a), std::forward<JPVT>(jpvt), std::forward<TAU>(tau), work);
  }
  */

  namespace detail {

    template <nda::MemoryArrayOfRank<3> A, MemoryMatrix JPVT, MemoryMatrix TAU, MemoryVector W>
    requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)    
    auto geqp3_impl(A &&a, JPVT &&jpvt, TAU &&tau, W &&work){
      
      EXPECTS(tau.size() >= a.extent(0));

      auto batchSize = a.extent(2);
      array<int, 1> info(batchSize, 0);

      // must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(a.extent(0) == a.extent(1));
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(jpvt.indexmap().min_stride() == 1);
      EXPECTS(tau.indexmap().min_stride() == 1);

      //get the optimal buffersize
      using value_type = get_value_t<A>;
      value_type bufferSize_T{};

      auto a0 = a(range::all, range::all, 0);
      array<remove_complex_t<value_type>, 1> rwork(2 * a.extent(1));
      lapack::f77::geqp3(a0.extent(0), a0.extent(1), a0.data(), get_ld(a0), jpvt.data(), tau.data(), &bufferSize_T, -1, rwork.data(), info(0));
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      if (work.size() < bufferSize) work.resize(bufferSize);
      for (int i = 0; i < batchSize; ++i) {
        auto a_b = a(range::all, range::all, i);
        lapack::f77::geqp3(a.extent(0), a.extent(1), a_b.data(), get_ld(a_b), jpvt.data() + i * get_ld(jpvt), 
                  tau.data() + i * get_ld(tau), work.data(), bufferSize, rwork.data(), info(i));
      }
      return info;
    }

  }

  template <nda::MemoryArrayOfRank<3> A, MemoryMatrix JPVT, MemoryMatrix TAU, MemoryVector W>
  requires(mem::on_host<A> and is_blas_lapack_v<get_value_t<A>> and have_same_value_type_v<A, TAU, W>
             and mem::have_compatible_addr_space<A, JPVT, TAU, W>)   
  auto geqp3_batch(A && a, JPVT && jpvt, TAU && tau, W && work){
    return detail::geqp3_impl(std::forward<A>(a),jpvt,tau,work);
  }

  namespace detail {
    template <nda::MemoryArrayOfRank<3> A, MemoryMatrix S, nda::MemoryArrayOfRank<3> U, nda::MemoryArrayOfRank<3> VT, MemoryVector W>
      requires(have_same_value_type_v<A, U, VT, W> and mem::have_compatible_addr_space<A, S, U, VT, W> and is_blas_lapack_v<get_value_t<A>>
              and has_F_layout<A> and has_F_layout<U> and has_F_layout<VT>)
    auto gesvd_impl(A &&a, S &&s, U &&u, VT &&vt, W &&work){

      auto batchSize = a.extent(2);
      array<int, 1> info(batchSize, 0);

      auto dm = std::min(a.extent(0), a.extent(1));
      EXPECTS(s.extent(0) == dm);      
      //if (s.size() < dm) s.resize(dm);

      // must be lapack compatible
      EXPECTS(a.indexmap().min_stride() == 1);
      EXPECTS(s.indexmap().min_stride() == 1);
      EXPECTS(u.indexmap().min_stride() == 1);
      EXPECTS(vt.indexmap().min_stride() == 1);

      // call host/device implementation depending on input type
      auto gesvd_call = []<typename... Ts>(Ts &&...args) {
        if constexpr (mem::have_device_compatible_addr_space<A, S, U, VT>) {
  #if defined(NDA_HAVE_DEVICE)
          lapack::device::gesvd(std::forward<Ts>(args)...);
  #else
          compile_error_no_gpu();
  #endif
        } else {
          lapack::f77::gesvd(std::forward<Ts>(args)...);
        }
      };

      // first call to get the optimal buffersize
      using value_type = get_value_t<A>;
      value_type bufferSize_T{};

      auto a0 = a(range::all, range::all, 0);
      auto s0 = s(range::all, 0);
      auto u0 = u(range::all, range::all, 0);
      auto vt0 = vt(range::all, range::all, 0);
      auto rwork = array<remove_complex_t<value_type>, 1, C_layout, heap<mem::get_addr_space<A>>>(5 * dm);
      gesvd_call('A', 'A', a0.extent(0), a0.extent(1), a0.data(), get_ld(a0), s0.data(), u0.data(), get_ld(u0), vt0.data(), get_ld(vt0), &bufferSize_T, -1,
                rwork.data(), info(0));
      int bufferSize = static_cast<int>(std::ceil(std::real(bufferSize_T)));

      // allocate work buffer and perform actual library call
      if (work.size() < bufferSize) work.resize(bufferSize);
      EXPECTS(work.indexmap().min_stride() == 1);
      for (int i = 0; i < batchSize; ++i) { 
        auto a_b = a(range::all, range::all, i);
        auto s_b = s(range::all, i);
        auto u_b = u(range::all, range::all, i);
        auto vt_b = vt(range::all, range::all, i);
        gesvd_call('A', 'A', a.extent(0), a.extent(1), a_b.data(), get_ld(a_b), s_b.data(), u_b.data(), get_ld(u_b), 
                  vt_b.data(), get_ld(vt_b), work.data(), bufferSize, rwork.data(), info(i));
      }
      return info;
    } 

  }
  
  template <nda::MemoryArrayOfRank<3> A, MemoryMatrix S, nda::MemoryArrayOfRank<3> U, nda::MemoryArrayOfRank<3> VT>
    requires(have_same_value_type_v<A, U, VT> and mem::have_compatible_addr_space<A, S, U, VT> and is_blas_lapack_v<get_value_t<A>>)
  auto gesvd_batch(A &&a, S &&s, U &&u, VT &&vt) { // NOLINT (temporary views are allowed here)
    using value_type = get_value_t<A>;
    nda::array<value_type, 1, C_layout, heap<mem::get_addr_space<A>>> work;
    return detail::gesvd_impl(std::forward<A>(a), std::forward<S>(s), std::forward<U>(u), std::forward<VT>(vt), work);
  }

} // namespace lapack

namespace linalg
{
  /**
   * @addtogroup linalg_tools
   * @{
   */

  template <Scalar T, typename LP = F_layout>
  auto get_permutation_matrix_qr(Vector auto const &jpvt) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(jpvt)>);
    int m  = jpvt.size();
    auto P = matrix<T, LP>::zeros(m, m);
    for (int i = 0; i < m; ++i){ 
      P(jpvt(i)-1, i) = T{1};
    }
    return P;
  }

  template <Scalar T, typename LP = F_layout>
  auto get_permutation_array(Matrix auto const &ipiv) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(ipiv)>);
    auto [m,nbatch]  = ipiv.shape();
    int dm = m;
    auto P = nda::array<T, 3, LP>::zeros(m, m, nbatch);
    for (int i = 0; i < nbatch; ++i) {
      P(nda::range::all,nda::range::all,i) = get_permutation_matrix<T,LP>(ipiv(nda::range::all,i),dm);
    }
    return P;
  }

  template <Scalar T, typename LP = F_layout>
  auto get_permutation_array_qr(Matrix auto const &jpvt) {
    static_assert(nda::mem::have_host_compatible_addr_space<decltype(jpvt)>);
    auto [m,nbatch]  = jpvt.shape();
    auto P = nda::array<T, 3, LP>::zeros(m, m, nbatch);
    for (int i = 0; i < nbatch; ++i) {
      P(nda::range::all,nda::range::all,i) = get_permutation_matrix_qr<T,LP>(jpvt(nda::range::all,i));
    }
    return P;
  }

    /*
      Routine to get Q, R matrices from QR decomposition
    */
    template <nda::MemoryArrayOfRank<3> A, nda::MemoryMatrix TAU>
      requires(nda::mem::have_host_compatible_addr_space<A, TAU> and nda::have_same_value_type_v<A, TAU> 
              and nda::is_blas_lapack_v<nda::get_value_t<A>>)
    auto get_qr_matrices(A const &a, TAU &tau, bool complete = false) {

      constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();

      auto all = nda::range::all;

      auto const [m, n, nbatch] = a.shape();
      auto const min_mn = std::min(m, n);
      auto const k      = (complete ? m : min_mn);
      auto Q            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(m, k, nbatch);
      auto R            = memory::buffered_array<MEM, nda::get_value_t<A>, 3, nda::F_layout>::zeros(k, n, nbatch);

      // compute Q matrix
      for(int b = 0; b < nbatch; ++b){
        Q(all, nda::range(min_mn),b) = a(all, nda::range(min_mn), b);
        int info{};
        if constexpr (nda::is_complex_v<nda::get_value_t<A>>) {
          info = nda::lapack::ungqr(Q(all,all,b), tau(all,b));
        } else {
          info = nda::lapack::orgqr(Q(all,all,b), tau(all,b));
        }
        if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::qr_in_place: orgqr/ungqr returned a non-zero value: info = " << info;

        // extract R matrix
        for (int i = 0; i < min_mn; ++i) R(nda::range(i + 1), i, b) = a(nda::range(i + 1), i, b);
        for (int i = min_mn; i < n; ++i) R(all, i, b) = a(all, i, b);
      }
      return std::make_tuple(Q, R);

    }

        /*
      Routine to get Q, R matrices from QR decomposition
    */
    template <nda::MemoryArrayOfRank<2> A, nda::MemoryVector TAU>
      requires(nda::mem::have_host_compatible_addr_space<A, TAU> and nda::have_same_value_type_v<A, TAU> 
              and nda::is_blas_lapack_v<nda::get_value_t<A>>)
    auto get_qr_matrices(A const &a, TAU &tau, bool complete = false) {

      constexpr MEMORY_SPACE MEM = memory::get_memory_space<A>();

      auto all = nda::range::all;

      auto const [m, n] = a.shape();
      auto const min_mn = std::min(m, n);
      auto const k      = (complete ? m : min_mn);
      auto Q            = memory::buffered_array<MEM, nda::get_value_t<A>, 2, nda::F_layout>::zeros(m, k);
      auto R            = memory::buffered_array<MEM, nda::get_value_t<A>, 2, nda::F_layout>::zeros(k, n);

      // compute Q matrix
      Q(all, nda::range(min_mn)) = a(all, nda::range(min_mn));
      int info{};
      if constexpr (nda::is_complex_v<nda::get_value_t<A>>) {
        info = nda::lapack::ungqr(Q(all,all), tau(all));
      } else {
        info = nda::lapack::orgqr(Q(all,all), tau(all));
      }
      if (info != 0) NDA_RUNTIME_ERROR << "Error in nda::qr_in_place: orgqr/ungqr returned a non-zero value: info = " << info;

      // extract R matrix
      for (int i = 0; i < min_mn; ++i) R(nda::range(i + 1), i) = a(nda::range(i + 1), i);
      for (int i = min_mn; i < n; ++i) R(all, i) = a(all, i);

      return std::make_tuple(Q, R);

    }

} // namespace linalg

} // namespace nda
