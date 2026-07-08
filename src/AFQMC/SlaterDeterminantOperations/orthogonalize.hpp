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
  
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/config.h"
#include "nda/nda.hpp"
#include "nda/tensor.hpp" 
#include "numerics/sparse/sparse.hpp"
#include "numerics/operations/determinants.hpp"
#include "numerics/operations/tensor.hpp"
#include "numerics/nda_linalg_lapack_extensions.hpp"
  
namespace sfqmc
{ 
namespace afqmc
{ 
namespace det_ops 
{ 

  namespace detail
  {

    template<nda::MemoryVector A_t, nda::MemoryVector B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    int partition(A_t && A, B_t && B, int low, int high) {

      // Selecting last element as the pivot
      auto pivot = A(high);

      // Index of element just before the last element
      // It is used for swapping
      int i = (low - 1);

      for (int j = low; j <= high - 1; j++) {
        // If current element is greater than pivot
        if (A(j).real() >= pivot.real()) {
            i++;
            std::swap(A(i), A(j));
            std::swap(B(i), B(j));
        }
      }

      // Put pivot to its position
      std::swap(A(i + 1), A(high));
      std::swap(B(i + 1), B(high));
    
      // Return the point of partition
      return (i + 1);
    }

    /*
    Routine to sort D vector and return sorted indices
    */
    template<nda::MemoryVector A_t, nda::MemoryVector B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    void quick_sort(A_t && A, B_t && B, int low, int high)
    {
      // Base case: This part will be executed until the starting
      // index low is lesser than the ending index high
      if (low < high) {

        // pi is Partitioning Index, arr[p] is now at
        // right place
        int pi = partition(A, B, low, high);

        // Separately sort elements before and after the
        // Partition Index pi
        quick_sort(A, B, low, pi - 1);
        quick_sort(A, B, pi + 1, high);
      }
    }

    template<nda::MemoryArrayOfRank<2> A_t, nda::MemoryArrayOfRank<2> B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_Fortran() and
              std::decay_t<B_t>::is_stride_order_Fortran())
    void quick_sort(A_t && A, B_t && B)
    {
      auto const [n, nbatch] = A.shape();

      for(int b = 0; b < nbatch; ++b){
        quick_sort(A(nda::range::all,b),B(nda::range::all,b),0,n-1);
      }
    }

    template<nda::MemoryArrayOfRank<2> A_t, nda::MemoryArrayOfRank<2> B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_C() and
              std::decay_t<B_t>::is_stride_order_C())
    void quick_sort(A_t && A, B_t && B)
    {
      auto const [n, nbatch] = A.shape();

      for(int b = 0; b < nbatch; ++b){
        quick_sort(A(b,nda::range::all),B(b,nda::range::all),0,n-1);
      }
    }

    template<nda::MemoryArrayOfRank<1> A_t, nda::MemoryArrayOfRank<1> B_t>
      requires(nda::mem::have_compatible_addr_space<A_t,B_t> and
              std::decay_t<A_t>::is_stride_order_C() and
              std::decay_t<B_t>::is_stride_order_C())
    void quick_sort(A_t && A, B_t && B)
    {
      auto const n = A.size();

      quick_sort(A,B,0,n-1);
    }

  }

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize(A_t && A, B_t && log_detR) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Nel] = A.shape();
  if(A.size()==0) return;
  utils::check( log_detR.extent(0) >= Nw, "Size mismatch");

  // transposing for now, can call geqlf if available in principle
  memory::buffered_array<MEM,Type,3> Q(Nw,Nel,M);
  memory::buffered_array<MEM,Type,2> tau(Nw,Nel);
  memory::buffered_array<MEM,Type,2> scl(Nw,Nel);
  memory::buffered_array<MEM,Type,1> work;

  nda::tensor::add(A,"nab",Q,"nba");  
  nda::lapack::geqrf(nda::transpose(Q),tau,work);

  // log(Det)
  math::log_determinant_from_geqrf(Q,scl,log_detR(nda::range(Nw)));
  
  // Q
  nda::lapack::gqr(nda::transpose(Q),tau,work);

  // copy back
  nda::tensor::add(Q,"nab",A,"nba");  

  // scale A by scl, to make sign of determinant consistent
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>) {
    nda::tensor::elementwise(ComplexType(1.0), scl, "wn", ComplexType(1.0), A, "win", nda::tensor::op::MUL);
  } else {
    for (int i = 0; i < M; ++i)
      A(nda::range::all,i,nda::range::all) *= scl();
  }
}


// Finite temperature w/QR
 /**
   * @brief Performs batched finite temperature stabilization procedure for product of propagator matrices
   * @details Given the arrays \f$ \mathbf{U}\f$, \f$ \mathbf{D}\f$ and \f$ \mathbf{V}\f$ representing the product
   * of propagator matrices at time-step \f$ \tau \f$, \f$ \mathbf{B}(\tau,0) = \mathbf{U}\mathbf{D}\mathbf{V} \f$, this
   * function stabilizes the matrices by first computing the pivoted-QR decomposition of 
   * \f$\mathbf{U}\mathbf{D} = \mathbf{Q}\mathbf{R}\mathbf{P}^{\mathsf T}\f$.
   *
   * Following the QR decomposition, the norm of each row of \f$ \mathbf{R} \f$ is computed and assigned to a vector, \f$ \mathbf{N}_i = \textrm{norm}(\mathbf{R}_i) \f$.
   * The stabilized matrices are then computed according to:
   * \f[
      \begin{align}
        \mathbf{U} & = \mathbf{Q}\mathbf{P}_s^{\mathsf T}\\
        \mathbf{D} & = \mathbf{P}_s\mathbf{N}\mathbf{P}_s^{\mathsf T}*\exp(-\xi) \\
        \mathbf{V} & = \left(\mathbf{P}_s\frac{1}{\mathbf{N}}\mathbf{R}\mathbf{P}^{\mathsf T}\right)\mathbf{V}\\
      \end{align}
    \f]
   * where the matrix \f$ \mathbf{P}_s \f$ is the permutation matrix that sorts the vector \f$ \mathbf{N} \f$ into
   * descending order, and \f$ \xi = 0.5 * \left[\log(\textrm{max}(\mathbf{N})) +\log(\textrm{min}(\mathbf{N}))\right] \f$.
   * 
   * The new scale factor, scl = scl \f$ + \xi \f$
   * @param U Input/Output. Array of propagator matrices.
   * @param D Input/Output. Matrix of propagator eigen/singular values.
   * @param V Input/Output. Array of propagator matrices.
   * @param scl Input/Output. Vector of scale factors for eigen/singular values for each walker in the batch.
   */
template<nda::MemoryArrayOfRank<3> U_t, nda::MemoryArrayOfRank<2> D_t,
         nda::MemoryArrayOfRank<3> V_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<U_t,D_t,V_t,B_t> and
          std::decay_t<U_t>::is_stride_order_C() and std::decay_t<D_t>::is_stride_order_C() and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize_wQR(U_t && U, D_t && D, V_t && V, B_t && scl) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<U_t>();
  using Type = nda::get_value_t<U_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, M2] = U.shape();
  if(U.size()==0) return;
  utils::check( scl.extent(0) >= Nw, "Size mismatch");

  memory::host_array<Type,2,nda::F_layout> UT(M,M);
  memory::host_array<Type,2,nda::F_layout> VT(M,M);
  memory::host_array<int,1,nda::F_layout> jpvt(M);
  memory::host_array<Type,1,nda::F_layout> tau(M);
  memory::host_array<Type,1,nda::F_layout> work;

  Type scl_new;

  // If running on GPU, copy to CPU
  if constexpr (nda::mem::have_device_compatible_addr_space<U_t>){

    auto Uh = nda::to_host(U);
    auto Dh = nda::to_host(D);
    auto Vh = nda::to_host(V);
    auto sclh = nda::to_host(scl);

    for(int b = 0; b < Nw; ++b)
    {
      jpvt() = 0; // NOTE : if jpvt != 0, columns are pre-permuted by geqp3
      for(int col = 0; col < M; ++col)
        UT(nda::range::all,col) = Dh(b,col) * Uh(b,nda::range::all,col);

      // pivoted QR : U*D = Q*R*P^T
      // NOTE : non-batched version of geqp3 returns jpvt indexed starting from 0
      nda::lapack::geqp3(UT,jpvt,tau,work);

      // get Q, R
      std::tie(UT,VT) = nda::linalg::get_qr_matrices(UT, tau, true);

      {
        // FIX : is this scope beneficial?
        memory::buffered_array<HOST_MEMORY,Type,2> M0(M,M);
        memory::buffered_array<HOST_MEMORY,int,1> P1(M);
      
        for(int i = 0; i < M; ++i){ 
          P1(jpvt(i)) = i;
          // D(i) = norm(R(i,:))
          Dh(b,i) = nda::norm(VT(i,nda::range(i,M)));
        }

        for(int i = 0; i < M; ++i)
          for(int j = 0; j < M; ++j)
            M0(i,j) = VT(i,P1(j))/Dh(b,i);

        P1(nda::range::all) = nda::arange(M);
        detail::quick_sort(Dh(b,nda::range::all),P1);

        for(int i = 0; i < M; ++i){
          for(int j = 0; j < M; ++j){
            Uh(b,i,j) = UT(i,P1(j));
            VT(i,j) = M0(P1(i),j);
          }
        }

      }

      scl_new = 0.5 * ( std::log(Dh(b,0).real()) + std::log(Dh(b,M-1).real()) );
      nda::blas::scal(std::exp(-scl_new),Dh(b,nda::range::all));
      sclh(b) += scl_new; 

      //nda::blas::gemm(ComplexType(1.0),VT(nda::range::all,nda::range::all),Vh(b,nda::ellipsis{}),
      //                ComplexType(0.0),Vh(b,nda::ellipsis{}));
      nda::blas::gemm(ComplexType(1.0),VT(nda::range::all,nda::range::all),Vh(b,nda::ellipsis{}),
                      ComplexType(0.0),UT);
              
      V(b,nda::ellipsis{}) = nda::to_device(UT);

    }

    U = nda::to_device(Uh);
    D = nda::to_device(Dh);
    //V = nda::to_device(Vh);
    scl = nda::to_device(sclh);

  }
  else{

    for(int b = 0; b < Nw; ++b)
    {
      jpvt() = 0;  // NOTE : if jpvt != 0, columns are pre-permuted by geqp3
      for(int col = 0; col < M; ++col)
        UT(nda::range::all,col) = D(b,col) * U(b,nda::range::all,col);

      // pivoted QR : U*D = Q*R*P^T
      // NOTE : non-batched version of geqp3 return jpvt indexed starting from 0
      nda::lapack::geqp3(UT,jpvt,tau,work);

      // get Q, R
      std::tie(UT,VT) = nda::linalg::get_qr_matrices(UT, tau, true);

      {
        // FIX : is this scope beneficial?
        memory::buffered_array<MEM,Type,2> M0(M,M);
        memory::buffered_array<MEM,int,1> P1(M);
      
        for(int i = 0; i < M; ++i){ 
          P1(jpvt(i)) = i;
          // D(i) = norm(R(i,:))
          D(b,i) = nda::norm(VT(i,nda::range(i,M)));
        }

        for(int i = 0; i < M; ++i)
          for(int j = 0; j < M; ++j)
            M0(i,j) = VT(i,P1(j))/D(b,i);

        P1(nda::range::all) = nda::arange(M);
        detail::quick_sort(D(b,nda::range::all),P1);

        for(int i = 0; i < M; ++i){
          for(int j = 0; j < M; ++j){
            U(b,i,j) = UT(i,P1(j));
            VT(i,j) = M0(P1(i),j);
          }
        }

      }

      scl_new = 0.5 * ( std::log(D(b,0).real()) + std::log(D(b,M-1).real()) );
      nda::blas::scal(std::exp(-scl_new),D(b,nda::range::all));
      scl(b) += scl_new; 

      //nda::blas::gemm(ComplexType(1.0),VT(nda::range::all,nda::range::all),V(b,nda::ellipsis{}),
      //                ComplexType(0.0),V(b,nda::ellipsis{}));

      nda::blas::gemm(ComplexType(1.0),VT(nda::range::all,nda::range::all),V(b,nda::ellipsis{}),
                      ComplexType(0.0),UT);
      
      V(b,nda::ellipsis{}) = UT;

    }
  }

}

// FIX: there is only a CPU version at the moment
// Finite temperature w/SVD
 /**
   * @brief Performs batched finite temperature stabilization procedure for product of propagator matrices
   * @details Given the arrays \f$ \mathbf{U}\f$, \f$ \mathbf{D}\f$ and \f$ \mathbf{V}\f$ representing the product
   * of propagator matrices at time-step \f$ \tau \f$, \f$ \mathbf{B}(\tau,0) = \mathbf{U}\mathbf{D}\mathbf{V} \f$, this
   * function stabilizes the matrices using the SVD decomposition of 
   * \f$\mathbf{U}\mathbf{D} = \tilde{\mathbf{U}}\tilde{\mathbf{D}}\tilde{\mathbf{V}}^\dagger\f$.
   *
   * The stabilized matrices are then:
   * \f[
      \begin{align}
        \mathbf{U} & = \tilde{\mathbf{U}}\\
        \mathbf{D} & = \tilde{\mathbf{D}}*\exp(-\xi) \\
        \mathbf{V} & = \tilde{\mathbf{V}}^\dagger\mathbf{V}\\
      \end{align}
    \f]
   * where \f$ \xi = 0.5 * \left[\log(\textrm{max}(\tilde{\mathbf{D}})) +\log(\textrm{min}(\tilde{\mathbf{D}}))\right] \f$.
   * @param U Input/Output. Array of propagator matrices.
   * @param D Input/Output. Matrix of propagator eigen/singular values.
   * @param V Input/Output. Array of propagator matrices.
   * @param scl Input/Output. Vector of scale factors for eigen/singular values for each walker in the batch.
   */
template<nda::MemoryArrayOfRank<3> U_t, nda::MemoryArrayOfRank<2> D_t,
         nda::MemoryArrayOfRank<3> V_t, nda::MemoryVector B_t>
requires( nda::mem::have_compatible_addr_space<U_t,D_t,V_t,B_t> and
          std::decay_t<U_t>::is_stride_order_C() and std::decay_t<D_t>::is_stride_order_C() and
          std::decay_t<V_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void orthogonalize_wSVD(U_t && U, D_t && D, V_t && V, B_t && scl) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<U_t>();
  using Type = nda::get_value_t<U_t>;
  static_assert( nda::is_complex_v<Type>, "Type mismatch");
  auto [Nw, M, Q] = U.shape();
  if(U.size()==0) return;
  utils::check( scl.extent(0) >= Nw, "Size mismatch");

  memory::host_array<Type,2,nda::F_layout> UT(M,M);
  memory::host_array<Type,2,nda::F_layout> VT(M,M);
  memory::host_array<nda::remove_complex_t<Type>,1,nda::F_layout> S(M);

  double scl_new;

  if constexpr (MEM==DEVICE_MEMORY)
  {  
    utils::check(false,"Orthogonalization routine with SVD only implemented for CPU");
  }
  else{
    for(int b = 0; b < Nw; ++b)
    {
      for(int row = 0; row < M; ++row)
        VT(nda::range::all,row) = D(b,row) * U(b,nda::range::all,row);

      nda::lapack::gesvd(VT,S,UT,VT);

      scl_new = 0.5 * ( std::log(S(0)) + std::log(S(M-1)) );
      nda::blas::scal(std::exp(-scl_new),S(nda::range::all));
      scl(b) += scl_new; 
      math::copy(S(nda::range::all),D(b,nda::range::all));

      U(b,nda::ellipsis{}) = UT;
      
      //nda::blas::gemm(ComplexType(1.0),VT,V(b,nda::ellipsis{}),
      //                ComplexType(0.0),V(b,nda::ellipsis{}));
      // UT used for temporary storage
      nda::blas::gemm(ComplexType(1.0),VT,V(b,nda::ellipsis{}),
                      ComplexType(0.0),UT);

      V(b,nda::ellipsis{}) = UT;
    }
  }

  /*
  //U*D --> to fortran order
  nda::tensor::contract(D, "nj", U, "nij", VT, "ijn");

  nda::lapack::gesvd_batch(VT,S,UT,VT);  

  for(int b = 0; b < Nw; ++b){
   scl_new = 0.5 * ( std::log(S(0,b)) + std::log(S(M-1,b)) );
   nda::blas::scal(std::exp(-scl_new),S(nda::range::all,b));
   scl(b) += scl_new; 
   math::copy(S(nda::range::all,b),D(b,nda::range::all));
  }

  nda::tensor::add(UT,"ijn",U,"nij");

  // FIX : need V^+ (not just V^T)
  nda::tensor::add(VT,"ijn",M1,"nij");

  // V <-- V'*V
  nda::tensor::contract(V,"nij",nda::conj(M1),"nki",V,"nkj");
  */  


}

template<typename WlkSet, nda::MemoryVector Vec>
requires( not nda::Array<WlkSet> )
void orthogonalize(WlkSet &wset, Vec && ldet, bool importance_sampling = true) 
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<Vec>();
  utils::check(MEM == wset.get_memory_space(), "Memory space mismatch");
  memory::check_memory_space<MEM>(ldet);
  auto walker_type = wset.getWalkerType();
  const int nwalk = wset.size();
  bool ft = wset.isFiniteTemperature();
  utils::check(ldet.size() >= nwalk, "Size mismatch");
  ldet() = ComplexType(0.0);
  if(!ft){
    if(importance_sampling) {
      orthogonalize( wset.SlaterMatrices(Alpha), ldet);
      if(walker_type == COLLINEAR)
        orthogonalize( wset.SlaterMatrices(Beta), ldet);
    } else {
      double scl = ( walker_type == CLOSED ? 2.0 : 1.0 );
      orthogonalize( wset.SlaterMatrices(Alpha), ldet);
      if(walker_type == COLLINEAR)
        orthogonalize( wset.SlaterMatrices(Beta), ldet);
      memory::buffered_array<MEM,ComplexType,1> wgt(nwalk);
      wset.getProperty(WEIGHT, wgt);
      auto wgt_h = nda::to_host(wgt);
      auto ldet_h = nda::to_host(ldet);
      wgt_h() *= nda::exp(scl*ldet_h());
      wgt() = wgt_h();
      wset.setProperty(WEIGHT, wgt);
    }
  }
  else {
    memory::buffered_array<MEM,ComplexType,1> scl_up(nwalk);
    wset.getProperty(LOGSCL_UP, scl_up);
    if(importance_sampling) {
      orthogonalize_wQR(wset.UMatrices(Alpha), wset.DMatrices(Alpha), wset.VMatrices(Alpha), scl_up);
      wset.setProperty(LOGSCL_UP, scl_up);
      if(walker_type == COLLINEAR){
        memory::buffered_array<MEM,ComplexType,1> scl_dn(nwalk);
        wset.getProperty(LOGSCL_DN, scl_dn);
        orthogonalize_wQR(wset.UMatrices(Beta), wset.DMatrices(Beta), wset.VMatrices(Beta), scl_dn);
        wset.setProperty(LOGSCL_DN, scl_dn);
      }
    } else {
        utils::check(false, "finite-T requires importance sampling");
    }
  }
}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc
