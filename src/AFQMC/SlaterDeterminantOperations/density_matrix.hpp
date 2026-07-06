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

#include "AFQMC/config.h"
#include "numerics/device_kernels/cuda/add_scalar.cuh"
#include "utilities/check_strides.hpp"
#include "numerics/operations/determinants.hpp"
#include "numerics/operations/product.hpp"
#include "numerics/operations/split_singular_vals.hpp"
#include "numerics/operations/add_diagonal.hpp"
#include "numerics/nda_functions.hpp"
#include "nda/tensor.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace det_ops 
{

using math::sparse::CSRMatrix;

namespace detail
{

//inverse and log(det)
template<typename A_t, nda::MemoryArrayOfRank<1> O_t, typename T_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, O_t, T_t> and
          std::decay_t<T_t>::is_stride_order_C()
        )
void inverse_logdet(A_t const& A, O_t && ovlp, T_t && TNN, int nbatch = 0, bool invert = true)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto NMO = A.shape()[0];

  utils::check(A.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(TNN.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,int,1> ipiv(NMO);
  memory::buffered_array<MEM,Type,1> work;
  memory::buffered_array<MEM,Type,1> res(1,Type(0.0));

  ipiv() = 0;

  TNN = A;

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,res(0));

  // FIX : is there a better solution for the GPU here?
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>){
    kernels::device::add_scalar(res,ovlp,nbatch);
  }
  else{
    for(int i = 0; i < nbatch; ++i)
      ovlp(i) += res(0);
  }

  // Invert
  if(invert)
    nda::lapack::getri_or_zero(TNN,ipiv,work);
}

template<nda::MemoryArrayOfRank<3> A_t, typename O_t, nda::MemoryArrayOfRank<3> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and
          std::decay_t<T_t>::is_stride_order_C()
        )
void inverse_logdet(A_t const& A, O_t && ovlp, T_t && TNN, bool invert = true)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto [nbatch, NMO, NMO2] = A.shape();

  utils::check(TNN.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NMO);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  TNN = A;

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  if(invert)
    nda::lapack::getri_or_zero(TNN,ipiv,work);
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t,
         nda::MemoryArrayOfRank<3> T_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t, T_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<T_t>::is_stride_order_C()
        )
void log_overlap_impl(A_t const& A, B_t const& B, O_t && ovlp, T_t && TNN, bool herm = true, bool invert = false)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  if(herm)
    utils::check(A.shape() == std::array<long,2>{NEL,NMO}, "Size mismatch");
  else
    utils::check(A.shape() == std::array<long,2>{NMO,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NEL,NEL}, "Size mismatch"); 

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  if(herm) 
    math::product(A,B,TNN);
  else
    math::product<'H'>(A,B,TNN);

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  if(invert) {
    nda::lapack::getri_or_zero(TNN,ipiv,work);
  }
}

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryArrayOfRank<3> T_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t,T_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t, T_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and
          std::decay_t<T_t>::is_stride_order_C() 
        )
void log_overlap_impl(A_t const& A, B_t const& B, O_t && ovlp, T_t && TNN, bool invert = false)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NEL,NEL}, "Size mismatch");

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  // T = dagger(A) * B
  math::product<'H'>(A,B,TNN);

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert
  if(invert) {
    nda::lapack::getri_or_zero(TNN,ipiv,work);
  }
}

//finite-T
template<typename UL_t, typename DL_t, typename VL_t,
             nda::MemoryArrayOfRank<3> UR_t, nda::MemoryArrayOfRank<2> DR_t,
             nda::MemoryArrayOfRank<3> VR_t, nda::MemoryArrayOfRank<1> O_t,
             nda::MemoryArrayOfRank<3> T_t, typename SL_t, nda::MemoryArrayOfRank<1> SR_t>
requires( nda::MemoryMatrix<UL_t> and nda::MemoryVector<DL_t> and nda::MemoryMatrix<VL_t> and
          nda::mem::have_compatible_addr_space<UL_t,DL_t,VL_t,UR_t,DR_t,VR_t,O_t,T_t,SR_t> and
          nda::have_same_value_type_v<UL_t,DL_t,VL_t,UR_t,DR_t,VR_t,O_t,T_t,SR_t> and
          std::decay_t<UR_t>::is_stride_order_C() and std::decay_t<DR_t>::is_stride_order_C()
          and std::decay_t<VR_t>::is_stride_order_C() and std::decay_t<T_t>::is_stride_order_C()
        )
void log_overlap_impl(UL_t const& UL, DL_t const& DL, VL_t const& VL,
                      UR_t && UR, DR_t && DR, VR_t && VR, SL_t const& sclL, SR_t const& sclR,
                      O_t && ovlp, T_t && TNN, bool unitaryL, bool unitaryR, bool invert = false)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<UL_t>();
  using Type = nda::get_value_t<UR_t>;

  auto [nbatch, NMO, NEL] = UR.shape();

  utils::check(UL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(TNN.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch"); 
  utils::check(DR.shape() == std::array<long,2>{nbatch,NMO}, "Size mismatch");
  utils::check(VR.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");

  utils::check(UL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(VL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,int,2> ipiv(nbatch,NMO);
  memory::buffered_array<MEM,Type,1> work;
  memory::buffered_array<MEM,Type,1> ovlp_loc(1,Type(0.0));

  // if running on GPU
  if constexpr (nda::mem::have_device_compatible_addr_space<UL_t>)
  {
    { // scopes to limit memory usage
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        kernels::device::add_scalar(ovlp_loc,ovlp,nbatch);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);

        // M0 <-- VL * DLmin
        //nda::tensor::contract(VL,"ij",DLmin,"j",M0,"ij");
        // FIX: copy because elementwise is in-place (is there an alternative?) 
        M0 = VL;
        nda::tensor::elementwise(ComplexType(1.0),DLmin,"j",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);

        // M1 <-- DRmin * VR
        //nda::tensor::contract(VR,"nij",DRmin,"ni",M1,"nij");
        // FIX: copy because elementwise is in-place (is there an alternative?)  
        M1 = VR;
        nda::tensor::elementwise(ComplexType(1.0),DRmin,"ni",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL);

        // G <-- DRmin*VR*VL*DLmin
        nda::tensor::contract(ComplexType(1.0),M1,"nij",M0,"jk",ComplexType(0.0),TNN,"nik");

        // M0 <-- UL^-1
        if(!unitaryL){
          detail::inverse_logdet(UL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(UL,ovlp,M0,nbatch,false);
          // M0() = nda::dagger(UL);
          nda::tensor::add(nda::conj(UL),"ji",M0,"ij");
        }
        // M1 <-- UR^-1
        if(!unitaryR){
          detail::inverse_logdet(UR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(UR,ovlp,M1,false); 
          nda::tensor::add(nda::conj(UR),"nij",M1,"nji");
        }

        // M0 <-- UL^-1*DLmax^-1
        //nda::tensor::contract(ComplexType(1.0),M0,"ij",DLmax_inv,"j",ComplexType(0.0),M0,"ij");  
        nda::tensor::elementwise(ComplexType(1.0),DLmax_inv,"j",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);

        // M1 <-- DRmax^-1*UR^-1
        //nda::tensor::contract(M1,"nij",DRmax_inv,"ni",M1,"nij"); 
        nda::tensor::elementwise(ComplexType(1.0),DRmax_inv,"ni",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL);

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- DRmax^-1*UR^-1*UL^-1*DLmax^-1 + DRmin*VR*VL*DLmin, i.e. (M1*M0 + G)
      nda::tensor::contract(ComplexType(1.0),M1,"nij",M0,"jk",ComplexType(1.0),TNN,"nik"); 

      // LU of [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]
      nda::lapack::getrf(TNN,ipiv,work);

      // Log(Ovlp)
      math::log_determinant_from_getrf(TNN,ipiv,ovlp);


    } // end of scope for M0, M1
  }
  else{

    {
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        for(int b = 0; b < nbatch; ++b) ovlp(b) += ovlp_loc(0);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);

        // M0 <-- VL * DLmin
        // FIX : is there a way to do this with BLAS/LAPACK?
        for(int col = 0; col < NMO; ++col)
          M0(nda::range::all,col) = VL(nda::range::all,col)*DLmin(col);

        for(int b = 0; b < nbatch; ++b){
          // G <-- DRmin * VR (G used as temporary storage)
          // FIX : is there a way to do this with BLAS/LAPACK?
          for(int row = 0; row < VR.extent(2); ++row)
            M1(b,row,nda::ellipsis{}) = DRmin(b,row) * VR(b,row,nda::range::all);

          // G <-- DRmin*VR*VL*DLmin
          nda::blas::gemm(ComplexType(1.0),M1(b,nda::ellipsis{}),M0,ComplexType(0.0),TNN(b,nda::ellipsis{}));
        }
        // M0 <-- UL^-1
        if(!unitaryL){
          detail::inverse_logdet(UL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(UL,ovlp,M0,nbatch,false);
          // U -> U^+ = U^-1 (for U unitary) 
          //M0() = nda::dagger(UL);
          nda::tensor::add(nda::conj(UL),"ji",M0,"ij");
        }
        // M1 <-- UR^-1
        if(!unitaryR){
          detail::inverse_logdet(UR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(UR,ovlp,M1,false);
          // U -> U^+ = U^-1 (for U unitary) 
          nda::tensor::add(nda::conj(UR),"nij",M1,"nji");
        }

        // M0 <-- UL^-1*DLmax^-1 
        for(int col = 0; col < NMO; ++col)
          nda::blas::scal(DLmax_inv(col),M0(nda::range::all,col)); 

        // M1 <-- DRmax^-1*UR^-1
        for(int b = 0; b < nbatch; ++b)
          for(int row = 0; row < NMO; ++row)
            nda::blas::scal(DRmax_inv(b,row),M1(b,row,nda::ellipsis{}));

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- DRmax^-1*UR^-1*UL^-1*DLmax^-1 + DRmin*VR*VL*DLmin, i.e. (M1*M0 + G)
      for(int b = 0; b < nbatch; ++b) 
        nda::blas::gemm(ComplexType(1.0),M1(b,nda::ellipsis{}),M0,ComplexType(1.0),TNN(b,nda::ellipsis{}));

      // LU of [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]
      nda::lapack::getrf(TNN,ipiv,work);

      // Log(Ovlp)
      math::log_determinant_from_getrf(TNN,ipiv,ovlp);
       
    }

  }
}


template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, typename O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and
          std::decay_t<B_t>::is_stride_order_C()
        )
void LUsolve(A_t && A, B_t && B, O_t && ovlp)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<A_t>;

  auto [nbatch, NMO, NMO2] = A.shape();

  utils::check(B.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch"); 

  memory::buffered_array<MEM,Type,3,nda::F_layout> T0(NMO,NMO,nbatch);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NMO);
  
  memory::buffered_array<MEM,Type,1> work;
 
  ipiv() = 0;

  //nda::tensor::add(B,"nij",T0,"ijn");
  //permute indices
  T0 = nda::permuted_indices_view<nda::encode(nda::permutations::cycle<3>(1))>(std::forward<B_t>(B));

  // LU 
  nda::lapack::getrf(A,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(A,ipiv,ovlp);

  // solve Ax = b
  nda::lapack::getrs(A,T0,ipiv);

  //nda::tensor::add(T0,"ijn",B,"nij");
  //permute indices
  B = nda::permuted_indices_view<nda::encode(nda::permutations::cycle<3>(2))>(T0);
}

}


template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and 
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<B_t>::is_stride_order_C() 
        ) 
void Log_Overlap(A_t const& A, B_t const& B, O_t && ovlp, bool herm = true)
{
  utils::check_strides(B,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>(); 
  using Type = nda::get_value_t<B_t>;
  auto [nbatch, NMO, NEL] = B.shape();
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL); 

  detail::log_overlap_impl(A,B,ovlp,TNN,herm);
}

template<nda::MemoryArrayOfRank<3> A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C()
        )
void Log_Overlap(A_t const& A, B_t const& B, O_t && ovlp)
{
  utils::check_strides(A,B,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;
  auto [nbatch, NMO, NEL] = A.shape();
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  detail::log_overlap_impl(A,B,ovlp,TNN);
}

// finite-T
template<typename UL_t, typename DL_t, typename VL_t, nda::MemoryArrayOfRank<3> UR_t,
         nda::MemoryArrayOfRank<2> DR_t, nda::MemoryArrayOfRank<3> VR_t, nda::MemoryArrayOfRank<1> O_t,
         typename SL_t, nda::MemoryArrayOfRank<1> SR_t>
requires( nda::mem::have_compatible_addr_space<UL_t,DL_t,VL_t,UR_t,DR_t,VR_t,O_t,SR_t> and
          nda::have_same_value_type_v<UL_t,DL_t,VL_t,UR_t,DR_t,VR_t,O_t,SL_t,SR_t> and
          std::decay_t<UR_t>::is_stride_order_C() and std::decay_t<DR_t>::is_stride_order_C()
          and std::decay_t<VR_t>::is_stride_order_C()        
        )
void Log_Overlap(UL_t const& UL, DL_t const& DL, VL_t const& VL,
                 UR_t && UR, DR_t && DR, VR_t && VR, SL_t const& sclL, SR_t const& sclR, O_t && ovlp, 
                 bool unitaryL = false, bool unitaryR = false)
{
  utils::check_strides(UR,DR,VR,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<UL_t>();
  using Type = nda::get_value_t<UR_t>;
  auto [nbatch, NMO, NEL] = UR.shape();
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NMO,NMO);

  detail::log_overlap_impl(UL,DL,VL,UR,DR,VR,sclL,sclR,ovlp,TNN,unitaryL,unitaryR);
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<1> O_t,
         nda::MemoryArrayOfRank<3> QQ0_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,QQ0_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, QQ0_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<QQ0_t>::is_stride_order_C()
        )
void Log_OverlapForWoodbury(A_t const& A, B_t const& B, O_t && ovlp, QQ0_t && QQ0, IVec && ref)
{
  auto all = nda::range::all;
  utils::check_strides(B,ovlp,QQ0,ref);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  utils::check(QQ0.shape() == std::array<long,3>{nbatch,NACT,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TMN(nbatch,NACT,NEL);
  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  // TMN = A*B
  math::product(A,B,TMN);

  // TNN(i,:) = TMN(ref(i),:)
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TMN(n,all,all), Type(0.0), TNN(n,all,all));

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri_or_zero(TNN,ipiv,work);

  // QQ0 = TMN * inv(TNN)
  math::product(TMN,TNN,QQ0);
}

// Density Matrices
template<typename A_t, 
         nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& A, B_t const& B, C_t && C, O_t && ovlp, bool compact = true, bool herm = true)
{
  utils::check_strides(B,C,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  if(herm)
    utils::check(A.shape() == std::array<long,2>{NEL,NMO}, "Size mismatch");
  else
    utils::check(A.shape() == std::array<long,2>{NMO,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  // A*B and overlap
  detail::log_overlap_impl(A,B,ovlp,TNN,herm,true);

  if(compact) {

    // C = T(TNN) * T(B)
    math::product<'T','T'>(TNN,B,C);

  } else {
    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    if (herm)
    {
      // T2 = T(T1) * T(B)
      math::product<'T','T'>(TNN,B,TNM);

      // C = conj(A) * T2
      math::product<'T'>(A,TNM,C);
    }
    else
    {
      // T2 = T1 * H(A)
      if constexpr (CSRMatrix<A_t>) {
        utils::check(false, "finish implementation!!!");
      } else {
        math::product<'N','H'>(TNN,A,TNM);
      }

      // T2 = T(T1) * T(B)
      // C = T( B * T2) = T(T2) * T(B)
      math::product<'T','T'>(TNM,B,C);
    }
  }
}

template<nda::MemoryArrayOfRank<3> A_t, 
         nda::MemoryArrayOfRank<3> B_t, 
         nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,O_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, O_t> and
          std::decay_t<A_t>::is_stride_order_C() and std::decay_t<B_t>::is_stride_order_C() and 
          std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& A, B_t const& B, C_t && C, O_t && ovlp, bool compact = true)
{
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = A.shape();
  utils::check(A.shape() == B.shape(), "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);

  // A*B and overlap
  detail::log_overlap_impl(A,B,ovlp,TNN,true);

  if(compact) {

    math::product<'T','T'>(TNN,B,C);

  } else {

    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    // T2 = T1 * H(A)
    math::product<'N','H'>(TNN,A,TNM);

    // T2 = T(T1) * T(B)
    // C = T( B * T2) = T(T2) * T(B)
    math::product<'T','T'>(TNM,B,C);

  }
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<3> C_t, 
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryArrayOfRank<3> QQ0_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,C_t,QQ0_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, C_t,QQ0_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<QQ0_t>::is_stride_order_C()
          and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrixForWoodbury(A_t const& A, B_t const& B, C_t &&C, O_t && ovlp, QQ0_t && QQ0, IVec && ref, bool compact)
{
  auto all = nda::range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");
  utils::check(QQ0.shape() == std::array<long,3>{nbatch,NACT,NEL}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,Type,3> TAB(nbatch,NACT,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  // TAB = A*B
  math::product(A,B,TAB);

  // TNN(i,:) = TAB(ref(i),:)
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TAB(n,all,all), Type(0.0), TNN(n,all,all));

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri_or_zero(TNN,ipiv,work);

  // QQ0 = TAB * inv(TNN)
  math::product(TAB,TNN,QQ0);

  if(compact) {

    // C = T(TNN) * T(B)
    math::product<'T','T'>(TNN,B,C);

  } else {

    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);

    // TNM = T(TNN) * T(B)
    math::product<'T','T'>(TNN,B,TNM);

    // C = conj(A) * TNM
    math::product<'T'>(A,TNM,C);

  } 
}

template<typename A_t, nda::MemoryArrayOfRank<3> B_t, nda::MemoryArrayOfRank<3> C_t,
         nda::MemoryArrayOfRank<1> O_t, nda::MemoryVector IVec>
requires( (CSRMatrix<A_t> or nda::MemoryMatrix<A_t>) and
          nda::mem::have_compatible_addr_space<A_t,B_t,O_t,C_t,IVec> and
          nda::have_same_value_type_v<A_t, B_t, O_t, C_t> and
          std::decay_t<B_t>::is_stride_order_C() and std::decay_t<C_t>::is_stride_order_C()
        )
void MixedDensityMatrixFromConfiguration(A_t const& A, B_t const& B, C_t &&C, O_t && ovlp, IVec && ref, bool compact)
{
  auto all = nda::range::all;
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = B.shape();
  auto NACT = A.extent(0);
  utils::check(A.shape() == std::array<long,2>{NACT,NMO}, "Size mismatch");
  if(compact)
    utils::check(C.shape() == std::array<long,3>{nbatch, NEL,NMO}, "Size mismatch");
  else
    utils::check(C.shape() == std::array<long,3>{nbatch, NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "Size mismatch");
  utils::check(ref.size() == NEL, "Size mismatch");

  memory::buffered_array<MEM,Type,3> TNN(nbatch,NEL,NEL);
  memory::buffered_array<MEM,Type,3> TAB(nbatch,NACT,NEL);
  memory::buffered_array<MEM,int,2> ipiv(nbatch,NEL);
  memory::buffered_array<MEM,Type,1> work;
  ipiv() = 0;

  // TAB = A*B
  math::product(A,B,TAB);

  // TNN(i,:) = TAB(ref(i),:)
  for(int n=0; n<nbatch; ++n)
    nda::copy_select(false, 0, ref, Type(1.0), TAB(n,all,all), Type(0.0), TNN(n,all,all));

  // LU 
  nda::lapack::getrf(TNN,ipiv,work);

  // Log(Ovlp)
  math::log_determinant_from_getrf(TNN,ipiv,ovlp);

  // Invert 
  nda::lapack::getri_or_zero(TNN,ipiv,work);

  // fill_if_zero()

  if(compact) {

    // C = T(TNN) * T(B)
    math::product<'T','T'>(TNN,B,C);

  } else {

    memory::buffered_array<MEM,Type,3> TNM(nbatch,NEL,NMO);
    // TNM = T(TNN) * T(B)
    math::product<'T','T'>(TNN,B,TNM);

    // C = conj(A) * TNM
    math::product<'T'>(A,TNM,C);

  } 
}

// Finite temperature Density Matrices
 /**
   * @brief Computes the equal-time density matrix \f$ \mathbf{G} \f$, with elements \f$ G^{\sigma\sigma^\prime}_{ij} \equiv \langle c^\dagger_{i\sigma}c_{j\sigma^\prime}\rangle\f$, at finite temperature for a batch of walkers.
   *
   * @details Given the arrays \f$ \mathbf{U}_R\f$, \f$ \mathbf{D}_R\f$, \f$ \mathbf{V}_R\f$, representing walkers,
   * and the arrays \f$ \mathbf{U}_L\f$, \f$ \mathbf{D}_L\f$, \f$ \mathbf{V}_L\f$, representing the trial propagator,
   * this function first computes \f$ \mathcal{G}\f$, with elements \f$ \mathcal{G}^{\sigma\sigma^\prime}_{ij} \equiv \langle c_{i\sigma}c^\dagger_{j\sigma^\prime}\rangle\f$ according to:
   *\f[
   *   \mathcal{G}=(\mathbf{U}_L)^{-1}(\mathbf{D}^\mathrm{max}_L)^{-1}\left[(\mathbf{D}^\mathrm{max}_R)^{-1}
   *   (\mathbf{U}_R)^{-1}(\mathbf{U}_L)^{-1}(\mathbf{D}^\mathrm{max}_L)^{-1}+\mathbf{D}^\mathrm{min}_R\mathbf{V}_R\mathbf{V}_L\mathbf{D}^\mathrm{min}_L\right]^{-1}
   *   (\mathbf{D}^\mathrm{max}_R)^{-1}(\mathbf{U}_R)^{-1},
    \f]
   * where \f$\mathbf{D}_R=\mathbf{D}^\mathrm{max}_R\mathbf{D}^\mathrm{min}_R\f$ and \f$\mathbf{D}_L=\mathbf{D}^\mathrm{min}_L\mathbf{D}^\mathrm{max}_L\f$, with,
   * \f[ \begin{align}
      \mathbf{D}^\textrm{max}_{ii} &= \left\{
      \begin{array}{ll}
      \vert \mathbf{D}_{ii}\vert\,,\,\,\, \textrm{if}\,\,\, \vert \mathbf{D}_{ii}\vert\geq 1 \\
       1\,,\,\,\, \textrm{otherwise}
    \end{array}
    \right. \\
    \mathbf{D}^\mathrm{min}_{ii} &= \left\{
    \begin{array}{ll}
       \textrm{sgn}\left[\mathbf{D}_{ii}\right]\,,\,\,\, \textrm{if}\,\,\, \vert \mathbf{D}_{ii}\vert\geq 1 \\
       \mathbf{D}_{ii}\,,\,\,\, \textrm{otherwise}
    \end{array}
    \right.
    \end{align}
   \f]
   * and returns \f$ \mathbf{G} = \mathbf{I} - \mathcal{G}^{\mathsf T} \f$.
   *
   * @tparam A_t nda::MemoryMatrix type
   * @tparam B_t nda::MemoryVector type
   * @tparam C_t nda::MemoryMatrix type
   * @param UL Trial propagator matrix
   * @param DL Trial propagator eigen/singular values
   * @param VL Trial propagator matrix
   * @param UR Array of walker matrices
   * @param DR Matrix of walker eigen/singular values
   * @param VR Array of walker matrices
   * @param G Output array for density matrices
   * @param ovlp Input/Output vector for overlaps
   * @param sclL Input scalar. Used to scale \f$ \mathbf{D}_L\f$.
   * @param sclR Input scalar Used to scale \f$ \mathbf{D}_R\f$.
   * @return The equal-time density matrix, \f$ \mathbf{G} = \mathbf{I} - \mathcal{G}^{\mathsf T} \f$
   */
template<typename A_t, typename B_t, typename C_t,
         nda::MemoryArrayOfRank<3> D_t,
         nda::MemoryArrayOfRank<2> E_t,
         nda::MemoryArrayOfRank<3> F_t,
         nda::MemoryArrayOfRank<3> G_t, 
         nda::MemoryArrayOfRank<1> O_t,
         typename SL_t,
         nda::MemoryArrayOfRank<1> SR_t>
requires(  nda::MemoryMatrix<A_t> and nda::MemoryVector<B_t> and nda::MemoryMatrix<C_t> and
          nda::mem::have_compatible_addr_space<A_t,B_t,C_t,D_t,E_t,F_t,G_t,O_t,SR_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, D_t, E_t, F_t, G_t, O_t, SR_t> and
          std::decay_t<D_t>::is_stride_order_C() and std::decay_t<E_t>::is_stride_order_C() and
          std::decay_t<F_t>::is_stride_order_C() and std::decay_t<G_t>::is_stride_order_C()
        )
void MixedDensityMatrix(A_t const& UL, B_t const& DL, C_t const& VL, 
                        D_t const& UR, E_t const& DR, F_t const& VR,
                        G_t && G, O_t && ovlp, SL_t const& sclL, SR_t const& sclR, 
                        bool unitaryL = false, bool unitaryR = false)
{

  // FIX : unitaryL, unitaryR are not currently used

  utils::check_strides(UR,DR,VR,G,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = UR.shape();

  //utils::check(UL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  //utils::check(DL.shape() == std::array<long,1>{NMO}, "Size mismatch");
  //utils::check(VL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(DR.shape() == std::array<long,2>{nbatch,NMO}, "Size mismatch");
  utils::check(VR.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(G.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,1> ovlp_loc(1,Type(0.0));

  // if running on GPU
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>)
  {
    {
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      // scopes to limit memory usage
      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        kernels::device::add_scalar(ovlp_loc,ovlp,nbatch);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);
      
        // M0 <-- VL * DLmin
        //nda::tensor::contract(VL,"ij",DLmin,"j",M0,"ij");
        // FIX: copy because elementwise is in-place (is there an alternative?)  
        M0 = VL;
        nda::tensor::elementwise(ComplexType(1.0),DLmin,"j",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);

        // M1 <-- DRmin * VR (M1 used as temporary storage)
        //nda::tensor::contract(VR,"nij",DRmin,"ni",M1,"nij");
        // FIX: copy because elementwise is in-place (is there an alternative?)  
        M1 = VR;
        nda::tensor::elementwise(ComplexType(1.0),DRmin,"ni",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL);

        // G <-- DRmin*VR*VL*DLmin
        nda::tensor::contract(ComplexType(1.0),M1,"nij",M0,"jk",ComplexType(0.0),G,"nik");

        // M0 <-- UL^-1
        if(!unitaryL){
          detail::inverse_logdet(UL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(UL,ovlp,M0,nbatch,false);
          //M0() = nda::dagger(UL);
          nda::tensor::add(nda::conj(UL),"ji",M0,"ij");
        }
        // M1 <-- UR^-1
        if(!unitaryR){
          detail::inverse_logdet(UR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(UR,ovlp,M1,false); 
          nda::tensor::add(nda::conj(UR),"nij",M1,"nji");
        }

        // M0 <-- UL^-1*DLmax^-1
        //nda::tensor::contract(ComplexType(1.0),M0,"ij",DLmax_inv,"j",ComplexType(0.0),M0,"ij");
        nda::tensor::elementwise(ComplexType(1.0),DLmax_inv,"j",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);  

        // M1 <-- DRmax^-1*UR^-1
        //nda::tensor::contract(M1,"nij",DRmax_inv,"ni",M1,"nij"); 
        nda::tensor::elementwise(ComplexType(1.0),DRmax_inv,"ni",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL); 

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- DRmax^-1*UR^-1*UL^-1*DLmax^-1 + DRmin*VR*VL*DLmin, i.e. (M1*M0 + G)
      nda::tensor::contract(ComplexType(1.0),M1,"nij",M0,"jk",ComplexType(1.0),G,"nik"); 

      // LU solve for [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      detail::LUsolve(G,M1,ovlp);

      // Gp^T = <c_i c_j^+>^T
      //    = UL^-1*DLmax^-1*[DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      nda::tensor::contract(ComplexType(-1.0),M0,"ij",M1,"njk",ComplexType(0.0),G,"nki");
      
      // G = I - Gp^T
      math::add_diagonal(ComplexType(1.0),G);

    } // end of scope for M0, M1
  }
  else //if running on CPU
  {
    {
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      // scopes to limit memory usage
      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        for(int b = 0; b < nbatch; ++b) ovlp(b) += ovlp_loc(0);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);

        // M0 <-- VL * DLmin
        // FIX : is there a way to do this with BLAS/LAPACK?
        for(int col = 0; col < NMO; ++col)
          M0(nda::range::all,col) = VL(nda::range::all,col)*DLmin(col);

        for(int b = 0; b < nbatch; ++b){
          // M1 <-- DRmin * VR (M1 used as temporary storage)
          // FIX : is there a way to do this with BLAS/LAPACK?
          for(int row = 0; row < VR.extent(2); ++row)
            M1(b,row,nda::ellipsis{}) = DRmin(b,row) * VR(b,row,nda::range::all);

          // G <-- DRmin*VR*VL*DLmin
          nda::blas::gemm(ComplexType(1.0),M1(b,nda::ellipsis{}),M0,ComplexType(0.0),G(b,nda::ellipsis{}));
        }
        // M0 <-- UL^-1
        if(!unitaryL){
          detail::inverse_logdet(UL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(UL,ovlp,M0,nbatch,false);
          // U -> U^+ = U^-1 (for U unitary) 
          //M0() = nda::dagger(UL);
          nda::tensor::add(nda::conj(UL),"ji",M0,"ij");
        }
        // M1 <-- UR^-1
        if(!unitaryR){
          detail::inverse_logdet(UR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(UR,ovlp,M1,false);
          // U -> U^+ = U^-1 (for U unitary) 
          nda::tensor::add(nda::conj(UR),"nij",M1,"nji");
        }

        // M0 <-- UL^-1*DLmax^-1 
        for(int col = 0; col < NMO; ++col)
          nda::blas::scal(DLmax_inv(col),M0(nda::range::all,col)); 

        // M1 <-- DRmax^-1*UR^-1
        for(int b = 0; b < nbatch; ++b)
          for(int row = 0; row < NMO; ++row)
            nda::blas::scal(DRmax_inv(b,row),M1(b,row,nda::ellipsis{}));

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- DRmax^-1*UR^-1*UL^-1*DLmax^-1 + DRmin*VR*VL*DLmin, i.e. (M1*M0 + G)
      for(int b = 0; b < nbatch; ++b) 
        nda::blas::gemm(ComplexType(1.0),M1(b,nda::ellipsis{}),M0,ComplexType(1.0),G(b,nda::ellipsis{})); 
      
      // LU solve for [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      detail::LUsolve(G,M1,ovlp);

      // Gp = <c_i c_j^+>
      //    = UL^-1*DLmax^-1*[DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      for(int b = 0; b < nbatch; ++b){
        nda::blas::gemm(ComplexType(-1.0),nda::transpose(M1(b,nda::ellipsis{})),nda::transpose(M0),
                        ComplexType(0.0),G(b,nda::ellipsis{}));
        //I-Gp^T  
        for(int i = 0; i < NMO; ++i){
          G(b,i,i) += Type(1.0);
        }
      }
    } // end of scope for M0, M1
  
  }

}

// Finite temperature Density Matrices
 /**
   * @brief Computes the equal-time density matrix \f$ \mathbf{G} \f$, with elements \f$ G^{\sigma\sigma^\prime}_{ij} \equiv \langle c^\dagger_{i\sigma}c_{j\sigma^\prime}\rangle\f$, at finite temperature for a batch of walkers.
   *
   * @details Given the arrays \f$ \mathbf{U}_R\f$, \f$ \mathbf{D}_R\f$, \f$ \mathbf{V}_R\f$, representing walkers,
   * and the arrays \f$ \mathbf{U}_L\f$, \f$ \mathbf{D}_L\f$, \f$ \mathbf{V}_L\f$, representing the trial propagator,
   * this function computes \f$ \mathbf{G}\f$ according to:
   * \f[
       \mathbf{G} = \mathbf{U}_L^{\mathsf T}\mathbf{D}^\mathrm{min}_L
            \left[\mathbf{D}^\mathrm{min}_L\mathbf{U}_L\mathbf{U}_R\mathbf{D}^\mathrm{min}_R
                +(\mathbf{D}^\mathrm{max}_L)^{-1}(\mathbf{V}_L)^{-1}(\mathbf{V}_R)^{-1}(\mathbf{D}^\mathrm{max}_R)^{-1}
                \right]^{-{\mathsf T}}
                \mathbf{D}^\mathrm{min}_R\mathbf{U}_R^{\mathsf T},
     \f]
   * where \f$\mathbf{D}_R=\mathbf{D}^\mathrm{max}_R\mathbf{D}^\mathrm{min}_R\f$ and \f$\mathbf{D}_L=\mathbf{D}^\mathrm{min}_L\mathbf{D}^\mathrm{max}_L\f$, with,
   * \f[ \begin{align}
      \mathbf{D}^\textrm{max}_{ii} &= \left\{
      \begin{array}{ll}
      \vert \mathbf{D}_{ii}\vert\,,\,\,\, \textrm{if}\,\,\, \vert \mathbf{D}_{ii}\vert\geq 1 \\
       1\,,\,\,\, \textrm{otherwise}
    \end{array}
    \right. \\
    \mathbf{D}^\mathrm{min}_{ii} &= \left\{
    \begin{array}{ll}
       \textrm{sgn}\left[\mathbf{D}_{ii}\right]\,,\,\,\, \textrm{if}\,\,\, \vert \mathbf{D}_{ii}\vert\geq 1 \\
       \mathbf{D}_{ii}\,,\,\,\, \textrm{otherwise}
    \end{array}
    \right.
    \end{align}
   \f]
   *
   * @tparam A_t nda::MemoryMatrix type
   * @tparam B_t nda::MemoryVector type
   * @tparam C_t nda::MemoryMatrix type
   * @param UL Trial propagator matrix
   * @param DL Trial propagator eigen/singular values
   * @param VL Trial propagator matrix
   * @param UR Array of walker matrices
   * @param DR Matrix of walker eigen/singular values
   * @param VR Array of walker matrices
   * @param G Output array for density matrices
   * @param ovlp Input/Output vector for overlaps
   * @param sclL Input scalar. Used to scale \f$ \mathbf{D}_L\f$.
   * @param sclR Input scalar Used to scale \f$ \mathbf{D}_R\f$.
   * @return The equal-time density matrix, \f$ \mathbf{G} \f$
   */
template<nda::MemoryMatrix A_t, 
         nda::MemoryVector B_t, 
         nda::MemoryMatrix C_t,
         nda::MemoryArrayOfRank<3> D_t,
         nda::MemoryArrayOfRank<2> E_t,
         nda::MemoryArrayOfRank<3> F_t,
         nda::MemoryArrayOfRank<3> G_t, 
         nda::MemoryArrayOfRank<1> O_t,
         typename SL_t,
         nda::MemoryArrayOfRank<1> SR_t>
requires( nda::mem::have_compatible_addr_space<A_t,B_t,C_t,D_t,E_t,F_t,G_t,O_t,SR_t> and
          nda::have_same_value_type_v<A_t, B_t, C_t, D_t, E_t, F_t, G_t, O_t, SR_t> and
          std::decay_t<D_t>::is_stride_order_C() and std::decay_t<E_t>::is_stride_order_C() and
          std::decay_t<F_t>::is_stride_order_C() and std::decay_t<G_t>::is_stride_order_C()
        )
void MixedDensityMatrix_v2(A_t const& UL, B_t const& DL, C_t const& VL, 
                        D_t const& UR, E_t const& DR, F_t const& VR,
                        G_t && G, O_t && ovlp, SL_t const& sclL, SR_t const& sclR, 
                        bool unitaryL = false, bool unitaryR = false)
{

  utils::check_strides(UR,DR,VR,G,ovlp);
  constexpr MEMORY_SPACE MEM = memory::get_memory_space<A_t>();
  using Type = nda::get_value_t<B_t>;

  auto [nbatch, NMO, NEL] = UR.shape();
  
  utils::check(UL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(DL.shape() == std::array<long,1>{NMO}, "Size mismatch");
  utils::check(VL.shape() == std::array<long,2>{NMO,NMO}, "Size mismatch");
  utils::check(DR.shape() == std::array<long,2>{nbatch,NMO}, "Size mismatch");
  utils::check(VR.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");
  utils::check(ovlp.size() >= nbatch, "");
  utils::check(G.shape() == std::array<long,3>{nbatch,NMO,NMO}, "Size mismatch");

  memory::buffered_array<MEM,Type,1> ovlp_loc(1,Type(0.0));

  // overlap is accumulated, so it must first be zeroed
  //ovlp() = Type(0.0);

  // if running on GPU
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>)
  {
    { // scopes to limit memory usage
      
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        kernels::device::add_scalar(ovlp_loc,ovlp,nbatch);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);

        // M0 <-- VL^-1
        if(!unitaryL){
          detail::inverse_logdet(VL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(VL,ovlp,M0,nbatch,false);
          //M0() = nda::dagger(VL);
          nda::tensor::add(nda::conj(VL),"ji",M0,"ij");
        }
        // M1 <-- VR^-1
        if(!unitaryR){
          detail::inverse_logdet(VR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(VR,ovlp,M1,false); 
          nda::tensor::add(nda::conj(VR),"nij",M1,"nji");
        }

        // M0 <-- DLmax^-1*VL^-1
        nda::tensor::elementwise(ComplexType(1.0),DLmax_inv,"i",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);

        // M1 <-- VR^-1*DRmax^-1   (M1 used as temporary storage)
        nda::tensor::elementwise(ComplexType(1.0),DRmax_inv,"nj",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL);

        // G <-- (DLmax^-1*VL^-1*VR^-1*DRmax^-1)^T
        nda::tensor::contract(ComplexType(1.0),M1,"nkj",M0,"ik",ComplexType(0.0),G,"nji");

        // M0 <-- UL^T*DLmin
        //nda::tensor::contract(ComplexType(1.0),UL,"ij",DLmin,"i",ComplexType(0.0),M0,"ji");  
        // FIX: copy because elementwise is in-place (is there an alternative?) 
        M0 = nda::transpose(UL);
        nda::tensor::elementwise(ComplexType(1.0),DLmin,"j",ComplexType(1.0),M0,"ij",nda::tensor::op::MUL);

        // M1 <-- DRmin*UR^T
        //nda::tensor::contract(ComplexType(1.0),UR,"nij",DRmin,"nj",ComplexType(0.0),M1,"nji");
        // FIX: copy because elementwise is in-place (is there an alternative?) 
        nda::tensor::add(UR,"nij",M1,"nji");
        nda::tensor::elementwise(ComplexType(1.0),DRmin,"ni",ComplexType(1.0),M1,"nij",nda::tensor::op::MUL);

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- (DLmin*UL*UR*DRmin + DLmax^-1*VL^-1*VR^-1*DRmax^-1)^T , i.e. G = (M1^T*M0^T + G)^T
      nda::tensor::contract(ComplexType(1.0),M1,"nij",M0,"jk",ComplexType(1.0),G,"nik"); 

      // LU solve for [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      detail::LUsolve(G,M1,ovlp);

      // G = <c_i^+ c_j>
      //    = UL^-1*DLmax^-1*[DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      nda::tensor::contract(ComplexType(1.0),M0,"ij",M1,"njk",ComplexType(0.0),G,"nik");

    } // end of scope for M0, M1
  }
  else //if running on CPU
  {
    {
      memory::buffered_array<MEM,Type,2> M0(NMO,NMO);
      memory::buffered_array<MEM,Type,3> M1(nbatch,NMO,NMO);

      // scopes to limit memory usage
      {
        memory::buffered_array<MEM,Type,2> DRmin(nbatch,NMO);
        memory::buffered_array<MEM,Type,2> DRmax_inv(nbatch,NMO);
        memory::buffered_array<MEM,Type,1> DLmin(NMO);
        memory::buffered_array<MEM,Type,1> DLmax_inv(NMO);

        math::splitDmatrix(DL, DLmin, DLmax_inv, ovlp_loc, sclL);
        for(int b = 0; b < nbatch; ++b) ovlp(b) += ovlp_loc(0);
        math::splitDmatrix(DR, DRmin, DRmax_inv, ovlp, sclR);

        // M0 <-- VL^-1
        if(!unitaryL){
          detail::inverse_logdet(VL,ovlp,M0,nbatch);  
        }
        else{
          // still need to compute log(det(UL)) if it is not stored, in the event UL is complex
          // and det(UL) has a phase (i.e. det(UL) =/= +-1)
          detail::inverse_logdet(VL,ovlp,M0,nbatch,false);
          // M0() = nda::dagger(VL);
          nda::tensor::add(nda::conj(VL),"ji",M0,"ij");
        }
        // M1 <-- VR^-1
        if(!unitaryR){
          detail::inverse_logdet(VR,ovlp,M1); 
        }
        else{
          // still need to compute log(det(UR)) if it is not stored, in the event UR is complex
          // and det(UR) has a phase (i.e. det(UR) =/= +-1)
          detail::inverse_logdet(VR,ovlp,M1,false); 
//          for(int b = 0; b < nbatch; ++b)
//            M1(b,nda::range::all,nda::range::all) = nda::dagger(VR(b,nda::ellipsis{}));
          nda::tensor::add(nda::conj(VR),"bji",M1,"bij");
        }

        // M0 <-- DLmax^-1*VL^-1
        for(int row = 0; row < NMO; ++row)
          nda::blas::scal(DLmax_inv(row),M0(row,nda::range::all));
          //M0(row,nda::range::all) *= DLmax_inv(row);

        // M1 <-- VR^-1*DRmax^-1   (M1 used as temporary storage)
        for(int b = 0; b < nbatch; ++b){
          for(int col = 0; col < NMO; ++col)
            nda::blas::scal(DRmax_inv(b,col),M1(b,nda::range::all,col));
            //M1(b,nda::range::all,col) *= DRmax_inv(b,col);

          // G <-- (DLmax^-1*VL^-1*VR^-1*DRmax^-1)^T
          nda::blas::gemm(ComplexType(1.0),nda::transpose(M1(b,nda::ellipsis{})),
                    nda::transpose(M0),ComplexType(0.0),G(b,nda::ellipsis{})); 
        }

        // M0 <-- UL^T*DLmin
        for(int row = 0; row < NMO; ++row)
          M0(nda::range::all,row) = DLmin(row) * UL(row,nda::range::all);

        // M1 <-- DRmin*UR^T
        for(int b = 0; b < nbatch; ++b)
          // FIX : is there a way to do this with BLAS/LAPACK?
          for(int row = 0; row < VR.extent(2); ++row)
            M1(b,row,nda::ellipsis{}) = DRmin(b,row) * UR(b,nda::range::all,row);

      } // end of scope for DRmin, DRmax_inv, DLmin, DLmax_inv 

      // G <-- (DLmin*UL*UR*DRmin + DRmax^-1*VL^-1*VR^-1*DRmax^-1)^T , i.e. G = (M1^T*M0^T + G)^T
      for(int b = 0; b < nbatch; ++b)
        nda::blas::gemm(ComplexType(1.0),M1(b,nda::ellipsis{}),M0,ComplexType(1.0),G(b,nda::ellipsis{})); 

      // LU solve for [DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      detail::LUsolve(G,M1,ovlp);

      // G = <c_i^+ c_j>
      //    = UL^-1*DLmax^-1*[DRmax^-1*UR^-1*UL^-1*DLmax^-1+DRmin*VR*VL*DLmin]^-1*DRmax^-1*UR^-1
      for(int b = 0; b < nbatch; ++b)
        nda::blas::gemm(ComplexType(1.0),M0,M1(b,nda::ellipsis{}),ComplexType(0.0),G(b,nda::ellipsis{}));

    }

  }
}

} // namespace det_ops 

} // namespace afqmc

} // namespace sfqmc

