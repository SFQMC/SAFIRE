/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <type_traits>

#include "config.h"
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Continuous_GeneralUJ
{
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>; 

public:

  Continuous_GeneralUJ() {
   utils::check(false, "Error in Continuous_GeneralUJ: Reached disabled default constructor.");
  }

  Continuous_GeneralUJ(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
                       WALKER_TYPES type,
                       PropagatorTypes ptype,  
                       memory::const_shared_array<MEM,ComplexType,1>&& h0_,
                       math::sparse::CSRMatrix auto&& vn_,
                       math::sparse::CSRMatrix auto&& vnT_,
                       bool shift_ = false,  
                       ComplexType e0 = 0
                      )
      : mpi(_mpi),
        walker_type(type),
        propg_type(ptype),
        nCV(0),
        shift_one_body_terms(shift_),
        E0(e0),
        h0(std::move(h0_)),
        SpVn(std::move(vn_)),
        SpVnT(std::move(vnT_))
  {
    if(propg_type != ContinuousChargePropagator and 
       propg_type != ContinuousSpinPropagator) 
      APP_ABORT(" Error: Wrong PropagatorTypes argument in Continuous_GeneralUJ. ");      
    nCV = SpVn.extent(1);
    utils::check(SpVn.extent(0) == SpVnT.extent(1), "Size mismatch");
    utils::check(SpVn.extent(1) == SpVnT.extent(0), "Size mismatch");
    utils::check(SpVn.extent(0) == h0.size(), "Size mismatch");
  }

  ~Continuous_GeneralUJ() {}

  Continuous_GeneralUJ(const Continuous_GeneralUJ<MEM>& other)            = default;
  Continuous_GeneralUJ& operator=(const Continuous_GeneralUJ<MEM>& other) = default;
  Continuous_GeneralUJ(Continuous_GeneralUJ<MEM>&& other)                 = default;
  Continuous_GeneralUJ& operator=(Continuous_GeneralUJ<MEM>&& other)      = default;

  /*
   * n2IJ maps an index in the ordering of the sparse structures to the ordering 
   * of H1 (the generic spin ordering of 1-body operators)
   * n2IJ is expected in host memory. 
   */ 
  void addOneBodyPropagatorMatrix(nda::array<ComplexType,3> & H1, double dt,
                                  [[maybe_unused]] nda::array<ComplexType, 1> const& vMF,
                                  nda::array<long,1> const& n2IJ)
  {
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int NMO = H1.extent(1) / npol;
    long nIJ = n2IJ.extent(0);

    utils::check(H1.shape() == std::array<long,3>{nspin, npol*NMO, npol*NMO}, "Shape mismatch");
    utils::check(n2IJ.extent(0) == h0.extent(0), "Size mismatch");

    memory::buffered_array<MEM,ComplexType,2> v_(1,nIJ);

    // turn off 1-body terms in vHS temporarily 
    bool shift_one_body_terms_ = shift_one_body_terms;
    shift_one_body_terms = true;
    {
      memory::buffered_array<MEM,ComplexType,2> vMF_(1,vMF.extent(0));
      vMF_(0,nda::range::all) = vMF(); 
      vHS(vMF_, v_, dt);
    }
    shift_one_body_terms = shift_one_body_terms_;

    auto v_host = nda::to_host(v_(0,nda::range::all));
    // if shift_one_body_terms is false, both h0 and hMF are included in vHS
    if(shift_one_body_terms) 
      v_host() += ComplexType(dt) * nda::to_host(h0()); 
    auto H1d = nda::flatten(H1);
    nda::copy_select(true, n2IJ, ComplexType(1.0), v_host, ComplexType(1.0), H1d);
  }

  void getFieldTypes(nda::MemoryVector auto && v) const {
    utils::check(v.size() == nCV, "Size mismatch");
    v() = int(propg_type);
  }
  // nothing to update in Continuous case!
  template<class... Args> void update([[maybe_unused]] Args&&... args) {}


  // v(w,IJ) = sum_n Vn(IJ,n) X(w,n)
  void vHS(nda::MemoryArrayOfRank<2> auto const& X, nda::MemoryMatrix auto& v, double dt)
  {
    auto all = nda::range::all;
    utils::check(X.extent(0) == v.extent(0), "Size mismatch");
    utils::check(SpVn.extent(1) == X.extent(1), "Size mismatch");
    utils::check(SpVn.extent(0) == v.extent(1), "Size mismatch");
    utils::check(v.extent(1) == h0.extent(0), "Size mismatch");

    ComplexType ia(std::sqrt(dt));
    math::sparse::csrmm<'N'>(ia, SpVn, nda::transpose(X),
                             ComplexType(1.0), nda::transpose(v));

    if(shift_one_body_terms) return;

    // multiply by '-i' to compensate for factor of 'i' implicit in the propagator
    // factor of -1 coming from the minus sign in exp(- t V) = exp( i * vHS )
    ia = ComplexType(0.0, dt);

    // v(w,n) = v(w,n) + ia*h0(n);
    if constexpr (MEM==HOST_MEMORY) 
      for(int iw=0; iw<v.extent(0); ++iw) v(iw,all) += ia*h0(); 
    else{
      //FIX: need a better solution here
      for(int iw=0; iw<v.extent(0); ++iw) nda::tensor::add(ia,h0(),"i",ComplexType(1.0),v(iw,all),"i");
      //nda::tensor::add(ia,h0(),"i",ComplexType(1.0),v,"wi");
    }
  }

  // v(n,w) = sum_IJ VnT(n,IJ) G(w,IJ)
  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto&& v, double dt)
  { 
    utils::check(SpVnT.extent(0) == v.extent(1), "Size mismatch");
    utils::check(SpVnT.extent(1) == G.extent(1), "Size mismatch");
    utils::check(G.extent(0) == v.extent(0), "Size mismatch");
    
    math::sparse::csrmm<'N'>(ComplexType(std::sqrt(dt)), SpVnT, nda::transpose(G),
                             ComplexType(1.0), nda::transpose(v));
  }
 
  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    utils::check(false," Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  int number_of_ke_vectors() const { return nCV; }
  int number_of_cholesky_vectors() const { return nCV; }

private:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  PropagatorTypes propg_type;

  int nCV = 0;

  // if shift_one_body_terms=true, h0 is added to the one body propagator.
  // otherwise it is added through vHS
  bool shift_one_body_terms = false;

  ComplexType E0 = 0.0;

  // All sparse matrices have a compact representation of IJ, as defined
  // by n2IJ in ModelHamOps. 

  // constant one-body term associated with the 
  // interacting term.
  memory::const_shared_array<MEM,ComplexType,1> h0;

  // HS operator 
  csrMat<ComplexType> SpVn;

  // transposed HS operator 
  csrMat<ComplexType> SpVnT;

};

} // namespace afqmc

} // namespace sfqmc

