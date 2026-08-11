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

#include <variant>

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"

namespace sfqmc
{
namespace afqmc
{
  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::runtime_optimization(memory::array_view<M,const ComplexType,2> G)
  {
    std::visit([&](auto&& a) { a.runtime_optimization(G); }, var);
  }

  template<MEMORY_SPACE M>
  nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix(double dt,
                       memory::array_view<HOST_MEMORY,const ComplexType,1> vMF)
  {
    return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(dt,vMF); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::energy(memory::array_view<M,ComplexType,2> E,
                                        memory::array_view<M,const ComplexType,2> G,
                                        int idet, bool addH1, bool addEJ, bool addEXX)
  {
    std::visit([&](auto&& a) { a.energy(E,G,idet,addH1,addEJ,addEXX); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::energy(SpinTypes spin, memory::array_view<M,ComplexType,2> E,
                                        memory::array_view<M,const ComplexType,2> G, int idet,
                                        memory::array_view<M,ComplexType,2> EJn,
                                        bool addH1, bool addEJ, bool addEXX)
  {
    std::visit([&](auto&& s) { s.energy(spin,E,G,idet,EJn,addH1,addEJ,addEXX); }, var);
  }

  template<MEMORY_SPACE M>
  memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS(
                       memory::array_view<M,ComplexType,2> X, double dt)
  {
    return std::visit([&](auto&& a) { return a.vHS(X,dt); }, var);
  }

  template<MEMORY_SPACE M>
  nda::array_view<math::sparse::csr_matrix<ComplexType,M,int,int>,1>
      HamiltonianOperations<M>::vHS_sparse(memory::array_view<M,const ComplexType,2> X, double dt)
  {
    return std::visit([&](auto&& a) { return a.vHS_sparse(X,dt); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::update_potentials(double dt,
                       memory::array_view<HOST_MEMORY,const ComplexType,1> nMF,
                       memory::array_view<M,ComplexType,1> vMF, bool natural_shift)
  {
    std::visit([&](auto&& s) { s.update_potentials(dt,nMF,vMF,natural_shift); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::ph_reference_energy(SpinTypes spin,
                       memory::array_view<M,ComplexType,2> E,
                       memory::array_view<M,const ComplexType,2> G,
                       memory::array_view<M,ComplexType,2> EJn, bool addH1)
  {
    std::visit([&](auto&& s) { s.ph_reference_energy(spin,E,G,EJn,addH1); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::ph_excited_energy(SpinTypes spin, int nelec,
                       memory::array_view<M,const int,1> iexcit,
                       memory::array_view<M,const int,1> refc,
                       memory::array_view<M,ComplexType,2> E,
                       memory::array_view<M,ComplexType,2> wgt,
                       memory::array_view<M,const ComplexType,4> R,
                       memory::array_view<M,ComplexType,3> K,
                       bool addH1)
  {
    std::visit([&](auto&& s) { s.ph_excited_energy(spin,nelec,iexcit,refc,E,wgt,R,K,addH1); }, var);
  }

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::vbias(memory::array_view<M,const ComplexType,2> G,
                                       memory::array_view<M,ComplexType,2> v, double dt)
  {
    std::visit([&](auto&& s) { s.vbias(G,v,dt); }, var);
  }

  // accessors
  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_ke_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_ke_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  std::tuple<int,int> HamiltonianOperations<M>::vHS_dims() const
  {
    return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
  }

  template<MEMORY_SPACE M>
  HamiltonianTypes HamiltonianOperations<M>::getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  template<MEMORY_SPACE M>
  nda::array<int,1> HamiltonianOperations<M>::getFieldTypes() const
  {
    return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
  }

  template class HamiltonianOperations<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
  template class HamiltonianOperations<DEVICE_MEMORY>;
#endif

} // namespace afqmc

} // namespace sfqmc
