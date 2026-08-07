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

#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/HamiltonianOperations/KPTHCOps.hpp"
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"


namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class HamiltonianOperations 
{

public:

  HamiltonianOperations() = delete;

  template<typename HOps>
  HamiltonianOperations(HOps&& other);

  template<typename HOps>
  HamiltonianOperations(HOps const& other);

  void runtime_optimization(nda::MemoryArrayOfRank<2> auto const& G)
  {
    std::visit([&](auto&& a) { a.runtime_optimization(G); }, var);
  }

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF) 
  {
    return getOneBodyPropagatorMatrix_impl(dt,vMF());
  }

  void energy(nda::MemoryArrayOfRank<2> auto && E, nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1  = true, bool addEJ  = true,bool addEXX = true)
  {
    auto E_=E();
    energy_impl(E_,G(),idet,addH1,addEJ,addEXX);
  }

  void energy(SpinTypes spin, nda::MemoryArrayOfRank<2> auto && E, nda::MemoryArrayOfRank<2> auto const& G, int idet, nda::MemoryArrayOfRank<2> auto && EJn,  bool addH1  = true, bool addEJ  = true,bool addEXX = true)
  { 
    std::visit([&](auto&& s) { s.energy(spin,E,G,idet,EJn,addH1,addEJ,addEXX); }, var);
  }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  }

  template<nda::MemoryMatrix X_t>
  //memory::buffered_array<memory::get_memory_space<X_t>(),ComplexType,4> vHS(X_t&& X, double dt)
  auto vHS(X_t&& X, double dt)
  {
    //auto X_ = X();
    //return vHS_impl(X_,dt);
    return std::visit([&](auto&& a) { return a.vHS(X,dt); }, var);
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    return std::visit([&](auto&& a) { return a.vHS_sparse(X,dt); }, var);
  }

  // instantiate!!!
  void update_potentials(double dt, nda::MemoryVector auto const& nMF, nda::MemoryVector auto&& vMF, bool natural_shift)
  {
    auto n_ = nMF();
    auto v_ = vMF();
    update_potentials_impl(dt,n_,v_,natural_shift);
  }

  template<class... Args>
  void ph_reference_energy(Args&&... args)
  { 
    std::visit([&](auto&& s) { s.ph_reference_energy(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void ph_excited_energy(Args&&... args)
  { 
    std::visit([&](auto&& s) { s.ph_excited_energy(std::forward<Args>(args)...); }, var);
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto&& v, double dt)
  {  
    auto v_ = v();
    vbias_impl(G(),v_,dt);
  }

  void energy_fullG(nda::MemoryArrayOfRank<2> auto && E, nda::MemoryArrayOfRank<2> auto const& G,
                    bool addH1 = true, bool addEJ = true, bool addEXX = true)
  {
    auto E_ = E();
    energy_fullG_impl(E_, G(), addH1, addEJ, addEXX);
  }

  // True iff vbias() accepts a FULL [nwalk, nspin*npol*NMO*npol*NMO] G as well as the half-rotated
  // compact one. See the per-operator definitions for what "full" costs each factorization.
  bool has_fullG_vbias() const;

  int number_of_cholesky_vectors() const;

  int number_of_ke_vectors() const;

  std::tuple<int,int> vHS_dims() const;

  HamiltonianTypes getHamType() const;

  nda::array<int,1> getFieldTypes() const;

  private:

  std::variant<THCOps<MEM,true>, THCOps<MEM,false>, KPTHCOps<MEM>,
               ModelHamOps<MEM,true>, ModelHamOps<MEM,false>, 
               Real3IndexFactorization<MEM>, KP3IndexFactorization<MEM> > var;

  // makes instantiations easier
  nda::array<ComplexType,3> getOneBodyPropagatorMatrix_impl(double dt,
                                                       nda::MemoryVector auto const& vMF); 

  void energy_impl(nda::MemoryArrayOfRank<2> auto& E, nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1, bool addEJ,bool addEXX);

  void vbias_impl(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt);

  void energy_fullG_impl(nda::MemoryArrayOfRank<2> auto& E, nda::MemoryArrayOfRank<2> auto const& G,
                         bool addH1, bool addEJ, bool addEXX);

  auto vHS_impl(nda::MemoryMatrix auto& X, double dt);

  void update_potentials_impl(double dt, nda::MemoryVector auto const& nMF, nda::MemoryVector auto& vMF, bool natural_shift);
};

} // namespace afqmc

} // namespace sfqmc

