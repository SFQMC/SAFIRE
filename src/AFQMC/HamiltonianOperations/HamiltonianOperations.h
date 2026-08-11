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
  template<typename HOps>
  HamiltonianOperations(HOps&& other) : var(std::forward<HOps>(other)) {}

  template<typename HOps>
  HamiltonianOperations& operator=(HOps&& other) {
    var = std::forward<HOps>(other);
    return *this;
  }

  void runtime_optimization(memory::array_view<MEM,const ComplexType,2> G);

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                       memory::array_view<HOST_MEMORY,const ComplexType,1> vMF);

  void energy(memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,const ComplexType,2> G,
              int idet, bool addH1  = true, bool addEJ  = true, bool addEXX = true);

  void energy(SpinTypes spin, memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,const ComplexType,2> G, int idet,
              memory::array_view<MEM,ComplexType,2> EJn,
              bool addH1  = true, bool addEJ  = true, bool addEXX = true);

  memory::buffered_array<MEM,ComplexType,4> vHS(memory::array_view<MEM,ComplexType,2> X, double dt);

  nda::array_view<math::sparse::csr_matrix<ComplexType,MEM,int,int>,1>
      vHS_sparse(memory::array_view<MEM,const ComplexType,2> X, double dt);

  // nMF is accumulated on the host even when the operations live on the device
  void update_potentials(double dt, memory::array_view<HOST_MEMORY,const ComplexType,1> nMF,
                         memory::array_view<MEM,ComplexType,1> vMF, bool natural_shift);

  void ph_reference_energy(SpinTypes spin, memory::array_view<MEM,ComplexType,2> E,
                           memory::array_view<MEM,const ComplexType,2> G,
                           memory::array_view<MEM,ComplexType,2> EJn, bool addH1 = true);

  void ph_excited_energy(SpinTypes spin, int nelec,
                         memory::array_view<MEM,const int,1> iexcit,
                         memory::array_view<MEM,const int,1> refc,
                         memory::array_view<MEM,ComplexType,2> E,
                         memory::array_view<MEM,ComplexType,2> wgt,
                         memory::array_view<MEM,const ComplexType,4> R,
                         memory::array_view<MEM,ComplexType,3> K,
                         bool addH1 = true);

  void vbias(memory::array_view<MEM,const ComplexType,2> G,
             memory::array_view<MEM,ComplexType,2> v, double dt);

  int number_of_cholesky_vectors() const;

  int number_of_ke_vectors() const;

  std::tuple<int,int> vHS_dims() const;

  HamiltonianTypes getHamType() const;

  nda::array<int,1> getFieldTypes() const;

  private:

  std::variant<THCOps<MEM,true>, THCOps<MEM,false>, KPTHCOps<MEM>,
               ModelHamOps<MEM,true>, ModelHamOps<MEM,false>,
               Real3IndexFactorization<MEM>, KP3IndexFactorization<MEM>> var;
};

} // namespace afqmc

} // namespace sfqmc

