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

#include <tuple>
#include <variant>
#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"
#include "AFQMC/Wavefunctions/PHMSD.hpp"
#include "AFQMC/Wavefunctions/NOMSD_FT.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Wavefunction 
{
public:
  template<typename Wfn>
  Wavefunction(Wfn&& other) : var(std::forward<Wfn>(other)) {}

  template<typename Wfn>
  Wavefunction& operator=(Wfn&& other) {
    var = std::forward<Wfn>(other);
    return *this;
  }

  /*
   * Returns the memory space.
   */
  MEMORY_SPACE get_memory_space() const;

  int number_of_cholesky_vectors() const;

  void runtime_optimization(WalkerSet<MEM>& wset);

  WALKER_TYPES getWalkerType() const;

  bool isFiniteTemperature() const;

  void vMF(memory::array_view<MEM,ComplexType,1> v, double dt);

  memory::const_shared_array<HOST_MEMORY,ComplexType,3> G_MF();

  // vbias/Energy/Log_Overlap come in two arities: the alternatives disagree on the
  // default for nt (0 for the zero-T wavefunctions, -1 for NOMSD_FT), so the facade
  // forwards without supplying one rather than picking a default here.
  void vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt);
  void vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v, double dt, int nt);

  memory::buffered_array<MEM,ComplexType,4> vHS(memory::array_view<MEM,ComplexType,2> X, double dt);

  nda::array_view<math::sparse::csr_matrix<ComplexType,MEM,int,int>,1>
      vHS_sparse(memory::array_view<MEM,const ComplexType,2> X, double dt);

  std::tuple<int,int> vHS_dims() const;

  void Energy(WalkerSet<MEM>& wset);
  void Energy(WalkerSet<MEM>& wset, int nt);
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,ComplexType,1> Ov);
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,ComplexType,1> Ov, int nt);

  void MixedDensityMatrix(WalkerSet<MEM> const& wset,
                          memory::array_view<MEM,ComplexType,2> G, bool compact);

  void Log_Overlap(WalkerSet<MEM>& wset);
  void Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov);

  // DensityMatrix, updateLogScale and accumulate_estimators keep the forwarding form.
  // The first two are not reachable through this facade, and the alternatives declare
  // them with incompatible parameter lists; accumulate_estimators takes pointers whose
  // type is fixed by the observable handlers, so pinning it here would only move the
  // instantiation up one level.
  template<class... Args>
  void DensityMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.DensityMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void updateLogScale(Args&&... args)
  {
    std::visit([&](auto&& a) { a.updateLogScale(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_estimators(std::forward<Args>(args)...); }, var);
  }

  int total_number_of_references() const;

  int getNMO() const;

  ComplexType getReferenceWeight(int i);

  void getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs);

  HamiltonianTypes getHamType() const;

  nda::array<int,1> getFieldTypes();

  void update_potentials(double dt, memory::array_view<HOST_MEMORY,const ComplexType,1> nMF,
                         memory::array_view<MEM,ComplexType,1> vMF, bool natural_shift);

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                         memory::array_view<HOST_MEMORY,const ComplexType,1> vMF);

  ComplexType getLogScale(SpinTypes s);

  void resetLogScale();

  void setLogPT0(memory::array_view<MEM,ComplexType,1> v);

  memory::array<MEM,ComplexType,1> getLogPT0();

  private:


  std::variant<NOMSD<MEM,PsiT_Matrix<MEM>>,
               NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               NOMSD_FT<MEM,PsiT_Matrix<MEM>>,
               NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               PHMSD<MEM>
              > var;

};

} // namespace afqmc

} // namespace sfqmc

