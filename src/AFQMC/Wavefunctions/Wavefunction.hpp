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

#include <variant>
#include "AFQMC/config.h"

#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"
#include "AFQMC/Wavefunctions/PHMSD.hpp"
#include "AFQMC/Wavefunctions/NOMSD_FT.hpp"
#include "AFQMC/Wavefunctions/StochasticWfn.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace wavefunction_detail
{
template<class T>
struct is_stochastic_wfn : std::false_type
{};
template<MEMORY_SPACE MEM, class devPsiT>
struct is_stochastic_wfn<StochasticWfn<MEM, devPsiT>> : std::true_type
{};
template<MEMORY_SPACE MEM, class MType2>
struct StochasticInnerStackImpl;
} // namespace wavefunction_detail

template<MEMORY_SPACE MEM>
class Wavefunction 
{
public:
  Wavefunction() { APP_ABORT(" Error: Reached default constructor of Wavefunction. "); }

  explicit Wavefunction(NOMSD<MEM,PsiT_Matrix<MEM>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<MEM,PsiT_Matrix<MEM>> const& other) : var(other) {} 

  explicit Wavefunction(NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>> const& other) : var(other) {}  

  explicit Wavefunction(PHMSD<MEM>&& other) : var(std::move(other)) {}
  explicit Wavefunction(PHMSD<MEM> const& other) : var(other) {} 
  
  // Add finite-T NOMSD wavefunctions
  explicit Wavefunction(NOMSD_FT<MEM,PsiT_Matrix<MEM>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD_FT<MEM,PsiT_Matrix<MEM>> const& other) = delete;

  explicit Wavefunction(NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>> const& other) = delete; 

  explicit Wavefunction(StochasticWfn<MEM, PsiT_Matrix<MEM>>&& other) : var(std::move(other)) {}
  explicit Wavefunction(StochasticWfn<MEM, memory::const_shared_array<MEM, ComplexType, 2>>&& other)
      : var(std::move(other))
  {}

  Wavefunction(Wavefunction const& other) = delete;
  Wavefunction(Wavefunction&& other)      = default;

  Wavefunction& operator=(Wavefunction const& other) = delete;
  Wavefunction& operator=(Wavefunction&& other) = default;

  /*
   * Returns the memory space.
   */
  auto get_memory_space() const 
  {
    return std::visit([&](auto&& a) { return a.get_memory_space(); }, var);
  }

  int number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<class WlkSet>
  void runtime_optimization(WlkSet& wset)
  {
    std::visit([&](auto&& a) { a.runtime_optimization(wset); }, var);
  }

  WALKER_TYPES getWalkerType() const
  {
    return std::visit([&](auto&& a) { return a.getWalkerType(); }, var);
  }

  bool isFiniteTemperature() const
  {
    return std::visit(
        [&](auto&& a) {
          using T = std::decay_t<decltype(a)>;
          if constexpr (requires(T const& x) { x.isFiniteTemperature(); })
            return a.isFiniteTemperature();
          else
            return false;
        },
        var);
  }

  template<class... Args>
  void vMF(Args&&... args)
  {
    std::visit([&](auto&& a) { a.vMF(std::forward<Args>(args)...); }, var);
  }

  auto G_MF()
  {
    return std::visit([&](auto&& a) { return a.G_MF(); }, var);
  }

  template<class... Args>
  void vbias(Args&&... args)
  {
    std::visit([&](auto&& a) { a.vbias(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto vHS(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.vHS(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.vHS_sparse(std::forward<Args>(args)...); }, var);
  }

  auto vHS_dims() const
  {
    return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
  }


  template<class... Args>
  void Energy(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Energy(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void DensityMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.DensityMatrix(std::forward<Args>(args)...); }, var);
  }

  
  template<class... Args>
  void MixedDensityMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.MixedDensityMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void Log_Overlap(Args&&... args)
  {
    std::visit([&](auto&& a) { a.Log_Overlap(std::forward<Args>(args)...); }, var);
  }
  

  auto total_number_of_references() const
  {
    return std::visit([&](auto&& a) { return a.total_number_of_references(); }, var);
  }

  template<class... Args>
  ComplexType getReferenceWeight(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getReferenceWeight(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void getReferences(Args&&... args) 
  {
    std::visit([&](auto&& a) { a.getReferences(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    std::visit([&](auto&& a) { a.accumulate_estimators(std::forward<Args>(args)...); }, var);
  }
/*
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    std::visit([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, var);
  } 
*/

  HamiltonianTypes getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  auto getFieldTypes()
  {
    return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    std::visit([&](auto&& a) { a.update_potentials(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  void updateLogScale(Args&&... args)
  {
    std::visit([&](auto&& a) { a.updateLogScale(std::forward<Args>(args)...); }, var);
  }

  template<class... Args>
  auto getLogScale(Args&&... args)
  {
    return std::visit([&](auto&& a) { return a.getLogScale(std::forward<Args>(args)...); }, var);
  }
  
  void resetLogScale()
  {
    std::visit([&](auto&& a) { a.resetLogScale(); }, var);
  }

  template<class... Args>
  void setLogPT0(Args&&... args)
  {
    std::visit([&](auto&& a) { a.setLogPT0(std::forward<Args>(args)...); }, var);
  }

  //auto getLogPT0()
  //{
  //  return std::visit([&](auto&& a) -> decltype(auto) { return a.getLogPT0(); }, var);
  //}

  auto getLogPT0()
  {
    using R = memory::array<MEM,ComplexType,1>;
    return std::visit([&](auto&& a) -> R { return a.getLogPT0(); }, var);
  }



  bool is_stochastic_wavefunction() const
  {
    return std::visit(
        [](auto&& a) { return wavefunction_detail::is_stochastic_wfn<std::decay_t<decltype(a)>>::value; }, var);
  }

  bool stochastic_inner_walkers_initialized() const
  {
    return std::visit(
        [](auto&& a) {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            return a.inner_walkers_initialized();
          return false;
        },
        var);
  }

  // Current inner trial-ensemble walker count: P (= inner_nwalkers) in the walker-independent P-sample
  // form, or nwalk*P after a conditioned/leapfrog resample; -1 for a non-stochastic or uninitialized
  // wavefunction. Read-only diagnostic (used by stochastic_mean_field_production_order).
  long stochastic_inner_ensemble_size() const
  {
    return std::visit(
        [](auto&& a) -> long {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            return a.inner_walkers_initialized() ? long(a.inner_wset().size()) : -1L;
          return -1L;
        },
        var);
  }

  void initialize_stochastic_inner_walkers(
      ptree const& walker_pt,
      memory::const_shared_array<HOST_MEMORY, ComplexType, 3> const& initial_guess,
      int NAEB)
  {
    std::visit(
        [&](auto&& a) {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            a.initialize_inner_walkers(walker_pt, initial_guess, NAEB);
        },
        var);
  }

  template<class WlkSet>
  void begin_inner_step(WlkSet& wset)
  {
    std::visit(
        [&](auto&& a) {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            a.begin_inner_step(wset);
        },
        var);
  }

  // Realign the conditioned inner ensemble with the outer walker set after an outer population-control
  // event. The driver calls this immediately after wset.popControl(). No-op for non-stochastic
  // wavefunctions and for stochastic trials that carry no slot-conditioned blocks.
  template<class WlkSet>
  void permute_inner_blocks_after_pop(const WlkSet& wset)
  {
    std::visit(
        [&](auto&& a) {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            a.permute_inner_blocks_after_pop(wset);
        },
        var);
  }

  template<MEMORY_SPACE MEM2, class MType2>
  friend struct wavefunction_detail::StochasticInnerStackImpl;

  private:


  std::variant<NOMSD<MEM,PsiT_Matrix<MEM>>,
               NOMSD<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               NOMSD_FT<MEM,PsiT_Matrix<MEM>>,
               NOMSD_FT<MEM,memory::const_shared_array<MEM,ComplexType,2>>,
               PHMSD<MEM>,
               StochasticWfn<MEM, PsiT_Matrix<MEM>>,
               StochasticWfn<MEM, memory::const_shared_array<MEM, ComplexType, 2>>
              > var;

};

} // namespace afqmc

} // namespace sfqmc

