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
#include "AFQMC/Wavefunctions/StochasticWfn.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace wavefunction_detail
{
/**
 * @brief Trait identifying the StochasticWfn alternatives of the Wavefunction variant.
 *
 * @details Only StochasticWfn carries an inner ensemble, so the variant's stochastic-only members
 * must compile away for every other alternative rather than requiring them all to grow no-op
 * overloads. This trait is what the `if constexpr` in those members tests.
 *
 * @param T the wavefunction alternative being tested
 */
template<class T>
struct is_stochastic_wfn : std::false_type
{};
template<MEMORY_SPACE MEM, class devPsiT>
struct is_stochastic_wfn<StochasticWfn<MEM, devPsiT>> : std::true_type
{};
/// @brief Concrete StochasticInnerStack, defined in WavefunctionFactory.cpp where the inner
/// wavefunction and propagator types are known.
template<MEMORY_SPACE MEM, class MType2>
struct StochasticInnerStackImpl;
} // namespace wavefunction_detail

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

  /// @brief True iff the underlying Hamiltonian operator accepts a FULL (un-rotated) G in vbias.
  /// Callers that build a mean field with no half-rotated form -- StochasticWfn::vMF at
  /// inner_n_samples > 1 -- must gate on this rather than discover the gap as a size mismatch deep in
  /// the operator.
  bool has_fullG_vbias() const;

  nda::array<int,1> getFieldTypes();

  void update_potentials(double dt, memory::array_view<HOST_MEMORY,const ComplexType,1> nMF,
                         memory::array_view<MEM,ComplexType,1> vMF, bool natural_shift);

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                         memory::array_view<HOST_MEMORY,const ComplexType,1> vMF);

  ComplexType getLogScale(SpinTypes s);

  void resetLogScale();

  void setLogPT0(memory::array_view<MEM,ComplexType,1> v);

  memory::array<MEM,ComplexType,1> getLogPT0();

  /// @brief True iff the held alternative is a StochasticWfn; false for every other trial.
  bool is_stochastic_wavefunction() const
  {
    return std::visit(
        [](auto&& a) { return wavefunction_detail::is_stochastic_wfn<std::decay_t<decltype(a)>>::value; }, var);
  }

  /// @brief True once a stochastic trial's inner ensemble has been allocated; false if non-stochastic.
  bool stochastic_inner_walkers_initialized() const
  {
    return visit_stochastic_or([](auto&& a) { return a.inner_walkers_initialized(); }, false);
  }

  /**
   * @brief Read-only diagnostic: size of a stochastic trial's inner ensemble.
   *
   * @details P in the walker-independent form, or nwalk*P once a conditioned resample has expanded it;
   * -1 if the trial is not stochastic or its ensemble is not yet initialized.
   */
  long stochastic_inner_ensemble_size() const
  {
    return visit_stochastic_or(
        [](auto&& a) -> long {
          return a.inner_walkers_initialized() ? long(a.inner_wset().size()) : -1L;
        },
        -1L);
  }

  /// @brief Cumulative field-chain Metropolis acceptance on this rank; -1 if non-stochastic.
  double stochastic_inner_chain_acceptance() const
  {
    return visit_stochastic_or([](auto&& a) { return a.inner_chain_acceptance(); }, -1.0);
  }

  /// @brief True when measure_energy() advances a live persistent pool, so replica-path tests can
  /// assert they are not passing vacuously; false if non-stochastic.
  bool stochastic_measure_advances_pool() const
  {
    return visit_stochastic_or([](auto&& a) { return a.measure_advances_pool(); }, false);
  }

  /// @brief Leapfrog conditioning-magnitude checksum; -1 if non-stochastic. See
  /// StochasticWfn::inner_cond_mag_sum.
  double stochastic_inner_cond_mag_sum() const
  {
    return visit_stochastic_or([](auto&& a) { return a.inner_cond_mag_sum(); }, -1.0);
  }

  /// @brief Trained inner timestep in force on a stochastic trial; -1 if non-stochastic. Exists so a
  /// test can prove the factory's stamp reached the built object. See StochasticWfn::inner_timestep.
  double stochastic_inner_timestep() const
  {
    return visit_stochastic_or([](auto&& a) { return a.inner_timestep(); }, -1.0);
  }

  /**
   * @brief Allocate a stochastic trial's inner ensemble; no-op for every other trial.
   *
   * @param walker_params the walker-set input block, shared with the outer walkers
   * @param initial_guess per-spin Slater matrices of the anchor determinant
   * @param NAEB number of spin-down electrons, sizing the beta block of a COLLINEAR anchor
   */
  void initialize_stochastic_inner_walkers(
      WalkerSetParameters const& walker_params,
      std::vector<nda::matrix<ComplexType>> const& initial_guess,
      int NAEB)
  {
    visit_stochastic([&](auto&& a) { a.initialize_inner_walkers(walker_params, initial_guess, NAEB); });
  }

  /**
   * @brief Open an outer propagation step on a stochastic trial; no-op for every other trial.
   *
   * @param wset the outer walker set, whose buffer holds the field-chain state
   */
  void begin_inner_step(WalkerSet<MEM>& wset);

  /**
   * @brief Estimator entry point for the local energy.
   *
   * @details A StochasticWfn averages over measurement replicas of its field pool; every other trial
   * is plain Energy(), as is a stochastic trial with a single replica.
   *
   * @param wset the outer walker set, non-const because advancing the pool writes its field block
   * @param E output energies, [nwalk, 3]
   * @param Ov output LOG overlap per walker
   * @param nt time slice index
   */
  void measure_energy(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> E,
                      memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /**
   * @brief Realign a stochastic trial's conditioned inner blocks after an outer population-control
   *        event.
   *
   * @details THE DRIVER MUST CALL THIS IMMEDIATELY AFTER wset.popControl(). That ordering is the
   * contract and it is not visible from the signature: any reduction taken between the two would pair
   * post-branch walkers with pre-branch inner blocks. No-op unless the trial is stochastic and
   * walker-conditioned.
   *
   * @param wset the post-population-control outer walker set
   */
  void permute_inner_blocks_after_pop(WalkerSet<MEM> const& wset);

  /**
   * @brief Snapshot the conditioned inner-ensemble magnitudes into the walkers before branching.
   *
   * @details THE DRIVER MUST CALL THIS IMMEDIATELY BEFORE wset.popControl(), paired with
   * permute_inner_blocks_after_pop. The magnitudes belong to phi_cond -- the walker the chains were
   * equilibrated against -- so they must travel with the walker that owns them; parked in the walker
   * buffer, branch() clones them and load balancing ships them, and the post-pop hook reads them back
   * exactly rather than re-deriving them. Non-const wset for that reason. No-op unless the trial is
   * stochastic and walker-conditioned.
   *
   * @param wset the pre-population-control outer walker set
   */
  void store_inner_blocks_before_pop(WalkerSet<MEM>& wset);

  template<MEMORY_SPACE MEM2, class MType2>
  friend struct wavefunction_detail::StochasticInnerStackImpl;

private:
  // Dispatch to the held StochasticWfn; no-op for every other variant alternative. Non-const only:
  // every mutating seam (initialize / begin_inner_step / permute) needs a mutable trial, and the
  // read-only accessors all want a return value, so they go through visit_stochastic_or instead.
  template<class F>
  void visit_stochastic(F&& f)
  {
    std::visit(
        [&](auto&& a) {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            f(a);
        },
        var);
  }

  // Dispatch to StochasticWfn and return its result; otherwise return fallback.
  template<class F, class R>
  R visit_stochastic_or(F&& f, R fallback) const
  {
    return std::visit(
        [&](auto&& a) -> R {
          using Wfn = std::decay_t<decltype(a)>;
          if constexpr (wavefunction_detail::is_stochastic_wfn<Wfn>::value)
            return f(a);
          else
            return fallback;
        },
        var);
  }

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
