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

#include <memory>

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/config.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Wavefunction;
template<MEMORY_SPACE MEM>
class Propagator;

inline ptree strip_stochastic_input_keys(ptree pt)
{
  for (auto const& key : {"stochastic", "inner_nwalkers", "inner_nsteps", "inner_seed", "inner_propagator",
                          "inner_conditioning", "inner_leapfrog"})
    pt.erase(key);
  return pt;
}

template<MEMORY_SPACE MEM, class devPsiT>
struct StochasticInnerStack
{
  virtual ~StochasticInnerStack() = default;

  virtual NOMSD<MEM, devPsiT>& nomsd()             = 0;
  virtual NOMSD<MEM, devPsiT> const& nomsd() const = 0;

  virtual Wavefunction<MEM>& wavefunction()             = 0;
  virtual Wavefunction<MEM> const& wavefunction() const = 0;

  virtual bool has_propagator() const         = 0;
  virtual Propagator<MEM>& propagator()             = 0;
  virtual Propagator<MEM> const& propagator() const = 0;
};

template<MEMORY_SPACE MEM, class devPsiT>
class StochasticWfn : public AFQMCInfo
{
public:
  StochasticWfn(AFQMCInfo& info,
                ptree pt_in,
                std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_in,
                HamiltonianOperations<MEM>&& outer_hop_,
                nda::array<ComplexType, 1>&& ci_,
                nda::array<devPsiT, 2>&& orbs_,
                std::unique_ptr<StochasticInnerStack<MEM, devPsiT>>&& inner_stack_in,
                WALKER_TYPES wlk,
                ComplexType nce,
                [[maybe_unused]] int targetNW = 1);

  static ptree interpret_inputs(const ptree pt0);

  ~StochasticWfn() = default;

  StochasticWfn(StochasticWfn const& other)            = delete;
  StochasticWfn& operator=(StochasticWfn const& other) = delete;
  StochasticWfn(StochasticWfn&& other)                   = default;
  StochasticWfn& operator=(StochasticWfn&& other)        = delete;

  void initialize_inner_walkers(ptree const& walker_pt,
                                memory::const_shared_array<HOST_MEMORY, ComplexType, 3> const& initial_guess,
                                int NAEB);

  bool inner_walkers_initialized() const { return inner_ensemble_.initialized; }
  int inner_nwalkers() const { return inner_nwalkers_; }
  int inner_nsteps() const { return inner_nsteps_; }
  bool inner_conditioning() const { return inner_conditioning_; }
  bool inner_leapfrog() const { return inner_leapfrog_; }

  WalkerSet<MEM>& inner_wset();
  WalkerSet<MEM> const& inner_wset() const;

  NOMSD<MEM, devPsiT>& inner_wfn() { return inner_stack_->nomsd(); }
  NOMSD<MEM, devPsiT> const& inner_wfn() const { return inner_stack_->nomsd(); }

  NOMSD<MEM, devPsiT>& inner_nomsd() { return inner_stack_->nomsd(); }
  NOMSD<MEM, devPsiT> const& inner_nomsd() const { return inner_stack_->nomsd(); }

  Wavefunction<MEM>& inner_wavefunction() { return inner_stack_->wavefunction(); }
  Wavefunction<MEM> const& inner_wavefunction() const { return inner_stack_->wavefunction(); }

  bool inner_propagator_built() const { return inner_stack_->has_propagator(); }
  Propagator<MEM>& inner_propagator();
  Propagator<MEM> const& inner_propagator() const;

  // Arms the per-outer-step inner-resample latch. In leapfrog mode (Phase 3c-ii) it also refreshes the
  // stored OVLP = ⟨Ψ_T|φ⟩ against the ensemble freshly resampled conditioned on the current (old)
  // walker φ, so the step's overlap RATIO new/old (Eq. 25) shares one ensemble and 𝒩(φ) cancels.
  template<class WlkSet>
  void begin_inner_step(WlkSet& wset);

  bool at_delegate_limit() const { return inner_nwalkers_ == 1 && inner_nsteps_ == 0; }

  NOMSD<MEM, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MEM, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int number_of_cholesky_vectors() const { return nomsd_.number_of_cholesky_vectors(); }

  template<class WlkSet>
  void runtime_optimization(WlkSet& wset) { nomsd_.runtime_optimization(wset); }

  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }
  constexpr auto get_memory_space() const { return MEM; }

  // Phase 6 (Tier 3): stochastic mean-field subtraction. vMF / G_MF are TRIAL-AGAINST-ITSELF
  // quantities (no outer walker), so -- unlike the Tier 1/2 mixed estimators that pair the inner
  // ensemble against the OUTER walkers -- they reduce the inner ensemble against ITSELF:
  //   G_MF = <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T>
  //        = [sum_{p,q} <psi_p|c+c|psi_q>] / [sum_{p,q} <psi_p|psi_q>]
  // over the P = inner_nwalkers_ walker-INDEPENDENT inner samples {psi_p} (uniform weight 1/P) -- the
  // inner-ensemble analogue of NOMSD's multi-determinant mean field, with the inner walkers playing the
  // role of the determinant expansion. vMF contracts that mean-field DM against the True Ham (estimator
  // 4, L.G_MF). Both collapse to the anchor density -- i.e. plain NOMSD::G_MF / vMF -- at the static
  // replicated limit (every psi_p == anchor). They DELEGATE to nomsd_ (the anchor mean field) (a) at the
  // single-determinant delegate limit, and (b) whenever the inner ensemble is NOT in its
  // walker-independent P-sample form -- i.e. after a conditioned/leapfrog resample (Phase 3c) has
  // expanded it to nwalk*P walker-CONDITIONED samples, which `begin_inner_step` can trigger BEFORE
  // `generateP1` calls vMF (see Propagate ordering). A conditioned ensemble has no walker-independent
  // subset to average for the trial mean field, and the anchor mean field is exact at the
  // static-limit-first scope (and a valid variance-reduction choice). See `mean_field_uses_inner_ensemble`.
  void vMF(nda::MemoryVector auto&& v, double dt);

  auto G_MF();

  template<class WlkSet, nda::MemoryMatrix MatA>
  void vbias(WlkSet& wset, MatA&& v, double dt, int nt = 0);

  template<class... Args>
  auto vHS(Args&&... args)
  {
    return nomsd_.vHS(std::forward<Args>(args)...);
  }

  template<class WlkSet, class Mat, class TVec>
  void Energy(const WlkSet& wset, Mat&& E, TVec&& Ov, int nt = 0);

  template<class WlkSet>
  void Energy(WlkSet& wset);

  template<class WlkSet>
  void Energy(WlkSet& wset, int nt)
  {
    (void)nt;
    Energy(wset);
  }

  // Phase 5 (Tier 2 observable): stochastic mixed density matrix for observable evaluation -- estimator
  // 3 of arXiv:2505.18519, G[w] = (sum_p w_p <psi_p|c+c|phi_w>/<psi_p|phi_w>) / sum_p w_p, reduced over
  // the inner ensemble and scored against the True Ham. This is the observable analogue of
  // MixedDensityMatrix_for_vbias: the same inner-ensemble reduction, but returned in the caller's
  // observable layout (and with the LOG overlap convention NOMSD's observable DM uses). Delegates to
  // nomsd_ at the single-determinant delegate limit. The 2-arg overload mirrors NOMSD -- it builds a
  // scratch overlap vector and routes through the 3-arg workhorse.
  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true)
  {
    int nw = wset.size();
    memory::buffered_array<MEM, ComplexType, 1> Ov(nw, ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), Ov, compact);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true);

  // Phase 5 (Tier 2): DM w.r.t. an EXTERNALLY supplied reference orbital set `Ref`. The bra is `Ref`
  // (not the stochastic trial) and the ket is the walker, so this is pure orbital algebra independent
  // of both the trial wavefunction and the Hamiltonian -- NOMSD's implementation never touches its own
  // OrbMats. The stochastic trial therefore plays no role, and delegating to `nomsd_` is exact. (Which
  // reference set a stochastic trial *exposes* for back-propagation was the Tier 6 question, resolved in
  // Phase 7: the outer-NOMSD set, inner-ensemble-agnostic; it does not change this Ref-parameterized method.)
  template<class WlkSet, class RVec, class MatG, class TVec>
  void DensityMatrix(const WlkSet& wset,
                     RVec&& Ref,
                     MatG&& G,
                     TVec&& Ov,
                     bool compact = true,
                     bool herm    = true)
  {
    nomsd_.DensityMatrix(wset, std::forward<RVec>(Ref), std::forward<MatG>(G), std::forward<TVec>(Ov), compact,
                         herm);
  }

  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G);

  template<class WlkSet, class TVec>
  void Log_Overlap(const WlkSet& wset, TVec&& Ov, int nt = 0);

  template<class WlkSet>
  void Log_Overlap(WlkSet& wset);

  // Phase 5 (Tier 2): accumulate observable contributions from the inner-ensemble-reduced (stochastic)
  // Green's function. The full mixed Green's function fed to the observables is the stochastic
  // `MixedDensityMatrix` (estimator 3, full NMO x NMO layout). For time-evolved / back-propagated
  // observables (X/Yc/M != null) that DM is transformed through the evolved operators exactly as NOMSD
  // does (`M + T(X) . G_full . Yc`); the transform is linear in the DM, so the inner-ensemble average
  // commutes with it. Delegates to `nomsd_` at the delegate limit. The 5-arg overload mirrors NOMSD
  // (null X/Yc/M, time_evolved = false) and routes through the full one.
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
                             std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
                             nda::MemoryArrayOfRank<4> auto* X, nda::MemoryArrayOfRank<4> auto* Yc,
                             nda::MemoryArrayOfRank<4> auto* M, bool time_evolved,
                             bool importanceSampling = true);

  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
                             std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
                             bool importanceSampling = true)
  {
    memory::buffered_array<MEM, ComplexType, 4>* X = nullptr;
    accumulate_estimators(iav, wset, wgt, properties_1body, properties, X, X, X, false, importanceSampling);
  }

  // Phase 5 (Tier 5 observable): generalized Fock matrix of a SUPPLIED density matrix against the True
  // Ham. Like NOMSD, this is a `HamOp` contraction on a caller-provided G (no trial reduction), so it
  // delegates to `nomsd_` (the True Ham). Reached only via the `generalizedFockMatrix` observable.
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  // Phase 7 (Tier 6): back-propagation reference set. CHOSEN SEMANTICS = OUTER-NOMSD DELEGATE,
  // INNER-ENSEMBLE-AGNOSTIC: the stochastic trial exposes exactly the OUTER nomsd_'s reference set (the
  // True-Ham trial's references) and ignores the inner ensemble entirely -- so these stay delegates to
  // nomsd_. For the paper's intended SINGLE-determinant anchor (Eq. 21) that set is {phi_T} = OrbMats(0)
  // with weight 1; for a multi-determinant outer trial it is the full CI expansion (nrefs=ndet, weights
  // ci[i]) -- i.e. identical to plain NOMSD in BOTH cases, hence the test asserts equality unconditionally
  // (no ndet==1 gate). Rationale:
  //  - The BP estimator (BackPropagatedEstimator/FullObsHandler) fills references for ONE walker and
  //    broadcasts them to ALL walkers (Refs(iw)=Refs(0)), so references MUST be walker-INDEPENDENT, and
  //    it captures them once per BP block and back-propagates over a FIXED window. The outer trial's
  //    references are walker-independent and time-stable (frozen while the inner ensemble resamples).
  //  - EXACT vs plain NOMSD by construction; the SCIENTIFIC caveat is that for a genuine inner_nsteps>0
  //    trial back-propagation is scored against the OUTER trial, NOT the field-sampled stochastic spread
  //    {psi_p} -- a documented approximation, consistent with vMF/G_MF collapsing to the anchor (Phase 6).
  // The faithful inner-ensemble reference set {psi_p} (nrefs=P, weights 1/P) is DEFERRED research: it is
  // incompatible with the walker-dependent conditioned/leapfrog ensemble, conflicts with BP's fixed
  // reference window (the ensemble resamples every step), and multiplies the back-propagation cost by P.
  // See Phase 7 in StochasticDevelopment.md.
  int total_number_of_references() const { return nomsd_.total_number_of_references(); }
  ComplexType getReferenceWeight(int i) const { return nomsd_.getReferenceWeight(i); }

  template<class... Args>
  void getReferences(Args&&... args)
  {
    nomsd_.getReferences(std::forward<Args>(args)...);
  }

  HamiltonianTypes getHamType() const { return nomsd_.getHamType(); }
  auto getFieldTypes() { return nomsd_.getFieldTypes(); }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    nomsd_.update_potentials(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return nomsd_.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return nomsd_.vHS_sparse(std::forward<Args>(args)...);
  }

  auto vHS_dims() const { return nomsd_.vHS_dims(); }

  template<class... Args>
  void updateLogScale(Args&&... args)
  {
    nomsd_.updateLogScale(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto getLogScale(Args&&... args)
  {
    return nomsd_.getLogScale(std::forward<Args>(args)...);
  }

private:
  static ptree nomsd_inputs(ptree const& pt0)
  {
    return NOMSD<MEM, devPsiT>::interpret_inputs(strip_stochastic_input_keys(pt0));
  }

  struct StochasticInnerEnsemble
  {
    std::unique_ptr<WalkerSet<MEM>> wset;
    std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng;
    bool initialized{false};
  };

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_;
  StochasticInnerEnsemble inner_ensemble_;
  int inner_nwalkers_{1};
  int inner_nsteps_{0};
  double inner_timestep_{0.01};
  bool inner_step_pending_{false};
  bool inner_conditioning_{false};
  bool inner_leapfrog_{false};
  nda::array<ComplexType, 3> inner_anchor_;
  // Phase 3c-ii (leapfrog): per inner walker q (slot-major q = ip*nwalk + w), the magnitude
  // |⟨ψ_q|φ_w^cond⟩| of its cross overlap with the walker its block was conditioned on. Set at each
  // conditioned resample; the leapfrog overlap reweights by 1/inner_cond_mag_ so the step ratio is Eq. 25.
  nda::array<RealType, 1> inner_cond_mag_;
  NOMSD<MEM, devPsiT> nomsd_;
  std::unique_ptr<StochasticInnerStack<MEM, devPsiT>> inner_stack_;

  void maybe_advance_inner_ensemble();

  // Phase 3c-i: resample the inner ensemble conditioned on each outer walker phi_w. Computes the
  // custom inner force bias x_bar(phi_w) = sqrt(dt)*L^var . <phi_T|c+c|phi_w>/<phi_T|phi_w> (the inner
  // trial IS the anchor phi_T, so this reuses inner_nomsd()'s mixed DM + vbias on the OUTER wset) and
  // drives the nw*P inner ensemble through the inner propagator's conditioned field-sampling seam.
  void advance_inner_ensemble_conditioned(memory::array<MEM, ComplexType, 2> const& X_bias, int nw);

  // Resample dispatch: conditioned (Phase 3c-i) when inner_conditioning_, else the walker-independent
  // free-projection path (Phase 3b). Honors the per-step latch armed by begin_inner_step().
  template<class WlkSet>
  void conditioned_resample(const WlkSet& wset);

  // Phase 3c-ii: fill inner_cond_mag_ with |⟨ψ_q|φ_w^cond⟩| after a conditioned resample (φ_cond = the
  // outer walkers the inner blocks were just conditioned on). The leapfrog overlap divides by this.
  template<class WlkSet>
  void compute_inner_cond_mag(const WlkSet& wset);

  template<class WlkSet, class TVecD, class TVecOv, class Accumulate>
  void reduce_inner_cross_dm(const WlkSet& wset,
                             bool compact,
                             int Gsize,
                             TVecD&& D,
                             TVecOv&& Ov,
                             Accumulate&& accumulate);

  // Phase 6 (Tier 3): whether vMF / G_MF should reduce the inner ensemble (true) or delegate to nomsd_'s
  // anchor mean field (false). The mean field <Psi_T|.|Psi_T> is trial-only (walker-independent), so it
  // is reduced ONLY when the inner ensemble is in its walker-INDEPENDENT P-sample form
  // (inner.size() == inner_nwalkers_) with P > 1. It is false (delegate) when:
  //  - inner_nwalkers_ == 1: a single sample is degenerate (its self-DM is the anchor at setup), so the
  //    P=1 reduction equals nomsd_ -- delegate rather than run it redundantly. This subsumes the
  //    single-determinant delegate limit (inner_nwalkers_==1 && inner_nsteps_==0) AND inner_nwalkers_==1
  //    with inner_nsteps_>0.
  //  - the ensemble has been expanded by a conditioned/leapfrog resample (Phase 3c) to nwalk*P
  //    walker-CONDITIONED samples (inner.size() != inner_nwalkers_): no walker-independent subset to
  //    average, so delegate to the anchor mean field.
  bool mean_field_uses_inner_ensemble() const
  {
    return inner_nwalkers_ > 1 && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr
           && int(inner_ensemble_.wset->size()) == inner_nwalkers_;
  }

  // Phase 6 (Tier 3): build the normalized stochastic trial mean-field one-body Green's function
  // <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T> into `Gsum` (full [1, nspin*npol*NMO*npol*NMO] layout) by reducing
  // the P = inner_nwalkers_ walker-independent inner samples against each other (the double sum above).
  // Shared by vMF (contracts it) and G_MF (returns it). Trial-only -- no outer walker set, no conditioned
  // resample. The caller MUST gate on mean_field_uses_inner_ensemble() (inner.size() == inner_nwalkers_).
  void reduce_inner_mean_field_dm(memory::buffered_array<MEM, ComplexType, 2>& Gsum);

  int dm_size(bool full) const;
  bool compact_G_for_vbias() const;
};

} // namespace afqmc
} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"
