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
#include <string>

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
  for (auto const& key : {"type", "inner_nwalkers", "inner_nsteps", "inner_seed", "inner_propagator",
                          "inner_conditioning", "inner_leapfrog", "inner_persistence", "inner_equil_steps",
                          "inner_mcmc", "inner_pool_burn_in", "inner_log_aggregate"})
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
  // Persistent (tethered) inner sampling. When on, the inner pool is kept across outer steps and
  // re-equilibrated by a short Metropolis MCMC instead of reset-then-redraw with Gaussian-shift resampling.
  bool inner_persistence() const { return inner_persistence_; }
  int inner_equil_steps() const { return inner_equil_steps_; }
  bool inner_log_aggregate() const { return inner_log_aggregate_; }

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

  // Arms the per-outer-step inner-resample latch. In leapfrog mode it also refreshes the stored OVLP = ⟨Ψ_T|φ⟩ against the ensemble freshly resampled conditioned on the current (old)
  // walker φ, so the step's overlap RATIO new/old (Eq. 25) shares one ensemble and 𝒩(φ) cancels.
  template<class WlkSet>
  void begin_inner_step(WlkSet& wset);

  // Realign the conditioned inner ensemble with the outer walker set after an outer population-control
  // event (branch + load balance); the driver calls this immediately after wset.popControl(). The
  // slot-major inner ensemble (index q = ip*nw + w, block w conditioned on outer walker phi_w) lives
  // OUTSIDE the outer walker buffer, so popControl's clone/shuffle would otherwise leave block w attached
  // to the OLD phi_w -- a reduction between popControl and the next propagation step (e.g.
  // accumulate_estimators when the measure and population-control intervals coincide) would then pair
  // post-branch outer walkers with pre-branch inner blocks. This permutes the inner blocks to follow the
  // walkers (via the per-walker SLOT_LINEAGE map recorded through branch/load-balance), or -- when a
  // walker arrived from another rank, whose inner block this rank does not hold -- rebuilds them with a
  // fresh conditioned resample on the next reduction.
  // No-op unless this is a conditioned dynamic trial (inner_conditioning_ && inner_nsteps_ > 0); static,
  // free-projection, and delegate-limit trials carry no slot-conditioned inner blocks.
  template<class WlkSet>
  void permute_inner_blocks_after_pop(const WlkSet& wset);

  bool at_delegate_limit() const { return inner_nwalkers_ == 1 && inner_nsteps_ == 0; }

  NOMSD<MEM, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MEM, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int number_of_cholesky_vectors() const { return nomsd_.number_of_cholesky_vectors(); }

  template<class WlkSet>
  void runtime_optimization(WlkSet& wset) { nomsd_.runtime_optimization(wset); }

  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }
  constexpr auto get_memory_space() const { return MEM; }

  // Stochastic mean-field subtraction. vMF / G_MF are TRIAL-AGAINST-ITSELF quantities (no outer walker),
  // so -- unlike the propagator hot-path mixed estimators that pair the inner ensemble against the OUTER
  // ensemble against the OUTER walkers -- they reduce the inner ensemble against ITSELF:
  //   G_MF = <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T>
  //        = [sum_{p,q} <psi_p|c+c|psi_q>] / [sum_{p,q} <psi_p|psi_q>]
  // over the P = inner_nwalkers_ walker-INDEPENDENT inner samples {psi_p} (uniform weight 1/P) -- the
  // inner-ensemble analogue of NOMSD's multi-determinant mean field, with the inner walkers playing the
  // role of the determinant expansion. vMF contracts that mean-field DM against the True Ham (estimator
  // 4, L.G_MF). Both collapse to the anchor density -- i.e. plain NOMSD::G_MF / vMF -- at the static
  // replicated limit (every psi_p == anchor). They DELEGATE to nomsd_ (the anchor mean field) (a) at the
  // single-determinant delegate limit, and (b) whenever the inner ensemble is NOT in its
  // walker-independent P-sample form -- i.e. after a conditioned/leapfrog resample has expanded it to
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

  // Stochastic mixed density matrix for observable evaluation -- estimator 3 of arXiv:2505.18519, G[w] = (sum_p w_p <psi_p|c+c|phi_w>/<psi_p|phi_w>) / sum_p w_p, reduced over
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

  // DM w.r.t. an EXTERNALLY supplied reference orbital set `Ref`. The bra is `Ref` (not the stochastic
  // trial) and the ket is the walker, so this is pure orbital algebra independent of both the trial
  // wavefunction and the Hamiltonian -- NOMSD's implementation never touches its own OrbMats. The
  // stochastic trial therefore plays no role, and delegating to `nomsd_` is exact. (Which reference set
  // a stochastic trial exposes for back-propagation is the outer-NOMSD set, inner-ensemble-agnostic; it
  // does not change this Ref-parameterized method.)
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

  // Accumulate observable contributions from the inner-ensemble-reduced (stochastic) Green's function. The full mixed Green's function fed to the observables is the stochastic
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

  // Generalized Fock matrix of a SUPPLIED density matrix against the True Ham. Like NOMSD, this is a `HamOp` contraction on a caller-provided G (no trial reduction), so it
  // delegates to `nomsd_` (the True Ham). Reached only via the `generalizedFockMatrix` observable.
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  // Back-propagation reference set. This is standard Motta-Zhang back-propagation (arXiv:1707.02684): the trial is back-propagated through the recorded OUTER fields and contracted
  // against the stored forward walker; the references are the determinants that represent <Psi_T|. Two
  // regimes, selected by bp_uses_inner_ensemble():
  //  - OUTER-NOMSD DELEGATE (static limit / P==1): expose the OUTER nomsd_'s reference set -- the anchor
  //    {phi_T} = OrbMats(0) (weight 1) for the single-determinant trial, or its CI expansion otherwise.
  //    Identical to plain NOMSD. Used at inner_nsteps==0 and inner_nwalkers==1.
  //  - DEDICATED FREE-PROJECTION DRAW (the faithful set): for a dynamic trial (inner_nsteps>0, P>1),
  //    getReferences draws a FRESH, walker-INDEPENDENT free-projection ensemble {psi_p = B_T(Y^[p])|phi_T>}
  //    (bare p_T(Y)) and exposes those P samples with uniform weight 1/P, so back-propagation scores
  //    against the true stochastic trial <Psi_T| ~ (1/P) sum_p <psi_p| (Eq. 24 of arXiv:2505.18519). The
  //    trial |Psi_T> is walker-INDEPENDENT, so its references must be too -- the forward walk's
  //    conditioning/leapfrog is a forward-only importance-sampling device, IRRELEVANT to the BP
  //    BP references; hence this draw is DECOUPLED from the forward inner ensemble and works for BOTH
  //    free-projection and conditioned/leapfrog forward walks. The estimator already carries the complex
  //    per-reference overlap exp(Ov_p)=<psi_p|phi_BP> (phase folded in) and uniform 1/P cancels in its
  //    normalization ratio; the explicit S_p/(1/|O_p|) reweighting of the forward path is the
  //    importance-sampling correction for WALKER-CONDITIONED draws (Eq. 23), absent here because these
  //    are bare free-projection samples. getReferences performs the draw and the estimator copies the
  //    references it reads, freezing them for the BP window (a fresh Monte-Carlo draw of the trial per
  //    block). Reduces to the delegate at inner_nsteps==0 / P==1.
  bool bp_uses_inner_ensemble() const
  {
    return inner_nsteps_ > 0 && inner_nwalkers_ > 1
           && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr;
  }

  int total_number_of_references() const
  {
    return bp_uses_inner_ensemble() ? inner_nwalkers_ : nomsd_.total_number_of_references();
  }
  ComplexType getReferenceWeight(int i) const
  {
    return bp_uses_inner_ensemble() ? ComplexType(1.0 / static_cast<double>(inner_nwalkers_), 0.0)
                                    : nomsd_.getReferenceWeight(i);
  }

  // Fills the [nref, npol*NMO, nel] reference Slater matrices (H-conjugated bras), as
  // BackPropagatedEstimator requests. For a dynamic trial (bp_uses_inner_ensemble()) it first performs a
  // dedicated free-projection draw (draw_bp_reference_ensemble()) and fills from those P samples;
  // otherwise it delegates to nomsd_ (the anchor / CI expansion). Defined in the .icc.
  template<class RefMat>
  void getReferences(int number_of_references, RefMat&& Refs);

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

  void resetLogScale() { nomsd_.resetLogScale(); }

  template<nda::MemoryArrayOfRank<1> T>
  void setLogPT0(T&& v)
  {
    nomsd_.setLogPT0(std::forward<T>(v));
  }

  auto getLogPT0() const { return nomsd_.getLogPT0(); }

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
  // Guards the back-propagation reference draw so it happens at most ONCE per BP window.
  // Set true by draw_bp_reference_ensemble() after a draw; a repeated getReferences in the same window
  // then reuses that draw (idempotent -- no silent re-draw). Reset to false by begin_inner_step(), i.e.
  // when the forward walk advances to the next step (the next BP window draws fresh).
  bool bp_refs_drawn_{false};
  bool inner_conditioning_{false};
  bool inner_leapfrog_{false};
  // Persistent-pool controls (default OFF => reset-then-redraw conditioned resample is byte-for-byte
  // unchanged). inner_persistence_: keep the inner pool across outer steps and re-equilibrate it in place.
  // inner_equil_steps_: Metropolis sweeps per outer step once the pool is primed. inner_pool_burn_in_:
  // extra sweeps at the one-time prime (fresh-from-anchor). inner_mcmc_: proposal kernel ("metropolis"
  // only for now; "hmc" reserved). inner_pool_primed_: latched true once the pool has been initialized so
  // subsequent steps persist rather than re-draw; cleared whenever the pool is invalidated (BP draw,
  // cross-rank pop-control move).
  bool inner_persistence_{false};
  int inner_equil_steps_{1};
  int inner_pool_burn_in_{0};
  std::string inner_mcmc_{"metropolis"};
  bool inner_pool_primed_{false};
  bool inner_log_aggregate_{false};
  nda::array<ComplexType, 3> inner_anchor_;
  // Leapfrog: per inner walker q (slot-major q = ip*nwalk + w), the magnitude |⟨ψ_q|φ_w^cond⟩| of its cross overlap with the walker its block was conditioned on. Set at each
  // conditioned resample; the leapfrog overlap reweights by 1/inner_cond_mag_ so the step ratio is Eq. 25.
  nda::array<RealType, 1> inner_cond_mag_;
  NOMSD<MEM, devPsiT> nomsd_;
  std::unique_ptr<StochasticInnerStack<MEM, devPsiT>> inner_stack_;

  void maybe_advance_inner_ensemble();

  // Draw a FRESH, walker-INDEPENDENT free-projection ensemble of P trial samples {psi_p = B_T(Y^[p])|phi_T>} into inner_ensemble_.wset for use as back-propagation references. Resets
  // to the anchor, sizes the ensemble to P, then advances inner_nsteps_ BARE free-projection steps via
  // inner_propagator().Propagate_free (which forces bare field sampling regardless of the forward
  // propagator's build mode -- so this is decoupled from any 3c conditioning/leapfrog of the forward
  // walk). Reuses inner_ensemble_.wset as scratch: every forward resample resets it to the anchor, so
  // transiently overwriting it here is safe (the next begin_inner_step resamples from scratch). Called
  // by getReferences. Defined in StochasticWfn.cpp (does not depend on the RefMat template).
  void draw_bp_reference_ensemble();

  // Reset the first `count` inner walkers in `inner` to the anchor |phi_T> (the per-spin Slater matrices
  // cached in inner_anchor_; handles CLOSED/COLLINEAR). The single place that knows the anchor/collinear
  // layout -- shared by the free-projection advance, the conditioned resample, and the BP reference draw.
  void reset_inner_to_anchor(WalkerSet<MEM>& inner, int count);

  // Propose a fresh bare free-projection sample psi* = B_T(Y*)|phi_T> into every inner walker
  // of the current (slot-major nw*P) pool -- reset to the anchor, then inner_nsteps_ Propagate_free steps.
  // Walker-INDEPENDENT (the proposal density is the trial's own p_T(Y)); the walker conditioning enters
  // only through the Metropolis accept/reject in metropolis_sweep_conditioned. Defined in StochasticWfn.cpp
  // (the propagator call needs the complete Propagator type, unavailable to the .icc templates).
  void propose_free_projection_pool();

  // Resample the inner ensemble conditioned on each outer walker phi_w. Computes the custom inner force bias x_bar(phi_w) = sqrt(dt)*L^var . <phi_T|c+c|phi_w>/<phi_T|phi_w> (the inner
  // trial IS the anchor phi_T, so this reuses inner_nomsd()'s mixed DM + vbias on the OUTER wset) and
  // drives the nw*P inner ensemble through the inner propagator's conditioned field-sampling seam.
  void advance_inner_ensemble_conditioned(memory::array<MEM, ComplexType, 2> const& X_bias, int nw);

  // Resample dispatch: walker-conditioned when inner_conditioning_, else the walker-independent
  // free-projection path. Honors the per-step latch armed by begin_inner_step().
  template<class WlkSet>
  void conditioned_resample(const WlkSet& wset);

  // Fill inner_cond_mag_ with |⟨ψ_q|φ_w^cond⟩| after a conditioned resample (φ_cond = the outer walker
  // its block was conditioned on). The leapfrog overlap divides by this.
  template<class WlkSet>
  void compute_inner_cond_mag(const WlkSet& wset);

  // Fill mag(q) = |⟨inner_q | φ_w⟩| for the slot-major (q = ip*nw + w) inner ensemble against the outer
  // walker set (same det(A^dag B) / CLOSED-doubling / COLLINEAR-product convention as the Log_Overlap
  // reduction). Shared by compute_inner_cond_mag (leapfrog reweight denominator) and the Metropolis
  // accept/reject (conditioned target ∝ p_T(Y)·|⟨ψ|φ_w⟩|). Per-rank-local. `mag` must be
  // sized nw*P; `inner` must be in the slot-major nw*P conditioned form.
  template<class WlkSet>
  void cross_overlap_magnitudes(const WlkSet& wset, WalkerSet<MEM>& inner, nda::array<RealType, 1>& mag);

  // Persistent conditioned resample. Replaces reset-then-redraw with a persistent pool:
  // on `prime` (first use / outer-count change / after invalidation) reset every slot to the anchor and
  // burn in; otherwise KEEP the previous pool and re-equilibrate. Runs inner_equil_steps_ (+ burn-in on
  // prime) Metropolis sweeps. Conditioned path only (inner_conditioning_ && inner_nsteps_ > 0).
  template<class WlkSet>
  void persistent_conditioned_resample(const WlkSet& wset, bool prime);

  // One Metropolis sweep over all nw*P inner slots for the persistent conditioned pool. Independence
  // proposal: a fresh bare free-projection draw ψ* = B̂_T(Y*)|φ_T⟩ (reset-to-anchor + inner_nsteps_
  // Propagate_free steps); per-slot accept with probability min(1, |⟨ψ*|φ_w⟩| / |⟨ψ_cur|φ_w⟩|), which
  // targets the EXACT conditioned distribution ∝ p_T(Y)·|⟨ψ|φ_w⟩| (no linear Gaussian-shift approximation,
  // unlike the force-biased Gaussian-shift draw used without persistence). Rejected slots keep their
  // previous sample -> persistence.
  template<class WlkSet>
  void metropolis_sweep_conditioned(const WlkSet& wset);

  template<class WlkSet, class TVecD, class TVecOv, class Accumulate>
  void reduce_inner_cross_dm(const WlkSet& wset,
                             bool compact,
                             int Gsize,
                             TVecD&& D,
                             TVecOv&& Ov,
                             Accumulate&& accumulate);

  // Whether vMF / G_MF should reduce the inner ensemble (true) or delegate to nomsd_'s anchor mean field
  // anchor mean field (false). The mean field <Psi_T|.|Psi_T> is trial-only (walker-independent), so it
  // is reduced ONLY when the inner ensemble is in its walker-INDEPENDENT P-sample form
  // (inner.size() == inner_nwalkers_) with P > 1. It is false (delegate) when:
  //  - inner_nwalkers_ == 1: a single sample is degenerate (its self-DM is the anchor at setup), so the
  //    P=1 reduction equals nomsd_ -- delegate rather than run it redundantly. This subsumes the
  //    single-determinant delegate limit (inner_nwalkers_==1 && inner_nsteps_==0) AND inner_nwalkers_==1
  //    with inner_nsteps_>0.
  //  - the ensemble has been expanded by a conditioned/leapfrog resample to nwalk*P walker-CONDITIONED
  //    walker-CONDITIONED samples (inner.size() != inner_nwalkers_): no walker-independent subset to
  //    average, so delegate to the anchor mean field.
  bool mean_field_uses_inner_ensemble() const
  {
    return inner_nwalkers_ > 1 && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr
           && int(inner_ensemble_.wset->size()) == inner_nwalkers_;
  }

  // Build the normalized stochastic trial mean-field one-body Green's function <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T> into `Gsum` (full [1, nspin*npol*NMO*npol*NMO] layout) by reducing
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
