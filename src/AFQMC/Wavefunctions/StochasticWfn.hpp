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
#include "numerics/device_kernels/kernels.h" // kernels::device::{row_accumulate,row_divide,inner_scalar_reduce,elementwise_log} (device only)

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
class Wavefunction;
template<MEMORY_SPACE MEM>
class Propagator;

// How the inner (trial) field ensemble is sampled. One key, because the modes are a strict chain
// (leapfrog requires conditioning, persistence requires conditioning, replicas require persistence):
// the enum makes the illegal combinations unrepresentable.
enum class SamplingTarget
{
  // inner_nsteps == 0. Replicated anchor ensemble; every reduction delegates to NOMSD. The exactness
  // limit the NOMSD-parity tests are written against.
  Static,
  // Dynamic, unconditioned draw from the bare prior p_T(Y). Correct but catastrophically noisy:
  // reference / test mode only (BP reference draw, dynamic-ensemble tests).
  Gaussian,
  // Dynamic, walker-conditioned: persistent field-space chains targeting p_T(Y)|<psi(Y)|phi_w>| with the
  // leapfrog reweight (Eq. 25). The production mode.
  WalkerOverlap,
};

inline std::string to_string(SamplingTarget m)
{
  switch (m)
  {
  case SamplingTarget::Static:
    return "static";
  case SamplingTarget::Gaussian:
    return "gaussian";
  case SamplingTarget::WalkerOverlap:
    return "walker_overlap";
  }
  return "unknown";
}

inline SamplingTarget parse_sampling_target(std::string const& s)
{
  if (s == "static")
    return SamplingTarget::Static;
  if (s == "gaussian")
    return SamplingTarget::Gaussian;
  if (s == "walker_overlap")
    return SamplingTarget::WalkerOverlap;
  APP_ABORT("Error in StochasticWfn: inner_sampling_target = '" + s +
            "' is not a sampling target. Choose one of: 'walker_overlap' (dynamic walker-conditioned "
            "persistent field chains with the leapfrog reweight -- the production sampler), 'gaussian' "
            "(dynamic unconditioned draw from the prior p_T -- a reference target; correct but "
            "catastrophically noisy), 'static' (inner_nsteps = 0 replicated anchor ensemble, delegates "
            "to NOMSD). "
            "NOTE: 'gaussian' here names the TARGET DENSITY, and is unrelated to inner_sampler = "
            "'gaussian', which names a random-walk PROPOSAL. hafqmc overloads the word the same way "
            "(sampling_target vs sampler_name); the two keys are independent.");
  return SamplingTarget::Static; // unreachable; APP_ABORT throws
}

// Resolve and validate the sampling mode from an input ptree. THE SINGLE OWNER of that translation:
// free-standing because WavefunctionFactory::interpret_inputs (which builds the inner propagator) and
// StochasticWfn::interpret_inputs (which selects the sampler) must never disagree. See
// The propagator is built first, so a second, independent resolution could pair a conditioned sampler
// with a free-projection propagator. `inner_nsteps` is passed in because callers have already validated
// it.
inline std::string resolve_sampling_target(ptree const& pt0, int inner_nsteps)
{
  auto target_opt = pt0.get_optional<std::string>("inner_sampling_target");

  // Deliberately NOT defaulted for a dynamic trial: defaulting it is how a run silently gets 'gaussian'.
  if (not target_opt && inner_nsteps > 0)
    APP_ABORT("Error in StochasticWfn: inner_nsteps > 0 selects a dynamic inner ensemble, so "
              "inner_sampling_target must be given explicitly -- 'walker_overlap' for the production "
              "walker-conditioned persistent chains, or 'gaussian' for the unconditioned prior-sampling "
              "reference target (correct, but its variance makes it unusable in production). It is NOT "
              "defaulted, because defaulting it is how a production run silently gets prior sampling.");

  const std::string inner_sampling_target = target_opt ? *target_opt : std::string("static");

  const SamplingTarget parsed = parse_sampling_target(inner_sampling_target);
  if (parsed == SamplingTarget::Static && inner_nsteps > 0)
    APP_ABORT("Error in StochasticWfn: inner_sampling_target = static is the inner_nsteps = 0 replicated "
              "anchor ensemble, but inner_nsteps = " +
              std::to_string(inner_nsteps) + " was given.");
  if (parsed != SamplingTarget::Static && inner_nsteps <= 0)
    APP_ABORT("Error in StochasticWfn: inner_sampling_target = " + inner_sampling_target +
              " is a dynamic sampler and requires inner_nsteps > 0.");
  return inner_sampling_target;
}

inline ptree strip_stochastic_input_keys(ptree pt)
{
  // Must list every [stochastic_wfn] key stripped before passing the ptree to the inner wavefunction.
  for (auto const& key : {"type", "inner_n_samples", "inner_nsteps", "inner_seed", "inner_propagator",
                          "inner_sampling_target", "inner_burn_in", "inner_sample_update_steps",
                          "inner_sampler", "inner_sampler_step", "inner_n_measure_samples"})
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
class StochasticWfn
{
public:
  StochasticWfn(std::string system,
                int NMO_,
                int nup_,
                int ndown_,
                ptree pt_in,
                std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_in,
                HamiltonianOperations<MEM>&& outer_hop_,
                nda::array<ComplexType, 1>&& ci_,
                nda::array<devPsiT, 2>&& orbs_,
                std::unique_ptr<StochasticInnerStack<MEM, devPsiT>>&& inner_stack_in,
                WALKER_TYPES wlk,
                [[maybe_unused]] int targetNW = 1);

  static ptree interpret_inputs(const ptree pt0);

  ~StochasticWfn() = default;

  StochasticWfn(StochasticWfn const& other)            = delete;
  StochasticWfn& operator=(StochasticWfn const& other) = delete;
  StochasticWfn(StochasticWfn&& other)                   = default;
  StochasticWfn& operator=(StochasticWfn&& other)        = delete;

  void initialize_inner_walkers(ptree const& walker_pt,
                                std::vector<nda::matrix<ComplexType>> const& initial_guess,
                                int NAEB);

  bool inner_walkers_initialized() const { return inner_ensemble_.initialized; }
  int inner_n_samples() const { return inner_n_samples_; }
  int inner_nsteps() const { return inner_nsteps_; }
  SamplingTarget inner_sampling_target() const { return inner_sampling_target_; }
  // The trained B_T timestep actually in force. Exposed so a test can assert the value the factory read
  // from inner_hamiltonian REACHED the wavefunction; a factory that drops it looks identical otherwise.
  double inner_timestep() const { return inner_timestep_; }
  // True for SamplingTarget::WalkerOverlap: conditioning, persistent field chains and the leapfrog
  // reweight are one algorithm, not independent knobs.
  bool is_conditioned() const { return inner_sampling_target_ == SamplingTarget::WalkerOverlap; }
  // MH sweeps per pool advance -- ONE knob for BOTH the propagation and measurement seams, matching
  // hafqmc's sample_update_steps. inner_burn_in_ is the extra one-time equilibration at the prime.
  int inner_burn_in() const { return inner_burn_in_; }
  int inner_sample_update_steps() const { return inner_sample_update_steps_; }
  std::string const& inner_sampler() const { return inner_sampler_; }
  double inner_sampler_step() const { return inner_sampler_step_; }
  // Number of measurement replicas averaged at fixed walkers.
  int inner_n_measure_samples() const { return inner_n_measure_samples_; }
  // True when the measurement seam advances the pool, i.e. whenever a live persistent chain exists.
  // ⚠️ Deliberately NOT gated on nm > 1: hafqmc advances its pool before EVERY block measurement, so nm
  // is purely the number of replicas AVERAGED and whether the pool advances is not a knob. Public so a
  // test can assert the path is LIVE instead of passing vacuously on the fallback.
  bool measure_advances_pool() const
  {
    return is_conditioned() && inner_chains_primed_ && not inner_dets_stale_;
  }
  // Cumulative Metropolis acceptance fraction of the field-space chain updates on this rank
  // (1.0 before any proposal has been made).
  double inner_chain_acceptance() const
  {
    return chain_proposed_ > 0 ? double(chain_accepted_) / double(chain_proposed_) : 1.0;
  }
  // Sum of the leapfrog conditioning magnitudes |<psi_q|phi_cond>| (test/diagnostic checksum; 0 when
  // unset). A realignment that only re-indexes the ensemble must leave it invariant, since it references
  // phi_cond and not the post-pop walker.
  double inner_cond_mag_sum() const
  {
    double s = 0.0;
    for (long i = 0; i < inner_cond_mag_.size(); ++i)
      s += inner_cond_mag_(i);
    return s;
  }

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

  // Arms the per-outer-step inner-resample latch, advances the persistent chain pool (the one seam with
  // non-const access to the outer walker set) and refreshes the stored OVLP so the step's overlap RATIO
  // new/old is Eq. 25.
  template<class WlkSet>
  void begin_inner_step(WlkSet& wset);

  // Realign the conditioned inner ensemble with the outer walker set after an outer population-control
  // event; the driver calls this immediately after wset.popControl(). Without it a reduction before the
  // next propagation step would pair post-branch outer walkers with pre-branch inner blocks. No-op
  // unless is_conditioned().
  template<class WlkSet>
  void permute_inner_blocks_after_pop(const WlkSet& wset);

  bool at_delegate_limit() const { return inner_n_samples_ == 1 && inner_nsteps_ == 0; }

  NOMSD<MEM, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MEM, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int number_of_cholesky_vectors() const { return nomsd_.number_of_cholesky_vectors(); }

  template<class WlkSet>
  void runtime_optimization(WlkSet& wset) { nomsd_.runtime_optimization(wset); }

  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }
  constexpr auto get_memory_space() const { return MEM; }

  // Stochastic mean-field subtraction. vMF / G_MF are TRIAL-AGAINST-ITSELF quantities (no outer walker):
  // they reduce the inner ensemble against ITSELF over the P walker-INDEPENDENT samples, and vMF contracts
  // that DM against the True Ham (estimator 4, L.G_MF). Which ensemble is reduced -- the forward one, a
  // dedicated scratch draw, or nomsd_'s anchor -- is decided by mean_field_uses_inner_ensemble() /
  // bp_uses_inner_ensemble().
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

  // Measurement entry point for the estimators: Energy() averaged over inner_n_measure_samples_ replicas
  // of the field pool at FIXED walkers. Non-const wset because the chain state lives in the outer walker
  // buffer. Reduces to exactly Energy(wset, E, Ov, nt) unless measure_advances_pool(). E is the mean of
  // per-replica energy ratios; Ov is the walker's STORED log overlap, not a replica's, so
  // EnergyEstimator's exp(ovlp - OVLP) factor stays 1.
  template<class WlkSet, class Mat, class TVec>
  void measure_energy(WlkSet& wset, Mat&& E, TVec&& Ov, int nt = 0);

  template<class WlkSet>
  void Energy(WlkSet& wset);

  template<class WlkSet>
  void Energy(WlkSet& wset, int nt)
  {
    (void)nt;
    Energy(wset);
  }

  // Stochastic mixed density matrix for observable evaluation -- estimator 3 of arXiv:2505.18519,
  // G[w] = (sum_p w_p <psi_p|c+c|phi_w>/<psi_p|phi_w>) / sum_p w_p, in the caller's observable layout and
  // with the LOG overlap convention NOMSD's observable DM uses. The 2-arg overload mirrors NOMSD:
  // scratch overlap vector, routed through the 3-arg workhorse.
  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true)
  {
    int nw = wset.size();
    memory::buffered_array<MEM, ComplexType, 1> Ov(nw, ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), Ov, compact);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true);

  // DM w.r.t. an EXTERNALLY supplied reference orbital set `Ref`: pure orbital algebra independent of
  // both the trial and the Hamiltonian, so delegating is exact.
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

  // Accumulate observable contributions from the inner-ensemble-reduced (stochastic) Green's function
  // (estimator 3, full NMO x NMO layout), transformed through the evolved operators as NOMSD does when
  // time_evolved. The 5-arg overload mirrors NOMSD (null X/Yc/M).
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

  // Generalized Fock matrix of a SUPPLIED density matrix: a HamOp contraction with no trial reduction,
  // so it delegates. Reached only via the `generalizedFockMatrix` observable.
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  // Selects the back-propagation reference set (standard Motta-Zhang BP, arXiv:1707.02684): false =
  // delegate to the OUTER nomsd_'s anchor / CI expansion (static limit or P == 1); true = a dedicated,
  // walker-INDEPENDENT free-projection draw of the trial. The trial is walker-independent, so its
  // references must be too -- the forward walk's conditioning is a forward-only device.
  bool bp_uses_inner_ensemble() const
  {
    return inner_nsteps_ > 0 && inner_n_samples_ > 1
           && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr;
  }

  int total_number_of_references() const
  {
    return bp_uses_inner_ensemble() ? inner_n_samples_ : nomsd_.total_number_of_references();
  }

  int getNMO() const { return NMO; }

  ComplexType getReferenceWeight(int i) const
  {
    return bp_uses_inner_ensemble() ? ComplexType(1.0 / static_cast<double>(inner_n_samples_), 0.0)
                                    : nomsd_.getReferenceWeight(i);
  }

  // Fills the [nref, npol*NMO, nel] reference Slater matrices (H-conjugated bras) BackPropagatedEstimator
  // requests, from a fresh free-projection draw for a dynamic trial and from nomsd_ otherwise.
  template<class RefMat>
  void getReferences(RefMat&& Refs);

  HamiltonianTypes getHamType() const { return nomsd_.getHamType(); }

  // True iff this wavefunction's Hamiltonian operator can contract a FULL (un-rotated) mean-field G.
  // StochasticWfn::vMF requires it at inner_n_samples > 1; see HamiltonianOperations::has_fullG_vbias.
  bool has_fullG_vbias() const { return nomsd_.has_fullG_vbias(); }
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

  std::string system_;
  int NMO{-1};
  int nup{-1};
  int ndown{-1};
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_;
  StochasticInnerEnsemble inner_ensemble_;
  int inner_n_samples_{1};
  int inner_nsteps_{0};
  // Set from the variational Hamiltonian's stamp (or, for a synthetic trial, the input).
  // Deliberately NOT defaulted to a plausible value -- see the constructor.
  double inner_timestep_{0.0};
  bool inner_step_pending_{false};
  // Guards the back-propagation reference draw to at most ONCE per BP window; cleared by
  // begin_inner_step() so the next window draws fresh.
  bool bp_refs_drawn_{false};
  // The inner-sampling mode; Static is the safe default (it needs no sampler settings at all).
  SamplingTarget inner_sampling_target_{SamplingTarget::Static};
  // Persistent-chain controls. There is no on/off member: the chains ARE the conditioned sampler
  // (is_conditioned()). inner_burn_in_: one-time sweeps at the prime. inner_sample_update_steps_: sweeps
  // per pool advance thereafter. inner_sampler_ / inner_sampler_step_: proposal kernel ("pcn" or
  // "gaussian") and its step size s (pcn: 0 < s <= 1, s = 1 an independence redraw; gaussian: s > 0).
  int inner_burn_in_{0};
  int inner_sample_update_steps_{1};
  std::string inner_sampler_{"pcn"};
  double inner_sampler_step_{0.5};
  // Measurement-replica count. There is no restore flag: the measurement advance FEEDS FORWARD into
  // propagation, matching hafqmc, which keeps the pool its measurement produced.
  int inner_n_measure_samples_{1};
  // inner_chains_primed_: the per-walker field blocks hold live chain states. inner_dets_stale_: the
  // cached inner determinants no longer match the chain fields and must be rebuilt from them
  // (deterministically) before use.
  bool inner_chains_primed_{false};
  bool inner_dets_stale_{false};
  // Cumulative per-rank Metropolis statistics of the field-space chain updates.
  long chain_proposed_{0};
  long chain_accepted_{0};
  long chain_updates_{0};
  nda::array<ComplexType, 3> inner_anchor_;
  // Construction inputs for the inner walker set, cached at initialize_inner_walkers so the mean-field
  // scratch ensemble can be built on demand via the SAME (known-good) WalkerSet constructor.
  // mf_scratch_wset_ holds a P-sample walker-independent draw used ONLY by vMF/G_MF, lazily allocated.
  ptree inner_walker_pt_;
  std::vector<nda::matrix<ComplexType>> inner_initial_guess_;
  std::unique_ptr<WalkerSet<MEM>> mf_scratch_wset_;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> mf_scratch_rng_;
  // Leapfrog: per inner walker q (slot-major q = ip*nwalk + w), the magnitude |<psi_q|phi_w^cond>| of its
  // cross overlap with the walker its block was conditioned on. The leapfrog overlap divides by it, so
  // the step ratio is Eq. 25.
  nda::array<RealType, 1> inner_cond_mag_;
  // Per inner sample q: accumulated log importance weight logsw = sum_step HW from conditioned resample.
  // Used for leapfrog reweighting exp(logsw) on the non-persistent path; empty on the persistent path.
  nda::array<ComplexType, 1> inner_logsw_;
  NOMSD<MEM, devPsiT> nomsd_;
  std::unique_ptr<StochasticInnerStack<MEM, devPsiT>> inner_stack_;

  void maybe_advance_inner_ensemble();

  // Draw a fresh walker-INDEPENDENT free-projection ensemble into inner_ensemble_.wset (used as scratch)
  // for use as back-propagation references, and set the flags that make the draw idempotent within a BP
  // window. Called by getReferences.
  void draw_bp_reference_ensemble();

  // Reset `target` to the anchor, size it to P, and advance inner_nsteps_ BARE free-projection steps to
  // produce a walker-INDEPENDENT P-sample draw {psi_p = B_T(Y^[p])|phi_T>} of the trial. THE one home of
  // the free-projection draw, shared by draw_bp_reference_ensemble and mean_field_scratch_ensemble.
  // Sets NO forward-walk flags -- callers own their own state.
  void draw_free_projection_samples(WalkerSet<MEM>& target);

  // Lazily build (once) a DEDICATED scratch walker set and draw a fresh walker-independent P-sample
  // ensemble into it, so vMF/G_MF can reduce the FULL trial's mean field without perturbing the
  // conditioned forward walk, whose ensemble and flags must not be perturbed by a trial-only quantity.
  WalkerSet<MEM>& mean_field_scratch_ensemble();

  // Copy the P inner-walker Slater matrices (post draw_bp_reference_ensemble) into Refs.
  template<class RefMat>
  void fill_references_from_draw(int number_of_references, RefMat&& Refs);

  // Reset the first `count` inner walkers to the anchor |phi_T> (from inner_anchor_; handles
  // CLOSED/COLLINEAR). The single place that knows the anchor/collinear layout.
  void reset_inner_to_anchor(WalkerSet<MEM>& inner, int count);

  // ---- Persistent field-space chain machinery. All defined in StochasticWfn.cpp, whose propagator
  // ---- calls need the complete Propagator type, unavailable to the .icc templates.

  // Draw the chain start Y ~ p_T into every chain's slot of the outer walker set's TrialFields block.
  // Requires the block to be sized.
  void prime_chain_fields(WalkerSet<MEM>& wset);

  // Rebuild the inner determinants from the stored chain fields. Deterministic -- the single source of
  // truth for what psi_q is, and therefore exact after population control or a BP reference draw.
  void rebuild_inner_dets_from_chain_fields(WalkerSet<MEM> const& wset);

  // One Metropolis-Hastings sweep over all nw*P chains in FIELD space, targeting p_T(Y)|<psi(Y)|phi_w>|
  // on slot q = ip*nw + w. Batched: one determinant build of the whole pool + one cross-overlap pass;
  // rejected slots restore their determinant row and keep their fields.
  void chain_pool_sweep(WalkerSet<MEM>& wset);

  // Per-outer-step persistent chain update, called from begin_inner_step (the non-const seam): size,
  // prime (+ burn-in) or repair, sweep against the CURRENT walkers, refresh the leapfrog magnitudes.
  // Purely rank-local: no communication, no collectives.
  void update_persistent_chain_pool(WalkerSet<MEM>& wset);

  // One measurement replica's pool motion: update_persistent_chain_pool's tail without the
  // priming/sizing/staleness handling, since measure_advances_pool() guarantees live chains.
  void advance_measure_pool(WalkerSet<MEM>& wset);

  // Bootstrap (non-persistent) resample of the inner ensemble conditioned on each outer walker phi_w,
  // driven by the caller-supplied conditioning force bias X_bias (Eq. 23).
  void advance_inner_ensemble_conditioned(memory::array<MEM, ComplexType, 2> const& X_bias, int nw);

  // Resample dispatch: walker-conditioned when is_conditioned(), else the walker-independent
  // free-projection path. Honors the per-step latch armed by begin_inner_step().
  template<class WlkSet>
  void conditioned_resample(const WlkSet& wset);

  // Fill inner_cond_mag_ after a conditioned resample. The leapfrog overlap divides by it.
  template<class WlkSet>
  void compute_inner_cond_mag(const WlkSet& wset);

  // Fill mag(q) = |<inner_q|phi_w>| for the slot-major inner ensemble against the outer walker set (same
  // det(A^dag B) / CLOSED-doubling / COLLINEAR-product convention as the Log_Overlap reduction). Shared
  // by compute_inner_cond_mag and the Metropolis accept/reject. Rank-local. `mag` must be sized nw*P and
  // `inner` must be in the slot-major nw*P conditioned form.
  template<class WlkSet>
  void cross_overlap_magnitudes(const WlkSet& wset, WalkerSet<MEM>& inner, nda::array<RealType, 1>& mag);

  template<class WlkSet, class TVecD, class TVecOv, class Accumulate>
  void reduce_inner_cross_dm(const WlkSet& wset,
                             bool compact,
                             int Gsize,
                             TVecD&& D,
                             TVecOv&& Ov,
                             Accumulate&& accumulate);

  // Whether vMF / G_MF can reduce the FORWARD inner ensemble directly, i.e. it is already in its
  // walker-INDEPENDENT P-sample form with P > 1. When false they do NOT simply fall back to the anchor --
  // they branch on bp_uses_inner_ensemble(): the anchor mean field when the trial IS the anchor, else a
  // dedicated free-projection scratch draw of the FULL trial.
  bool mean_field_uses_inner_ensemble() const
  {
    // is_conditioned() is gated out EXPLICITLY rather than by size: at nwalk == 1 the conditioned
    // nwalk*P form collapses to inner_n_samples_ and would spoof the size test below.
    return inner_n_samples_ > 1 && not is_conditioned()
           && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr
           && int(inner_ensemble_.wset->size()) == inner_n_samples_;
  }

  // Build the normalized trial mean-field Green's function <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T> into `Gsum`
  // (full [1, nspin*npol*NMO*npol*NMO] layout) by reducing the P samples in `inner` against each other.
  // Shared by vMF (contracts it) and G_MF (returns it). `inner` MUST be in the walker-INDEPENDENT
  // P-sample form -- asserted.
  void reduce_inner_mean_field_dm(WalkerSet<MEM>& inner, memory::buffered_array<MEM, ComplexType, 2>& Gsum);

  int dm_size(bool full) const;
  bool compact_G_for_vbias() const;
};

} // namespace afqmc
} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"
