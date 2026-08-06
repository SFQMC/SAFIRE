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

// How the inner (trial) field ensemble is sampled. ONE key -- inner_sampling_target -- replaces the
// inner_conditioning / inner_leapfrog / inner_persistence booleans. Those were never independent: their
// legal combinations formed a strict chain (leapfrog required conditioning, persistence required
// conditioning, replicas required persistence) enforced by four separate runtime APP_ABORTs, so 2^3
// spellings selected 3 behaviours and the illegal 5 were only caught after construction. An enum makes
// them unrepresentable instead.
enum class SamplingTarget
{
  // inner_nsteps == 0. Replicated anchor ensemble; every reduction delegates to NOMSD, so observables are
  // independent of inner_n_samples. This is the EXACTNESS LIMIT the NOMSD-parity tests are written against
  // -- keep it: it is what proves the estimator right, not a production sampler.
  Static,
  // Dynamic (inner_nsteps > 0) with an unconditioned free-projection draw: fields from the bare prior
  // p_T(Y), the plain (1/P) sum reduction, no reweight. REFERENCE / TEST MODE, correct but not usable:
  // <psi(Y)|phi> is sharply peaked in Y, so prior sampling has catastrophic variance. Needed by the
  // back-propagation reference-draw and dynamic-ensemble tests.
  Gaussian,
  // Dynamic, walker-conditioned. Persistent per-slot field-space Markov chains targeting
  // p_T(Y)|<psi(Y)|phi_w>|, with the leapfrog 1/|<psi_q|phi_cond>| reweight so the step's overlap ratio
  // is exactly Eq. 25 and N(phi) cancels. THE PRODUCTION MODE, and the only one with a convergence story.
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

// Resolve the sampling mode from an input ptree, translating the four removed booleans it replaces.
//
// THIS IS THE SINGLE OWNER OF THAT TRANSLATION, and it is a free function precisely because there are
// TWO consumers that must never disagree: WavefunctionFactory::interpret_inputs (whose output selects how
// the inner PROPAGATOR is built) and StochasticWfn::interpret_inputs (which selects the sampler). The
// propagator is built first, so if the two resolved the mode independently -- or if only the later one
// knew how to read a legacy input -- a conditioned sampler would be paired with a free-projection
// propagator and abort at the first step. One function, two call sites, no second source of truth.
//
// `inner_nsteps` is passed in rather than read here: callers have already validated it.
inline std::string resolve_sampling_target(ptree const& pt0, int inner_nsteps)
{
  // Resolve and validate the sampling target. Shared by WavefunctionFactory::interpret_inputs (whose
  // output decides how the inner PROPAGATOR is built) and StochasticWfn::interpret_inputs (which selects
  // the sampler). The propagator is built first, so resolving in one place is what stops the two
  // disagreeing and pairing a conditioned sampler with a free-projection propagator.
  auto target_opt = pt0.get_optional<std::string>("inner_sampling_target");

  // NOT DEFAULTED for a dynamic trial: 'gaussian' is correct but has catastrophic variance, so it must
  // never be reachable by forgetting a key. Static needs no sampler settings and is the safe default.
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
  // inner_conditioning, inner_leapfrog, inner_persistence and inner_pool_burn_in are REMOVED options
  // (see interpret_inputs) but stay listed here: this list is "keys StochasticWfn owns OR HAS EVER
  // OWNED", so a legacy input is rejected by interpret_inputs rather than leaking through to the inner
  // wavefunction as an unknown key.
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
  // The three predicates below are SYNONYMS for inner_sampling_target_ == Conditioned, kept because the algorithm
  // bodies read better naming the mechanism in play at each site ("conditioned resample" vs "leapfrog
  // reweight" vs "persistent pool") than repeating the mode test. They are no longer independent knobs:
  // one mode turns on the conditioned target, the leapfrog reweight that makes it unbiased, and the
  // persistent chains that draw it -- the three are one algorithm, and any two without the third was
  // either rejected at construction or (conditioning without leapfrog) an estimator with no N(phi)
  // cancellation, i.e. wrong. Do NOT reintroduce them as separate inputs.
  bool inner_conditioning() const { return inner_sampling_target_ == SamplingTarget::WalkerOverlap; }
  bool inner_leapfrog() const { return inner_sampling_target_ == SamplingTarget::WalkerOverlap; }
  // Persistent (tethered) inner sampling -- the ONLY conditioned inner sampler, not an option. Each
  // (outer walker, p) slot owns a Markov chain whose STATE is the auxiliary-field configuration Y that
  // generates its trial sample psi = B_T(Y)|phi_T>; chains are kept across outer steps and
  // re-equilibrated by a short Metropolis-Hastings walk in field space targeting
  // p_T(Y)*|<psi(Y)|phi_w>|. The field configurations live in a per-walker block of the OUTER walker
  // buffer (WalkerSetBase::TrialFields), so population control clones and ships the chains with their
  // walkers and the determinants can be rebuilt exactly wherever a walker lands.
  // True exactly when the persistent chains are the active sampler, i.e. in Conditioned mode; Static and
  // Free have no chains.
  bool inner_persistence() const { return inner_sampling_target_ == SamplingTarget::WalkerOverlap; }
  // MH sweeps applied to the field-chain pool each time the pool is advanced. ONE knob for BOTH seams,
  // matching hafqmc's sample_update_steps, which drives its propagation-side re-tether
  // (update_tethered_samples_state) and its measurement-side advance (measure_block_energy_state) from
  // the same number. It replaced inner_equil_steps + inner_measure_stride, which counted the identical
  // thing -- chain_pool_sweep() calls -- and which production always set equal anyway, since the stride
  // defaulted to the equil count.
  //
  // The two seams do differ in what the sweeps are chasing, and that is worth knowing even though it
  // does not warrant two inputs: at the propagation seam the walker has just moved, so the chain is
  // tracking a MOVING target and its lag has a floor no sweep count removes; at the measurement seam the
  // walkers are fixed, so the lag decays geometrically to zero and extra sweeps also decorrelate
  // successive replicas.
  int inner_burn_in() const { return inner_burn_in_; }
  int inner_sample_update_steps() const { return inner_sample_update_steps_; }
  std::string const& inner_sampler() const { return inner_sampler_; }
  double inner_sampler_step() const { return inner_sampler_step_; }
  // Measurement-side replica averaging (inner_n_measure_samples). inner_equil_steps equilibrates the pool
  // against walkers that move every propagation step; these knobs advance the pool at fixed walkers and
  // average the local energy over inner_n_measure_samples such advances.
  int inner_n_measure_samples() const { return inner_n_measure_samples_; }
  // True when the measurement seam advances the pool, i.e. whenever a live persistent chain exists.
  //
  // ⚠️ THIS NO LONGER REQUIRES nm > 1, and that is a deliberate hafqmc-matching change. hafqmc advances
  // its pool by sample_update_steps sweeps against the CURRENT walkers before EVERY block measurement
  // (trial/stochastic_runtime_methods.py, measure_block_energy_state -> advance_steps), including the
  // n_measure_samples == 1 case. SAFIRE used to skip the advance entirely at nm == 1, so it measured with
  // a pool tethered to the walker as of the START of the last propagation step -- several steps, an
  // orthogonalisation and possibly a population-control event stale. nm is now purely the number of
  // replicas AVERAGED; whether the pool advances is not a knob.
  //
  // Public so a test can assert the path is LIVE instead of passing vacuously on the fallback.
  bool measure_advances_pool() const
  {
    return inner_persistence() && inner_chains_primed_ && not inner_dets_stale_;
  }
  // Cumulative Metropolis acceptance fraction of the field-space chain updates on this rank
  // (1.0 before any proposal has been made).
  double inner_chain_acceptance() const
  {
    return chain_proposed_ > 0 ? double(chain_accepted_) / double(chain_proposed_) : 1.0;
  }
  // Sum of the leapfrog conditioning magnitudes |<psi_q|phi_cond>| (test/diagnostic checksum; 0 when
  // unset). A pop-control realignment that only re-indexes the ensemble must leave this sum invariant
  // -- it references phi_cond (the walker the chains were equilibrated against), not the post-pop
  // walker -- so a test can pin "permute must not recompute cond_mag against the current walker".
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
  // No-op unless this is a conditioned dynamic trial (SamplingTarget::WalkerOverlap); static,
  // free-projection, and delegate-limit trials carry no slot-conditioned inner blocks.
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

  // Stochastic mean-field subtraction. vMF / G_MF are TRIAL-AGAINST-ITSELF quantities (no outer walker),
  // so -- unlike the propagator hot-path mixed estimators that pair the inner ensemble against the OUTER
  // ensemble against the OUTER walkers -- they reduce the inner ensemble against ITSELF:
  //   G_MF = <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T>
  //        = [sum_{p,q} <psi_p|c+c|psi_q>] / [sum_{p,q} <psi_p|psi_q>]
  // over the P = inner_n_samples_ walker-INDEPENDENT inner samples {psi_p} (uniform weight 1/P) -- the
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

  // Measurement entry point for the estimators: Energy() averaged over inner_n_measure_samples_ replicas
  // of the field pool at FIXED walkers (see inner_n_measure_samples()). Needs non-const wset because the
  // chain state lives in the outer walker buffer's TrialFields block, so advancing it writes there.
  //
  // Reduces to exactly Energy(wset, E, Ov, nt) -- same call, same values -- unless the replica path is
  // active (measure_advances_pool()).
  //
  // Estimator convention: E is the mean of per-replica energy ratios; Ov is the walker's stored log
  // overlap (the pool weights were accumulated against), so EnergyEstimator's exp(ovlp - OVLP) stays 1.
  // Do not return a replica's own overlap -- replicas are measured after advance_measure_pool.
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
  //    Identical to plain NOMSD. Used at inner_nsteps==0 and inner_n_samples==1.
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

  // Fills the [nref, npol*NMO, nel] reference Slater matrices (H-conjugated bras), as
  // BackPropagatedEstimator requests. For a dynamic trial (bp_uses_inner_ensemble()) it first performs a
  // dedicated free-projection draw (draw_bp_reference_ensemble()) and fills from those P samples;
  // otherwise it delegates to nomsd_ (the anchor / CI expansion). Defined in the .icc.
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
  // Guards the back-propagation reference draw so it happens at most ONCE per BP window.
  // Set true by draw_bp_reference_ensemble() after a draw; a repeated getReferences in the same window
  // then reuses that draw (idempotent -- no silent re-draw). Reset to false by begin_inner_step(), i.e.
  // when the forward walk advances to the next step (the next BP window draws fresh).
  bool bp_refs_drawn_{false};
  // The inner-sampling mode. Static is the safe default: it is the only mode that needs no sampler
  // settings at all, and a dynamic trial must name its mode explicitly (see interpret_inputs) so that
  // Free -- correct but catastrophically noisy -- can never be reached by forgetting a key.
  SamplingTarget inner_sampling_target_{SamplingTarget::Static};
  // Persistent-chain controls. There is no on/off member: the chains ARE the conditioned sampler, so
  // inner_persistence() is derived from inner_sampling_target_ == Conditioned. inner_sample_update_steps_: MH sweeps per pool
  // advance, applied from the one-time prime onwards -- there is no separate burn-in count, because the
  // prime is followed by inner_sample_update_steps_ sweeps every step and the outer equilibration window discards
  // those steps anyway.
  // inner_sampler_: proposal kernel -- "pcn" (preconditioned Crank-Nicolson, prior-preserving:
  // Y* = sqrt(1-s^2) Y + s xi) or "gaussian" (random walk: Y* = Y + s xi, prior ratio in the
  // acceptance). inner_sampler_step_: the proposal step size s (pcn: 0 < s <= 1, s = 1 is an
  // independence redraw; gaussian: s > 0).
  // MH sweeps run ONCE when the chains are primed, before the walk starts -- hafqmc's burn_in
  // (--trial-burn-in, default 100 in the production runner). Distinct from inner_sample_update_steps,
  // which is paid on every pool advance thereafter.
  int inner_burn_in_{0};
  int inner_sample_update_steps_{1};
  std::string inner_sampler_{"pcn"};
  double inner_sampler_step_{0.5};
  // Measurement-replica count. There is no restore flag: the measurement advance FEEDS FORWARD into
  // propagation, matching hafqmc, which keeps the pool its measurement produced.
  int inner_n_measure_samples_{1};
  // inner_chains_primed_: the per-walker field blocks hold live chain states (set at the first
  // persistent pool update, which runs inside begin_inner_step -- the one seam with non-const access
  // to the outer walker set). inner_dets_stale_: the cached inner determinants no longer match the
  // chain fields (the BP reference draw reused the pool storage, or population control moved chains
  // between slots/ranks) and must be rebuilt -- deterministically -- from the fields before use.
  bool inner_chains_primed_{false};
  bool inner_dets_stale_{false};
  // Cumulative per-rank Metropolis statistics of the field-space chain updates.
  long chain_proposed_{0};
  long chain_accepted_{0};
  long chain_updates_{0};
  nda::array<ComplexType, 3> inner_anchor_;
  // Construction inputs for the inner walker set, cached at initialize_inner_walkers so a dedicated
  // mean-field scratch ensemble (mf_scratch_wset_) can be built on demand via the SAME (known-good)
  // WalkerSet constructor, without touching the forward ensemble. mf_scratch_wset_ holds a P-sample
  // walker-independent draw of the trial used ONLY by vMF/G_MF (lazily allocated on first use).
  ptree inner_walker_pt_;
  std::vector<nda::matrix<ComplexType>> inner_initial_guess_;
  std::unique_ptr<WalkerSet<MEM>> mf_scratch_wset_;
  std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> mf_scratch_rng_;
  // Leapfrog: per inner walker q (slot-major q = ip*nwalk + w), the magnitude |⟨ψ_q|φ_w^cond⟩| of its cross overlap with the walker its block was conditioned on. Set at each
  // conditioned resample; the leapfrog overlap reweights by 1/inner_cond_mag_ so the step ratio is Eq. 25.
  nda::array<RealType, 1> inner_cond_mag_;
  // Per inner sample q: accumulated log importance weight logsw = sum_step HW from conditioned resample.
  // Used for leapfrog reweighting exp(logsw) on the non-persistent path; empty on the persistent path.
  nda::array<ComplexType, 1> inner_logsw_;
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

  // Reset `target` to the anchor |phi_T>, size it to P = inner_n_samples_, then advance inner_nsteps_ BARE
  // free-projection steps (inner_propagator().Propagate_free) to produce a walker-INDEPENDENT P-sample
  // draw {psi_p = B_T(Y^[p])|phi_T>} of the trial. The one place the free-projection draw lives; shared by
  // draw_bp_reference_ensemble (back-propagation references) and mean_field_scratch_ensemble (the trial
  // mean field). Sets NO forward-walk flags -- callers own their own state. Defined in StochasticWfn.cpp
  // (needs the complete Propagator type). Device-safe (reset_inner_to_anchor + device-ported Propagate_free).
  void draw_free_projection_samples(WalkerSet<MEM>& target);

  // Lazily build (once) a DEDICATED scratch walker set, draw a fresh walker-independent P-sample
  // free-projection ensemble into it, and return it. Used by vMF/G_MF to reduce the FULL-trial mean field
  // <Psi_T|.|Psi_T> when the forward inner ensemble has been expanded to the conditioned nw*P form (so
  // reducing it would be wrong AND clobbering it would corrupt the forward walk). Keeping the draw in its
  // own scratch leaves inner_ensemble_.wset and every forward-walk flag untouched -- the mean field is a
  // trial-only quantity and must not perturb the conditioned/persistent/leapfrog walk. Defined in
  // StochasticWfn.cpp (needs the complete Propagator type).
  WalkerSet<MEM>& mean_field_scratch_ensemble();

  // Copy the P inner-walker Slater matrices (post draw_bp_reference_ensemble) into Refs.
  template<class RefMat>
  void fill_references_from_draw(int number_of_references, RefMat&& Refs);

  // Reset the first `count` inner walkers in `inner` to the anchor |phi_T> (the per-spin Slater matrices
  // cached in inner_anchor_; handles CLOSED/COLLINEAR). The single place that knows the anchor/collinear
  // layout -- shared by the free-projection advance, the conditioned resample, and the BP reference draw.
  void reset_inner_to_anchor(WalkerSet<MEM>& inner, int count);

  // ---- Persistent field-space chain machinery (all defined in StochasticWfn.cpp: the propagator
  // ---- calls need the complete Propagator type, unavailable to the .icc templates).

  // Draw a fresh field configuration Y ~ p_T (i.i.d. standard normals) into every chain's slot of the
  // outer walker set's TrialFields block -- the chain start. Requires the block to be sized.
  void prime_chain_fields(WalkerSet<MEM>& wset);

  // Rebuild the inner determinants from the chain fields: reset the (slot-major nw*P) pool to the
  // anchor |phi_T>, then apply the inner propagator's B_T with the STORED fields, one inner step at a
  // time (Propagate_given_fields). Deterministic -- the single source of truth for what psi_q is --
  // and therefore exact after population control moved chains between slots or ranks, and after the
  // back-propagation reference draw reused the pool storage.
  void rebuild_inner_dets_from_chain_fields(WalkerSet<MEM> const& wset);

  // One Metropolis-Hastings sweep over all nw*P chains, in FIELD space (the chain state is the field
  // configuration Y, not the determinant). Proposal per inner_sampler_: pCN (prior-preserving) or
  // random-walk gaussian (prior ratio in the acceptance). Target on slot q = ip*nw + w:
  // p_T(Y) * |<psi(Y)|phi_w>|. Batched: one Propagate_given_fields build of all proposal determinants
  // + one cross-overlap pass; rejected slots restore their determinant row (fields untouched).
  void chain_pool_sweep(WalkerSet<MEM>& wset);

  // Per-outer-step persistent chain update, called from begin_inner_step (the non-const seam):
  // lazily size the TrialFields block, prime the chains on first use (+ burn-in), rebuild stale
  // determinants, run inner_sample_update_steps_ sweeps against the CURRENT (old) walkers, and refresh the
  // leapfrog conditioning magnitudes. Purely rank-local: no communication, no collectives.
  void update_persistent_chain_pool(WalkerSet<MEM>& wset);

  // Advance the persistent pool by inner_sample_update_steps_ sweeps against the CURRENT (fixed) walkers and
  // refresh the leapfrog conditioning magnitudes -- one measurement replica's worth of pool motion.
  // Same tail as update_persistent_chain_pool, minus the priming/sizing/staleness handling: the caller
  // (measure_energy) only runs when the chains are already live.
  void advance_measure_pool(WalkerSet<MEM>& wset);

  // Resample the inner ensemble conditioned on each outer walker phi_w. Computes the custom inner force bias x_bar(phi_w) = sqrt(dt)*L^var . <phi_T|c+c|phi_w>/<phi_T|phi_w> (the inner
  // trial IS the anchor phi_T, so this reuses inner_nomsd()'s mixed DM + vbias on the OUTER wset) and
  // drives the nw*P inner ensemble through the inner propagator's conditioned field-sampling seam.
  void advance_inner_ensemble_conditioned(memory::array<MEM, ComplexType, 2> const& X_bias, int nw);

  // Resample dispatch: walker-conditioned in Conditioned mode, else the walker-independent
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

  template<class WlkSet, class TVecD, class TVecOv, class Accumulate>
  void reduce_inner_cross_dm(const WlkSet& wset,
                             bool compact,
                             int Gsize,
                             TVecD&& D,
                             TVecOv&& Ov,
                             Accumulate&& accumulate);

  // Whether vMF / G_MF can reduce the FORWARD inner ensemble directly (true), i.e. it is already in its
  // walker-INDEPENDENT P-sample form (inner.size() == inner_n_samples_) with P > 1. When false, vMF/G_MF do
  // NOT simply fall back to the anchor -- they branch on bp_uses_inner_ensemble():
  //  - inner_n_samples_ == 1 OR inner_nsteps_ == 0 (bp_uses_inner_ensemble() false): the trial IS the
  //    anchor (a single/degenerate sample, or no free projection), so delegate to nomsd_'s anchor mean
  //    field -- exact, and avoids a redundant reduction. Subsumes the single-determinant delegate limit.
  //  - inner_nsteps_ > 0 && P > 1 but the ensemble has been expanded by a conditioned/leapfrog/persistent
  //    resample to the nwalk*P walker-CONDITIONED form (bp_uses_inner_ensemble() true): there is no
  //    walker-independent subset to average here, so vMF/G_MF draw a DEDICATED free-projection scratch
  //    ensemble (mean_field_scratch_ensemble) and reduce the FULL trial's mean field from that -- NOT the
  //    anchor. Reducing the anchor here (the historical behavior) made the HS-contour shift inconsistent
  //    with the full-trial force bias/energy and biased the phaseless constraint toward overbinding.
  bool mean_field_uses_inner_ensemble() const
  {
    // A conditioned dynamic trial's forward ensemble is the slot-major nwalk*P walker-CONDITIONED form.
    // At nwalk==1 (a single outer walker per rank -- the normal MPI layout, and an explicitly-tested
    // regime) that size collapses to 1*P == inner_n_samples_ and would SPOOF the size test below, making
    // vMF/G_MF reduce the phi_w-conditioned ensemble as if it were the walker-independent trial. So gate
    // conditioned dynamic trials OUT explicitly: they always take the dedicated free-projection scratch
    // draw (mean_field_scratch_ensemble) regardless of nwalk. Persistence IS Conditioned mode, so
    // this covers the persistent path too. (The same nwalk==1 size ambiguity is handled in
    // conditioned_resample via the latch/flags rather than size -- see that routine.)
    return inner_n_samples_ > 1 && not inner_conditioning()
           && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr
           && int(inner_ensemble_.wset->size()) == inner_n_samples_;
  }

  // Build the normalized stochastic trial mean-field one-body Green's function <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T> into `Gsum` (full [1, nspin*npol*NMO*npol*NMO] layout) by reducing
  // the P = inner_n_samples_ walker-independent samples in `inner` against each other (the double sum
  // above). Shared by vMF (contracts it) and G_MF (returns it). Trial-only -- no outer walker set, no
  // conditioned resample. `inner` MUST be in the walker-INDEPENDENT P-sample form (inner.size() ==
  // inner_n_samples_): either the forward ensemble when mean_field_uses_inner_ensemble() is true, or the
  // dedicated mean-field scratch draw (mean_field_scratch_ensemble()) when the forward ensemble has been
  // expanded to the conditioned nw*P form.
  void reduce_inner_mean_field_dm(WalkerSet<MEM>& inner, memory::buffered_array<MEM, ComplexType, 2>& Gsum);

  int dm_size(bool full) const;
  bool compact_G_for_vbias() const;
};

} // namespace afqmc
} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"
