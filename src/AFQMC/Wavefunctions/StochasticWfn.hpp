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

#include "AFQMC/parameters.hpp"
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

/**
 * @brief How the inner (trial) auxiliary-field ensemble of a StochasticWfn is sampled.
 *
 * @details Alias of StochasticSamplingTarget (AFQMC/parameters.hpp), kept under this name because the
 * modes form a strict chain (the leapfrog reweight requires conditioning, persistent chains require
 * conditioning, measurement replicas require persistence): one key rather than independent booleans
 * makes the illegal combinations unrepresentable instead of rejecting them at runtime after
 * construction.
 */
using SamplingTarget = StochasticSamplingTarget;

/**
 * @brief Input spelling of a SamplingTarget.
 *
 * @param m the sampling target to spell
 */
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

/**
 * @brief Resolve and validate the inner sampling mode from the (already schema-parsed) wavefunction
 *        parameters.
 *
 * @details Single owner of that translation for WavefunctionFactory (propagator build) and
 * StochasticWfn (sampler selection). The propagator is built first, so resolving it twice risks
 * pairing a conditioned sampler with a free-projection propagator, which aborts on the first step.
 * Not defaulted when inner_nsteps > 0: defaulting to "gaussian" would silently give prior sampling in
 * production. "static" is the safe default at inner_nsteps == 0.
 *
 * @param params the wavefunction input block
 */
inline StochasticSamplingTarget resolve_sampling_target(WavefunctionParameters const& params)
{
  if (not params.inner_sampling_target && params.inner_nsteps > 0)
    APP_ABORT("Error in StochasticWfn: inner_nsteps > 0 selects a dynamic inner ensemble, so "
              "inner_sampling_target must be given explicitly -- 'walker_overlap' for the production "
              "walker-conditioned persistent chains, or 'gaussian' for the unconditioned prior-sampling "
              "reference target (correct, but its variance makes it unusable in production). It is NOT "
              "defaulted, because defaulting it is how a production run silently gets prior sampling.");
  const StochasticSamplingTarget target =
      params.inner_sampling_target.value_or(StochasticSamplingTarget::Static);
  if (target == StochasticSamplingTarget::Static && params.inner_nsteps > 0)
    APP_ABORT("Error in StochasticWfn: inner_sampling_target = static is the inner_nsteps = 0 replicated "
              "anchor ensemble, but inner_nsteps = " + std::to_string(params.inner_nsteps) + " was given.");
  if (target != StochasticSamplingTarget::Static && params.inner_nsteps <= 0)
    APP_ABORT("Error in StochasticWfn: inner_sampling_target = " + to_string(target) +
              " is a dynamic sampler and requires inner_nsteps > 0.");
  return target;
}

/**
 * @brief Type-erased handle on the inner (variational) stack a StochasticWfn samples its trial from.
 *
 * @details The inner stack is a NOMSD whose trial IS the anchor determinant, the Wavefunction variant
 * wrapping it, and the propagator carrying the trained B_T. StochasticWfn holds it behind this
 * interface so it does not have to name the concrete inner wavefunction/propagator types, which are
 * only known to WavefunctionFactory (see StochasticInnerStackImpl in WavefunctionFactory.cpp).
 *
 * @param MEM memory space of the inner stack, matching the owning StochasticWfn
 * @param devPsiT storage type of the trial orbital matrices
 */
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

/**
 * @brief Trial wavefunction represented by a Monte-Carlo ensemble of auxiliary-field samples rather
 *        than by a fixed determinant expansion.
 *
 * @details |Psi_T> ~ (1/P) sum_p B_T(Y^[p]) |phi_T> with P = inner_n_samples (arXiv:2505.18519). An
 * outer NOMSD scores against the True Hamiltonian; a StochasticInnerStack holds the variational
 * sampler. Overrides are either CROSS (inner vs outer walkers: Energy, Log_Overlap, mixed DM) or SELF
 * (inner vs itself: vMF, G_MF); everything else delegates. At at_delegate_limit() the trial is the
 * anchor and every override forwards to NOMSD. Do not distinguish walker-independent vs
 * walker-conditioned layouts by size alone: they coincide at nwalk == 1.
 *
 * @param MEM memory space the wavefunction and its ensembles live in
 * @param devPsiT storage type of the trial orbital matrices
 */
template<MEMORY_SPACE MEM, class devPsiT>
class StochasticWfn
{
public:
  /**
   * @brief Build the stochastic trial: outer NOMSD from the True Hamiltonian, sampler settings from
   *        `params` (validated here via validate_stochastic_inputs), inner stack adopted from the
   *        factory.
   *
   * @details Validates the sampler settings via validate_stochastic_inputs and aborts on an
   * inconsistent combination. A dynamic trial (inner_nsteps > 0) requires CLOSED or COLLINEAR walkers
   * and an explicit inner propagator timestep -- there is no default, because a wrong one is
   * indistinguishable from a working run.
   *
   * @param system name of the system this wavefunction belongs to
   * @param NMO_ number of molecular orbitals
   * @param nup_ number of spin-up electrons
   * @param ndown_ number of spin-down electrons
   * @param params the wavefunction input block
   * @param mpi_in MPI context shared with the rest of the calculation
   * @param outer_hop_ HamiltonianOperations of the TRUE Hamiltonian, moved into the outer NOMSD
   * @param ci_ CI coefficients of the anchor expansion
   * @param orbs_ orbital matrices of the anchor expansion
   * @param inner_stack_in the inner (variational) NOMSD + propagator; must not be null
   * @param wlk walker type; a dynamic trial supports CLOSED and COLLINEAR only
   * @param targetNW target walker count, forwarded to the outer NOMSD
   */
  StochasticWfn(std::string system,
                int NMO_,
                int nup_,
                int ndown_,
                WavefunctionParameters params,
                std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_in,
                HamiltonianOperations<MEM>&& outer_hop_,
                nda::array<ComplexType, 1>&& ci_,
                nda::array<devPsiT, 2>&& orbs_,
                std::unique_ptr<StochasticInnerStack<MEM, devPsiT>>&& inner_stack_in,
                WALKER_TYPES wlk,
                [[maybe_unused]] int targetNW = 1);

  /**
   * @brief Validate `params` in place, defaulting and resolving every stochastic field.
   *
   * @details Unknown-key rejection is the JSON schema's job, not this function's -- the inner_* surface
   * is closed at the schema level. This only fills defaults that depend on other fields (e.g. the
   * sampling target, the sampler step) and checks cross-field consistency.
   *
   * @param params the wavefunction input block, updated in place
   */
  static void validate_stochastic_inputs(WavefunctionParameters& params);

  ~StochasticWfn() = default;

  StochasticWfn(StochasticWfn const& other)            = delete;
  StochasticWfn& operator=(StochasticWfn const& other) = delete;
  StochasticWfn(StochasticWfn&& other)                   = default;
  StochasticWfn& operator=(StochasticWfn&& other)        = delete;

  /**
   * @brief Allocate the inner walker set and cache the anchor determinant it is reset to.
   *
   * @details Must be called once before any reduction. The construction inputs are cached so the
   * mean-field scratch ensemble can later be built through this same known-good WalkerSet
   * constructor.
   *
   * @param walker_params the walker-set input block, shared with the outer walkers
   * @param initial_guess per-spin Slater matrices of the anchor |phi_T>
   * @param NAEB number of spin-down electrons, used to size the beta block of a COLLINEAR anchor
   */
  void initialize_inner_walkers(WalkerSetParameters const& walker_params,
                                std::vector<nda::matrix<ComplexType>> const& initial_guess,
                                int NAEB);

  /// @brief True once initialize_inner_walkers() has run.
  bool inner_walkers_initialized() const { return inner_ensemble_.initialized; }
  /// @brief P, the number of auxiliary-field samples representing the trial.
  int inner_n_samples() const { return inner_n_samples_; }
  /// @brief Length of the inner auxiliary-field path; 0 is the static (anchor) ensemble.
  int inner_nsteps() const { return inner_nsteps_; }
  /// @brief How the inner ensemble is sampled.
  SamplingTarget inner_sampling_target() const { return inner_sampling_target_; }
  /**
   * @brief The trained B_T timestep actually in force.
   *
   * @details Exposed so a test can assert the value the factory read from the variational
   * Hamiltonian's inner_timestep attribute REACHED the wavefunction; a factory that reads the stamp
   * and drops it is indistinguishable from one that works unless the built object is asked.
   */
  double inner_timestep() const { return inner_timestep_; }
  /**
   * @brief True for SamplingTarget::WalkerOverlap, the walker-conditioned production sampler.
   *
   * @details There is no separate persistence or leapfrog predicate: conditioning, the persistent
   * field chains and the leapfrog reweight are one algorithm, not independent knobs.
   */
  bool is_conditioned() const { return inner_sampling_target_ == SamplingTarget::WalkerOverlap; }
  /// @brief One-time Metropolis-Hastings sweeps applied when the field chains are primed (hafqmc's burn_in).
  int inner_burn_in() const { return inner_burn_in_; }
  /**
   * @brief Metropolis-Hastings sweeps per pool advance.
   *
   * @details ONE knob for BOTH the propagation and the measurement seam, matching hafqmc's
   * sample_update_steps, which drives its re-tether and its block-measurement advance from the same
   * number. The seams differ in what the sweeps chase -- a moving target at propagation, where the lag
   * has a floor no sweep count removes, and a fixed one at measurement, where it decays geometrically
   * -- but not enough to warrant two inputs.
   */
  int inner_sample_update_steps() const { return inner_sample_update_steps_; }
  /// @brief Field-space proposal kernel: "pcn" (prior-preserving) or "gaussian" (random walk).
  std::string const& inner_sampler() const { return inner_sampler_; }
  /// @brief Proposal step size s; pcn takes 0 < s <= 1 (s = 1 is an independence redraw), gaussian s > 0.
  double inner_sampler_step() const { return inner_sampler_step_; }
  /// @brief Number of measurement replicas averaged at fixed walkers.
  int inner_n_measure_samples() const { return inner_n_measure_samples_; }
  /**
   * @brief True when the measurement seam advances the field pool, i.e. whenever live chains exist.
   *
   * @details Deliberately NOT gated on inner_n_measure_samples > 1: hafqmc advances its pool before
   * EVERY block measurement, including the single-replica case, so the replica count is purely how
   * many measurements are AVERAGED and whether the pool advances is not a knob. Public so a test can
   * assert the replica path is LIVE rather than passing vacuously on the fallback.
   */
  bool measure_advances_pool() const
  {
    return is_conditioned() && inner_chains_primed_ && not inner_dets_stale_;
  }
  /// @brief Cumulative Metropolis acceptance fraction of the field-space chain updates on this rank; 1.0
  /// before any proposal has been made.
  double inner_chain_acceptance() const
  {
    return chain_proposed_ > 0 ? double(chain_accepted_) / double(chain_proposed_) : 1.0;
  }
  /**
   * @brief Checksum over the leapfrog conditioning magnitudes |<psi_q|phi_cond>|; 0 when unset.
   *
   * @details Test/diagnostic only. A population-control realignment that merely re-indexes the
   * ensemble must leave this invariant, since the magnitudes reference the walker the chains were
   * equilibrated against and not the post-pop walker.
   */
  double inner_cond_mag_sum() const
  {
    double s = 0.0;
    for (long i = 0; i < inner_cond_mag_.size(); ++i)
      s += inner_cond_mag_(i);
    return s;
  }

  /// @brief The inner ensemble; aborts if initialize_inner_walkers() has not run.
  WalkerSet<MEM>& inner_wset();
  WalkerSet<MEM> const& inner_wset() const;

  /**
   * @name Inner stack accessors
   *
   * @brief The variational NOMSD whose trial is the anchor |phi_T>, and the Wavefunction wrapping it.
   *
   * @details inner_wfn() and inner_nomsd() are the same object under two names, kept because callers
   * read naturally either way. The inner NOMSD is what supplies the conditioning force bias, since
   * the inner trial IS the anchor.
   * @{
   */
  NOMSD<MEM, devPsiT>& inner_wfn() { return inner_stack_->nomsd(); }
  NOMSD<MEM, devPsiT> const& inner_wfn() const { return inner_stack_->nomsd(); }

  NOMSD<MEM, devPsiT>& inner_nomsd() { return inner_stack_->nomsd(); }
  NOMSD<MEM, devPsiT> const& inner_nomsd() const { return inner_stack_->nomsd(); }

  Wavefunction<MEM>& inner_wavefunction() { return inner_stack_->wavefunction(); }
  Wavefunction<MEM> const& inner_wavefunction() const { return inner_stack_->wavefunction(); }
  /// @}

  /// @brief True once the factory has built the inner propagator carrying B_T.
  bool inner_propagator_built() const { return inner_stack_->has_propagator(); }
  /// @brief The propagator carrying B_T; aborts if the factory did not build one.
  Propagator<MEM>& inner_propagator();
  Propagator<MEM> const& inner_propagator() const;

  /**
   * @brief Open an outer propagation step: arm the resample latch, advance the persistent field-chain
   *        pool, and refresh the stored overlap.
   *
   * @details Sole seam with non-const access to the outer walker buffer (chain fields). The updated
   * pool serves every reduction of the step. Runs on every rank; no communication.
   *
   * @param wset the outer walker set, taken non-const because the chain state lives in its buffer
   */
  void begin_inner_step(WalkerSet<MEM>& wset);

  /**
   * @brief Realign the conditioned inner ensemble with the outer walkers after population control.
   *
   * @details The driver calls this immediately after wset.popControl(). The slot-major inner ensemble
   * lives OUTSIDE the outer walker buffer, so popControl's clone/shuffle would otherwise leave block w
   * attached to the old phi_w, and a reduction before the next propagation step would pair post-branch
   * walkers with pre-branch inner blocks. No-op unless is_conditioned(): the other modes carry no
   * slot-conditioned blocks.
   *
   * @param wset the post-population-control outer walker set
   */
  void permute_inner_blocks_after_pop(WalkerSet<MEM> const& wset);

  /**
   * @brief Snapshot inner_cond_mag_ into the walkers' TRIAL_COND_MAG block before population control.
   *
   * @details The driver calls this immediately before wset.popControl(). Each magnitude belongs to the
   * phi_cond its chain was equilibrated against, so it must move with that walker: in the walker buffer
   * branch() clones it and load balancing ships it, which lets the post-pop hook read every slot back
   * exactly -- including a walker that arrived from another rank, whose value used to be unavailable
   * here and was approximated by a recompute against the post-pop walker. No-op unless is_conditioned().
   *
   * @param wset the pre-population-control outer walker set
   */
  void store_inner_blocks_before_pop(WalkerSet<MEM>& wset);

  /// @brief True when the trial IS the anchor determinant (P == 1 and no free projection), so every override
  /// delegates and the class is exactly a single-determinant NOMSD.
  bool at_delegate_limit() const { return inner_n_samples_ == 1 && inner_nsteps_ == 0; }

  /// @brief The outer NOMSD carrying the TRUE Hamiltonian, against which every reduction is scored.
  NOMSD<MEM, devPsiT>& outer_nomsd() { return nomsd_; }
  NOMSD<MEM, devPsiT> const& outer_nomsd() const { return nomsd_; }

  int number_of_cholesky_vectors() const { return nomsd_.number_of_cholesky_vectors(); }

  /**
   * @brief Per-run optimization hook, delegated to the outer NOMSD.
   *
   * @param wset the outer walker set
   */
  void runtime_optimization(WalkerSet<MEM>& wset) { nomsd_.runtime_optimization(wset); }

  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }
  constexpr auto get_memory_space() const { return MEM; }

  /**
   * @brief Mean-field expectation of the Cholesky/HS potentials for the FULL stochastic trial,
   *        v_n = L_n . G_MF (estimator 4).
   *
   * @details Self-reduction over the P walker-independent samples (no outer walker). Must use the
   * full trial's mean field, not the anchor's, for consistency with the force bias and energy.
   * Ensemble choice follows mean_field_uses_inner_ensemble() / bp_uses_inner_ensemble(). Requires
   * has_fullG_vbias().
   *
   * @param v output potential vector, one entry per Cholesky vector
   * @param dt timestep the potentials are scaled by
   */
  void vMF(nda::MemoryVector auto&& v, double dt);

  /**
   * @brief Mean-field one-body Green's function of the FULL stochastic trial,
   *        <Psi_T|c+c|Psi_T>/<Psi_T|Psi_T>, in the full [nspin][npol*NMO][npol*NMO] layout.
   *
   * @details Same self-reduction and same ensemble choice as vMF, but returned rather than contracted,
   * and deliberately NOT gated on the Hamiltonian's full-G capability because it needs no Hamiltonian
   * operator at all. Only the discrete/model-Hamiltonian propagator setup consumes it; continuous
   * Cholesky propagators use vMF only.
   */
  auto G_MF();

  /**
   * @brief Force bias of the outer walkers against the stochastic trial.
   *
   * @details Off the anchor the inner-ensemble-reduced density matrix has no half-rotated form, so
   * this builds a full un-rotated G and requires a Hamiltonian operator that can contract one.
   *
   * @param wset the outer walker set
   * @param v output force bias, [nwalk, number_of_cholesky_vectors]
   * @param dt timestep the potentials are scaled by
   * @param nt time slice index, unused for a ground-state trial
   */
  template<class WlkSet, nda::MemoryMatrix MatA>
  void vbias(WlkSet& wset, MatA&& v, double dt, int nt = 0);

  template<class... Args>
  auto vHS(Args&&... args)
  {
    return nomsd_.vHS(std::forward<Args>(args)...);
  }

  /**
   * @brief Local energy of each outer walker against the stochastic trial.
   *
   * @details Inner ensemble vs outer walkers; leapfrog-reweighted under the conditioned sampler.
   * Complete on every rank (HamOps reduces internally; do not all_reduce).
   *
   * @param wset the outer walker set
   * @param E output energies, [nwalk, 3] as (one-body, exchange, Coulomb)
   * @param Ov output LOG overlap per walker, matching the OVLP property convention
   * @param nt time slice index, unused for a ground-state trial
   */
  void Energy(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> E,
              memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /**
   * @brief Measurement entry: Energy() averaged over inner_n_measure_samples_ pool advances at fixed
   *        walkers.
   *
   * @details Reduces to Energy() unless measure_advances_pool(). The pool advances before every
   * replica including the first, and the advance is kept (hafqmc convention).
   *
   * @param wset the outer walker set, non-const because advancing the pool writes its field block
   * @param E output energies, [nwalk, 3]; mean of per-replica energy ratios
   * @param Ov output LOG overlap; the walker's STORED overlap (not a replica's), so
   *        EnergyEstimator's exp(ovlp - OVLP) stays 1
   * @param nt time slice index, unused for a ground-state trial
   */
  void measure_energy(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> E,
                      memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /// @brief Energy of every walker, written back into the walker set's OVLP / E1_ / EXX_ / EJ_ properties.
  void Energy(WalkerSet<MEM>& wset);

  /// @brief Overload ignoring the time slice index, for interface parity with the finite-temperature trials.
  void Energy(WalkerSet<MEM>& wset, int nt)
  {
    (void)nt;
    Energy(wset);
  }

  /**
   * @brief Stochastic mixed density matrix for observables (estimator 3 of arXiv:2505.18519).
   *
   * @details Same inner-ensemble reduction as MixedDensityMatrix_for_vbias, in the observable layout
   * with NOMSD's LOG-overlap convention. Compact layout is static-ensemble only: off the anchor each
   * sample lives in a different orbital space. The 2-arg overload builds a scratch overlap and routes
   * to the 3-arg workhorse.
   *
   * @param wset the outer walker set
   * @param G output density matrix, [nwalk, dm_size]
   * @param compact request the half-rotated layout; rejected once inner_nsteps > 0
   */
  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true)
  {
    int nw = wset.size();
    memory::buffered_array<MEM, ComplexType, 1> Ov(nw, ComplexType(0.0));
    MixedDensityMatrix(wset, std::forward<MatG>(G), Ov, compact);
  }

  /**
   * @brief Stochastic mixed density matrix, also returning the per-walker LOG overlap.
   *
   * @param wset the outer walker set
   * @param G output density matrix, [nwalk, dm_size]
   * @param Ov output LOG overlap per walker
   * @param compact request the half-rotated layout; rejected once inner_nsteps > 0
   */
  void MixedDensityMatrix(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,2> G,
                          memory::array_view<MEM,ComplexType,1> Ov, bool compact = true);

  /**
   * @brief Density matrix against an EXTERNALLY supplied reference orbital set.
   *
   * @details The bra is Ref and the ket the walker, so this is pure orbital algebra independent of
   * both the trial wavefunction and the Hamiltonian -- the stochastic trial plays no role and
   * delegating to the outer NOMSD is exact.
   *
   * @param wset the outer walker set
   * @param Ref the reference orbital set forming the bra
   * @param G output density matrix
   * @param Ov output overlap per walker
   * @param compact request the half-rotated layout
   * @param herm treat Ref as already conjugate-transposed
   */
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

  /**
   * @brief Inner-ensemble-reduced mixed density matrix in the layout vbias consumes.
   *
   * @details Forces the full un-rotated layout once the ensemble has moved off the anchor, for the
   * same basis reason as MixedDensityMatrix.
   *
   * @param wset the outer walker set
   * @param G output density matrix, [nwalk, dm_size]
   */
  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G);

  /**
   * @brief Log overlap of each outer walker with the stochastic trial.
   *
   * @details Per-rank local and complete (like NOMSD): each rank scores its own walkers against the
   * full inner ensemble. Do not all_reduce per-walker overlaps.
   *
   * @param wset the outer walker set
   * @param Ov output LOG overlap per walker
   * @param nt time slice index, unused for a ground-state trial
   */
  void Log_Overlap(WalkerSet<MEM> const& wset, memory::array_view<MEM,ComplexType,1> Ov, int nt = 0);

  /// @brief Log overlap of every walker, written back into the walker set's OVLP property.
  void Log_Overlap(WalkerSet<MEM>& wset);

  /**
   * @brief Accumulate observables from the inner-ensemble-reduced Green's function (estimator 3).
   *
   * @details Same accumulation loop as NOMSD's single-determinant path. Time-evolved transforms are
   * linear in G, so averaging then transforming matches transforming per sample then averaging.
   *
   * @param iav averaging slot the observables accumulate into
   * @param wset the outer walker set
   * @param wgt per-walker weights
   * @param properties_1body one-body observables to accumulate
   * @param properties general observables to accumulate
   * @param X evolved creation-operator transform, or null when not time-evolved
   * @param Yc evolved annihilation-operator transform, or null when not time-evolved
   * @param M operator state added once, or null when not time-evolved
   * @param time_evolved whether X/Yc/M carry a back-propagated transform
   * @param importanceSampling must be true; the non-IS path is unfinished, as in NOMSD
   */
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
                             std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
                             nda::MemoryArrayOfRank<4> auto* X, nda::MemoryArrayOfRank<4> auto* Yc,
                             nda::MemoryArrayOfRank<4> auto* M, bool time_evolved,
                             bool importanceSampling = true);

  /// @brief Overload for observables with no back-propagated transform (null X/Yc/M), mirroring NOMSD.
  template<class WlkSet, class Observable>
  void accumulate_estimators(int iav, WlkSet& wset, nda::MemoryVector auto const& wgt,
                             std::vector<Observable>& properties_1body, std::vector<Observable>& properties,
                             bool importanceSampling = true)
  {
    memory::buffered_array<MEM, ComplexType, 4>* X = nullptr;
    accumulate_estimators(iav, wset, wgt, properties_1body, properties, X, X, X, false, importanceSampling);
  }

  /**
   * @brief Generalized Fock matrix of a SUPPLIED density matrix against the True Hamiltonian.
   *
   * @details A Hamiltonian-operator contraction on a caller-provided G with no trial reduction, so
   * delegating is exact. Reached only via the generalizedFockMatrix observable.
   *
   * @param args forwarded to NOMSD::generalizedFockMatrix
   */
  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  /**
   * @brief Selects which back-propagation reference set getReferences() exposes.
   *
   * @details False delegates to the outer NOMSD's anchor / CI expansion (static limit or P == 1);
   * true takes a dedicated walker-INDEPENDENT free-projection draw of the trial, so back-propagation
   * scores against <Psi_T| ~ (1/P) sum_p <psi_p| (Eq. 24). That draw is decoupled from the forward
   * ensemble on purpose: the trial is walker-independent, so its references must be too, and the
   * forward walk's conditioning is a forward-only importance-sampling device.
   */
  bool bp_uses_inner_ensemble() const
  {
    return inner_nsteps_ > 0 && inner_n_samples_ > 1
           && inner_ensemble_.initialized && inner_ensemble_.wset != nullptr;
  }

  /// @brief Number of back-propagation references: P for a dynamic trial, the anchor expansion otherwise.
  int total_number_of_references() const
  {
    return bp_uses_inner_ensemble() ? inner_n_samples_ : nomsd_.total_number_of_references();
  }

  int getNMO() const { return NMO; }

  bool isFiniteTemperature() const { return false; }

  /**
   * @brief Weight of back-propagation reference `i`: uniform 1/P for a dynamic trial.
   *
   * @param i reference index
   */
  ComplexType getReferenceWeight(int i) const
  {
    return bp_uses_inner_ensemble() ? ComplexType(1.0 / static_cast<double>(inner_n_samples_), 0.0)
                                    : nomsd_.getReferenceWeight(i);
  }

  /**
   * @brief Fill the reference Slater matrices BackPropagatedEstimator requests.
   *
   * @details For a dynamic trial this first performs a fresh free-projection draw, giving one
   * Monte-Carlo realization of the trial per back-propagation block; otherwise it delegates.
   *
   * @param Refs output references, [nref, npol*NMO, nel], as H-conjugated bras
   */
  template<class RefMat>
  void getReferences(RefMat&& Refs);

  /// @brief True iff this wavefunction's Hamiltonian operator can contract a FULL (un-rotated) mean-field G,
  /// which vMF requires once inner_n_samples > 1. See HamiltonianOperations::has_fullG_vbias.
  bool has_fullG_vbias() const { return nomsd_.has_fullG_vbias(); }

  /**
   * @name Pass-through interface
   *
   * @brief Members that carry no inner-ensemble reduction and forward unchanged to the outer NOMSD.
   *
   * @details These describe the True Hamiltonian, the walker layout, or the Hubbard-Stratonovich
   * machinery -- none of which the stochastic representation of the trial changes -- so delegating is
   * exact at every sampling target, not only at the delegate limit. See the corresponding NOMSD
   * members for their semantics.
   * @{
   */
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
  /// @}

private:
  struct StochasticInnerEnsemble
  {
    std::unique_ptr<WalkerSet<MEM>> wset;
    std::shared_ptr<utils::RandomGenerator_t<HOST_MEMORY>> rng;
    bool initialized{false};
  };

  /**
   * @brief True iff there is an actual beta block to operate on.
   *
   * @details COLLINEAR alone is NOT enough. A fully polarized system can be carried as COLLINEAR with
   * ndown == 0 (upstream reclassified the Li rohf_nomsd_polarized fixture that way), leaving a
   * zero-column beta block. Every beta operation must be gated on THIS, not on the walker type: an
   * empty block is a no-op mathematically (an empty determinant has det 1, hence log-overlap 0 and no
   * density-matrix columns) but is a hard error in nda/det_ops/cuTENSOR --
   * `Precondition !a.empty()` in get_block_layout, a zero-extent divide in the cross-space copy, or
   * CUTENSOR_STATUS_NOT_SUPPORTED. All three are invisible on a CPU build.
   */
  bool has_beta() const { return nomsd_.getWalkerType() == COLLINEAR and ndown > 0; }

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
  //
  // These initializers are NOT the input defaults and must not be read as documenting them -- the
  // constructor assigns every one of them from WavefunctionParameters unconditionally, so the live
  // defaults are the ones in parameters.hpp and nothing here has to track them.
  int inner_burn_in_{0};
  int inner_sample_update_steps_{0};
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
  WalkerSetParameters inner_walker_params_;
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
  void conditioned_resample(WalkerSet<MEM> const& wset);

  // Fill inner_cond_mag_ after a conditioned resample. The leapfrog overlap divides by it.
  void compute_inner_cond_mag(WalkerSet<MEM> const& wset);

  // Fill mag(q) = |<inner_q|phi_w>| for the slot-major inner ensemble against the outer walker set (same
  // det(A^dag B) / CLOSED-doubling / COLLINEAR-product convention as the Log_Overlap reduction). Shared
  // by compute_inner_cond_mag and the Metropolis accept/reject. Rank-local. `mag` must be sized nw*P and
  // `inner` must be in the slot-major nw*P conditioned form.
  void cross_overlap_magnitudes(WalkerSet<MEM> const& wset, WalkerSet<MEM>& inner,
                                nda::array<RealType, 1>& mag);

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
