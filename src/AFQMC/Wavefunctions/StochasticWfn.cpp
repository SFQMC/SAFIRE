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

#include "AFQMC/Wavefunctions/StochasticWfn.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Utilities/probit.h"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, class devPsiT>
Propagator<MEM>& StochasticWfn<MEM, devPsiT>::inner_propagator()
{
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::inner_propagator: inner propagator not built.");
  return inner_stack_->propagator();
}

template<MEMORY_SPACE MEM, class devPsiT>
Propagator<MEM> const& StochasticWfn<MEM, devPsiT>::inner_propagator() const
{
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::inner_propagator: inner propagator not built.");
  return inner_stack_->propagator();
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::reset_inner_to_anchor(WalkerSet<MEM>& inner, int count)
{
  // The single place that knows the anchor/collinear layout, so future spin/collinear fixes live in one
  // spot; shared by every routine that rebuilds the pool.
  const bool collinear = has_beta();
  auto all             = nda::range::all;
  // The anchor stays on the host: the per-walker assignments below cross into MEM on their own, the
  // same way populate_from_guess{,_ft} write a host guess into the walker buffer. (The old
  // ENABLE_DEVICE branch staged it through to_device only to copy it straight back into this host
  // array, which was a no-op round trip.)
  nda::array<ComplexType, 3> const& anchor_on_mem = inner_anchor_;
  for (int q = 0; q < count; ++q)
  {
    inner.SlaterMatrices(Alpha)(q, all, all) = anchor_on_mem(0, all, all);
    if (collinear)
    {
      int naeb = int(inner.SlaterMatrices(Beta).extent(2));
      // naeb == 0 is a REAL case, not a degenerate one: a fully polarized system carried as COLLINEAR
      // (e.g. the Li rohf_nomsd_polarized fixture, dims NAEA=3 NAEB=0). Copying zero columns is a
      // no-op, but nda's cross-address-space assign_from_ndarray divides by the extent to lay out the
      // host->device transfer and raises SIGFPE on a zero-extent view. Host->host survives it, so this
      // only ever shows up on a GPU build.
      if (naeb > 0)
        inner.SlaterMatrices(Beta)(q, all, all) = anchor_on_mem(1, all, nda::range(naeb));
    }
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::maybe_advance_inner_ensemble()
{
  if (inner_nsteps_ <= 0)
  {
    inner_step_pending_ = false;
    return;
  }
  if (not inner_step_pending_)
    return;
  inner_step_pending_ = false;

  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::maybe_advance_inner_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::maybe_advance_inner_ensemble: inner propagator not built.");

  WalkerSet<MEM>& inner = *inner_ensemble_.wset;
  reset_inner_to_anchor(inner, int(inner.size()));

  RealType eshift(0);
  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
    inner_propagator().Propagate(inner, eshift, dt, 0);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::draw_bp_reference_ensemble()
{
  // A fresh, walker-INDEPENDENT free-projection draw of the trial for the BP references, decoupled from
  // the forward inner ensemble. inner_ensemble_.wset is reused as scratch and resized to P: every forward
  // resample resets it to the anchor, so transiently overwriting it is safe.
  if (inner_nsteps_ <= 0 || inner_n_samples_ <= 1)
    return;
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner propagator not built.");

  const int P           = inner_n_samples_;
  // Idempotency guard: at most ONE draw per BP window, so a repeated getReferences cannot silently
  // produce a different ensemble. If the flag is set but the ensemble was resized, redraw (safe).
  if (bp_refs_drawn_ && int(inner_ensemble_.wset->size()) == P)
    return;

  draw_free_projection_samples(*inner_ensemble_.wset);
  // Force a fresh forward resample next step so the reductions never see this transient P-sized draw.
  // Only the CACHED DETERMINANTS were overwritten -- the chain fields are untouched in the outer walker
  // buffer -- so flag them for a deterministic rebuild; the chains survive every BP draw.
  bp_refs_drawn_      = true;
  inner_step_pending_ = true;
  inner_dets_stale_   = true;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::draw_free_projection_samples(WalkerSet<MEM>& target)
{
  // Propagate_free forces bare field sampling regardless of the forward propagator's build mode, which is
  // what decouples this draw from any conditioning/leapfrog of the forward walk. Sets NO forward-walk
  // flags -- the caller owns its own state.
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::draw_free_projection_samples: inner propagator not built.");
  const int P = inner_n_samples_;
  if (int(target.size()) != P)
  {
    target.resize(P, inner_anchor_);
  }
  reset_inner_to_anchor(target, P);
  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
    inner_propagator().Propagate_free(target, dt, 0);
}

template<MEMORY_SPACE MEM, class devPsiT>
WalkerSet<MEM>& StochasticWfn<MEM, devPsiT>::mean_field_scratch_ensemble()
{
  // A walker-INDEPENDENT P-sample draw of the trial for vMF/G_MF that leaves inner_ensemble_.wset, the
  // persistent chain pool, the latch and the leapfrog magnitudes untouched -- at the point vMF is
  // consumed those are live state of the running walk.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::mean_field_scratch_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::mean_field_scratch_ensemble: inner propagator not built.");
  if (mf_scratch_wset_ == nullptr)
  {
    auto wt = inner_walker_params_.walker_type; // identical to initialize_inner_walkers
    // Its OWN walker-set RNG, so it cannot perturb the stream the persistent chains draw from. Unused
    // beyond construction (that is deterministic, and the draw uses the propagator's RNG) -- a fresh one
    // just makes the decoupling explicit.
    if (mf_scratch_rng_ == nullptr)
      mf_scratch_rng_ = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
    mf_scratch_wset_ = std::make_unique<WalkerSet<MEM>>(mpi_, inner_walker_params_, mf_scratch_rng_, wt,
                                                        inner_initial_guess_, inner_n_samples_);
  }
  draw_free_projection_samples(*mf_scratch_wset_);
  return *mf_scratch_wset_;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::prime_chain_fields(WalkerSet<MEM>& wset)
{
  // Chain start: Y ~ p_T (i.i.d. standard normals, the same probit(uniform) convention as construct_X's
  // bare field assembly). Uniforms come from the wavefunction's own rank-decorrelated inner RNG -- the
  // chain machinery never touches the propagator's stream, whose draws stay synchronized across ranks.
  // The HOST draw + copy into the (device) TrialFields view is what keeps the field values, and so the
  // persistent parity tests, bitwise-comparable between CPU and GPU builds.
  const int nw    = int(wset.size());
  const int block = wset.trial_fields_size();
  auto Yw         = wset.TrialFields();
  nda::array<double, 1> u(long(nw) * block);
  inner_ensemble_.rng->sampleUniformFields(u);
  nda::array<ComplexType, 2> Yh(nw, block);
  long k = 0;
  for (int w = 0; w < nw; ++w)
    for (int j = 0; j < block; ++j)
      Yh(w, j) = ComplexType(probit(u(k++)), 0.0);
  Yw() = Yh(); // host -> device copy into the strided TrialFields view (no-op-ish on HOST_MEMORY)
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::rebuild_inner_dets_from_chain_fields(WalkerSet<MEM> const& wset)
{
  // psi_q = B_T(Y_q)|phi_T> for every chain, from the fields stored in the outer walker buffer.
  // Deterministic (no RNG), so it reproduces the pool exactly wherever the fields came from -- after
  // population control moved chains between slots or ranks, or after the BP draw reused the pool
  // storage. Rank-local; no communication.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::rebuild_inner_dets_from_chain_fields: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::rebuild_inner_dets_from_chain_fields: inner propagator not built.");
  const int nw      = int(wset.size());
  const int P       = inner_n_samples_;
  const long ntot   = long(nw) * P;
  const int nCV     = inner_nomsd().number_of_cholesky_vectors();
  const int pathlen = inner_nsteps_ * nCV;
  utils::check(wset.has_trial_fields() && wset.trial_fields_size() == P * pathlen,
               "rebuild_inner_dets_from_chain_fields: TrialFields block missing or mis-sized.");

  WalkerSet<MEM>& inner = *inner_ensemble_.wset;
  if (inner.size() != ntot)
  {
    inner.resize(int(ntot), inner_anchor_);
  }
  reset_inner_to_anchor(inner, int(ntot));

  auto Yw = wset.TrialFields(); // MEM view [nw][P*pathlen]
  memory::array<MEM, ComplexType, 2> X(ntot, nCV);
  RealType dt(inner_timestep_);
  for (int s = 0; s < inner_nsteps_; ++s)
  {
    for (long q = 0; q < ntot; ++q)
    {
      const int w  = int(q % nw);
      const int ip = int(q / nw);
      X(q, nda::range::all) = Yw(w, nda::range(long(ip) * pathlen + long(s) * nCV,
                                               long(ip) * pathlen + long(s + 1) * nCV));
    }
    inner_propagator().Propagate_given_fields(inner, X, dt);
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::chain_pool_sweep(WalkerSet<MEM>& wset)
{
  // One Metropolis-Hastings sweep of all nw*P field-space chains against their tethered walkers. Chain
  // q = ip*nw + w targets p_T(Y)*|<psi(Y)|phi_w>| -- the distribution the estimator reductions assume,
  // and against whose normalization theirs cancels. The proposal moves the FIELDS; determinants are
  // derived, one batched B_T(Y*) build + one cross-overlap pass per sweep. Rank-local. The per-chain
  // proposal and accept/reject run on host over to_host copies, using the HOST inner RNG so a GPU build
  // draws the identical mt19937 stream; only the big-array updates use device copies.
  {
    WalkerSet<MEM>& inner = *inner_ensemble_.wset;
    const int nw          = int(wset.size());
    const int P           = inner_n_samples_;
    const long ntot       = long(nw) * P;
    const int nCV         = inner_nomsd().number_of_cholesky_vectors();
    const int pathlen     = inner_nsteps_ * nCV;
    const WALKER_TYPES wt = nomsd_.getWalkerType();
    // has_beta(), not (wt == COLLINEAR): a zero-column beta block must be skipped entirely.
    const bool coll       = has_beta();
    auto all              = nda::range::all;

    // 1. current-chain target magnitudes |<psi_cur_q | phi_w>|.
    nda::array<RealType, 1> mag_cur(ntot);
    cross_overlap_magnitudes(wset, inner, mag_cur);

    // 2. snapshot the current determinants -- rejected proposals restore from here. Beta extents are
    //    read only when COLLINEAR (the ternary short-circuits for a CLOSED walker set).
    auto SMa = inner.SlaterMatrices(Alpha);
    memory::buffered_array<MEM, ComplexType, 3> old_a(SMa.extent(0), SMa.extent(1), SMa.extent(2));
    old_a() = SMa();
    const long b0 = coll ? inner.SlaterMatrices(Beta).extent(0) : 1;
    const long b1 = coll ? inner.SlaterMatrices(Beta).extent(1) : 1;
    const long b2 = coll ? inner.SlaterMatrices(Beta).extent(2) : 1;
    memory::buffered_array<MEM, ComplexType, 3> old_b(b0, b1, b2);
    if (coll)
      old_b() = inner.SlaterMatrices(Beta)();

    // 3. propose fields per chain -- pcn Y* = sqrt(1-s^2) Y + s xi (prior-preserving, so the acceptance
    //    is the bare magnitude ratio) or gaussian random walk Y* = Y + s xi (prior ratio in prior_lr).
    //    Slot-major scratch; written back only on acceptance.
    auto Yw        = wset.TrialFields();
    auto Yw_h      = nda::to_host(Yw); // host copy for the per-element proposal reads (no-op-ish on host)
    const bool pcn = (inner_sampler_ == "pcn");
    const double s = inner_sampler_step_;
    const double keep = pcn ? std::sqrt(std::max(0.0, 1.0 - s * s)) : 1.0;
    nda::array<ComplexType, 2> Ystar_h(ntot, pathlen);
    nda::array<double, 1> prior_lr(ntot);
    prior_lr() = 0.0;
    {
      nda::array<double, 1> u(ntot * long(pathlen));
      inner_ensemble_.rng->sampleUniformFields(u);
      long k = 0;
      for (long q = 0; q < ntot; ++q)
      {
        const int w  = int(q % nw);
        const int ip = int(q / nw);
        for (int j = 0; j < pathlen; ++j)
        {
          const double y  = real(Yw_h(w, long(ip) * pathlen + j));
          const double yp = keep * y + s * probit(u(k++));
          Ystar_h(q, j)   = ComplexType(yp, 0.0);
          if (not pcn)
            prior_lr(q) += 0.5 * (y * y - yp * yp);
        }
      }
    }
    // Proposal fields to device (used for the determinant build and the accepted-field write-back below).
    memory::array<MEM, ComplexType, 2> Ystar(ntot, pathlen);
    Ystar() = Ystar_h();

    // 4. build the proposal determinants psi(Y*) in place (snapshot holds the current ones).
    reset_inner_to_anchor(inner, int(ntot));
    {
      memory::array<MEM, ComplexType, 2> X(ntot, nCV);
      RealType dt(inner_timestep_);
      for (int st = 0; st < inner_nsteps_; ++st)
      {
        for (long q = 0; q < ntot; ++q)
          X(q, all) = Ystar(q, nda::range(long(st) * nCV, long(st + 1) * nCV));
        inner_propagator().Propagate_given_fields(inner, X, dt);
      }
    }

    // 5. proposal target magnitudes |<psi(Y*_q) | phi_w>|.
    nda::array<RealType, 1> mag_prop(ntot);
    cross_overlap_magnitudes(wset, inner, mag_prop);

    // 6. accept/reject per chain. A zero current magnitude (walker orthogonal to the chain's sample)
    //    always accepts, moving off it.
    nda::array<double, 1> u_acc(ntot);
    inner_ensemble_.rng->sampleUniformFields(u_acc);
    auto SMa_now = inner.SlaterMatrices(Alpha);
    auto SMb_now = coll ? inner.SlaterMatrices(Beta) : inner.SlaterMatrices(Alpha);
    for (long q = 0; q < ntot; ++q)
    {
      const double acc = (mag_cur(q) > RealType(0))
                             ? std::exp(prior_lr(q)) * double(mag_prop(q) / mag_cur(q))
                             : 1.0;
      ++chain_proposed_;
      if (u_acc(q) < acc)
      {
        ++chain_accepted_; // accept: keep psi(Y*) already in `inner`, store Y*
        const int w  = int(q % nw);
        const int ip = int(q / nw);
        Yw(w, nda::range(long(ip) * pathlen, long(ip + 1) * pathlen)) = Ystar(q, all);
      }
      else
      {
        SMa_now(q, all, all) = old_a(q, all, all); // reject: fields untouched, restore psi_cur
        if (coll)
          SMb_now(q, all, all) = old_b(q, all, all);
      }
    }
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::update_persistent_chain_pool(WalkerSet<MEM>& wset)
{
  // Per-outer-step persistent chain update. Unconditional per step and
  // rank-local, so every rank performs the same sequence of propagator/reduction calls -- no
  // rank-dependent control flow, no communication.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::update_persistent_chain_pool: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::update_persistent_chain_pool: inner propagator not built.");

  // Device-safe: delegates to the ported chain helpers (prime_chain_fields / rebuild_inner_dets_from_chain_fields
  // / chain_pool_sweep / compute_inner_cond_mag).
  const int nw    = int(wset.size());
  const int P     = inner_n_samples_;
  const int nCV   = inner_nomsd().number_of_cholesky_vectors();
  const int block = P * inner_nsteps_ * nCV;
  if (not wset.has_trial_fields())
    wset.resize_trial_fields(block);
  else
    utils::check(wset.trial_fields_size() == block,
                 "update_persistent_chain_pool: TrialFields block size mismatch.");

  if (not inner_chains_primed_)
  {
    prime_chain_fields(wset);
    inner_chains_primed_ = true;
    rebuild_inner_dets_from_chain_fields(wset);
    inner_dets_stale_ = false;
    // One-time burn-in of the freshly primed chains (hafqmc's burn_in). The only equilibration that
    // happens before the walk has moved a walker, so it is what stops the first steps sampling a pool
    // still at its prior draw; the per-advance sweeps below are paid every step thereafter.
    for (int sweep = 0; sweep < inner_burn_in_; ++sweep)
      chain_pool_sweep(wset);
  }
  else if (inner_dets_stale_ || inner_ensemble_.wset->size() != long(nw) * P)
  {
    rebuild_inner_dets_from_chain_fields(wset);
    inner_dets_stale_ = false;
  }

  for (int sweep = 0; sweep < inner_sample_update_steps_; ++sweep)
    chain_pool_sweep(wset);

  if (is_conditioned())
    compute_inner_cond_mag(wset);

  if (++chain_updates_ % 200 == 0 && chain_proposed_ > 0)
    app_log(2, "StochasticWfn field-chain MCMC ({}, step {}): cumulative acceptance {:.3f} ({} / {})",
            inner_sampler_, inner_sampler_step_, inner_chain_acceptance(), chain_accepted_, chain_proposed_);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::advance_measure_pool(WalkerSet<MEM>& wset)
{
  // One measurement replica's pool motion: sweeps against the CURRENT walkers, then a refresh of the
  // leapfrog magnitudes so the next reduction's weights match the pool it is about to measure with.
  // update_persistent_chain_pool's tail without the priming/sizing/staleness branches -- the caller
  // guarantees live chains, and a measurement must never be the thing that first creates them.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: inner propagator not built.");
  if (not inner_chains_primed_)
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: persistent chains not primed; a measurement "
              "replica may not be the first thing to start them.");

  for (int sweep = 0; sweep < inner_sample_update_steps_; ++sweep)
    chain_pool_sweep(wset);

  if (is_conditioned())
    compute_inner_cond_mag(wset);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::advance_inner_ensemble_conditioned(
    memory::array<MEM, ComplexType, 2> const& X_bias, int nw)
{
  // Walker-conditioned resample: grow the ensemble to nw*P (slot-major q = ip*nw + w), reset every walker
  // to the anchor, then advance inner_nsteps_ conditioned field-sampling steps. Block w shares the
  // conditioning bias X_bias(w,:), so its P samples are importance-sampled toward phi_w. Recovers the
  // static anchor exactly at inner_nsteps_ == 0.
  inner_step_pending_ = false;

  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::advance_inner_ensemble_conditioned: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::advance_inner_ensemble_conditioned: inner propagator not built.");

  const int P     = inner_n_samples_;
  const long ntot = long(nw) * P;
  auto all        = nda::range::all;

  WalkerSet<MEM>& inner = *inner_ensemble_.wset;
  if (inner.size() != ntot)
  {
    inner.resize(int(ntot), inner_anchor_);
  }

  // reset every inner walker to the anchor
  reset_inner_to_anchor(inner, int(ntot));

  // broadcast the per-outer-walker bias to its P inner samples (slot-major: q = ip*nw + w)
  const int nCV = int(X_bias.extent(1));
  utils::check(X_bias.extent(0) == nw, "advance_inner_ensemble_conditioned: X_bias row count mismatch.");
  memory::buffered_array<MEM, ComplexType, 2> X_inner(int(ntot), nCV);
  for (int ip = 0; ip < P; ++ip)
    X_inner(nda::range(long(ip) * nw, long(ip + 1) * nw), all) = X_bias(nda::range(0, nw), all);

  // Accumulate per-sample log importance weight logsw = sum_step HW over inner_nsteps_ conditioned steps.
  // Leapfrog reweighting uses exp(logsw) instead of dividing by |<psi|phi_cond>|.
  if (inner_logsw_.size() != ntot)
    inner_logsw_ = nda::array<ComplexType, 1>(int(ntot));
  inner_logsw_() = ComplexType(0.0);
  auto hw_step = nda::array<ComplexType, 1>(int(ntot));

  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
  {
    inner_propagator().Propagate_conditioned(inner, X_inner, dt, 0, &hw_step);
    inner_logsw_() += hw_step();
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::begin_inner_step(WalkerSet<MEM>& wset)
{
  inner_step_pending_ = true;
  // A forward step closes the current BP window: invalidate any cached BP reference draw so the next
  // back-propagation block draws a fresh free-projection ensemble (and so getReferences sees the forward
  // ensemble, not a stale draw, after a resample).
  bp_refs_drawn_ = false;
  // Persistent chains: this is the one seam with NON-const access to the outer walker set (whose buffer
  // holds the chain fields), so ALL chain mutation happens here. The updated pool then serves every
  // reduction of this step (latch cleared), so the step's overlap RATIO new/old shares one ensemble
  // (Eq. 25). Runs on every rank at every step -- uniform control flow, no communication.
  if (is_conditioned())
  {
    update_persistent_chain_pool(wset);
    inner_step_pending_ = false;
  }
  // Refresh the stored OVLP against the ensemble tied to the current (old) walker, so the
  // post-propagation Log_Overlap sees the SAME ensemble and N(phi) cancels in the ratio.
  //
  // 🔴 NO POOL-CHANGE HANDOFF. The pool advanced while the walker did not, so the stored OVLP jumps by
  // r = O_pool_new(phi)/O_pool_old(phi); we re-measure and do NOT charge WEIGHT for it. Deliberate, and
  // the only divergence from hafqmc in the sampler: the sole uncancelled piece of a pool change is a
  // ratio of conditioned-density normalizations, which is real and positive, so arg r is pure finite-P
  // noise and charging max(0, cos arg r) is a physics-free penalty. A paired N2 A/B measured -2.44 +/-
  // 1.67 mHa (1.46 sigma, not significant) with the predicted weight decay. Do not add a handoff back
  // without a per-system probe -- and then restore |r|, not the phase-only remnant.
  if (is_conditioned())
    Log_Overlap(wset);
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::store_inner_blocks_before_pop(WalkerSet<MEM>& wset)
{
  // Park inner_cond_mag_ on the walkers, so population control moves each magnitude with the walker
  // whose phi_cond it refers to. Slot-major (ip*nw + w) member -> walker-major (w, ip) block; the
  // transpose costs nw*P and runs once per pop event.
  if (not is_conditioned())
    return;
  if (not inner_chains_primed_)
    return;
  const int nw = int(wset.size());
  const int P  = inner_n_samples_;
  if (inner_cond_mag_.size() != long(nw) * P)
    return; // not yet sized (no conditioned resample has run); nothing to preserve
  if (not wset.has_trial_cond_mag())
    wset.resize_trial_cond_mag(P);
  else
    utils::check(wset.trial_cond_mag_size() == P,
                 "store_inner_blocks_before_pop: TrialCondMag block size mismatch.");
  nda::array<ComplexType, 2> Mh(nw, P);
  for (int w = 0; w < nw; ++w)
    for (int ip = 0; ip < P; ++ip)
      Mh(w, ip) = ComplexType(inner_cond_mag_(long(ip) * nw + w), 0.0);
  auto Mw = wset.TrialCondMag();
  Mw()    = Mh(); // host -> device copy into the strided view (no-op-ish on HOST_MEMORY)
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::permute_inner_blocks_after_pop(WalkerSet<MEM> const& wset)
{
  // Only WalkerOverlap trials carry a slot-conditioned inner ensemble; the other modes have nothing to
  // realign. THE CONTRACT: both halves of a walker's conditioned state ride inside walker_buffer -- the
  // chain FIELDS, from which the determinants are rebuilt, and the TRIAL_COND_MAG magnitudes, snapshot
  // by store_inner_blocks_before_pop. branch() clones both with the walker and load balancing ships
  // both, so every slot is recovered EXACTLY and no slot needs a recompute against the post-pop wset --
  // which is the walker after this step's propagation and pop control, not the phi_cond the value means.
  // That exactness is what keeps the leapfrog "old" overlap in sync.
  if (not is_conditioned())
    return;
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    return;

  const int nw = int(wset.size());
  const int P  = inner_n_samples_;

  if (not inner_chains_primed_)
    return; // chains not started yet (no pop event can precede the first propagation step)
  rebuild_inner_dets_from_chain_fields(wset);
  inner_dets_stale_ = false;
  if (inner_cond_mag_.size() == long(nw) * P)
  {
    // FAIL CLOSED ON AN UNPAIRED CALL. store_inner_blocks_before_pop and this function are a PAIR: the
    // store is the only thing that puts the magnitudes where branching can carry them, so reaching here
    // without it means they were left indexed by a slot layout that popControl has already invalidated.
    // Silently skipping the realignment would then divide the leapfrog overlap by another walker's
    // magnitude -- wrong physics, no error, and the caller cannot see it. Caught by
    // stochastic_persistent_permute_after_pop_control, whose clone case produced overlaps ~1e7 when this
    // was a silent no-op.
    utils::check(wset.has_trial_cond_mag(),
                 "permute_inner_blocks_after_pop: the TrialCondMag block is absent, so "
                 "store_inner_blocks_before_pop was not called before popControl. The two are a pair; "
                 "calling this one alone leaves the conditioned magnitudes misaligned with the walkers.");
    utils::check(wset.trial_cond_mag_size() == P,
                 "permute_inner_blocks_after_pop: TrialCondMag block size mismatch.");
    auto Mh = nda::to_host(wset.TrialCondMag());
    for (int w = 0; w < nw; ++w)
      for (int ip = 0; ip < P; ++ip)
        inner_cond_mag_(long(ip) * nw + w) = RealType(real(Mh(w, ip)));
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::Energy(WalkerSet<MEM>& wset)
{
  if (at_delegate_limit())
  {
    nomsd_.Energy(wset);
    return;
  }
  // Off-anchor: delegates to the device-safe Energy(wset, eloc, ovlp) reduction.
  int nw = wset.size();
  memory::buffered_array<MEM, ComplexType, 1> ovlp(nw, ComplexType(0.0));
  memory::buffered_array<MEM, ComplexType, 2> eloc(nw, 3);
  Energy(wset, eloc, ovlp);
  wset.setProperty(OVLP, ovlp);
  wset.setProperty(E1_, eloc(nda::range::all, 0));
  wset.setProperty(EXX_, eloc(nda::range::all, 1));
  wset.setProperty(EJ_, eloc(nda::range::all, 2));
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::Log_Overlap(WalkerSet<MEM>& wset)
{
  if (at_delegate_limit())
  {
    nomsd_.Log_Overlap(wset);
    return;
  }
  // Off-anchor: delegates to the device-safe Log_Overlap(wset, ovlp) (static-anchor ported).
  int nw = wset.size();
  memory::buffered_array<MEM, ComplexType, 1> ovlp(nw, ComplexType(0.0));
  Log_Overlap(wset, ovlp);
  wset.setProperty(OVLP, ovlp);
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::conditioned_resample(WalkerSet<MEM> const& wset)
{
  if (is_conditioned())
  {
    const int nw = int(wset.size());
    // Once the chains are live they are advanced ONLY in begin_inner_step; a reduction's job here is
    // REPAIR, not sampling -- rebuild the cached determinants when the BP draw reused the pool storage
    // or a pop event flagged them stale. Rank-local, read-only on the outer walker set.
    if (inner_chains_primed_)
    {
      if (inner_dets_stale_ || inner_ensemble_.wset->size() != long(nw) * inner_n_samples_)
      {
        rebuild_inner_dets_from_chain_fields(wset);
        inner_dets_stale_ = false;
        compute_inner_cond_mag(wset); // |<psi_q|phi_w^cond>| for the leapfrog reweighting (Eq. 25)
      }
      return;
    }
    // Resample when the per-step latch is armed (the first reduction of an outer step owns the resample;
    // later ones reuse the ensemble), or when the ensemble is not yet sized/conditioned for the current
    // outer-walker count. The (not inner_chains_primed_) term is load-bearing, not redundant: at nw == 1
    // the size check cannot tell a fresh ensemble (sized to P) from a conditioned one (sized nw*P), so a
    // driver's pre-loop initial-energy call would leave inner_cond_mag_ unsized and trip the leapfrog
    // check. That call instead takes the bootstrap resample below and is reweighted by exp(logsw).
    const bool need_resample =
        inner_step_pending_ || (not inner_chains_primed_) ||
        (inner_ensemble_.wset->size() != long(nw) * inner_n_samples_);
    if (not need_resample)
      return;
    // Conditioning force bias x_bar(phi_w) = sqrt(dt) * L^var . <phi_T|c+c|phi_w>/<phi_T|phi_w> (Eq. 23).
    // The inner trial IS the anchor, so the inner NOMSD's own mixed DM + vbias on the OUTER walker set
    // yields it directly -- no bespoke contraction, and the prefactor/normalization match the field-shift
    // convention assemble_X expects.
    const int nc  = inner_nomsd().dm_size(false); // compact cross-DM size (inner trial is single-det)
    const int nCV = inner_nomsd().number_of_cholesky_vectors();
    memory::buffered_array<MEM, ComplexType, 2> Gcross(nw, nc);
    memory::buffered_array<MEM, ComplexType, 1> ovcross(nw);
    Gcross()  = ComplexType(0.0);
    ovcross() = ComplexType(0.0);
    inner_nomsd().MixedDensityMatrix(wset, Gcross, ovcross, true);
    memory::array<MEM, ComplexType, 2> X_bias(nw, nCV);
    X_bias() = ComplexType(0.0);
    inner_nomsd().vbias_from_G(Gcross, X_bias, inner_timestep_);
    advance_inner_ensemble_conditioned(X_bias, nw);
    compute_inner_cond_mag(wset); // |<psi_q|phi_w^cond>| for the leapfrog overlap reweighting (Eq. 25)
  }
  else
  {
    // Walker-independent free-projection resample (honors the latch + static no-op).
    maybe_advance_inner_ensemble();
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::compute_inner_cond_mag(WalkerSet<MEM> const& wset)
{
  // |<psi_q|phi_w^cond>| for each inner walker q, against the outer walker its block was conditioned on
  // in the resample just completed. The leapfrog overlap divides each per-sample term by this, making the
  // step ratio Eq. 25. Sizes the member and delegates the magnitude loop, which is shared with the
  // Metropolis accept/reject.
  inner_cond_mag_ = nda::array<RealType, 1>(long(wset.size()) * inner_n_samples_);
  cross_overlap_magnitudes(wset, *inner_ensemble_.wset, inner_cond_mag_);
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::cross_overlap_magnitudes(WalkerSet<MEM> const& wset, WalkerSet<MEM>& inner,
                                                           nda::array<RealType, 1>& mag)
{
  const int nw          = int(wset.size());
  const int P           = inner_n_samples_;
  const WALKER_TYPES wt = nomsd_.getWalkerType();
  auto all              = nda::range::all;
  utils::check(mag.size() == long(nw) * P, "cross_overlap_magnitudes: mag must be sized nw*P.");
  utils::check(inner.size() == long(nw) * P, "cross_overlap_magnitudes: inner must be slot-major nw*P.");
  memory::buffered_array<MEM, ComplexType, 1> logov(nw);
  memory::buffered_array<MEM, ComplexType, 1> logov_b(nw);
  for (int ip = 0; ip < P; ++ip)
  {
    auto inner_a = inner.SlaterMatrices(Alpha)(nda::range(long(ip) * nw, long(ip + 1) * nw), all, all);
    logov() = ComplexType(0.0);
    det_ops::Log_Overlap(inner_a, wset.SlaterMatrices(Alpha), logov);
    if (wt == CLOSED)
      nda::tensor::scale(ComplexType(2.0), logov);
    else if (has_beta())
    {
      auto inner_b = inner.SlaterMatrices(Beta)(nda::range(long(ip) * nw, long(ip + 1) * nw), all, all);
      logov_b()    = ComplexType(0.0);
      det_ops::Log_Overlap(inner_b, wset.SlaterMatrices(Beta), logov_b);
    }
    // mag is a host array and the per-slot log-overlaps are small ([nw]), so finish |exp(.)| on host --
    // device-safe without a bespoke magnitude kernel.
    auto logov_h = nda::to_host(logov);
    nda::array<ComplexType, 1> logov_b_h;
    // has_beta(), not (wt == COLLINEAR): at ndown == 0 the beta leg above never ran, so logov_b holds
    // UNINITIALIZED data -- reading it would corrupt the magnitude, not merely crash.
    if (has_beta())
      logov_b_h = nda::to_host(logov_b);
    for (int w = 0; w < nw; ++w)
    {
      ComplexType lo = logov_h(w);
      if (has_beta())
        lo += logov_b_h(w);
      mag(long(ip) * nw + w) = std::abs(std::exp(lo));
    }
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::Energy(WalkerSet<MEM> const& wset,
                                         memory::array_view<MEM,ComplexType,2> E,
                                         memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  (void)nt;
  memory::check_memory_space<MEM>(E, Ov);
  utils::check(E.shape() == std::array<long, 2>{wset.size(), 3}, "Size mismatch");
  utils::check(Ov.size() == wset.size(), "Size mismatch");
  if (at_delegate_limit())
  {
    nomsd_.Energy(wset, E, Ov, nt);
    return;
  }
  const int nwalk         = wset.size();
  const ComplexType zero(0.0);
  const bool full         = (inner_nsteps_ > 0);
  const bool compact      = not full;
  const int Gsize         = dm_size(full);

  memory::buffered_array<MEM, ComplexType, 2> eloc2(nwalk, 3);
  memory::buffered_array<MEM, ComplexType, 1> D(nwalk, zero);
  E()  = zero;
  Ov() = zero;

  reduce_inner_cross_dm(
      wset, compact, Gsize, D, Ov,
      [&](auto& Gp, auto& /*ov_*/, auto& Sp, int /*ip*/, int /*r0*/, int /*rN*/) {
        eloc2() = zero;
        if (full)
          nomsd_.energy_from_fullG(eloc2, Gp);
        else
          nomsd_.energy_from_G(eloc2, Gp, 0);
        // E[w,:] += Sp[w] * eloc2[w,:]  (per-outer-walker scaled accumulate over the P inner samples).
        if constexpr (MEM == HOST_MEMORY)
          for (int w = 0; w < nwalk; ++w)
            E(w, nda::range::all) += Sp(w) * eloc2(w, nda::range::all);
        else
        {
#if defined(ENABLE_DEVICE)
          kernels::device::row_accumulate(Sp, eloc2, E);
#endif
        }
      });

  // NO all_reduce over E here. energy_from_G / energy_from_fullG returns the COMPLETE per-walker energy
  // on every rank (as NOMSD::Energy relies on) and reduce_inner_cross_dm replicates its loop on every
  // rank, so E and D are already complete; an all_reduce would double-count E by comm.size().
  if constexpr (MEM == HOST_MEMORY)
    for (int w = 0; w < nwalk; ++w)
      E(w, nda::range::all) /= D(w);
  else
  {
#if defined(ENABLE_DEVICE)
    kernels::device::row_divide(D, E);
#endif
  }

  // reduce_inner_cross_dm fills Ov with the LINEAR effective overlap; OVLP is the LOG overlap (7.3).
  if constexpr (MEM == HOST_MEMORY)
    for (int w = 0; w < nwalk; ++w)
      Ov(w) = std::log(Ov(w));
  else
  {
#if defined(ENABLE_DEVICE)
    kernels::device::elementwise_log(Ov);
#endif
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::measure_energy(WalkerSet<MEM>& wset,
                                                 memory::array_view<MEM,ComplexType,2> E,
                                                 memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  // E = (1/nm) sum_j Energy_j over nm pool advances at fixed walkers. The pool advances before EVERY
  // replica including the first, and the advance is KEPT (both match hafqmc). Falls through to plain
  // Energy only when there is no live chain to advance.
  if (not measure_advances_pool())
  {
    Energy(wset, E, Ov, nt);
    return;
  }

  const int nwalk = int(wset.size());
  const int nm    = inner_n_measure_samples_;

  memory::buffered_array<MEM, ComplexType, 2> E_j(nwalk, 3);
  memory::buffered_array<MEM, ComplexType, 1> Ov_j(nwalk);
  // Device path reuses the existing reduction kernels: row_accumulate(ones, ...) is the unweighted
  // E += E_j and row_divide(nm_vec, E) the 1/nm scale.
  memory::buffered_array<MEM, ComplexType, 1> ones(nwalk, ComplexType(1.0));
  memory::buffered_array<MEM, ComplexType, 1> nm_vec(nwalk, ComplexType(double(nm)));
  E()  = ComplexType(0.0);
  Ov() = ComplexType(0.0);

  for (int j = 0; j < nm; ++j)
  {
    // Advance the pool before every replica, including the first.
    advance_measure_pool(wset);
    E_j()  = ComplexType(0.0);
    Ov_j() = ComplexType(0.0);
    Energy(wset, E_j, Ov_j, nt);
    if constexpr (MEM == HOST_MEMORY)
    {
      for (int w = 0; w < nwalk; ++w)
        E(w, nda::range::all) += E_j(w, nda::range::all);
    }
    else
    {
#if defined(ENABLE_DEVICE)
      kernels::device::row_accumulate(ones, E_j, E);
#endif
    }
    // Ov_j is DISCARDED on purpose -- see the OVLP note after the loop.
  }

  if constexpr (MEM == HOST_MEMORY)
  {
    const ComplexType inv_nm(1.0 / double(nm));
    for (int w = 0; w < nwalk; ++w)
      E(w, nda::range::all) *= inv_nm;
  }
  else
  {
#if defined(ENABLE_DEVICE)
    kernels::device::row_divide(nm_vec, E);
#endif
  }

  // Return the walker's stored OVLP so EnergyEstimator's exp(ovlp - OVLP) factor stays 1.
  // Replica overlaps are intentionally discarded; see StochasticWfn.hpp.
  if constexpr (MEM == HOST_MEMORY)
    wset.getProperty(OVLP, Ov);
  else
  {
#if defined(ENABLE_DEVICE)
    wset.getProperty(OVLP, Ov);
#endif
  }
}
// Stochastic mixed density matrix for observable evaluation: the same inner-ensemble reduction as
// MixedDensityMatrix_for_vbias (estimator 3), in the caller's observable layout and with the LOG overlap
// NOMSD's observable DM reports.
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::MixedDensityMatrix(WalkerSet<MEM> const& wset,
                                                     memory::array_view<MEM,ComplexType,2> G,
                                                     memory::array_view<MEM,ComplexType,1> Ov,
                                                     bool compact)
{
  if (at_delegate_limit())
  {
    nomsd_.MixedDensityMatrix(wset, G, Ov, compact);
    return;
  }
  memory::check_memory_space<MEM>(G, Ov);
  const int nwalk = wset.size();
  const ComplexType zero(0.0);

  // Off the anchor the per-pair COMPACT DM lands in psi_p's OWN occupied basis, which cannot be averaged
  // across the ensemble; only the full layout is basis-independent there. Mirrors
  // MixedDensityMatrix_for_vbias forcing the full layout once inner_nsteps > 0.
  utils::check(not(compact and inner_nsteps_ > 0),
               "StochasticWfn::MixedDensityMatrix: a compact observable density matrix is only "
               "supported for the static inner ensemble (inner_nsteps == 0); request the full "
               "(compact == false) layout for a dynamic ensemble.");

  const int Gsize = dm_size(not compact);
  utils::check(G.shape() == std::array<long, 2>{nwalk, Gsize}, "Size mismatch");
  utils::check(Ov.size() == nwalk, "Size mismatch");

  memory::buffered_array<MEM, ComplexType, 2> Gnum(nwalk, Gsize);
  Gnum() = zero;
  memory::buffered_array<MEM, ComplexType, 1> D(nwalk, zero);
  Ov() = zero;

  reduce_inner_cross_dm(
      wset, compact, Gsize, D, Ov,
      [&](auto& Gp, auto& /*ov_*/, auto& Sp, int /*ip*/, int /*r0*/, int /*rN*/) {
        if constexpr (MEM == HOST_MEMORY)
          for (int w = 0; w < nwalk; ++w)
            Gnum(w, nda::range::all) += Sp(w) * Gp(w, nda::range::all);
        else
        {
#if defined(ENABLE_DEVICE)
          kernels::device::row_accumulate(Sp, Gp, Gnum);
#endif
        }
      });

  if constexpr (MEM == HOST_MEMORY)
    for (int w = 0; w < nwalk; ++w)
      Gnum(w, nda::range::all) /= D(w);
  else
  {
#if defined(ENABLE_DEVICE)
    kernels::device::row_divide(D, Gnum);
#endif
  }
  G() = Gnum();

  // Linear effective overlap -> LOG overlap, as the Energy / Log_Overlap overrides do (7.3).
  if constexpr (MEM == HOST_MEMORY)
    for (int w = 0; w < nwalk; ++w)
      Ov(w) = std::log(Ov(w));
  else
  {
#if defined(ENABLE_DEVICE)
    kernels::device::elementwise_log(Ov);
#endif
  }
}
template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::Log_Overlap(WalkerSet<MEM> const& wset,
                                              memory::array_view<MEM,ComplexType,1> Ov, int nt)
{
  (void)nt;
  memory::check_memory_space<MEM>(Ov);
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::Log_Overlap: inner walkers not initialized.");
  if (at_delegate_limit())
  {
    nomsd_.Log_Overlap(wset, Ov, nt);
    return;
  }
  if constexpr (MEM == HOST_MEMORY)
  {
  conditioned_resample(wset);

  WalkerSet<MEM>& inner  = *inner_ensemble_.wset;
  const int nw           = wset.size();
  // Conditioned: inner is slot-major nw*P, and pair tk maps to outer walker iw = tk % nw and inner walker
  // q = tk (its own conditioned sample). Unconditioned: P shared walkers paired against all outer
  // walkers. The leapfrog reweight is part of WalkerOverlap, not a separate mode.
  const bool conditioned = is_conditioned();
  const bool leapfrog    = conditioned;
  const int P            = (conditioned ? inner_n_samples_ : int(inner.size()));
  utils::check(Ov.size() >= nw, "Size mismatch");
  utils::check(P >= 1, "inner ensemble must be non-empty");
  if (conditioned)
    utils::check(inner.size() == long(nw) * P, "conditioned inner ensemble size mismatch (expect nw*P)");
  if (leapfrog)
    utils::check(inner_cond_mag_.size() == long(nw) * P,
                 "leapfrog inner_cond_mag_ not sized -- conditioned resample must run first");
  const ComplexType inv_P(1.0 / static_cast<double>(P), 0.0);
  const WALKER_TYPES wt = nomsd_.getWalkerType();
  auto all              = nda::range::all;

  Ov() = ComplexType(0.0);

  // Per-rank-local and per-rank-COMPLETE, exactly like NOMSD::Log_Overlap: every rank loops all nw*P
  // pairs of its OWN walkers, with no cross-rank distribution and NO all_reduce. Do not reintroduce
  // either: the outer walkers are DISTRIBUTED in the real driver, so all-reducing per-walker Ov(iw) sums
  // different physical walkers across ranks (correct only for the replicated wset of a unit test).
  const int tk0 = 0, tkN = nw * P;

  memory::buffered_array<MEM, ComplexType, 1> log_ov(1, ComplexType(0.0));
  memory::buffered_array<MEM, ComplexType, 3> inner_one(1, NMO, nup);
  memory::buffered_array<MEM, ComplexType, 3> outer_one(1, NMO, nup);

  for (int tk = tk0; tk < tkN; ++tk)
  {
    const int iw = (conditioned ? tk % nw : tk / P);
    const int ip = (conditioned ? tk / nw : tk % P);
    const int inner_idx = conditioned ? tk : ip;
    inner_one(0, all, all) = inner.SlaterMatrices(Alpha)(inner_idx, all, all);
    outer_one(0, all, all) = wset.SlaterMatrices(Alpha)(iw, all, all);
    log_ov(0)              = ComplexType(0.0);
    det_ops::Log_Overlap(inner_one, outer_one, log_ov);
    if (wt == CLOSED)
      nda::tensor::scale(ComplexType(2.0), log_ov);
    ComplexType ov = std::exp(log_ov(0));
    if (wt == COLLINEAR and ndown > 0)
    {
      // ndown == 0 (fully polarized carried as COLLINEAR, e.g. the Li rohf_nomsd_polarized fixture)
      // leaves an EMPTY beta block. It contributes nothing -- an empty determinant has det 1, so
      // log-overlap 0 -- but det_ops/nda trip `Precondition !a.empty()` on a zero-column operand.
      memory::buffered_array<MEM, ComplexType, 3> inner_beta(1, NMO, ndown);
      memory::buffered_array<MEM, ComplexType, 3> outer_beta(1, NMO, ndown);
      inner_beta(0, all, all) = inner.SlaterMatrices(Beta)(inner_idx, all, all);
      outer_beta(0, all, all) = wset.SlaterMatrices(Beta)(iw, all, all);
      log_ov(0)               = ComplexType(0.0);
      det_ops::Log_Overlap(inner_beta, outer_beta, log_ov);
      ov *= std::exp(log_ov(0));
    }
    if (leapfrog)
    {
      // Keyed on inner_chains_primed_ (a state fact), not on a mode flag: the reweight must match WHICH
      // SAMPLER produced the ensemble in hand. Bootstrap-resample samples carry exp(logsw) = p_T/q; chain
      // samples are already distributed as p_T(Y)|<psi|phi>| and divide by inner_cond_mag_.
      if ((not inner_chains_primed_) && inner_logsw_.size() == inner_cond_mag_.size())
      {
        Ov(iw) += std::exp(inner_logsw_(tk)) * ov;
      }
      else
      {
        double mag = double(inner_cond_mag_(tk));
        if (mag > 0.0)
        {
          Ov(iw) += ov / ComplexType(mag, 0.0);
        }
      }
    }
    else
    {
      Ov(iw) += inv_P * ov;
    }
  }

    // The pair loop accumulates the LINEAR effective overlap; OVLP stores the log overlap.
    for (int iw = 0; iw < nw; ++iw)
      Ov(iw) = std::log(Ov(iw));

  }
  else
  {
#if defined(ENABLE_DEVICE)
    // Device: reduce_inner_cross_dm accumulates exactly the linear effective overlap
    // Ov = (1/P) sum_p <psi_p|phi_w>; a no-op accumulate discards the mixed DM it also builds. The log
    // is the OVLP convention, as in the host pair loop above.
    memory::buffered_array<MEM, ComplexType, 1> Ddummy(wset.size(), ComplexType(0.0));
    reduce_inner_cross_dm(wset, /*compact=*/true, dm_size(false), Ddummy, Ov,
                          [](auto&, auto&, auto&, int, int, int) {});
    kernels::device::elementwise_log(Ov);
#endif
  }
}

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
