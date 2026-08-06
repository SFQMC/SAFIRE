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
  const bool collinear = (inner_stack_->nomsd().getWalkerType() == COLLINEAR);
  auto all             = nda::range::all;
  nda::array<ComplexType, 3> anchor_on_mem = inner_anchor_;
#if defined(ENABLE_DEVICE)
  if constexpr (MEM == DEVICE_MEMORY)
    anchor_on_mem = nda::to_device(inner_anchor_);
#endif
  for (int q = 0; q < count; ++q)
  {
    inner.SlaterMatrices(Alpha)(q, all, all) = anchor_on_mem(0, all, all);
    if (collinear)
    {
      int naeb = int(inner.SlaterMatrices(Beta).extent(2));
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
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY)
      target.resize(P, nda::to_device(inner_anchor_));
    else
#endif
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
    auto wt = WalkerSet<MEM>::parse_walker_type(inner_walker_pt_); // identical to initialize_inner_walkers
    // Its OWN walker-set RNG, so it cannot perturb the stream the persistent chains draw from. Unused
    // beyond construction (that is deterministic, and the draw uses the propagator's RNG) -- a fresh one
    // just makes the decoupling explicit.
    if (mf_scratch_rng_ == nullptr)
      mf_scratch_rng_ = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
    mf_scratch_wset_ = std::make_unique<WalkerSet<MEM>>(mpi_, inner_walker_pt_, mf_scratch_rng_, wt,
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
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY)
      inner.resize(int(ntot), nda::to_device(inner_anchor_));
    else
#endif
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
    const bool coll       = (wt == COLLINEAR);
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
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY)
      inner.resize(int(ntot), nda::to_device(inner_anchor_));
    else
#endif
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

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
