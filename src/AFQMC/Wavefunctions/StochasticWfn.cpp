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
  // Reset the first `count` inner walkers to the anchor |phi_T>. The single place that knows the
  // anchor/collinear layout -- shared by maybe_advance_inner_ensemble, advance_inner_ensemble_conditioned,
  // and draw_bp_reference_ensemble (so future spin/collinear fixes live in one spot).
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
  // A FRESH, walker-INDEPENDENT free-projection draw of P trial samples {psi_p = B_T(Y^[p])|phi_T>} for back-propagation references (standard Motta-Zhang BP with the trial
  // represented stochastically). Decoupled from the forward inner ensemble: the trial |Psi_T> is
  // walker-independent, and the forward walk's conditioning/leapfrog is only a forward-overlap
  // importance-sampling device -- irrelevant to the references. So we always draw BARE free-projection
  // samples here via Propagate_free (which forces bare field sampling regardless of the forward
  // propagator's build mode). inner_ensemble_.wset is reused as scratch and sized to P: every forward
  // resample resets it to the anchor (maybe_advance_inner_ensemble / advance_inner_ensemble_conditioned),
  // so transiently overwriting it (and resizing P <-> nwalk*P) is safe -- the next begin_inner_step
  // resamples it from scratch. getReferences calls this each BP block, giving a fresh MC draw per block.
  if (inner_nsteps_ <= 0 || inner_nwalkers_ <= 1)
    return;
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner propagator not built.");

  const int P           = inner_nwalkers_;
  // Idempotency guard: at most ONE draw per BP window. If this window already drew (and the ensemble is
  // still at the P-sample form), reuse it -- a repeated getReferences in the same window must NOT silently
  // produce a different ensemble. The flag is reset by begin_inner_step() (forward walk advances => next
  // window). If the flag is set but the ensemble was somehow resized, fall through and redraw (safe).
  if (bp_refs_drawn_ && int(inner_ensemble_.wset->size()) == P)
    return;

  WalkerSet<MEM>& inner = *inner_ensemble_.wset;
  if (int(inner.size()) != P)
  {
#if defined(ENABLE_DEVICE)
    if constexpr (MEM == DEVICE_MEMORY)
      inner.resize(P, nda::to_device(inner_anchor_));
    else
#endif
      inner.resize(P, inner_anchor_);
  }

  // reset every reference sample to the anchor |phi_T>, then advance bare free-projection steps
  reset_inner_to_anchor(inner, P);

  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
    inner_propagator().Propagate_free(inner, dt, 0);
  // Mark this window as drawn (idempotency guard) and force a fresh forward resample on the next outer
  // step so the reductions never see this transient P-sized draw. For persistent chains only the CACHED
  // DETERMINANTS were overwritten -- the chain fields live in the outer walker buffer and are untouched
  // -- so flag the determinants stale for a deterministic rebuild from the fields; the chains themselves
  // survive every back-propagation draw.
  bp_refs_drawn_      = true;
  inner_step_pending_ = true;
  inner_dets_stale_   = true;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::prime_chain_fields(WalkerSet<MEM>& wset)
{
  // Chain start: draw Y ~ p_T (i.i.d. standard normals, the same probit(uniform) convention as
  // construct_X's bare field assembly) into every chain's slot of the walker set's TrialFields block.
  // Uniforms come from the wavefunction's own rank-decorrelated inner RNG -- the chain machinery never
  // touches the propagator's RNG stream, whose draws are kept synchronized across ranks.
  if constexpr (MEM != HOST_MEMORY)
  {
    (void)wset;
    APP_ABORT("Error in StochasticWfn::prime_chain_fields: persistent field-space chains are CPU-only "
              "(the dynamic stochastic trial is gated to host builds).");
  }
  else
  {
    const int nw    = int(wset.size());
    const int block = wset.trial_fields_size();
    auto Yw         = wset.TrialFields();
    nda::array<double, 1> u(long(nw) * block);
    utils::sampleUniformFields(u, *inner_ensemble_.rng);
    long k = 0;
    for (int w = 0; w < nw; ++w)
      for (int j = 0; j < block; ++j)
        Yw(w, j) = ComplexType(probit(u(k++)), 0.0);
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::rebuild_inner_dets_from_chain_fields(WalkerSet<MEM> const& wset)
{
  // psi_q = B_T(Y_q)|phi_T> for every chain, from the fields stored in the outer walker buffer: reset
  // the slot-major nw*P pool to the anchor and apply the inner propagator with the stored fields one
  // inner step at a time. Deterministic (no RNG), so it reproduces the pool exactly wherever the fields
  // came from -- after population control moved chains between slots or ranks, or after the
  // back-propagation reference draw reused the pool storage. Rank-local; no communication.
  if constexpr (MEM != HOST_MEMORY)
  {
    (void)wset;
    APP_ABORT("Error in StochasticWfn::rebuild_inner_dets_from_chain_fields: persistent field-space "
              "chains are CPU-only (the dynamic stochastic trial is gated to host builds).");
  }
  else
  {
    if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
      APP_ABORT("Error in StochasticWfn::rebuild_inner_dets_from_chain_fields: inner walkers not initialized.");
    if (not inner_stack_->has_propagator())
      APP_ABORT("Error in StochasticWfn::rebuild_inner_dets_from_chain_fields: inner propagator not built.");
    const int nw      = int(wset.size());
    const int P       = inner_nwalkers_;
    const long ntot   = long(nw) * P;
    const int nCV     = inner_nomsd().number_of_cholesky_vectors();
    const int pathlen = inner_nsteps_ * nCV;
    utils::check(wset.has_trial_fields() && wset.trial_fields_size() == P * pathlen,
                 "rebuild_inner_dets_from_chain_fields: TrialFields block missing or mis-sized.");

    WalkerSet<MEM>& inner = *inner_ensemble_.wset;
    if (inner.size() != ntot)
      inner.resize(int(ntot), inner_anchor_);
    reset_inner_to_anchor(inner, int(ntot));

    auto Yw = wset.TrialFields(); // const host view [nw][P*pathlen]
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
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::chain_pool_sweep(WalkerSet<MEM>& wset)
{
  // One Metropolis-Hastings sweep of all nw*P field-space chains against their tethered walkers.
  // Chain q = ip*nw + w targets p_T(Y)*|<psi(Y)|phi_w>| -- the walker-conditioned field distribution
  // the estimator reductions assume (their normalizations cancel against it). The proposal moves the
  // FIELDS; determinants are derived: one batched B_T(Y*) build for the whole pool + one cross-overlap
  // pass per sweep. Kernels:
  //   pcn:      Y* = sqrt(1-s^2) Y + s xi  -- preserves the Gaussian prior exactly, so the acceptance
  //             is the bare overlap-magnitude ratio; s = 1 is an independence redraw.
  //   gaussian: Y* = Y + s xi              -- random walk; the prior ratio exp((|Y|^2-|Y*|^2)/2)
  //             multiplies the acceptance.
  // A zero current magnitude (walker orthogonal to the chain's sample) always accepts, moving off it.
  // Rejected chains keep their fields and restore their determinant row from the snapshot. Rank-local.
  if constexpr (MEM != HOST_MEMORY)
  {
    (void)wset;
    APP_ABORT("Error in StochasticWfn::chain_pool_sweep: persistent field-space chains are CPU-only "
              "(the dynamic stochastic trial is gated to host builds).");
  }
  else
  {
    WalkerSet<MEM>& inner = *inner_ensemble_.wset;
    const int nw          = int(wset.size());
    const int P           = inner_nwalkers_;
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

    // 3. propose fields per chain (slot-major scratch; written back only on acceptance).
    auto Yw        = wset.TrialFields();
    const bool pcn = (inner_mcmc_ == "pcn");
    const double s = inner_mcmc_step_;
    const double keep = pcn ? std::sqrt(std::max(0.0, 1.0 - s * s)) : 1.0;
    nda::array<ComplexType, 2> Ystar(ntot, pathlen);
    nda::array<double, 1> prior_lr(ntot);
    prior_lr() = 0.0;
    {
      nda::array<double, 1> u(ntot * long(pathlen));
      utils::sampleUniformFields(u, *inner_ensemble_.rng);
      long k = 0;
      for (long q = 0; q < ntot; ++q)
      {
        const int w  = int(q % nw);
        const int ip = int(q / nw);
        for (int j = 0; j < pathlen; ++j)
        {
          const double y  = real(Yw(w, long(ip) * pathlen + j));
          const double yp = keep * y + s * probit(u(k++));
          Ystar(q, j)     = ComplexType(yp, 0.0);
          if (not pcn)
            prior_lr(q) += 0.5 * (y * y - yp * yp);
        }
      }
    }

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

    // 6. accept/reject per chain.
    nda::array<double, 1> u_acc(ntot);
    utils::sampleUniformFields(u_acc, *inner_ensemble_.rng);
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
  // Per-outer-step persistent chain update, called from begin_inner_step (the one seam with non-const
  // access to the outer walker set): size the TrialFields block on first use, start the chains (prime +
  // burn-in) or repair stale determinants, then run inner_equil_steps_ MH sweeps against the CURRENT
  // walkers and refresh the leapfrog conditioning magnitudes. Unconditional per step and rank-local, so
  // every rank performs the same sequence of propagator/reduction calls -- no rank-dependent control
  // flow, no communication.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::update_persistent_chain_pool: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::update_persistent_chain_pool: inner propagator not built.");

  const int nw    = int(wset.size());
  const int P     = inner_nwalkers_;
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
    for (int sweep = 0; sweep < inner_pool_burn_in_; ++sweep)
      chain_pool_sweep(wset);
  }
  else if (inner_dets_stale_ || inner_ensemble_.wset->size() != long(nw) * P)
  {
    rebuild_inner_dets_from_chain_fields(wset);
    inner_dets_stale_ = false;
  }

  for (int sweep = 0; sweep < inner_equil_steps_; ++sweep)
    chain_pool_sweep(wset);

  if (inner_leapfrog_)
    compute_inner_cond_mag(wset);

  if (++chain_updates_ % 200 == 0 && chain_proposed_ > 0)
    app_log(2, "StochasticWfn field-chain MCMC ({}, step {}): cumulative acceptance {:.3f} ({} / {})",
            inner_mcmc_, inner_mcmc_step_, inner_chain_acceptance(), chain_accepted_, chain_proposed_);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::advance_inner_ensemble_conditioned(
    memory::array<MEM, ComplexType, 2> const& X_bias, int nw)
{
  // Walker-conditioned resample. The inner ensemble is grown to nw*P walkers (slot-major index q = ip*nw + w), every walker reset to the anchor |phi_T>, then advanced inner_nsteps_
  // conditioned field-sampling steps. Block w shares the conditioning bias x_bar(phi_w) = X_bias(w,:)
  // (computed by the caller from the anchor-to-outer-walker cross DM), so its P samples are
  // importance-sampled toward phi_w. Recovers the static anchor exactly at inner_nsteps_ == 0.
  inner_step_pending_ = false;

  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::advance_inner_ensemble_conditioned: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::advance_inner_ensemble_conditioned: inner propagator not built.");

  const int P     = inner_nwalkers_;
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

  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
    inner_propagator().Propagate_conditioned(inner, X_inner, dt, 0);
}

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
