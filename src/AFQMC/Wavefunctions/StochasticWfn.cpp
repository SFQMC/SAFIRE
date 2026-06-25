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
  mpi_->comm.barrier();
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::draw_bp_reference_ensemble()
{
  // Phase 7 (Tier 6): a FRESH, walker-INDEPENDENT free-projection draw of P trial samples
  // {psi_p = B_T(Y^[p])|phi_T>} for back-propagation references (standard Motta-Zhang BP with the trial
  // represented stochastically). Decoupled from the forward inner ensemble: the trial |Psi_T> is
  // walker-independent, and the forward walk's 3c conditioning/leapfrog is only a forward-overlap
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
  // step so the reductions never see this transient P-sized draw.
  bp_refs_drawn_      = true;
  inner_step_pending_ = true;
  mpi_->comm.barrier();
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::advance_inner_ensemble_conditioned(
    memory::array<MEM, ComplexType, 2> const& X_bias, int nw)
{
  // Phase 3c-i: walker-conditioned resample. The inner ensemble is grown to nw*P walkers (slot-major
  // index q = ip*nw + w), every walker reset to the anchor |phi_T>, then advanced inner_nsteps_
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
  mpi_->comm.barrier();
}

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
