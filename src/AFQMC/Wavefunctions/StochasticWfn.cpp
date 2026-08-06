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

#include <cstdlib>
#include <fstream>
#include <iomanip>

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
  if (inner_nsteps_ <= 0 || inner_n_samples_ <= 1)
    return;
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::draw_bp_reference_ensemble: inner propagator not built.");

  const int P           = inner_n_samples_;
  // Idempotency guard: at most ONE draw per BP window. If this window already drew (and the ensemble is
  // still at the P-sample form), reuse it -- a repeated getReferences in the same window must NOT silently
  // produce a different ensemble. The flag is reset by begin_inner_step() (forward walk advances => next
  // window). If the flag is set but the ensemble was somehow resized, fall through and redraw (safe).
  if (bp_refs_drawn_ && int(inner_ensemble_.wset->size()) == P)
    return;

  // Draw the walker-independent P-sample free-projection ensemble into inner_ensemble_.wset (used as
  // scratch; every forward resample resets it to the anchor, so transiently overwriting it is safe).
  draw_free_projection_samples(*inner_ensemble_.wset);
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
void StochasticWfn<MEM, devPsiT>::draw_free_projection_samples(WalkerSet<MEM>& target)
{
  // Reset `target` to the anchor |phi_T>, size it to P, and advance inner_nsteps_ BARE free-projection
  // steps: this materializes a walker-INDEPENDENT P-sample draw {psi_p = B_T(Y^[p])|phi_T>} of the trial
  // (Propagate_free forces bare field sampling regardless of the forward propagator's build mode, so the
  // draw is decoupled from any conditioning/leapfrog of the forward walk). The single home of the
  // free-projection draw, shared by draw_bp_reference_ensemble and mean_field_scratch_ensemble. Sets NO
  // forward-walk flags -- the caller owns its own state (BP window idempotency, mean-field scratch, ...).
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
  // Provide a walker-INDEPENDENT P-sample draw of the trial for vMF/G_MF WITHOUT perturbing the forward
  // inner ensemble. The trial mean field <Psi_T|.|Psi_T> is a walker-independent quantity, so it must be
  // reduced from a fresh free-projection draw -- but at the point vMF is consumed (generateP1, after
  // begin_inner_step) the forward ensemble is in the conditioned nw*P form and its determinants/flags are
  // live state of the running walk. Drawing into a DEDICATED scratch set (mf_scratch_wset_, lazily built
  // once via the same known-good WalkerSet constructor initialize_inner_walkers used) keeps
  // inner_ensemble_.wset, the persistent chain pool, the latch and the leapfrog magnitudes all untouched.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::mean_field_scratch_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::mean_field_scratch_ensemble: inner propagator not built.");
  if (mf_scratch_wset_ == nullptr)
  {
    auto wt = WalkerSet<MEM>::parse_walker_type(inner_walker_pt_); // identical to initialize_inner_walkers
    // Give the scratch set its OWN walker-set RNG so it never shares (and cannot perturb) the forward
    // inner ensemble's RNG stream -- the persistent field-space chains draw from inner_ensemble_.rng.
    // Construction is deterministic (populate_from_guess sets every walker to the anchor) and the
    // free-projection draw uses the inner PROPAGATOR's RNG, so the scratch's own generator is unused
    // beyond construction; a fresh one just makes the decoupling explicit.
    if (mf_scratch_rng_ == nullptr)
      mf_scratch_rng_ = std::make_shared<utils::RandomGenerator_t<HOST_MEMORY>>();
    mf_scratch_wset_ = std::make_unique<WalkerSet<MEM>>(mpi_, inner_walker_pt_, mf_scratch_rng_, wt,
                                                        inner_initial_guess_, inner_n_samples_);
  }
  draw_free_projection_samples(*mf_scratch_wset_);
  return *mf_scratch_wset_;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::dump_persample_row(long call, int ip, ComplexType lin_ov, ComplexType s,
                                                     ComplexType e0, ComplexType e1, ComplexType e2)
{
  // Debug-only: append one per-inner-sample row to $SAFIRE_DUMP_PERSAMPLE (first 32 Energy events only).
  // No-op unless the env var is set. Columns: call ip Re(ov) Im(ov) Re(s) Im(s) Re(e0) Im(e0) Re(e1) Im(e1)
  // Re(e2) Im(e2), where ov=<psi_ip|phi_0>, s is the weight actually used by the estimator, e0..e2 are the
  // per-sample energy components (e0 total). Offline: SAFIRE E = sum_p s*e0 / sum_p s; plain ratio-of-sums
  // reference = sum_p ov*e0 / sum_p ov.
  const char* path = std::getenv("SAFIRE_DUMP_PERSAMPLE");
  if (path == nullptr || call >= 32)
    return;
  std::ofstream f(path, std::ios::app);
  if (!f)
    return;
  f << call << ' ' << ip << ' ' << std::setprecision(14) << std::scientific << lin_ov.real() << ' '
    << lin_ov.imag() << ' ' << s.real() << ' ' << s.imag() << ' ' << e0.real() << ' ' << e0.imag() << ' '
    << e1.real() << ' ' << e1.imag() << ' ' << e2.real() << ' ' << e2.imag() << '\n';
}

template<MEMORY_SPACE MEM, class devPsiT>
bool StochasticWfn<MEM, devPsiT>::want_vbias_dump()
{
  return std::getenv("SAFIRE_DUMP_VBIAS") != nullptr;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::dump_vbias_row(long call, int rows, int naea, ComplexType const* phi,
                                                 int nCV, ComplexType const* vb)
{
  // APPEND, for call `call`: a "VB <call>" header, then rows*naea "re im" lines of walker-0's phi
  // (row-major), then nCV "re im" lines of its force bias. Header "rows naea nCV" written on call 0.
  const char* path = std::getenv("SAFIRE_DUMP_VBIAS");
  if (path == nullptr)
    return;
  std::ofstream f(path, call == 0 ? std::ios::trunc : std::ios::app);
  if (!f)
    return;
  f << std::setprecision(15) << std::scientific;
  if (call == 0)
    f << rows << ' ' << naea << ' ' << nCV << '\n';
  f << "VB " << call << '\n';
  const long mat = long(rows) * naea;
  for (long k = 0; k < mat; ++k)
    f << phi[k].real() << ' ' << phi[k].imag() << '\n';
  for (int m = 0; m < nCV; ++m)
    f << vb[m].real() << ' ' << vb[m].imag() << '\n';
}

template<MEMORY_SPACE MEM, class devPsiT>
bool StochasticWfn<MEM, devPsiT>::want_slater_dump()
{
  return std::getenv("SAFIRE_DUMP_SLATER") != nullptr;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::dump_slater_snapshot(int rows, int naea, int P, ComplexType const* phi,
                                                       ComplexType const* psis)
{
  // Write phi (the walker-0 ket) then the P inner determinants psi_ip as text; row-major [rows, naea].
  // Format: "rows naea P" ; then rows*naea "re im" lines for phi ; then per ip: "PSI ip" + rows*naea lines.
  const char* path = std::getenv("SAFIRE_DUMP_SLATER");
  if (path == nullptr)
    return;
  std::ofstream f(path);
  if (!f)
    return;
  f << std::setprecision(15) << std::scientific;
  f << rows << ' ' << naea << ' ' << P << '\n';
  const long mat = long(rows) * naea;
  for (long k = 0; k < mat; ++k)
    f << phi[k].real() << ' ' << phi[k].imag() << '\n';
  for (int ip = 0; ip < P; ++ip)
  {
    f << "PSI " << ip << '\n';
    ComplexType const* p = psis + long(ip) * mat;
    for (long k = 0; k < mat; ++k)
      f << p[k].real() << ' ' << p[k].imag() << '\n';
  }
}

template<MEMORY_SPACE MEM, class devPsiT>
bool StochasticWfn<MEM, devPsiT>::want_walker_dump()
{
  return std::getenv("SAFIRE_DUMP_WALKERS") != nullptr;
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::dump_walker_row(long call, int rows, int naea, ComplexType const* phi)
{
  // APPEND walker-0's ket phi for measurement event `call`. Header "rows naea" written on call 0; then a
  // "WALKER <call>" block + rows*naea "re im" lines (row-major). Reader loops WALKER blocks until EOF.
  const char* path = std::getenv("SAFIRE_DUMP_WALKERS");
  if (path == nullptr)
    return;
  std::ofstream f(path, call == 0 ? std::ios::trunc : std::ios::app);
  if (!f)
    return;
  f << std::setprecision(15) << std::scientific;
  if (call == 0)
    f << rows << ' ' << naea << '\n';
  f << "WALKER " << call << '\n';
  const long mat = long(rows) * naea;
  for (long k = 0; k < mat; ++k)
    f << phi[k].real() << ' ' << phi[k].imag() << '\n';
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::prime_chain_fields(WalkerSet<MEM>& wset)
{
  // Chain start: draw Y ~ p_T (i.i.d. standard normals, the same probit(uniform) convention as
  // construct_X's bare field assembly) into every chain's slot of the walker set's TrialFields block.
  // Uniforms come from the wavefunction's own rank-decorrelated inner RNG -- the chain machinery never
  // touches the propagator's RNG stream, whose draws are kept synchronized across ranks.
  // Device port (hybrid): draw the uniforms on the HOST inner RNG (same rank-decorrelated mt19937 stream
  // as a CPU build -> identical field values, so persistent parity tests stay bitwise-comparable), assemble
  // the fields on a host buffer, then copy into the (device) TrialFields view. The field block is small; a
  // device-cuRAND draw would be a localized perf follow-up but would break CPU/GPU reproducibility.
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
  // psi_q = B_T(Y_q)|phi_T> for every chain, from the fields stored in the outer walker buffer: reset
  // the slot-major nw*P pool to the anchor and apply the inner propagator with the stored fields one
  // inner step at a time. Deterministic (no RNG), so it reproduces the pool exactly wherever the fields
  // came from -- after population control moved chains between slots or ranks, or after the
  // back-propagation reference draw reused the pool storage. Rank-local; no communication.
  // Device-safe: deterministic rebuild from the stored fields. The field gather X(q,:) = Yw(w, slice) is a
  // device->device slice-copy (both are MEM), and Propagate_given_fields runs on device.
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
  // Device port (hybrid): the per-chain proposal and accept/reject are small scalar logic -- run them on
  // host over to_host copies using the HOST inner RNG (identical mt19937 stream to a CPU build, so the
  // persistent parity tests stay bitwise-comparable), and apply the big-array updates (proposal-field
  // determinant build, accepted-field write-back, rejected-determinant restore) with device copies.
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

    // 3. propose fields per chain (slot-major scratch; written back only on acceptance).
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

    // 6. accept/reject per chain.
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
  // Per-outer-step persistent chain update, called from begin_inner_step (the one seam with non-const
  // access to the outer walker set): size the TrialFields block on first use, start the chains (prime +
  // burn-in) or repair stale determinants, then run inner_sample_update_steps_ MH sweeps against the CURRENT
  // walkers and refresh the leapfrog conditioning magnitudes. Unconditional per step and rank-local, so
  // every rank performs the same sequence of propagator/reduction calls -- no rank-dependent control
  // flow, no communication.
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
    // Prime only. There is no separate burn-in count: the priming step falls through to the
    // inner_sample_update_steps_ sweeps below, and every subsequent outer step sweeps again, so the chains have
    // taken inner_sample_update_steps_ * (steps so far) sweeps by the time any measurement is kept -- the outer
    // equilibration window discards the early steps regardless.
    prime_chain_fields(wset);
    inner_chains_primed_ = true;
    rebuild_inner_dets_from_chain_fields(wset);
    inner_dets_stale_ = false;
    // One-time burn-in of the freshly primed chains, matching hafqmc's burn_in. The per-advance sweeps
    // below are paid every step thereafter; this is the only equilibration that happens before the walk
    // has moved a walker, so it is what stops the first steps sampling a pool still at its prior draw.
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

  if (inner_leapfrog())
    compute_inner_cond_mag(wset);

  if (++chain_updates_ % 200 == 0 && chain_proposed_ > 0)
    app_log(2, "StochasticWfn field-chain MCMC ({}, step {}): cumulative acceptance {:.3f} ({} / {})",
            inner_sampler_, inner_sampler_step_, inner_chain_acceptance(), chain_accepted_, chain_proposed_);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::advance_measure_pool(WalkerSet<MEM>& wset)
{
  // One measurement replica's pool motion: inner_sample_update_steps_ sweeps against the CURRENT walkers,
  // then refresh the leapfrog conditioning magnitudes so the next reduction's weights match the pool it
  // is about to measure with. This is update_persistent_chain_pool's tail with the priming, sizing and
  // staleness branches removed -- measure_advances_pool() already guarantees primed, non-stale chains,
  // and a measurement must never be the thing that first creates them.
  //
  // The walkers do not move across these sweeps, so unlike the propagation-side advance the chain is
  // relaxing toward a FIXED target: the lag decays geometrically with no floor.
  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: inner propagator not built.");
  if (not inner_chains_primed_)
    APP_ABORT("Error in StochasticWfn::advance_measure_pool: persistent chains not primed; a measurement "
              "replica may not be the first thing to start them.");

  for (int sweep = 0; sweep < inner_sample_update_steps_; ++sweep)
    chain_pool_sweep(wset);

  if (inner_leapfrog())
    compute_inner_cond_mag(wset);
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::snapshot_chain_fields(WalkerSet<MEM> const& wset,
                                                        nda::array<ComplexType, 2>& save) const
{
  // The chain state is the field configuration, so this is a complete snapshot. Host-side buffer: the
  // TrialFields view is strided over the walker buffer, and on device this is the same small host copy
  // prime_chain_fields already round-trips.
  utils::check(wset.has_trial_fields(), "snapshot_chain_fields: TrialFields block missing.");
  auto Yw = wset.TrialFields();
  save.resize(std::array<long, 2>{Yw.extent(0), Yw.extent(1)});
  save() = Yw();
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::restore_chain_fields(WalkerSet<MEM>& wset,
                                                       nda::array<ComplexType, 2> const& save)
{
  // Put the fields back and rebuild the determinants deterministically from them, which is exactly the
  // repair path population control already relies on -- so the pool returns to the state the measurement
  // found it in, bit for bit. Then refresh the conditioning magnitudes, which the replica sweeps moved.
  utils::check(wset.has_trial_fields(), "restore_chain_fields: TrialFields block missing.");
  auto Yw = wset.TrialFields();
  utils::check(save.extent(0) == Yw.extent(0) && save.extent(1) == Yw.extent(1),
               "restore_chain_fields: snapshot shape mismatch (population control moved between the "
               "snapshot and the restore?).");
  Yw() = save();
  rebuild_inner_dets_from_chain_fields(wset);
  inner_dets_stale_ = false;
  if (inner_leapfrog())
    compute_inner_cond_mag(wset);
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
