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

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "IO/AppAbort.hpp"
#include "AFQMC/parameters.hpp"
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "nda/h5.hpp"

#include <string>
#include <vector>
#include <complex>
#include <random>
#include <functional>
#include <optional>
#include <memory>

#include "test_common.hpp"
#include "test_stochastic_common.hpp"
#include "utilities/check.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/readWfn.h"

#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/Observables/full1rdm.hpp"

#include "numerics/sparse/sparse.hpp"

using std::complex;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

namespace {

struct StochasticWfnOptions
{
  int inner_n_samples     = 1;
  int inner_nsteps       = 0;
  // unset = let the caller's inner_nsteps decide (static when 0). Otherwise Gaussian or WalkerOverlap.
  std::optional<StochasticSamplingTarget> inner_sampling_target{};
  // Persistent field chains are not an option: they ARE the conditioned sampler, so a trial with
  // inner_sampling_target = conditioned is persistent by construction. inner_sample_update_steps is the only pool-mixing knob:
  // one sweep count for BOTH the propagation-side and measurement-side pool advances, and no separate
  // one-time burn-in count. 1, deliberately BELOW the production default of 32: these decks assert pool
  // MOTION and layout, which a short chain shows just as well and much faster. Always forwarded below,
  // so this stays 1 no matter what WavefunctionParameters defaults to.
  int inner_sample_update_steps         = 1;
  std::string inner_sampler   = ""; // empty = input default ("pcn")
  double inner_sampler_step   = 0.0; // <= 0 = kernel default
  double inner_prop_timestep = 0.01;
  int inner_n_measure_samples = 1;  // nm; 1 => measure_energy IS Energy, byte-for-byte
};

WavefunctionParameters make_stochastic_wfn_params(std::string const& name, std::string const& wfn_file,
                                                   StochasticWfnOptions const& opt = {})
{
  WavefunctionParameters pt{.name = name, .filename = wfn_file, .inner_n_samples = opt.inner_n_samples};
  utils::mark_stochastic_wfn_input(pt);
  if (opt.inner_nsteps > 0)
    pt.inner_nsteps = opt.inner_nsteps;
  if (opt.inner_sampling_target)
    pt.inner_sampling_target = opt.inner_sampling_target;
  else if (opt.inner_nsteps > 0)
    pt.inner_sampling_target = StochasticSamplingTarget::Gaussian; // dynamic trials must name a mode; these decks want the bare draw
  // Assigned unconditionally, NOT elided against a literal: this used to skip the assignment at 1
  // because 1 was also WavefunctionParameters' default, so when that default moved to 32 every deck
  // asking for 1 silently got 32 -- a behaviour change no assertion could see.
  pt.inner_sample_update_steps = opt.inner_sample_update_steps;
  if (not opt.inner_sampler.empty())
    pt.inner_sampler = opt.inner_sampler;
  if (opt.inner_sampler_step > 0.0)
    pt.inner_sampler_step = opt.inner_sampler_step;
  // Only emitted when non-default, so every pre-existing test's params match the old behaviour.
  if (opt.inner_n_measure_samples != 1)
    pt.inner_n_measure_samples = opt.inner_n_measure_samples;
  if (opt.inner_nsteps > 0)
    pt.inner_propagator = PropagatorParameters{.timestep = opt.inner_prop_timestep};
  return pt;
}

WavefunctionParameters make_nomsd_wfn_params(std::string const& name, std::string const& wfn_file)
{
  return WavefunctionParameters{.name = name, .filename = wfn_file};
}

template<MEMORY_SPACE MEM>
struct StochasticHamWfnEnv
{
  std::unique_ptr<HamiltonianFactory> ham_fac;
  Hamiltonian* ham = nullptr;
  WavefunctionFactory<MEM> wfn_fac;
  WalkerSetParameters wlk_pt{};
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  int NMO = 0;
  int nup = 0;
  int ndown = 0;
  WALKER_TYPES type = CLOSED;

  static std::optional<StochasticHamWfnEnv> build(
      std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string const& hamil_file,
      std::string const& wfn_file, std::function<bool(WALKER_TYPES)> accept_type = {})
  {
    if (getWavefunctionType(wfn_file) != NOMSD_WFN)
      return std::nullopt;

    StochasticHamWfnEnv env;
    const auto info = read_info_from_wfn(wfn_file, "any");
    env.NMO         = std::get<0>(info);
    env.nup         = std::get<1>(info);
    env.ndown       = std::get<2>(info);
    env.type        = afqmc::getWalkerType(wfn_file, "any");
    if (accept_type && not accept_type(env.type))
      return std::nullopt;

    env.ham_fac = std::make_unique<HamiltonianFactory>();
    env.ham_fac->push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    env.ham = &env.ham_fac->getHamiltonian(mpi, "ham0");

    env.wlk_pt = WalkerSetParameters{.name = "wset0", .walker_type = env.type};
    return env;
  }

  Wavefunction<MEM>& push_stochastic_wfn(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                         std::string const& name, std::string const& wfn_file,
                                         StochasticWfnOptions const& opt, int nwalk, bool init_inner_walkers = true)
  {
    WavefunctionParameters pt = make_stochastic_wfn_params(name, wfn_file, opt);
    utils::apply_wfn_defaults(pt, *ham);
    wfn_fac.push(name, pt);
    auto& wfn = wfn_fac.getWavefunction(mpi, name, type, false, ham, nwalk);
    if (init_inner_walkers)
      wfn_fac.maybe_initialize_stochastic_inner_walkers(wfn, name, type, wlk_pt);
    return wfn;
  }

  Wavefunction<MEM>& push_nomsd_wfn(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                    std::string const& name, std::string const& wfn_file, int nwalk)
  {
    auto nomsd_pt = make_nomsd_wfn_params(name, wfn_file);
    utils::apply_wfn_defaults(nomsd_pt, *ham);
    wfn_fac.push(name, nomsd_pt);
    return wfn_fac.getWavefunction(mpi, name, type, false, ham, nwalk);
  }

  WalkerSet<MEM> make_resized_walker_set(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, int nwalk,
                                         std::string const& wfn_id)
  {
    auto const& initial_guess = wfn_fac.getInitialGuess(wfn_id);
    return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
  }
};

} // namespace

// Build + initialize the inner ensemble only (no outer walkers / reductions). Widest fixture mask in
// the suite (incl. solids/lattice that dynamic cases exclude); also covers the old narrow
// `wfn_factory: stochasticwfn` smoke.
template<MEMORY_SPACE MEM>
void stochastic_build_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                            std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};
  WavefunctionParameters stoch_pt{.name = "wfn_stoch", .filename = wfn_file, .inner_n_samples = 1};
  utils::mark_stochastic_wfn_input(stoch_pt);
  utils::apply_wfn_defaults(stoch_pt, ham);
  WfnFac.push("wfn_stoch", stoch_pt);

  app_log(0, "[stochastic_build_smoke] building stochastic wavefunction");
  auto& wfn_stoch = WfnFac.getWavefunction(mpi, "wfn_stoch", type, false, &ham, 11);
  app_log(0, "[stochastic_build_smoke] built; initializing inner walkers");
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_stoch, "wfn_stoch", type, wlk_pt);
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());
  app_log(0, "[stochastic_build_smoke] inner walkers initialized OK");
}

TEST_CASE("stochastic_wfn: build and inner init", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  app_log(0,"StochasticWfn build + inner-init smoke test.");

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_build_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);

}

// ============================================================================
// Stochastic hot-path overrides: static delegate-limit parity (Log_Overlap, Energy, vbias) and the
// dynamic free-projection ensemble (inner_nsteps > 0), tag [stochastic_wfn].
//
// These exercise the StochasticWfn reductions through the public Wavefunction API only (the overhaul
// variant keeps its typed internals private): Log_Overlap, Energy, and vbias on the OUTER walkers, plus
// the dynamic free-projection drive. Two invariants are checked, mirroring the develop reference suite:
//   (1) inner_n_samples invariance -- a static replicated ensemble (inner_nsteps = 0) gives observables
//       independent of inner_n_samples (holds for ANY trial);
//   (2) delegate limit -- for a single-determinant trial the stochastic reduction equals the plain NOMSD
//       result. Multi-determinant trials diverge by design (a single-determinant inner ensemble cannot
//       reproduce a CI-weighted NOMSD), so (2) is gated on ndet == 1.
// ============================================================================

// One-line premise gate for the inner_n_samples-invariance comparisons. Returns true when the test should
// SKIP because the fixture's initial guess is not the trial reference.
//
// Why P matters at all, given both arms are stochastic and share the anchor: at inner_n_samples == 1 the
// stochastic path is at its DELEGATE LIMIT and hands the reduction to NOMSD, which scores against PsiT;
// at inner_n_samples > 1 it scores against the replicated anchor, which is Psi0. On the two BH UHF
// fixtures where Psi0 != PsiT those are different wavefunctions, and the "invariance" check fails by the
// fixture's ~1e-4 Psi0/PsiT subspace rotation -- MEASURED at ~1e-5 relative on the overlaps, far above
// any floating-point explanation and completely independent of the code under test.
//
// Defined after anchor_reference_mismatch (below), which does the measuring.
template<MEMORY_SPACE MEM>
bool anchor_is_not_reference(Wavefunction<MEM>& wfn_nomsd, WalkerSet<MEM>& wset_at_guess, WALKER_TYPES type,
                            int NMO, const char* tag);

// PREMISE CHECK for every test that compares a StochasticWfn path against NOMSD.
//
// The inner ensemble's anchor is the INITIAL GUESS (the wfn file's Psi0_alpha/Psi0_beta, via
// StochasticWfn::initialize_inner_walkers), while NOMSD scores against PsiT. Those are independent
// datasets, so such a comparison is only meaningful when they are the same determinant. Two fixtures
// deliberately break that -- BH afqmc_uhf_nomsd.h5 and afqmc_uhf_nomsd_init_rhf.h5 -- shipping a Psi0
// that differs from PsiT by a sign flip plus a ~1e-4 subspace rotation (max|Psi0-conj(PsiT)^T| = 2.0,
// |det(PsiT^H Psi0)| = 1-4.7e-9). Against those, a "stochastic != NOMSD" failure says nothing about the
// code under test: it is comparing two different wavefunctions. Every other NOMSD fixture in the suite
// has Psi0 == PsiT bit-for-bit, as does every production trial the noci_comp decks write.
//
// Measuring it without new plumbing: at the unperturbed initial guess phi == Psi0, so NOMSD returns
// G = phi inv(PsiT^H phi) PsiT^H, an oblique projector. It is idempotent EITHER WAY -- so idempotency
// cannot discriminate -- but Hermitian exactly when PsiT and Psi0 span the same subspace. (Hermiticity is
// preserved by the transposed storage convention.) Returns max|G - G^H| over NOMSD's own DM: ~1e-16 when
// the premise holds, ~1e-4 when it does not.
template<MEMORY_SPACE MEM>
double anchor_reference_mismatch(Wavefunction<MEM>& wfn_nomsd, WalkerSet<MEM>& wset_at_guess,
                                 WALKER_TYPES type, int NMO)
{
  const int nspin = (type == COLLINEAR ? 2 : 1);
  const int npol  = (type == NONCOLLINEAR ? 2 : 1);
  const int nwalk = int(wset_at_guess.size());
  const int rows = npol * NMO, cols = npol * NMO;
  memory::array<MEM, ComplexType, 2> G(nwalk, nspin * rows * cols);
  wfn_nomsd.MixedDensityMatrix(wset_at_guess, G, false);
  nda::array<ComplexType, 2> Gh(nda::to_host(G));
  double herm = 0.0;
  for (int w = 0; w < nwalk; ++w)
    for (int sp = 0; sp < nspin; ++sp)
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
        {
          const long a = long(sp * rows + r) * cols + c, b = long(sp * rows + c) * cols + r;
          herm         = std::max(herm, std::abs(Gh(w, a) - std::conj(Gh(w, b))));
        }
  return herm;
}

// Definition of the gate declared above: measure the premise, log it, and say whether to skip.
template<MEMORY_SPACE MEM>
bool anchor_is_not_reference(Wavefunction<MEM>& wfn_nomsd, WalkerSet<MEM>& wset_at_guess, WALKER_TYPES type,
                            int NMO, const char* tag)
{
  const double herm = anchor_reference_mismatch<MEM>(wfn_nomsd, wset_at_guess, type, NMO);
  app_log(0, "  {}: anchor/reference premise  max|G - G^H| = {:.6e}", tag, herm);
  if (herm <= 1e-10)
    return false;
  app_log(0, "  {}: SKIPPED -- initial guess (inner anchor) is not the trial reference.", tag);
  return true;
}

// The OVLP property stores a complex LOG overlap whose imaginary part (phase) is defined only mod
// 2*pi. NOMSD::Log_Overlap accumulates the unwrapped log-det phase, while the stochastic reduction
// sums in linear space and then takes the log, so it returns the principal branch -- the two can
// differ by an integer multiple of 2*pi*i while describing the SAME overlap. Compare the physical
// (linear) overlaps exp(log_ov), which are branch-independent (this also matches the develop
// reference suite, which compared linear overlaps directly).
inline nda::array<ComplexType, 1> linear_overlap(nda::array<ComplexType, 1> const& log_ov)
{
  nda::array<ComplexType, 1> lin(log_ov.size());
  for (long i = 0; i < log_ov.size(); ++i)
    lin(i) = std::exp(log_ov(i));
  return lin;
}

// Delegate-limit parity for every hot-path reduction, in ONE case.
//
// At the delegate limit (inner_n_samples = 1, inner_nsteps = 0) the inner ensemble collapses to the single
// trial-determinant anchor, so each stochastic override must reproduce plain NOMSD on the same outer
// walkers. At inner_nsteps = 0 with P > 1 the ensemble is P exact replicas of that anchor, so every
// observable must additionally be independent of inner_n_samples. This case checks both invariants for
// Log_Overlap, Energy (including the 3-arg propagator entry point), vbias, and the observable
// MixedDensityMatrix in both layouts -- plus the production call ORDER against a single walker set.
//
// One case rather than five per-quantity cases, following `wfn_factory: sdet`. Blocks are braced and
// INFO-tagged, so a failure still names the quantity.
//
// TWO GATES, and they are deliberately NOT the same gate:
//   - single_det (ndet == 1): required for ANY stochastic-vs-NOMSD comparison, because a CI-weighted NOMSD
//     cannot be reproduced by a single-determinant inner ensemble.
//   - premise (Psi0 == PsiT): required ONLY for the P-invariance comparisons. At P = 1 the stochastic path
//     is at its delegate limit and hands the reduction to NOMSD, scoring against PsiT; at P > 1 it scores
//     against the replicated anchor, which is Psi0. On the two BH UHF fixtures those are different
//     wavefunctions. s1-vs-NOMSD is unaffected (both score against PsiT), so it runs on every fixture --
//     which is what the old per-quantity cases got wrong by returning early on both.
template<MEMORY_SPACE MEM>
void stochastic_delegate_limit_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                             std::string hamil_file, std::string wfn_file)
{
  // PR-3: every off-anchor path below is device-ported (row_accumulate / row_divide /
  // inner_scalar_reduce / elementwise_log kernels), so this runs on DEVICE_MEMORY too.
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file);
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int    nwalk = 11; // prime: forces non-trivial splits in shared routines
  const double dt(0.01);
  const int    nspin = (env.type == COLLINEAR ? 2 : 1);
  const int    npol  = (env.type == NONCOLLINEAR ? 2 : 1);
  const int    nel   = (env.type == COLLINEAR ? env.nup + env.ndown : env.nup);

  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd", wfn_file, nwalk);

  // Static limit (inner_nsteps = 0) at P = 1 and P = 3.
  auto build_stoch = [&](const std::string& name, int inner_n_samples) -> Wavefunction<MEM>& {
    StochasticWfnOptions opt;
    opt.inner_n_samples = inner_n_samples;
    auto& w             = env.push_stochastic_wfn(mpi, name, wfn_file, opt, nwalk);
    REQUIRE(w.is_stochastic_wavefunction());
    REQUIRE(w.stochastic_inner_walkers_initialized());
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_p1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_p3", 3);
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() == wfn_s1.number_of_cholesky_vectors());
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() == wfn_s3.number_of_cholesky_vectors());

  const bool single_det = (wfn_nomsd.total_number_of_references() == 1);

  // Every arm reads a walker set built from the SAME initial guess and given the SAME deterministic
  // perturbation, so all three see bit-for-bit identical outer walkers going into the reductions.
  // Constructs the walker set in its own scope and passes it BY REFERENCE, so a WalkerSet is never
  // moved or copied (only make_resized_walker_set's elided return builds one).
  auto with_fresh_walkers = [&](auto&& body) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd");
    utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    return body(wset);
  };
  // Discrete (model) propagators must initialize potentials before any L.G contraction.
  auto prime_model_potentials = [&](Wavefunction<MEM>& wfn) {
    if (wfn.getHamType() == ModelHamiltonian)
    {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * env.NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
  };
  auto exp_of = [&](nda::array<ComplexType, 1> const& Ov) {
    nda::array<ComplexType, 1> e(Ov.size());
    for (int w = 0; w < int(Ov.size()); ++w)
      e(w) = std::exp(Ov(w));
    return e;
  };

  // Measured ONCE (the five old cases each measured it separately, on their own walker set). The walker
  // set here MUST stay unperturbed: the measurement relies on phi == Psi0 at the initial guess, which is
  // what makes NOMSD's DM an oblique projector whose hermiticity discriminates Psi0 == PsiT.
  const bool premise = [&] {
    auto premise_wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd"); // UNPERTURBED
    return not anchor_is_not_reference<MEM>(wfn_nomsd, premise_wset, env.type, env.NMO, "delegate limit");
  }();

  // ---- production call ORDER, against a single walker set -------------------------------------------
  // Log_Overlap -> runtime_optimization -> Energy -> vbias, all on ONE wset: the only block that can
  // catch cross-call coupling (Energy overwriting OVLP, a reduction leaving cached state behind), and the
  // suite's only call to StochasticWfn::runtime_optimization. No premise needed -- both arms score
  // against PsiT at P = 1.
  {
    INFO("delegate limit: production call order");
    auto harvest = [&](Wavefunction<MEM>& wfn) {
      return with_fresh_walkers([&](auto& wset) {
        wfn.Log_Overlap(wset);
        wfn.runtime_optimization(wset);
        wfn.Energy(wset);
        nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
        wset.getProperty(OVLP, ov);
        wset.getProperty(E1_, e1);
        wset.getProperty(EXX_, exx);
        wset.getProperty(EJ_, ej);
        prime_model_potentials(wfn);
        memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
        wfn.vbias(wset, X, dt);
        return std::make_tuple(std::move(ov), std::move(e1), std::move(exx), std::move(ej),
                               nda::array<ComplexType, 2>(nda::to_host(X)));
      });
    };
    auto [ov_n, e1_n, exx_n, ej_n, X_n] = harvest(wfn_nomsd);
    auto [ov_s, e1_s, exx_s, ej_s, X_s] = harvest(wfn_s1);
    if (single_det)
    {
      CHECK_THAT(ov_s, utils::Approx(ov_n));
      CHECK_THAT(e1_s, utils::Approx(e1_n));
      CHECK_THAT(exx_s, utils::Approx(exx_n));
      CHECK_THAT(ej_s, utils::Approx(ej_n));
      CHECK_THAT(X_s, utils::Approx(X_n));
    }
  }

  // ---- Log_Overlap ---------------------------------------------------------------------------------
  // Eq. 24 of arXiv:2505.18519 (static-ensemble limit). Read WITHOUT a following Energy call -- Energy
  // overwrites the OVLP property and would mask the override.
  {
    INFO("delegate limit: Log_Overlap");
    auto collect = [&](Wavefunction<MEM>& wfn) {
      return with_fresh_walkers([&](auto& wset) {
        wfn.Log_Overlap(wset);
        nda::array<ComplexType, 1> ov(nwalk);
        wset.getProperty(OVLP, ov);
        return ov;
      });
    };
    auto ov_ref = collect(wfn_nomsd);
    auto ov_s1  = collect(wfn_s1);
    auto ov_s3  = collect(wfn_s3);
    if (single_det)
      CHECK_THAT(linear_overlap(ov_s1), utils::Approx(linear_overlap(ov_ref)));
    if (premise)
      CHECK_THAT(linear_overlap(ov_s3), utils::Approx(linear_overlap(ov_s1)));
  }

  // ---- Energy -------------------------------------------------------------------------------------
  // Eq. 27 of arXiv:2505.18519: effective local energy (E1, EXX, EJ) plus overlap per outer walker.
  {
    INFO("delegate limit: Energy");
    struct WalkerEnergies
    {
      nda::array<ComplexType, 1> ov, e1, exx, ej;
    };
    auto collect = [&](Wavefunction<MEM>& wfn) {
      return with_fresh_walkers([&](auto& wset) {
        wfn.Energy(wset);
        WalkerEnergies out;
        out.ov.resize(nwalk);
        out.e1.resize(nwalk);
        out.exx.resize(nwalk);
        out.ej.resize(nwalk);
        wset.getProperty(OVLP, out.ov);
        wset.getProperty(E1_, out.e1);
        wset.getProperty(EXX_, out.exx);
        wset.getProperty(EJ_, out.ej);
        return out;
      });
    };
    WalkerEnergies ref = collect(wfn_nomsd);
    WalkerEnergies s1  = collect(wfn_s1);
    WalkerEnergies s3  = collect(wfn_s3);
    if (single_det)
    {
      CHECK_THAT(linear_overlap(s1.ov), utils::Approx(linear_overlap(ref.ov)));
      CHECK_THAT(s1.e1, utils::Approx(ref.e1));
      CHECK_THAT(s1.exx, utils::Approx(ref.exx));
      CHECK_THAT(s1.ej, utils::Approx(ref.ej));
    }
    if (premise)
    {
      CHECK_THAT(linear_overlap(s3.ov), utils::Approx(linear_overlap(s1.ov)));
      CHECK_THAT(s3.e1, utils::Approx(s1.e1));
      CHECK_THAT(s3.exx, utils::Approx(s1.exx));
      CHECK_THAT(s3.ej, utils::Approx(s1.ej));
    }

    // Energy's Ov is the same reduction as Log_Overlap through a different code path, so the two must
    // agree. Same arm, so neither gate applies.
    with_fresh_walkers([&](auto& wset) {
      wfn_s1.Log_Overlap(wset);
      nda::array<ComplexType, 1> ov_ovlp(nwalk);
      wset.getProperty(OVLP, ov_ovlp);
      CHECK_THAT(linear_overlap(ov_ovlp), utils::Approx(linear_overlap(s1.ov)));
      return 0;
    });

    // The 3-arg Energy(wset, E, Ov) -- the propagator's local-energy entry point, not the
    // property-setter form -- agrees with Energy(wset). Exercises caller-allocated buffers.
    with_fresh_walkers([&](auto& wset) {
      memory::array<MEM, ComplexType, 2> E_direct(nwalk, 3);
      memory::array<MEM, ComplexType, 1> Ov_direct(nwalk);
      wfn_s1.Energy(wset, E_direct, Ov_direct);
      auto E_h  = nda::to_host(E_direct);
      auto Ov_h = nda::to_host(Ov_direct);
      nda::array<ComplexType, 1> e1_col(nwalk), exx_col(nwalk), ej_col(nwalk);
      for (int n = 0; n < nwalk; ++n)
      {
        e1_col(n)  = E_h(n, 0);
        exx_col(n) = E_h(n, 1);
        ej_col(n)  = E_h(n, 2);
      }
      CHECK_THAT(linear_overlap(Ov_h), utils::Approx(linear_overlap(s1.ov)));
      CHECK_THAT(e1_col, utils::Approx(s1.e1));
      CHECK_THAT(exx_col, utils::Approx(s1.exx));
      CHECK_THAT(ej_col, utils::Approx(s1.ej));
      return 0;
    });
  }

  // ---- vbias --------------------------------------------------------------------------------------
  // MixedDensityMatrix_for_vbias (estimator 3) reduced, then contracted (estimator 4,
  // x_gamma[w] = L_gamma . G[w]) against the True-Ham Cholesky. vbias(wset, X, dt) drives the DM
  // internally, so compare the force bias X directly -- the intermediate G is not exposed on the variant.
  {
    INFO("delegate limit: vbias");
    auto collect = [&](Wavefunction<MEM>& wfn) {
      return with_fresh_walkers([&](auto& wset) {
        prime_model_potentials(wfn);
        memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
        wfn.vbias(wset, X, dt);
        return nda::array<ComplexType, 2>(nda::to_host(X));
      });
    };
    auto X_ref = collect(wfn_nomsd);
    auto X_s1  = collect(wfn_s1);
    auto X_s3  = collect(wfn_s3);
    if (single_det)
      CHECK_THAT(X_s1, utils::Approx(X_ref));
    if (premise)
      CHECK_THAT(X_s3, utils::Approx(X_s1));
  }

  // ---- observable MixedDensityMatrix, both layouts -------------------------------------------------
  // The observable analogue of MixedDensityMatrix_for_vbias, and unlike vbias it IS exposed on the
  // variant, so G is compared directly in the compact [nel*NMO] and full [NMO*NMO] layouts.
  {
    INFO("delegate limit: observable MixedDensityMatrix");
    auto collect = [&](Wavefunction<MEM>& wfn, bool compact) {
      return with_fresh_walkers([&](auto& wset) {
        const int Gsize = compact ? nel * npol * env.NMO : nspin * npol * env.NMO * npol * env.NMO;
        memory::array<MEM, ComplexType, 2> G(nwalk, Gsize);
        memory::array<MEM, ComplexType, 1> Ov(nwalk);
        // G and Ov come from two calls, not one fused call: the second reduction of an outer step
        // reuses the ensemble the first one resampled (the inner_step_pending_ latch in
        // StochasticWfn::conditioned_resample), so the pair is consistent. Measured bit-identical to
        // the fused form. Do NOT insert a begin_inner_step between these two lines.
        wfn.MixedDensityMatrix(wset, G, compact);
        wfn.Log_Overlap(wset, Ov);
        return std::make_pair(nda::array<ComplexType, 2>(nda::to_host(G)),
                              nda::array<ComplexType, 1>(nda::to_host(Ov)));
      });
    };
    for (bool compact : {true, false})
    {
      INFO((compact ? "compact layout" : "full layout")); // parens: INFO expands to `<< x`, which binds tighter than ?:
      auto [G_ref, Ov_ref] = collect(wfn_nomsd, compact);
      auto [G_s1, Ov_s1]   = collect(wfn_s1, compact);
      auto [G_s3, Ov_s3]   = collect(wfn_s3, compact);
      if (single_det)
      {
        CHECK_THAT(G_s1, utils::Approx(G_ref));
        CHECK_THAT(exp_of(Ov_s1), utils::Approx(exp_of(Ov_ref)));
      }
      if (premise)
      {
        CHECK_THAT(G_s3, utils::Approx(G_s1));
        CHECK_THAT(exp_of(Ov_s3), utils::Approx(exp_of(Ov_s1)));
      }
    }
  }
}

TEST_CASE("stochastic_wfn: delegate limit matches nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn delegate-limit parity: order, Log_Overlap, Energy, vbias, mixed DM.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_delegate_limit_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// StochasticWfn::vMF / G_MF are the trial's OWN mean-field quantities <Psi_T|.|Psi_T>/<Psi_T|Psi_T>,
// built by reducing the inner ensemble against ITSELF (a double sum over inner-walker pairs -- the
// inner-ensemble analogue of NOMSD's multi-determinant mean field). Unlike the propagator hot-path mixed
// estimators there is no outer walker. At the static replicated limit every inner walker
// == the anchor, so both collapse to the anchor mean field == plain NOMSD::vMF / G_MF. We compare the
// mean-field bias vMF (= L . G_MF, a [nCV] vector) and the mean-field DM G_MF directly: inner_n_samples
// invariance (static replicated 1 vs 3) and delegate-limit equality (ndet==1).
template<MEMORY_SPACE MEM>
void stochastic_mean_field_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: this off-anchor path is device-ported; runs on DEVICE_MEMORY too.

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  const double dt(0.01);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};

  auto nomsd_pt_1 = WavefunctionParameters{.name = "wfn_nomsd_mf", .filename = wfn_file};
  utils::apply_wfn_defaults(nomsd_pt_1, ham);
  WfnFac.push("wfn_nomsd_mf", nomsd_pt_1);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_mf", type, false, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_n_samples) -> Wavefunction<MEM>& {
    WavefunctionParameters pt{.name = name, .filename = wfn_file, .inner_n_samples = inner_n_samples};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_mf1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_mf3", 3);

  // Mean-field bias vMF = L . G_MF (a [nCV] vector).
  auto collect_vMF = [&](Wavefunction<MEM>& wfn) {
    // Discrete (model) propagators must initialize potentials before the L.G contraction.
    if (wfn.getHamType() == ModelHamiltonian)
    {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    memory::array<MEM, ComplexType, 1> v(wfn.number_of_cholesky_vectors(), ComplexType(0.0, 0.0));
    wfn.vMF(v, dt);
    return nda::to_host(v);
  };
  // At inner_n_samples > 1 the trial's own mean field is a reduction of the inner ensemble against
  // itself, so it exists only as a FULL un-rotated G. THCOps / KPTHCOps / KP3IndexFactorization
  // implement vbias for the half-rotated compact G only, and StochasticWfn::vMF refuses rather than
  // reinterpreting the buffer. Assert the refusal explicitly: a capability gap that is PINNED BY A TEST
  // cannot silently turn back into a wrong number, which is exactly what the lattice path used to do.
  // (wfn_s1 hits the P==1 delegate limit and never needs the full G, so it is unaffected.)
  const bool fullG_vbias = wfn_s3.has_fullG_vbias();
  app_log(0, "  vMF full-G vbias supported by this Hamiltonian operator: {}", fullG_vbias);

  auto v_ref = collect_vMF(wfn_nomsd);
  auto v_s1  = collect_vMF(wfn_s1);

  // Localize before asserting. inner_nsteps is unset here => 0 => the STATIC limit, where the inner
  // ensemble is P exact replicas of the anchor. So these are deterministic identities, not stochastic
  // estimates: any nonzero deviation is a real inner_n_samples dependence, not sampling noise.
  auto amax = [](auto const& a, auto const& b) {
    double d = 0.0, s = 0.0;
    for (long i = 0; i < a.size(); ++i)
    {
      d = std::max(d, std::abs(a(i) - b(i)));
      s = std::max(s, std::abs(b(i)));
    }
    return std::make_pair(d, s);
  };
  // PREMISE: skip when Psi0 != PsiT. At inner_n_samples == 1 the stochastic path hits the
  // delegate limit (-> NOMSD -> PsiT) while s3 scores against the anchor (-> Psi0), so on those
  // fixtures this compares two wavefunctions. See anchor_is_not_reference.
  auto premise_guess = WfnFac.getInitialGuess("wfn_nomsd_mf");
  auto premise_wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, premise_guess, nwalk);
  const bool premise_holds = not anchor_is_not_reference<MEM>(wfn_nomsd, premise_wset, type, NMO, "vMF");

  if (fullG_vbias)
  {
    auto v_s3 = collect_vMF(wfn_s3);
    auto [d31, s31] = amax(v_s3, v_s1);
    auto [d1r, s1r] = amax(v_s1, v_ref);
    app_log(0, "  vMF static-limit: max|v_s3-v_s1| = {:.6e} (rel {:.3e})   max|v_s1-v_ref| = {:.6e} (rel {:.3e})",
            d31, s31 > 0 ? d31 / s31 : 0.0, d1r, s1r > 0 ? d1r / s1r : 0.0);
    if (premise_holds)
    {
      // (1) inner_n_samples invariance of the mean-field bias.
      CHECK_THAT(v_s3, utils::Approx(v_s1));
      // (2) delegate limit: single-determinant trial => stochastic vMF == NOMSD.
      if (wfn_nomsd.total_number_of_references() == 1)
        CHECK_THAT(v_s1, utils::Approx(v_ref));
    }
  }
  else
  {
    // Capability gap, asserted rather than skipped. This is the ONLY assertion standing between an
    // unimplemented full-G contraction and a quietly wrong vMF: it fails the moment an operator starts
    // accepting the buffer without contracting it correctly. Unconditional -- it does not depend on the
    // Psi0 == PsiT premise, since nothing is being compared against the anchor.
    REQUIRE_THROWS_AS(collect_vMF(wfn_s3), AppAbortException);
  }
  if (not premise_holds)
    return;

  // Mean-field one-body Green's function G_MF ([nspin][npol*NMO][npol*NMO]).
  auto collect_GMF = [&](Wavefunction<MEM>& wfn) {
    auto Gshm = wfn.G_MF();
    return nda::to_host(Gshm());
  };
  auto G_ref = collect_GMF(wfn_nomsd);
  auto G_s1  = collect_GMF(wfn_s1);
  auto G_s3  = collect_GMF(wfn_s3);

  {
    auto fl = [](auto const& A) { return nda::reshape(A, std::array<long, 1>{A.size()}); };
    auto [d31, s31] = amax(fl(G_s3), fl(G_s1));
    auto [d1r, s1r] = amax(fl(G_s1), fl(G_ref));
    app_log(0, "  G_MF static-limit: max|G_s3-G_s1| = {:.6e} (rel {:.3e})   max|G_s1-G_ref| = {:.6e} (rel {:.3e})",
            d31, s31 > 0 ? d31 / s31 : 0.0, d1r, s1r > 0 ? d1r / s1r : 0.0);
  }
  CHECK_THAT(G_s3, utils::Approx(G_s1));
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(G_s1, utils::Approx(G_ref));
}

TEST_CASE("stochastic_wfn: vMF and G_MF match nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF mean-field delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
    // ALL_SYSTEMS. At inner_n_samples > 1 in a non-conditioned mode, vMF/G_MF reduce a walker-independent
    // P-sample ensemble (mean_field_uses_inner_ensemble) into a FULL un-rotated G, which vMF then hands
    // to HamOp::vbias. Not every factorization can contract that, and the two outcomes used to be:
    //   - solids  (THCOps, KP3IndexFactorization) -> APP_ABORT "vbias: Size mismatch"          [loud]
    //   - lattice (Discrete_GeneralUJ / Hubbard)  -> returned a P-DEPENDENT vMF                [SILENT]
    // The lattice case was the dangerous one: at the static limit the inner ensemble is P copies of the
    // anchor, so vMF is P-independent BY CONSTRUCTION, and molecules confirmed it (max|v_s3-v_s1| = 1e-16
    // to 1e-7); the Hubbard fixtures gave 1.8e-2 to 1.3e-1 with the Psi0 == PsiT premise HOLDING
    // (max|G-G^H| = 8.3e-17). Cause: ModelHamOps::vbias branched on the trial determinant count instead
    // of the G layout, so a full G reached a compact array_view over the same buffer -- in bounds,
    // because a compact G is smaller, hence quiet. FIXED: it now dispatches on the layout.
    // The solids gap is real and remains; the test now ASSERTS the refusal (has_fullG_vbias() == false
    // => vMF throws) instead of excluding the fixtures, so it cannot regress into a silent wrong number.
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

namespace
{
// Shared helpers for the mean-field + back-propagation reference tests: max |A-B| over two views (1D/2D),
// finiteness checks (1D/3D), and tr(G) (electron count -- a physical invariant, exact and noise-free
// regardless of the stochastic sampling). Defined before their first use (`stochastic_wfn: mean field production order`).
template<class A, class B>
double max_abs_diff2d(A const& X, B const& Y)
{
  double m = 0.0;
  for (long i = 0; i < X.extent(0); ++i)
    for (long j = 0; j < X.extent(1); ++j)
      m = std::max(m, std::abs(X(i, j) - Y(i, j)));
  return m;
}
template<class A>
bool all_finite3d(A const& X)
{
  for (long p = 0; p < X.extent(0); ++p)
    for (long i = 0; i < X.extent(1); ++i)
      for (long j = 0; j < X.extent(2); ++j)
        if (not std::isfinite(X(p, i, j).real()) or not std::isfinite(X(p, i, j).imag()))
          return false;
  return true;
}
template<class A, class B>
double max_abs_diff1d(A const& X, B const& Y)
{
  double m = 0.0;
  for (long i = 0; i < X.extent(0); ++i)
    m = std::max(m, std::abs(X(i) - Y(i)));
  return m;
}
template<class A>
bool all_finite1d(A const& X)
{
  for (long i = 0; i < X.extent(0); ++i)
    if (not std::isfinite(X(i).real()) or not std::isfinite(X(i).imag()))
      return false;
  return true;
}
template<class A>
bool all_finite2d(A const& X)
{
  for (long i = 0; i < X.extent(0); ++i)
    for (long j = 0; j < X.extent(1); ++j)
      if (not std::isfinite(X(i, j).real()) or not std::isfinite(X(i, j).imag()))
        return false;
  return true;
}
// Electron count = tr(G) summed over spin blocks (a physical invariant of any N-electron trial, exact and
// noise-free regardless of the stochastic sampling).
template<class A>
ComplexType g_trace3d(A const& X)
{
  ComplexType t(0.0, 0.0);
  for (long s = 0; s < X.extent(0); ++s)
    for (long i = 0; i < std::min(X.extent(1), X.extent(2)); ++i)
      t += X(s, i, i);
  return t;
}
} // namespace

// Production-order regression: vMF / G_MF are trial-only quantities (<Psi_T|.|Psi_T>/<Psi_T|Psi_T>), but
// in the real Propagate loop begin_inner_step(wset) runs BEFORE generateP1 calls vMF, and in leapfrog mode
// begin_inner_step eagerly resamples -- expanding the inner ensemble to nwalk*P walker-CONDITIONED samples.
// The mean field must NOT be a double sum over that conditioned ensemble (wrong normalization + a spurious
// outer-walker dependence), NOR the ANCHOR single-determinant mean field (the historical fallback -- it is
// only correct for a static trial; for a dynamic trial it makes the HS-contour shift inconsistent with the
// full-trial force bias/energy and biases the phaseless constraint toward overbinding). It must be the FULL
// trial's mean field, reduced from a dedicated walker-INDEPENDENT free-projection draw
// (mean_field_scratch_ensemble). This test reproduces Propagate ordering (begin_inner_step before vMF/G_MF)
// and asserts: (1) finite result with preserved electron count tr(G); (2) result differs from anchor/NOMSD;
// (3) the scratch draw leaves the forward conditioned ensemble untouched.
template<MEMORY_SPACE MEM>
void stochastic_mean_field_production_order(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: conditioned + leapfrog inner sampling is device-ported; runs on DEVICE_MEMORY too.
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;
    const double dt(0.01);
    auto all = nda::range::all;

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    const int inner_n_samples = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};

    auto nomsd_pt_2 = WavefunctionParameters{.name = "wfn_nomsd_mfp", .filename = wfn_file};
    utils::apply_wfn_defaults(nomsd_pt_2, ham);
    WfnFac.push("wfn_nomsd_mfp", nomsd_pt_2);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_mfp", type, false, &ham, nwalk);

    // Leapfrog stochastic trial: begin_inner_step will conditioned-resample (expand to nwalk*P).
    WavefunctionParameters pt{.name             = "wfn_stoch_mfp",
                              .filename         = wfn_file,
                              .inner_n_samples  = inner_n_samples,
                              .inner_nsteps     = 1,
                              .inner_sampling_target = StochasticSamplingTarget::WalkerOverlap,
                              .inner_propagator = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_mfp", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_mfp", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_mfp", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_mfp");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    // Reproduce the Propagate ordering: begin_inner_step (leapfrog => conditioned resample to nwalk*P)
    // BEFORE the mean-field calls. With the fix, vMF/G_MF see the non-P-sample (conditioned) ensemble and
    // draw a dedicated walker-independent free-projection ensemble to reduce the FULL trial's mean field.
    wfn_s.begin_inner_step(wset);

    // The leapfrog begin_inner_step must actually have expanded the ensemble to nwalk*P (otherwise the
    // scenario the fix handles is not exercised). This pins that the mean-field call happens while the
    // forward ensemble is in the conditioned form.
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

    // Anchor (NOMSD) reference mean field -- what the buggy code delegated to.
    memory::array<MEM, ComplexType, 1> v_ref(wfn_nomsd.number_of_cholesky_vectors(), ComplexType(0.0, 0.0));
    memory::array<MEM, ComplexType, 1> v_s(wfn_s.number_of_cholesky_vectors(), ComplexType(0.0, 0.0));
    wfn_nomsd.vMF(v_ref, dt);
    wfn_s.vMF(v_s, dt);
    auto v_ref_h = nda::to_host(v_ref);
    auto v_s_h   = nda::to_host(v_s);

    auto Gmf_ref = wfn_nomsd.G_MF();
    auto Gmf_s   = wfn_s.G_MF();
    auto G_ref_h = nda::to_host(Gmf_ref());
    auto G_s_h   = nda::to_host(Gmf_s());

    // (3) The dedicated scratch draw must leave the forward conditioned ensemble (nwalk*P) intact -- the
    // mean-field call is trial-only and must not perturb the running conditioned/leapfrog walk. (Unlike the
    // back-propagation reference draw, which reuses inner_ensemble_.wset as scratch and relies on the next
    // begin_inner_step to re-expand it, the mean-field draw uses a SEPARATE scratch and never touches it.)
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

    // (1) finite + electron-count (trace) preserved.
    CHECK(all_finite1d(v_s_h));
    CHECK(all_finite3d(G_s_h));
    CHECK_THAT(g_trace3d(G_s_h), utils::Approx(g_trace3d(G_ref_h)));

    // (2) regression guard: the DYNAMIC trial's mean field is genuinely OFF the anchor -- the buggy code
    // returned the anchor (bit-for-bit == NOMSD via nomsd_.vMF), so a nonzero difference proves vMF/G_MF now
    // reduce the full stochastic trial. (Deterministic: the inner free-projection draw is seeded.)
    double dv = max_abs_diff1d(v_s_h, v_ref_h);
    double dG = max_abs_diff2d(G_s_h(0, all, all), G_ref_h(0, all, all));
    app_log(1, "  stochastic_mean_field_production_order: |vMF_stoch - vMF_anchor|_max = {:.3e}, "
               "|G_MF_stoch - G_MF_anchor|_max = {:.3e}", dv, dG);
    CHECK(dv > 1e-6);
    CHECK(dG > 1e-6);
  }
}

TEST_CASE("stochastic_wfn: mean field production order", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF survive the begin_inner_step-before-generateP1 order.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_production_order<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Back-propagation reference set and outer-facing layout queries AT THE STATIC LIMIT (inner_nsteps = 0 -- the default in this test). There bp_uses_inner_ensemble() is false, so
// the stochastic trial DELEGATES its reference set to the outer nomsd_ (= {phi_T}, weight 1, for the
// single-determinant anchor; the full CI expansion for a multi-det outer trial), ignoring the inner
// ensemble. So total_number_of_references / getReferenceWeight / getReferences equal plain NOMSD's
// UNCONDITIONALLY (not just at ndet==1) and INDEPENDENT of inner_n_samples (1 and 3 both delegate). (For a
// DYNAMIC trial, inner_nsteps > 0 with P > 1, getReferences instead performs a dedicated free-projection
// draw -- exercised by `stochastic_wfn: bp free projection ensemble` and the production-order cases.) This test verifies
// that static-limit reference parity (COUNT, per-reference WEIGHT, reference Slater matrices) plus
// layout/metadata parity (Cholesky count, Ham type, walker type). NOTE: reference-API + layout only;
// the integration smokes below exercise the full estimator/driver BP path (static at inner_nsteps = 0;
// dynamic, with the conditioned+leapfrog forward walk + dedicated draw, in _dynamic_smoke).
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: free-projection BP / accumulate is device-ported (getReferences + accumulate_estimators); runs on DEVICE_MEMORY.

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};

  auto nomsd_pt_3 = WavefunctionParameters{.name = "wfn_nomsd_bp", .filename = wfn_file};
  utils::apply_wfn_defaults(nomsd_pt_3, ham);
  WfnFac.push("wfn_nomsd_bp", nomsd_pt_3);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bp", type, false, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_n_samples) -> Wavefunction<MEM>& {
    WavefunctionParameters pt{.name = name, .filename = wfn_file, .inner_n_samples = inner_n_samples};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_bp1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_bp3", 3);

  // (1) reference COUNT: anchor-only => matches NOMSD regardless of inner_n_samples.
  const int nrefs = wfn_nomsd.total_number_of_references();
  CHECK(wfn_s1.total_number_of_references() == nrefs);
  CHECK(wfn_s3.total_number_of_references() == nrefs);

  // (2) per-reference WEIGHT matches NOMSD.
  for (int i = 0; i < nrefs; ++i)
  {
    CHECK_THAT(wfn_s1.getReferenceWeight(i), utils::Approx(wfn_nomsd.getReferenceWeight(i)));
    CHECK_THAT(wfn_s3.getReferenceWeight(i), utils::Approx(wfn_nomsd.getReferenceWeight(i)));
  }

  // (3) the reference Slater matrices themselves match NOMSD (shape [nrefs, npol*NMO, nel], as
  // BackPropagatedEstimator requests them).
  auto collect_refs = [&](Wavefunction<MEM>& wfn) {
    memory::buffered_array<MEM, ComplexType, 3> Refs;
    wfn.getReferences(Refs);
    return nda::to_host(Refs);
  };
  auto R_ref = collect_refs(wfn_nomsd);
  CHECK_THAT(collect_refs(wfn_s1), utils::Approx(R_ref));
  CHECK_THAT(collect_refs(wfn_s3), utils::Approx(R_ref));

  // (4) Layout/metadata parity (not merely "by construction"): the outer-facing queries stay on nomsd_ (True Ham) and so equal NOMSD's, independent of inner_n_samples.
  CHECK(wfn_s1.number_of_cholesky_vectors() == wfn_nomsd.number_of_cholesky_vectors());
  CHECK(wfn_s3.number_of_cholesky_vectors() == wfn_nomsd.number_of_cholesky_vectors());
  CHECK(wfn_s1.getHamType() == wfn_nomsd.getHamType());
  CHECK(wfn_s3.getHamType() == wfn_nomsd.getHamType());
  CHECK(wfn_s1.getWalkerType() == wfn_nomsd.getWalkerType());
  CHECK(wfn_s3.getWalkerType() == wfn_nomsd.getWalkerType());
}

TEST_CASE("stochastic_wfn: bp refs match nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn back-propagation reference API (outer-NOMSD delegate) parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Production-order regression: the BP references are drawn DURING the run -- after begin_inner_step has
// DURING the run -- after begin_inner_step has resampled and EXPANDED the (conditioned/leapfrog) inner
// ensemble to nwalk*P. This pins that the dedicated free-projection reference draw is DECOUPLED from that
// walker-conditioned forward ensemble. It builds a conditioned trial (inner_sampling_target = conditioned,
// true, inner_nsteps = 1), calls begin_inner_step(wset) -- which advances AND expands the inner ensemble
// to nwalk*P -- then asserts the BP draw still returns P walker-INDEPENDENT references at weight 1/P (NOT
// the anchor, NOT the conditioned ensemble), each propagated off the anchor; and that the next
// begin_inner_step RE-EXPANDS the forward ensemble to nwalk*P (the draw reused it as scratch but left the
// forward walk unharmed). REQUIREs the expansion so the scenario is pinned. CLOSED+CPU (leapfrog is
// CPU-only today).
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_production_order(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: conditioned + leapfrog inner sampling is device-ported; runs on DEVICE_MEMORY too.
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    const int inner_n_samples = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};

    auto nomsd_pt_4 = WavefunctionParameters{.name = "wfn_nomsd_bpp", .filename = wfn_file};
    utils::apply_wfn_defaults(nomsd_pt_4, ham);
    WfnFac.push("wfn_nomsd_bpp", nomsd_pt_4);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bpp", type, false, &ham, nwalk);

    WavefunctionParameters pt{.name             = "wfn_stoch_bpp",
                              .filename         = wfn_file,
                              .inner_n_samples  = inner_n_samples,
                              .inner_nsteps     = 1,
                              .inner_sampling_target = StochasticSamplingTarget::WalkerOverlap,
                              .inner_propagator = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_bpp", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_bpp", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_bpp", type, wlk_pt);

    auto get_refs = [&](Wavefunction<MEM>& wfn) {
      memory::buffered_array<MEM, ComplexType, 3> Refs;
      wfn.getReferences(Refs);
      return nda::to_host(Refs);
    };
    auto all = nda::range::all;
    auto R_anchor = get_refs(wfn_nomsd); // NOMSD delegate -> the single anchor reference

    // Drive the inner ensemble: leapfrog begin_inner_step resamples + expands it to nwalk*P.
    auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_bpp");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

    // The BP references are DECOUPLED from the (walker-conditioned, nwalk*P) forward ensemble: even for a
    // conditioned + leapfrog trial, back-propagation exposes a dedicated walker-INDEPENDENT
    // free-projection draw of the P trial samples at weight 1/P (dedicated reference draw), NOT the anchor.
    CHECK(wfn_s.total_number_of_references() == inner_n_samples);
    for (int p = 0; p < inner_n_samples; ++p)
      CHECK_THAT(wfn_s.getReferenceWeight(p), utils::Approx(ComplexType(1.0 / inner_n_samples, 0.0)));

    auto R_draw1 = get_refs(wfn_s);
    CHECK(all_finite3d(R_draw1));
    for (int p = 0; p < inner_n_samples; ++p) // each sample is propagated off the anchor (one B_T step)
      CHECK(max_abs_diff2d(R_draw1(p, all, all), R_anchor(0, all, all)) > 1e-6);

    // The dedicated draw left the forward ensemble at size P, but the next begin_inner_step re-expands it
    // to nwalk*P (the forward walk is unharmed by the BP reference draw).
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

    // Across windows: begin_inner_step opened a new BP window (resetting the idempotency guard), so
    // getReferences now draws a FRESH free-projection ensemble -- different from the previous window's.
    auto R_draw2 = get_refs(wfn_s);
    CHECK(all_finite3d(R_draw2));
    double cross_window = 0.0;
    for (int p = 0; p < inner_n_samples; ++p)
      cross_window = std::max(cross_window, max_abs_diff2d(R_draw1(p, all, all), R_draw2(p, all, all)));
    CHECK(cross_window > 1e-6);
  }
}

TEST_CASE("stochastic_wfn: bp refs production order", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn BP references are a dedicated free-projection draw, decoupled from a "
             "conditioned/leapfrog forward ensemble.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_production_order<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Dedicated free-projection BP references: for a dynamic trial (inner_nsteps > 0, P > 1) back-propagation
// back-propagation draws a FRESH, walker-independent free-projection ensemble {psi_p = B_T(Y^[p])|phi_T>}
// and exposes those P samples as references with weight 1/P, rather than the anchor. This test (CLOSED/
// CPU) builds a non-conditioned dynamic trial and verifies (1) P references each at weight 1/P, (2) the
// references are a finite free-projection draw, each PROPAGATED off the anchor (one B_T step), and (3)
// the draw is IDEMPOTENT within a BP window (a repeated getReferences reuses the same ensemble; cross-
// window freshness after begin_inner_step is checked in the production-order cases). (At inner_nsteps == 0 / P == 1,
// BP delegates to the anchor == NOMSD -- covered by
// `stochastic_wfn: bp refs match nomsd`; the conditioned/leapfrog decoupling is covered by
// the production-order cases.)
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_inner_refs(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: free-projection BP references are device-ported (getReferences); runs on DEVICE_MEMORY.
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;

    const int P    = 3;

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};

    auto nomsd_pt_5 = WavefunctionParameters{.name = "wfn_nomsd_bpir", .filename = wfn_file};
    utils::apply_wfn_defaults(nomsd_pt_5, ham);
    WfnFac.push("wfn_nomsd_bpir", nomsd_pt_5);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bpir", type, false, &ham, nwalk);

    // Non-conditioned, free-projection, dynamic (inner_nsteps = 1) trial -> dedicated reference draw active.
    WavefunctionParameters pt{.name             = "wfn_stoch_bpir",
                              .filename         = wfn_file,
                              .inner_n_samples  = P,
                              .inner_nsteps     = 1,
                              .inner_sampling_target = StochasticSamplingTarget::Gaussian,
                              .inner_propagator = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_bpir", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_bpir", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_bpir", type, wlk_pt);

    // (1) Dedicated reference draw active for a dynamic trial: P references at weight 1/P (vs NOMSD's single anchor).
    CHECK(wfn_s.total_number_of_references() == P);
    CHECK(wfn_nomsd.total_number_of_references() == 1);
    for (int p = 0; p < P; ++p)
      CHECK_THAT(wfn_s.getReferenceWeight(p), utils::Approx(ComplexType(1.0 / P, 0.0)));

    auto get_refs = [&](Wavefunction<MEM>& wfn) {
      memory::buffered_array<MEM, ComplexType, 3> Refs;
      wfn.getReferences(Refs);
      return nda::to_host(Refs);
    };
    auto all = nda::range::all;

    // (2) getReferences performs a fresh free-projection draw {psi_p = B_T(Y^[p])|phi_T>}: P finite
    // references, each PROPAGATED off the anchor (inner_nsteps = 1 bare B_T step).
    auto R_anchor = get_refs(wfn_nomsd); // NOMSD delegate -> the single anchor reference
    auto R_draw1  = get_refs(wfn_s);
    CHECK(all_finite3d(R_draw1));
    for (int p = 0; p < P; ++p)
      CHECK(max_abs_diff2d(R_draw1(p, all, all), R_anchor(0, all, all)) > 1e-6);

    // (3) IDEMPOTENT within a BP window: a repeated getReferences with no intervening forward step
    // (begin_inner_step) reuses the SAME draw -- the guard prevents a silent re-draw within a window
    // (cross-window freshness, after begin_inner_step, is checked in the production-order cases).
    auto R_draw2 = get_refs(wfn_s);
    CHECK(all_finite3d(R_draw2));
    CHECK_THAT(R_draw2, utils::Approx(R_draw1));
  }
}

TEST_CASE("stochastic_wfn: bp free projection ensemble", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn back-propagation draws a fresh free-projection reference ensemble.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_inner_refs<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

template<MEMORY_SPACE MEM>
void stochastic_accumulate_estimators_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: free-projection BP / accumulate is device-ported (getReferences + accumulate_estimators); runs on DEVICE_MEMORY.

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  const int  nup    = std::get<1>(info);
  const int  ndown  = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

  WavefunctionFactory<MEM> WfnFac{};

  auto nomsd_pt_6 = WavefunctionParameters{.name = "wfn_nomsd_ae", .filename = wfn_file};
  utils::apply_wfn_defaults(nomsd_pt_6, ham);
  WfnFac.push("wfn_nomsd_ae", nomsd_pt_6);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_ae", type, false, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_n_samples) -> Wavefunction<MEM>& {
    WavefunctionParameters pt{.name = name, .filename = wfn_file, .inner_n_samples = inner_n_samples};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_ae1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_ae3", 3);

  const int nspin = (type == COLLINEAR ? 2 : 1);
  const int npol  = (type == NONCOLLINEAR ? 2 : 1);

  // Deterministic non-trivial time-evolution operators: X (-> c+), Yc (= conj(Y), -> c) and the operator
  // state M, shape [nw][nspin][npol*NMO][npol*NMO]. Identical across wfns, so the stochastic-vs-NOMSD
  // parity holds for ANY choice (both apply the same linear M + T(X).G_full.Yc transform to their full
  // mixed DM); a near-identity-plus-offsets choice exercises the gemms without ill-conditioning.
  auto make_op = [&](double diag, double off) {
    // Assemble on host (element-by-element), then move to MEM: writing individual elements into a device
    // array from host segfaults (this test now runs on DEVICE_MEMORY).
    nda::array<ComplexType, 4> A_h(nwalk, nspin, npol * NMO, npol * NMO);
    for (int w = 0; w < nwalk; ++w)
      for (int s = 0; s < nspin; ++s)
        for (int i = 0; i < npol * NMO; ++i)
          for (int j = 0; j < npol * NMO; ++j)
            A_h(w, s, i, j) = ComplexType(i == j ? diag : off * double((i + 3 * j) % 5), 0.0);
    if constexpr (MEM == HOST_MEMORY)
      return memory::array<MEM, ComplexType, 4>(A_h);
#if defined(ENABLE_DEVICE)
    else
      return memory::array<MEM, ComplexType, 4>(nda::to_device(A_h));
#else
    else
    {
      static_assert(MEM == HOST_MEMORY, "Device memory requires ENABLE_DEVICE");
      return memory::array<MEM, ComplexType, 4>{};
    }
#endif
  };
  auto Xop  = make_op(1.0, 0.05);
  auto Ycop = make_op(1.0, 0.03);
  auto Mop  = make_op(0.0, 0.01);

  // Accumulate the one-body RDM for one walker block, then read back the printed one_rdm. full1rdm is
  // used directly as the Observable template type (same v.accumulate(...) path as the variant). With
  // time_evolved the operators above transform the (stochastic) mixed DM as in BPWithTimeEvolvedOperators.
  auto collect_one_rdm = [&](Wavefunction<MEM>& wfn, const std::string& tag, bool time_evolved) {
    auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_ae");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    nda::array<ComplexType, 1> wgt(nwalk);
    wgt() = ComplexType(1.0, 0.0);

    std::vector<full1rdm> props1;
    props1.emplace_back(mpi, OneRDMParameters{}, type, NMO, 1);
    std::vector<full1rdm> props; // empty

    if (time_evolved)
      wfn.accumulate_estimators(0, wset, wgt, props1, props, &Xop, &Ycop, &Mop, true);
    else
      wfn.accumulate_estimators(0, wset, wgt, props1, props);

    // Rank-unique temp filename: at -np > 1 every rank runs this round-trip, and a shared filename
    // makes the ranks collide on the same HDF5 file (file-lock error: "unable to lock the file").
    const std::string fname =
        "stochastic_accumulate_" + tag + "_r" + std::to_string(mpi->comm.rank()) + ".h5";
    std::remove(fname.c_str());
    nda::array<ComplexType, 1> Wsum(1);
    Wsum(0) = ComplexType(double(nwalk), 0.0);
    {
      h5::file file(fname, 'w');
      h5::group grp(file);
      props1[0].print(0, &grp, Wsum);
    }
    // full1rdm::print writes only on mpi->comm.root(), so only root's file has the RDM. Read it back
    // (and compare, below) on root only; non-root ranks still called accumulate_estimators + print
    // (the code under test) but skip the read-back. Returns empty on non-root.
    nda::array<ComplexType, 1> data;
    if (mpi->comm.root())
    {
      h5::file file(fname, 'r');
      h5::group grp(file);
      h5::group og = grp.open_group("FullOneRDM").open_group("Average_0");
      nda::h5_read(og, "one_rdm_000000000", data);
    }
    std::remove(fname.c_str());
    return data;
  };

  const bool single_det = (wfn_nomsd.total_number_of_references() == 1);

  // Mixed (non-time-evolved) 1RDM.
  auto rdm_ref = collect_one_rdm(wfn_nomsd, "nomsd", false);
  auto rdm_s1  = collect_one_rdm(wfn_s1, "s1", false);
  auto rdm_s3  = collect_one_rdm(wfn_s3, "s3", false);
  // Time-evolved (back-propagated operators) 1RDM -- same parity, exercising the M + T(X).G_full.Yc path.
  auto trdm_ref = collect_one_rdm(wfn_nomsd, "nomsd_te", true);
  auto trdm_s1  = collect_one_rdm(wfn_s1, "s1_te", true);
  auto trdm_s3  = collect_one_rdm(wfn_s3, "s3_te", true);
  // Comparisons only on root, where the 1RDM was read back (full1rdm::print is root-only).
  if (mpi->comm.root())
  {
    // PREMISE: skip when Psi0 != PsiT. At inner_n_samples == 1 the stochastic path hits the
    // delegate limit (-> NOMSD -> PsiT) while s3 scores against the anchor (-> Psi0), so on those
    // fixtures this compares two wavefunctions. See anchor_is_not_reference.
    auto premise_guess = WfnFac.getInitialGuess("wfn_nomsd_ae");
    auto premise_wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, premise_guess, nwalk);
    if (anchor_is_not_reference<MEM>(wfn_nomsd, premise_wset, type, NMO, "1RDM"))
      return;
    // (1) inner_n_samples invariance: a static replicated ensemble gives an inner_n_samples-independent 1RDM.
    CHECK_THAT(rdm_s3, utils::Approx(rdm_s1));
    // (2) delegate limit: single-determinant trial => stochastic accumulated 1RDM == NOMSD.
    if (single_det)
      CHECK_THAT(rdm_s1, utils::Approx(rdm_ref));
    CHECK_THAT(trdm_s3, utils::Approx(trdm_s1));
    if (single_det)
      CHECK_THAT(trdm_s1, utils::Approx(trdm_ref));
  }
}

TEST_CASE("stochastic_wfn: accumulate_estimators matches nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn accumulate_estimators (one_rdm) delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_accumulate_estimators_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// ============================================================================

// Dynamic ensemble + un-rotated full-G kernels (CLOSED/CPU only today).
//
// inner_nsteps > 0 routes the reductions through the un-rotated full-G energy/force-bias kernels
// (energy_from_fullG / the full-G layout in vbias_from_G) instead of the compact nd = 0 half-rotated
// path. The construction rejects inner_nsteps > 0 unless the trial is CLOSED (RHF) and the build is
// CPU, so these tests skip on non-CLOSED inputs and on DEVICE_MEMORY.
// ============================================================================

// Build the un-rotated full-G path WITHOUT ever resampling (inner_nsteps = 1 but begin_inner_step is
// never called), so the inner ensemble stays at the anchor |phi_T>. There the full-G kernels must
// reproduce the compact nd = 0 path (inner_nsteps = 0, which delegates to NOMSD) and NOMSD itself.
template<MEMORY_SPACE MEM>
void stochastic_full_g_matches_compact(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                       std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: the un-rotated full-G energy is device-ported; this runs on DEVICE_MEMORY too (free-projection,
  // ensemble held at the anchor, so full-G must reproduce the compact/NOMSD result).
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // This test is the only thing comparing an un-rotated full-G energy kernel against the
    // compact/NOMSD reference, so which walker types reach it is load-bearing. Gated on the engine's
    // capability, not on CLOSED -- but energy_collinear's two-spin path is still NOT covered here: the
    // only premise-passing COLLINEAR fixture has an empty beta block. See the coverage note on
    // dynamic_inner_supports (test_common.hpp).
    if (not utils::dynamic_inner_supports(type))
      return;
    const double dt(0.01);

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};

    auto nomsd_pt_7 = WavefunctionParameters{.name = "wfn_nomsd_fg", .filename = wfn_file};
    utils::apply_wfn_defaults(nomsd_pt_7, ham);
    WfnFac.push("wfn_nomsd_fg", nomsd_pt_7);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_fg", type, false, &ham, nwalk);

    auto build_stoch = [&](const std::string& name, int inner_n_samples, int inner_nsteps) -> Wavefunction<MEM>& {
      // inner_nsteps is the parameter under test here: 0 selects the compact/static arm, 1 the
      // un-rotated full-G representation. Only the dynamic arm needs a mode, and it is held at the
      // anchor (begin_inner_step is never called), so make_stochastic_wfn_params's bare Gaussian draw
      // default is the right one. No inner_hamiltonian here, so there is no stamped trained timestep to
      // read; the default 0.01 inner_propagator timestep stands in for a synthetic trial's own input.
      StochasticWfnOptions opt;
      opt.inner_n_samples = inner_n_samples;
      opt.inner_nsteps    = inner_nsteps;
      WavefunctionParameters pt = make_stochastic_wfn_params(name, wfn_file, opt);
      utils::apply_wfn_defaults(pt, ham);
      WfnFac.push(name, pt);
      auto& w = WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
      return w;
    };
    auto& wfn_compact = build_stoch("wfn_stoch_fg0", 1, 0); // compact nd = 0 (delegates to NOMSD)
    auto& wfn_full    = build_stoch("wfn_stoch_fg1", 1, 1); // un-rotated full-G; NOT resampled

    // The two arms score against DIFFERENT determinants unless Psi0 == PsiT: wfn_compact (inner_nsteps=0)
    // delegates to NOMSD -> PsiT, while wfn_full uses the inner-ensemble anchor -> Psi0. See
    // anchor_reference_mismatch. Without this guard the BH UHF fixtures produce a ~2e-5 relative energy
    // disagreement that is purely the fixture's Psi0/PsiT difference and not a kernel defect.
    {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_fg");
      auto wset_guess           = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
      const double herm         = anchor_reference_mismatch<MEM>(wfn_nomsd, wset_guess, type, NMO);
      app_log(0, "  full-G vs compact: anchor/reference premise  max|G - G^H| = {:.6e}", herm);
      if (herm > 1e-10)
      {
        app_log(0, "  full-G vs compact: SKIPPED -- initial guess (inner anchor) is not the trial reference.");
        return;
      }
    }

    struct WalkerEnergies
    {
      nda::array<ComplexType, 1> ov, e1, exx, ej;
    };
    auto collect_energies = [&](Wavefunction<MEM>& wfn) {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_fg");
      auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
      utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
      wfn.Energy(wset); // never calls begin_inner_step -> full-G evaluated at the anchor
      WalkerEnergies out;
      out.ov.resize(nwalk);
      out.e1.resize(nwalk);
      out.exx.resize(nwalk);
      out.ej.resize(nwalk);
      wset.getProperty(OVLP, out.ov);
      wset.getProperty(E1_, out.e1);
      wset.getProperty(EXX_, out.exx);
      wset.getProperty(EJ_, out.ej);
      return out;
    };
    WalkerEnergies ref  = collect_energies(wfn_nomsd);
    WalkerEnergies comp = collect_energies(wfn_compact);
    WalkerEnergies full = collect_energies(wfn_full);

    // full-G energy at the anchor == compact nd = 0 == NOMSD (single-determinant delegate limit).
    CHECK_THAT(full.e1, utils::Approx(comp.e1));
    CHECK_THAT(full.exx, utils::Approx(comp.exx));
    CHECK_THAT(full.ej, utils::Approx(comp.ej));
    CHECK_THAT(linear_overlap(full.ov), utils::Approx(linear_overlap(comp.ov)));
    if (wfn_nomsd.total_number_of_references() == 1)
    {
      CHECK_THAT(full.e1, utils::Approx(ref.e1));
      CHECK_THAT(full.exx, utils::Approx(ref.exx));
      CHECK_THAT(full.ej, utils::Approx(ref.ej));
      CHECK_THAT(linear_overlap(full.ov), utils::Approx(linear_overlap(ref.ov)));
    }

    // Force bias: the full-Likn contraction at the anchor == the compact half-rotated one. Compare
    // the bias X = L.G ([nwalk][nCV] in both), which is layout-independent (G layouts differ).
    auto collect_vbias = [&](Wavefunction<MEM>& wfn) {
      auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_fg");
      auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
      utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt);
      return nda::to_host(X);
    };
    auto X_comp = collect_vbias(wfn_compact);
    auto X_full = collect_vbias(wfn_full);
    CHECK_THAT(X_full, utils::Approx(X_comp));
  }
}

TEST_CASE("stochastic_wfn: full_g matches compact", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn un-rotated full-G vs compact at the anchor.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_full_g_matches_compact<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Dynamic MixedDensityMatrix vs NOMSD at the anchor (inner_nsteps > 0, begin_inner_step never called).
// Isolates G assembly from the energy kernels; static mixed_dm_matches_nomsd does not cover this path.
template<MEMORY_SPACE MEM>
void stochastic_dynamic_full_g_dm_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file);
  if (not env_opt)
    return;
  auto& env = *env_opt;
  if (not utils::dynamic_inner_supports(env.type))
    return; // the dynamic full-G path is CLOSED/COLLINEAR only

  const int nspin = (env.type == COLLINEAR ? 2 : 1);
  const int npol  = (env.type == NONCOLLINEAR ? 2 : 1);
  const int nwalk = 11;

  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_dyndm", wfn_file, nwalk);

  StochasticWfnOptions opt;
  opt.inner_n_samples = 1;
  opt.inner_nsteps   = 1; // dynamic => un-rotated full-G representation
  auto& wfn_dyn      = env.push_stochastic_wfn(mpi, "wfn_stoch_dyndm", wfn_file, opt, nwalk);

  // Non-compact layout only: that IS the full-G layout the energy kernel contracts.
  const int Gsize = nspin * npol * env.NMO * npol * env.NMO;
  // PREMISE (load-bearing -- read anchor_reference_mismatch before trusting a failure here): this test
  // asserts that the inner anchor and NOMSD's reference are the same determinant, which is a property of
  // the FIXTURE (Psi0 vs PsiT), not of the code under test. Guarded below.
  //
  // `perturb` is a conditioning knob. perturb_stochastic_walkers drives walkers far off the anchor
  // (|<psi|phi>| down to ~1e-9, |G| up to ~80). Both settings are asserted where the premise holds --
  // CLOSED passes the perturbed comparison at 6.8e-16 and the COLLINEAR Hubbard fixtures at ~1e-15, which
  // is what rules out ill-conditioning as an explanation for any disagreement seen here.
  auto collect_dm = [&](Wavefunction<MEM>& wfn, bool perturb) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_dyndm");
    if (perturb)
      utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    memory::array<MEM, ComplexType, 2> G(nwalk, Gsize);
    memory::array<MEM, ComplexType, 1> Ov(nwalk);
    // never begin_inner_step => ensemble at the anchor, so both reductions see the same ensemble
    wfn.MixedDensityMatrix(wset, G, false);
    wfn.Log_Overlap(wset, Ov);
    return std::make_pair(nda::array<ComplexType, 2>(nda::to_host(G)),
                          nda::array<ComplexType, 1>(nda::to_host(Ov)));
  };
  auto exp_of = [&](nda::array<ComplexType, 1> const& Ov) {
    nda::array<ComplexType, 1> e(Ov.size());
    for (int w = 0; w < int(Ov.size()); ++w)
      e(w) = std::exp(Ov(w));
    return e;
  };

  if (wfn_nomsd.total_number_of_references() != 1)
    return;

  // G is [nwalk][nspin*npol*NMO][npol*NMO] flattened, so spin sigma occupies rows
  // [sigma*npol*NMO, (sigma+1)*npol*NMO).
  const int rows = npol * env.NMO, cols = npol * env.NMO;
  auto report = [&](char const* tag, nda::array<ComplexType, 2> const& G_ref,
                    nda::array<ComplexType, 1> const& Ov_ref, nda::array<ComplexType, 2> const& G_dyn,
                    nda::array<ComplexType, 1> const& Ov_dyn) {
    // Per-walker, because the aggregate max is dominated by whichever walker is worst conditioned and so
    // hides whether the error is uniform (a defect) or concentrated on near-singular walkers (roundoff).
    for (int w = 0; w < nwalk; ++w)
    {
      double dmax = 0.0, gmax = 0.0;
      for (int sp = 0; sp < nspin; ++sp)
        for (int r = 0; r < rows; ++r)
          for (int c = 0; c < cols; ++c)
          {
            const long idx = long(sp * rows + r) * cols + c;
            dmax           = std::max(dmax, std::abs(G_dyn(w, idx) - G_ref(w, idx)));
            gmax           = std::max(gmax, std::abs(G_ref(w, idx)));
          }
      const double aov = std::abs(std::exp(Ov_ref(w)));
      const double dov = std::abs(Ov_dyn(w) - Ov_ref(w)); // log space: this IS the relative overlap error
      app_log(0, "  {} w{:<2d} |ovlp| = {:.3e}  d(log ovlp) = {:.3e}   max|G| = {:.3e}  relG = {:.3e}", tag, w,
              aov, dov, gmax, gmax > 0.0 ? dmax / gmax : 0.0);
    }
  };

  auto [G_clean_ref, Ov_clean_ref] = collect_dm(wfn_nomsd, false);
  auto [G_clean_dyn, Ov_clean_dyn] = collect_dm(wfn_dyn, false);

  auto wset_guess   = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_dyndm"); // unperturbed == Psi0
  const double herm = anchor_reference_mismatch<MEM>(wfn_nomsd, wset_guess, env.type, env.NMO);
  app_log(0, "  dyn-full-G DM: anchor/reference premise  max|G - G^H| = {:.6e}", herm);
  if (herm > 1e-10)
  {
    // Psi0 != PsiT for this fixture: the anchor and NOMSD's reference are different determinants, so
    // there is nothing to compare. Report and skip rather than record a meaningless failure.
    report("dyn-full-G DM [clean]", G_clean_ref, Ov_clean_ref, G_clean_dyn, Ov_clean_dyn);
    app_log(0, "  dyn-full-G DM: SKIPPED -- initial guess (inner anchor) is not the trial reference.");
    return;
  }

  report("dyn-full-G DM [clean]", G_clean_ref, Ov_clean_ref, G_clean_dyn, Ov_clean_dyn);
  CHECK_THAT(G_clean_dyn, utils::Approx(G_clean_ref));
  CHECK_THAT(exp_of(Ov_clean_dyn), utils::Approx(exp_of(Ov_clean_ref)));

  // Perturbed walkers too: the premise holds, so this is a genuine assertion, and it is the stronger of
  // the two (walkers far off the anchor, |ovlp| down to ~5e-7 and |G| up to ~80). CLOSED passes it at
  // 6.8e-16, which is also what rules out ill-conditioning as an explanation for anything seen here.
  auto [G_pert_ref, Ov_pert_ref] = collect_dm(wfn_nomsd, true);
  auto [G_pert_dyn, Ov_pert_dyn] = collect_dm(wfn_dyn, true);
  report("dyn-full-G DM [pert ]", G_pert_ref, Ov_pert_ref, G_pert_dyn, Ov_pert_dyn);
  CHECK_THAT(G_pert_dyn, utils::Approx(G_pert_ref));
  CHECK_THAT(exp_of(Ov_pert_dyn), utils::Approx(exp_of(Ov_pert_ref)));
}

TEST_CASE("stochastic_wfn: dynamic full_g dm matches nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn dynamic (inner_nsteps>0) full-G mixed DM vs NOMSD at the anchor.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_dynamic_full_g_dm_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Drive the dynamic ensemble: with inner_nsteps > 0 and begin_inner_step() before each step, the
// inner ensemble is reset to the anchor and advanced inner_nsteps free-projection B_T steps, then
// scored with the un-rotated full-G kernels on the MOVED walkers. Exercises resample + all four
// overrides end to end; asserts finiteness (the values are stochastic, not fixed).
template<MEMORY_SPACE MEM>
void stochastic_dynamic_ensemble_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                       std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: free-projection dynamic path is device-ported (full-G energy + inner-propagator advance); runs on
  // DEVICE_MEMORY too.
  {
    const double dt(0.01);

    auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                   [](WALKER_TYPES t) { return t == CLOSED; });
    if (not env_opt)
      return;
    auto& env = *env_opt;

    const int nwalk = 11;

    StochasticWfnOptions stoch_opt;
    stoch_opt.inner_n_samples = 4;
    stoch_opt.inner_nsteps   = 1;
    auto& wfn                = env.push_stochastic_wfn(mpi, "wfn_stoch_dyn", wfn_file, stoch_opt, nwalk);

    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_dyn");
    utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);

    // Each "step" mimics one outer propagator step: arm the latch, then run the hot-path overrides in
    // order. begin_inner_step() + the first reduction resamples the inner ensemble once; the rest
    // score the same moved ensemble. Assert finiteness (values are stochastic, not fixed).
    for (int step = 0; step < 3; ++step)
    {
      wfn.begin_inner_step(wset);
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt); // first reduction -> resamples; full-G force bias on the moved ensemble
      wfn.Energy(wset);       // energy_fullG on the same ensemble
      wfn.Log_Overlap(wset);
      nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
      wset.getProperty(OVLP, ov);
      wset.getProperty(E1_, e1);
      wset.getProperty(EXX_, exx);
      wset.getProperty(EJ_, ej);
      auto X_h = nda::to_host(X);
      // One assertion per array, not per element: these are stochastic values, so the only claim is
      // "nothing blew up". A per-element loop reports the same single fact nwalk*nCV times.
      REQUIRE(all_finite1d(ov));
      REQUIRE(all_finite1d(e1));
      REQUIRE(all_finite1d(exx));
      REQUIRE(all_finite1d(ej));
      REQUIRE(all_finite2d(X_h));
    }
  }
}

TEST_CASE("stochastic_wfn: dynamic free projection ensemble", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn dynamic free-projection ensemble smoke.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_dynamic_ensemble_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Persistent (tethered) inner pool under production conditioning + leapfrog. Instead of the
// reset-then-redraw of the Gaussian-shift conditioned path, each (walker, p) slot owns a FIELD-space
// Markov chain (state = the auxiliary-field configuration Y behind psi = B_T(Y)|phi_T>, stored in the
// outer walker buffer's TrialFields block) re-equilibrated each step by Metropolis-Hastings sweeps
// targeting the conditioned distribution p_T(Y)*|<psi(Y)|phi_w>|. Drives several outer steps through
// all four hot-path overrides for BOTH proposal kernels (pcn and gaussian) and asserts finiteness
// (values are stochastic, not fixed), that the slot-major nw*P pool layout is preserved across steps
// (persistence, not a resize to P or a collapse), that the acceptance counters advance sanely, and --
// via the equil_steps = 0 leg -- that the zero-equilibration prime path is well-behaved. The chains are
// not opt-in: every conditioned dynamic trial has them, and the static / free-projection cases (every
// other [stochastic_wfn] case) have no pool at all.
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                      std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent field chains device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;
    const double dt(0.01);

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 7;
    const int inner_n_samples = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    // Persistent decks: for each proposal kernel, a genuine short MCMC (update_steps = 1), plus the two
    // degenerate ends of the sweep schedule. `burn_in` is explicit in every leg because it is NOT
    // independent of the acceptance claim below: it defaults to 100, so a deck that sets only
    // update_steps = 0 is not frozen at all -- it still pays 100 proposing sweeps at the prime.
    auto run_persistent = [&](const std::string& name, int equil_steps, int burn_in,
                              const std::string& mcmc, double mcmc_step) {
      WavefunctionFactory<MEM> WfnFac{};
      WavefunctionParameters pt{.name                       = name,
                                .filename                   = wfn_file,
                                .inner_n_samples            = inner_n_samples,
                                .inner_nsteps               = 1,
                                .inner_sampling_target      = StochasticSamplingTarget::WalkerOverlap,
                                .inner_sample_update_steps  = equil_steps,
                                .inner_burn_in              = burn_in,
                                .inner_sampler              = mcmc,
                                .inner_sampler_step         = mcmc_step > 0.0 ? std::optional<double>(mcmc_step) : std::nullopt,
                                .inner_propagator           = PropagatorParameters{.timestep = 0.01}};
      utils::mark_stochastic_wfn_input(pt);
      utils::apply_wfn_defaults(pt, ham);
      WfnFac.push(name, pt);
      auto& wfn = WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, name, type, wlk_pt);

      auto const& initial_guess = WfnFac.getInitialGuess(name);
      auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
      utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

      for (int step = 0; step < 4; ++step)
      {
        wfn.begin_inner_step(wset); // leapfrog: equilibrates the persistent pool + stores the old overlap
        memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
        wfn.vbias(wset, X, dt);
        wfn.Energy(wset);
        wfn.Log_Overlap(wset);
        // Persistence: the pool must stay in the slot-major nw*P conditioned layout across steps (it is
        // re-equilibrated in place, never resized to P or collapsed).
        REQUIRE(wfn.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);
        nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
        wset.getProperty(OVLP, ov);
        wset.getProperty(E1_, e1);
        wset.getProperty(EXX_, exx);
        wset.getProperty(EJ_, ej);
        auto X_h = nda::to_host(X);
        // One assertion per array, not per element (see the same note in the free-projection smoke):
        // this case alone used to emit ~45k of the suite's ~58k assertions restating one fact.
        REQUIRE(all_finite1d(ov));
        REQUIRE(all_finite1d(e1));
        REQUIRE(all_finite1d(exx));
        REQUIRE(all_finite1d(ej));
        REQUIRE(all_finite2d(X_h));
      }
      // Acceptance bookkeeping. inner_chain_acceptance() is accepted/proposed, or exactly 1.0 when
      // NOTHING was ever proposed -- so on a deck that makes no proposals the value is a sentinel, not a
      // measurement, and the two are only distinguishable if the deck's sweep counts are known. Hence
      // the split: a deck with zero sweeps of either kind must report the sentinel EXACTLY, which a
      // stray proposal (or a burn-in that ignores its count) would break.
      const double acc = wfn.stochastic_inner_chain_acceptance();
      if (equil_steps == 0 && burn_in == 0)
        CHECK_THAT(acc, utils::Approx(1.0)); // no proposals possible => the sentinel, bit-exact
      else
      {
        // Sweeps ran. A valid fraction is all that is claimed here: with no public proposal counter,
        // "at least one proposal happened" is not separable from "every proposal was accepted", and the
        // pool motion that proves the sweeps did something is asserted in
        // stochastic_wfn: measure replicas nm1 advances pool via the cond_mag checksum.
        REQUIRE(acc >= 0.0);
        REQUIRE(acc <= 1.0);
      }
    };

    run_persistent("wfn_stoch_persist_pcn", 1, 1, "pcn", 0.5);
    run_persistent("wfn_stoch_persist_pcn_indep", 1, 1, "pcn", 1.0); // s = 1: independence redraw limit
    run_persistent("wfn_stoch_persist_gauss", 1, 1, "gaussian", 0.05);
    // Zero per-advance sweeps but the default-sized burn-in: chains equilibrate once, then hold.
    run_persistent("wfn_stoch_persist_noequil", 0, 100, "pcn", 0.5);
    // Genuinely frozen: no burn-in, no per-advance sweeps. The only leg that can report the sentinel.
    run_persistent("wfn_stoch_persist_frozen", 0, 0, "pcn", 0.5);
  }
}

TEST_CASE("stochastic_wfn: persistent inner pool", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent conditioned/leapfrog inner pool smoke.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Regression: a driver's pre-loop initial-energy call (Energy/Log_Overlap called BEFORE the first
// begin_inner_step, i.e. before the persistent chains exist) with a single outer walker per rank
// (nwalk == 1). The freshly-initialized inner ensemble has size == inner_n_samples_ (= P), which
// coincidentally already equals nwalk*P at nwalk == 1, so the conditioned_resample bootstrap size-check
// alone cannot tell an un-conditioned ensemble from a properly resampled one -- the persistent path's
// chains-not-yet-primed fallback (a one-off Gaussian-shift resample) must catch it. Without that guard
// this aborts ("leapfrog inner_cond_mag_ not sized") the first time SAFIRE reports the starting local
// energy on a single-walker-per-rank run (e.g. n_walkers_per_mpi_task=1).
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_nwalk1_bootstrap(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                 std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent / log-aggregate device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                   [](WALKER_TYPES t) { return t == CLOSED; });
    if (not env_opt)
      return;
    auto& env = *env_opt;

    const int nwalk          = 1; // the bug's trigger: nwalk*P aliases the pre-resample ensemble size at P.
    const int inner_n_samples = 4;

    StochasticWfnOptions stoch_opt;
    stoch_opt.inner_n_samples     = inner_n_samples;
    stoch_opt.inner_nsteps       = 1;
    stoch_opt.inner_sampling_target = StochasticSamplingTarget::WalkerOverlap;
    stoch_opt.inner_sample_update_steps = 1;
    auto& wfn                    = env.push_stochastic_wfn(mpi, "wfn_stoch_nwalk1", wfn_file, stoch_opt, nwalk);

    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nwalk1");
    utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);

    // The driver's initial-energy report: Energy() called directly, with NO prior begin_inner_step.
    REQUIRE(wfn.stochastic_inner_ensemble_size() == inner_n_samples); // pre-resample: still the P-sized anchor.
    wfn.Energy(wset);
    REQUIRE(wfn.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples); // now properly conditioned.
    nda::array<ComplexType, 1> e1(nwalk), exx(nwalk), ej(nwalk);
    wset.getProperty(E1_, e1);
    wset.getProperty(EXX_, exx);
    wset.getProperty(EJ_, ej);
    REQUIRE(all_finite1d(e1));
    REQUIRE(all_finite1d(exx));
    REQUIRE(all_finite1d(ej));
  }
}

TEST_CASE("stochastic_wfn: persistent pool nwalk1 bootstrap", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent/leapfrog pool: nwalk=1 pre-begin_inner_step bootstrap regression.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_nwalk1_bootstrap<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Non-persistent permute tests removed with the input mode they required; local-clone contract lives in
// persistent_permute_after_pop below (incl. cross-rank sentinel SECTION).

// Persistent pool x popControl: production order is popControl -> permute_inner_blocks_after_pop ->
// accumulate_step. Cases below pin both ends of that coupling.

// Case 1 -- pop control transports persistent chains with their walkers; post-pop hook rebuilds dets
// from transported fields. Clone-3-into-0 overlap identity (fields live in the outer walker buffer;
// see `sharedwset: stochastic branch lineage`).
template<MEMORY_SPACE MEM>
void stochastic_persistent_permute_after_pop_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent field chains device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;
    if (mpi->comm.size() != 1)
      return; // per-rank-local permutation; the exact cross-slot equalities assume no cross-rank mixing.

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_n_samples = 3;
    const int clone_src      = 3;
    const int clone_dst      = 0;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};
    WavefunctionParameters pt{.name                      = "wfn_stoch_pp_persist",
                              .filename                  = wfn_file,
                              .inner_n_samples           = inner_n_samples,
                              .inner_nsteps              = 1,
                              .inner_sampling_target     = StochasticSamplingTarget::WalkerOverlap,
                              .inner_sample_update_steps = 1,
                              .inner_propagator          = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_pp_persist", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_pp_persist", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_pp_persist", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_pp_persist");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Prime + equilibrate the persistent pool conditioned on the distinct walkers, then record overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);
    memory::buffered_array<MEM, ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before); // latch consumed by begin_inner_step -> reuses the primed pool

    // Simulate a count-preserving popControl clone: outer slot clone_dst becomes a copy of clone_src.
    // Production branch() clones the ENTIRE walker_buffer row, so the copy includes the walker's chain
    // fields (TrialFields row) alongside its Slater matrix -- and, for COLLINEAR, BOTH spin blocks (see
    // the same note in stochastic_inner_permute_after_pop_control); the lineage scalar is set for
    // completeness (the persistent path rebuilds from the fields and does not consume it).
    // Mirror the driver: store_inner_blocks_before_pop -> popControl -> permute_inner_blocks_after_pop.
    // The store must precede the simulated branch, because it captures the magnitudes for the walker
    // layout that exists BEFORE anything moves. Skipping it is not a harmless omission -- the post-pop
    // hook now fails closed on the unpaired call, since silently leaving the magnitudes indexed by a
    // stale slot layout divides the leapfrog overlap by another walker's magnitude (this case produced
    // overlaps ~1e7 while that skip was silent).
    wfn_s.store_inner_blocks_before_pop(wset);
    {
      auto all = nda::range::all;
      auto SM  = wset.SlaterMatrices(Alpha);
      SM(clone_dst, all, all) = SM(clone_src, all, all);
      if (type == COLLINEAR)
      {
        auto SMb = wset.SlaterMatrices(Beta);
        SMb(clone_dst, all, all) = SMb(clone_src, all, all);
      }
      auto TF = wset.TrialFields();
      TF(clone_dst, all) = TF(clone_src, all);
      // branch() copies WHOLE walker rows, so a real clone carries the conditioned magnitudes with the
      // fields. This hand-built clone must copy that row too or it is not modelling branch().
      auto CM = wset.TrialCondMag();
      CM(clone_dst, all) = CM(clone_src, all);
    }
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      lin(clone_dst) = ComplexType(double(clone_src), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    }

    // The post-pop hook rebuilds the pool determinants from the transported chain fields.
    wfn_s.permute_inner_blocks_after_pop(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

    // Score again -- latch consumed + chains live => NO resample, so this reads the rebuilt persistent
    // pool against the cloned walkers.
    memory::buffered_array<MEM, ComplexType, 1> ov_after(nwalk);
    wfn_s.Log_Overlap(wset, ov_after);

    auto lin_before = linear_overlap(nda::to_host(ov_before));
    auto lin_after  = linear_overlap(nda::to_host(ov_after));
    // Tethered block clone_src moved to slot clone_dst, paired with phi_clone_dst == phi_clone_src.
    CHECK_THAT(lin_after(clone_dst), utils::Approx(lin_before(clone_src)));
    CHECK_THAT(lin_after(clone_src), utils::Approx(lin_before(clone_src)));
    for (int w = 0; w < nwalk; ++w)
      if (w != clone_dst)
        CHECK_THAT(lin_after(w), utils::Approx(lin_before(w)));

    // CROSS-RANK ARRIVAL. A walker imported from another rank carries the SLOT_LINEAGE sentinel -1,
    // because this rank holds no pre-pop inner block for the slot it came from.
    //
    // The persistent contract is the OPPOSITE of the deleted reset-then-redraw one, and that inversion is
    // the whole point of this block: BOTH halves of a walker's conditioned state ride inside the outer
    // walker buffer, so they arrive with the walker -- the chain FIELDS, from which the pool determinants
    // are rebuilt exactly, and the TRIAL_COND_MAG magnitudes, snapshot by store_inner_blocks_before_pop.
    // NOTHING IS RESAMPLED and NOTHING IS RECOMPUTED, so every overlap is preserved and the foreign
    // column is as exact as a local one.
    //
    // ⚠️ This block previously asserted the opposite for the magnitudes -- that inner_cond_mag_ "does not
    // ride along" and "is recomputed for the foreign column". That recompute was the defect: it evaluated
    // the magnitude against the POST-pop walker rather than the phi_cond the value means, which is the
    // very thing stochastic_persistent_cond_mag_invariant_under_permute pins as wrong. It was reachable
    // only at np>1, which is why single-rank runs never saw it.
    //
    // The store call below is not decoration: it mirrors the driver's order
    // (store_inner_blocks_before_pop -> popControl -> permute_inner_blocks_after_pop). Without it the
    // block is unallocated, the read-back is skipped, and this section silently tests nothing.
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      lin(1) = ComplexType(-1.0, 0.0); // slot 1 arrived from another rank
      wset.setProperty(SLOT_LINEAGE, lin);

      wfn_s.store_inner_blocks_before_pop(wset);
      wfn_s.permute_inner_blocks_after_pop(wset);
      REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);

      memory::buffered_array<MEM, ComplexType, 1> ov_xr(nwalk);
      wfn_s.Log_Overlap(wset, ov_xr);
      auto lin_xr = linear_overlap(nda::to_host(ov_xr));
      for (int w = 0; w < nwalk; ++w)
        CHECK_THAT(lin_xr(w), utils::Approx(lin_after(w)));
    }
  }
}

TEST_CASE("stochastic_wfn: persistent permute after pop", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn permute_inner_blocks_after_pop realigns the PERSISTENT inner pool after pop control.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_permute_after_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Case 2 -- chain-transport semantics of the coupling: the pool determinants always follow the chain
// FIELDS stored in the outer walker buffer, with no chain restart in either direction. Made observable
// with inner_sample_update_steps = 0 (zero sweeps per advance, so the only thing that can change the pool is the
// post-pop rebuild itself):
//   (A) identity lineage, fields untouched -> the rebuild reproduces the same pool -> overlaps UNCHANGED
//       (an all-or-nothing "re-prime on pop" would have destroyed them);
//   (B) one slot's fields replaced (simulating a migrated walker arriving with a DIFFERENT chain, with
//       the push_walkers sentinel set) -> the rebuild derives that slot's determinants from the NEW
//       fields -> its overlap CHANGES while every other slot's stays exactly fixed.
// This pins that persistence and population control neither reset nor freeze each other: samples survive
// any local branch exactly, and a migrant's samples are reconstructed from its transported chain rather
// than redrawn from scratch.
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_survives_pop_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent field chains device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;
    if (mpi->comm.size() != 1)
      return; // synthetic single-rank sentinel check (real cross-rank moves need no special harness).

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_n_samples = 3;
    const int sentinel_slot  = 2;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};
    // equil_steps = 0 freezes the chains at the prime draw, so the only thing that can change the pool
    // across the pop event is the post-pop rebuild itself -- transported fields => identical
    // determinants; replaced fields => different determinants at exactly that slot. (How well-mixed the
    // frozen pool is does not matter to the identity being checked, only that it cannot move.)
    WavefunctionParameters pt{.name                      = "wfn_stoch_persist_pop",
                              .filename                  = wfn_file,
                              .inner_n_samples           = inner_n_samples,
                              .inner_nsteps              = 1,
                              .inner_sampling_target     = StochasticSamplingTarget::WalkerOverlap,
                              .inner_sample_update_steps = 0,
                              .inner_propagator          = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_persist_pop", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_persist_pop", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_persist_pop", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_persist_pop");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    auto identity_lineage = [&]() {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    };
    auto sentinel_lineage = [&]() {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      lin(sentinel_slot) = ComplexType(-1.0, 0.0); // foreign-rank arrival
      wset.setProperty(SLOT_LINEAGE, lin);
    };

    // Prime the persistent pool (first use: reset-to-anchor + burn_in sweeps) and record its overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);
    memory::buffered_array<MEM, ComplexType, 1> ov0(nwalk);
    wfn_s.Log_Overlap(wset, ov0);
    auto lin0 = linear_overlap(nda::to_host(ov0));

    // (A) A local branch (identity lineage, fields untouched): the post-pop rebuild reproduces the same
    // pool from the same fields, and the next begin_inner_step (0 equil sweeps) leaves it alone -- the
    // tethered samples carry across the pop event UNCHANGED.
    identity_lineage();
    // Paired with the post-pop hook, in the driver's order; the hook fails closed without it.
    wfn_s.store_inner_blocks_before_pop(wset);
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);
    memory::buffered_array<MEM, ComplexType, 1> ovA(nwalk);
    wfn_s.Log_Overlap(wset, ovA);
    auto linA = linear_overlap(nda::to_host(ovA));
    for (int w = 0; w < nwalk; ++w)
      CHECK_THAT(linA(w), utils::Approx(lin0(w))); // persistence held: no restart, no re-equilibration

    // (B) A migrated walker arrives with a DIFFERENT chain: replace slot sentinel_slot's fields
    // (negation is a deterministic, valid, different chain state) and set the push_walkers sentinel.
    // The rebuild must derive that slot's determinants from the NEW fields (overlap changes) and leave
    // every other slot's exactly untouched -- no whole-rank restart.
    {
      // Negate the sentinel slot's chain fields (a valid, deterministic, different chain state). TrialFields
      // is a MEM view, so on device do the negation via a host round-trip of that single row.
      auto TF   = wset.TrialFields();
      auto row  = nda::to_host(TF(sentinel_slot, nda::range::all));
      for (long j = 0; j < row.extent(0); ++j)
        row(j) = -row(j);
      TF(sentinel_slot, nda::range::all) = row; // host -> device (no-op-ish on HOST_MEMORY)
    }
    sentinel_lineage();
    // Paired with the post-pop hook, in the driver's order; the hook fails closed without it.
    wfn_s.store_inner_blocks_before_pop(wset);
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_n_samples);
    memory::buffered_array<MEM, ComplexType, 1> ovB(nwalk);
    wfn_s.Log_Overlap(wset, ovB);
    auto linB = linear_overlap(nda::to_host(ovB));
    REQUIRE(std::abs(linB(sentinel_slot) - lin0(sentinel_slot)) > 1e-6); // dets follow the new fields
    for (int w = 0; w < nwalk; ++w)
      if (w != sentinel_slot)
        CHECK_THAT(linB(w), utils::Approx(lin0(w))); // every other chain carried exactly
  }
}

TEST_CASE("stochastic_wfn: persistent pool survives pop", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent chains survive pop control; a replaced chain rebuilds its own slot only.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_survives_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// Regression: the leapfrog conditioning magnitudes inner_cond_mag_ = |<psi_q|phi_cond>| reference the
// walker the chains were EQUILIBRATED against (phi_cond, set in begin_inner_step), and a
// pop-control realignment that only re-indexes the ensemble must PERMUTE them, never recompute them
// against the current (post-propagation, post-pop) walker. The distinction is invisible to the other
// permute tests because they never move the walker between begin_inner_step and the permute -- so a
// recompute lands on the same phi and looks correct. Here we deliberately perturb the walkers AFTER
// begin_inner_step (mimicking the driver: begin_inner_step -> Propagate moves phi -> popControl ->
// permute), then apply an IDENTITY-lineage permute. Under the correct permutation the magnitudes are
// unchanged (identity re-indexing of values still tied to phi_cond); a stray recompute against the
// moved walker changes them. This is the unit-level analogue of the driver's EnergyEstimator deno_real
// bookkeeping invariant (must be 1 to floating precision), which the recompute broke from the first
// measured block on the trained N2 trial.
template<MEMORY_SPACE MEM>
void stochastic_persistent_cond_mag_invariant_under_permute(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent field chains device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Walker types the dynamic inner path supports; see dynamic_inner_supports in test_common.hpp for
    // the COLLINEAR-coverage note (including what widening this gate did NOT buy).
    if (not utils::dynamic_inner_supports(type))
      return;
    if (mpi->comm.size() != 1)
      return; // manipulates SLOT_LINEAGE directly; the identity-lineage permute assumes no cross-rank mixing.

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_n_samples = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    WalkerSetParameters wlk_pt{.name = "wset0", .walker_type = type};

    WavefunctionFactory<MEM> WfnFac{};
    WavefunctionParameters pt{.name                      = "wfn_stoch_condmag",
                              .filename                  = wfn_file,
                              .inner_n_samples           = inner_n_samples,
                              .inner_nsteps              = 1,
                              .inner_sampling_target     = StochasticSamplingTarget::WalkerOverlap,
                              .inner_sample_update_steps = 1,
                              .inner_propagator          = PropagatorParameters{.timestep = 0.01}};
    utils::mark_stochastic_wfn_input(pt);
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push("wfn_stoch_condmag", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_condmag", type, false, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_condmag", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_condmag");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown); // distinct walkers => distinct cond_mag

    // Equilibrate the chains against the CURRENT walkers phi_cond and set inner_cond_mag_ (leapfrog).
    wfn_s.begin_inner_step(wset);
    const double sum_cond = wfn_s.stochastic_inner_cond_mag_sum();
    REQUIRE(sum_cond > 0.0); // magnitudes were actually set

    // Move the walkers (mimic the propagation that runs before pop control) WITHOUT re-equilibrating
    // the chains -- the leapfrog ensemble and its phi_cond-referenced magnitudes must stay fixed.
    utils::perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // A count-preserving local pop event: identity lineage (no clone, no cross-rank arrival).
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    }
    // Paired with the post-pop hook, in the driver's order. The store captures the magnitudes while
    // they still refer to phi_cond, which is exactly the value this test asserts survives.
    wfn_s.store_inner_blocks_before_pop(wset);
    wfn_s.permute_inner_blocks_after_pop(wset);

    // The magnitudes still reference phi_cond, so an identity re-indexing leaves their sum EXACTLY
    // unchanged. A recompute against the moved walker would change it (the fixed bug).
    CHECK_THAT(wfn_s.stochastic_inner_cond_mag_sum(), utils::Approx(sum_cond));
  }
}

TEST_CASE("stochastic_wfn: cond_mag permute invariant", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn leapfrog inner_cond_mag_ is permuted (not recomputed) across a pop-control realignment.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_cond_mag_invariant_under_permute<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// End-to-end propagator integration: build a real OUTER AFQMCBasePropagator (default hybrid) bound to
// the dynamic stochastic trial (inner_nsteps = 1) and run Propagate() steps. Drives the full hot path
// THROUGH the propagator (vbias -> vHS -> apply -> Log_Overlap), validating that the stochastic
// overrides plug into a real propagation step. Asserts the walkers stay finite.

// At inner_n_measure_samples == 1, measure_energy must ADVANCE THE POOL exactly as it does at nm > 1.
//
// This test used to assert the opposite -- that nm == 1 made measure_energy bit-identical to Energy --
// and that invariant was deliberately removed to match hafqmc, which advances its pool by
// sample_update_steps sweeps before EVERY block measurement including the single-sample case. nm is now
// only the number of replicas averaged. Pinned via inner_cond_mag_sum(), the |<psi_q|phi_cond>| checksum:
// advance_measure_pool recomputes it, so it moves iff the pool moved.
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_nm1_advances_pool(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                    std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                 [](WALKER_TYPES t) { return t == CLOSED; });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk          = 3;
  const int inner_n_samples = 4;

  StochasticWfnOptions opt;
  opt.inner_n_samples     = inner_n_samples;
  opt.inner_nsteps       = 1;
  opt.inner_sampling_target = StochasticSamplingTarget::WalkerOverlap;
  opt.inner_sample_update_steps = 1;
  // nm left at its default of 1 -- that IS the case under test.
  auto& wfn = env.push_stochastic_wfn(mpi, "wfn_stoch_nm1", wfn_file, opt, nwalk);

  auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm1");
  utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
  wfn.begin_inner_step(wset); // prime the chains
  REQUIRE(wfn.stochastic_measure_advances_pool()); // live at nm == 1, which is the point of this test

  const double cond_before = wfn.stochastic_inner_cond_mag_sum();
  // MEM-space buffers: Energy/measure_energy take memory::array_view<MEM,...>, so a host nda::array
  // does not bind in a DEVICE_MEMORY instantiation (it converts, so it cannot bind).
  memory::array<MEM, ComplexType, 2> E_m(nwalk, 3), E_e(nwalk, 3);
  memory::array<MEM, ComplexType, 1> Ov_m(nwalk), Ov_e(nwalk);
  wfn.measure_energy(wset, E_m, Ov_m);
  const double cond_after = wfn.stochastic_inner_cond_mag_sum();
  // The pool advanced, and it was KEPT (no restore) -- so the checksum moved and stays moved.
  CHECK(std::abs(cond_after - cond_before) > 1e-12);
  wfn.Energy(wset, E_e, Ov_e);

  // Energy() run afterwards sees the SAME (advanced, retained) pool, so it reproduces measure_energy's
  // per-walker estimate exactly. That is the check that the advance fed forward rather than being undone.
  nda::array<ComplexType, 2> E_m_h(nda::to_host(E_m)), E_e_h(nda::to_host(E_e));
  for (int w = 0; w < nwalk; ++w)
    for (int k = 0; k < 3; ++k)
    {
      REQUIRE(real(E_m_h(w, k)) == real(E_e_h(w, k)));
      REQUIRE(imag(E_m_h(w, k)) == imag(E_e_h(w, k)));
    }
}

TEST_CASE("stochastic_wfn: measure replicas nm1 advances pool", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn measure_energy advances the pool at nm=1 too (hafqmc-matching).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_nm1_advances_pool<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// At inner_n_measure_samples > 1, measure_energy must return the walker's stored log overlap so
// EnergyEstimator's exp(ovlp - OVLP) reweight stays 1 (deno_real == 1). Replica overlaps belong to
// pools the walker weights never saw and must not be returned.
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_ovlp_is_stored(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                 [](WALKER_TYPES t) { return t == CLOSED; });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk          = 3;
  const int inner_n_samples = 4;

  StochasticWfnOptions opt;
  opt.inner_n_samples         = inner_n_samples;
  opt.inner_nsteps           = 1;
  opt.inner_sampling_target = StochasticSamplingTarget::WalkerOverlap;
  opt.inner_sample_update_steps = 1;
  opt.inner_n_measure_samples = 4;
  auto& wfn = env.push_stochastic_wfn(mpi, "wfn_stoch_nm4", wfn_file, opt, nwalk);

  auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm4");
  utils::perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
  wfn.begin_inner_step(wset);
  // Without this the test would pass vacuously through the plain-Energy fallback, where the invariant
  // holds trivially and the fixed bug could return unnoticed.
  REQUIRE(wfn.stochastic_measure_advances_pool());

  nda::array<ComplexType, 1> ovlp_before(nwalk);
  wset.getProperty(OVLP, ovlp_before);

  memory::array<MEM, ComplexType, 2> E(nwalk, 3);
  memory::array<MEM, ComplexType, 1> Ov(nwalk);
  wfn.measure_energy(wset, E, Ov);
  nda::array<ComplexType, 1> Ov_h(nda::to_host(Ov));

  nda::array<ComplexType, 1> ovlp_after(nwalk);
  wset.getProperty(OVLP, ovlp_after);

  for (int w = 0; w < nwalk; ++w)
  {
    // The returned Ov IS the stored OVLP: this is deno_real == 1 exactly, by construction.
    REQUIRE(real(Ov_h(w)) == real(ovlp_before(w)));
    REQUIRE(imag(Ov_h(w)) == imag(ovlp_before(w)));
    // ...and measuring left the stored property alone, so the next step's reweight is inert too.
    REQUIRE(real(ovlp_after(w)) == real(ovlp_before(w)));
    REQUIRE(imag(ovlp_after(w)) == imag(ovlp_before(w)));
  }
  // The replica-averaged energy is still a usable number (the averaging ran, nothing overflowed).
  REQUIRE(all_finite2d(nda::array<ComplexType, 2>(nda::to_host(E))));
}

TEST_CASE("stochastic_wfn: measure replicas ovlp stored", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn measure_energy at nm>1 returns the STORED OVLP (deno_real == 1 invariant).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_ovlp_is_stored<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// nm energy: measure_energy must return the unweighted mean of nm consecutive replicas. Two nm=2
// calls with the same inner_seed visit the same four pool states as one nm=4, so
// (E1+E2)/2 == E_nm4 (deterministic; catches wrong divisor / accumulator / off-by-one).
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_average_is_true_mean(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                     std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // CLOSED *and* COLLINEAR -- i.e. everything the dynamic path supports, so this asks the capability
  // predicate rather than open-coding the same two types. Both checks here are reference-free (SAFIRE
  // against itself), so unlike the parity tests neither is affected by the Psi0 != PsiT fixtures, and
  // this is one of the few dynamic cases that genuinely exercises a two-spin COLLINEAR trial.
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                 utils::dynamic_inner_supports);
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 3;

  StochasticWfnOptions opt;
  opt.inner_n_samples         = 4;
  opt.inner_nsteps           = 1;
  opt.inner_sampling_target = StochasticSamplingTarget::WalkerOverlap;
  opt.inner_sample_update_steps = 1;
  // Restore OFF is what makes the two arms comparable: the pool must carry over between the split calls.

  auto measure = [&](Wavefunction<MEM>& wfn, WalkerSet<MEM>& wset,
                     memory::array<MEM, ComplexType, 2>& E) {
    memory::array<MEM, ComplexType, 1> Ov(nwalk);
    E() = ComplexType(0.0); // a caller-side zero, so a missing zero INSIDE measure_energy still shows up
    wfn.measure_energy(wset, E, Ov);
  };

  // Arm 1: nm = 2, measured twice. Four advances total, in two batches.
  StochasticWfnOptions opt2 = opt;
  opt2.inner_n_measure_samples = 2;
  auto& wfn2  = env.push_stochastic_wfn(mpi, "wfn_stoch_nm2_split", wfn_file, opt2, nwalk);
  auto wset2  = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm2_split");
  utils::perturb_stochastic_walkers<MEM>(wset2, env.type, env.NMO, env.nup, env.ndown);
  wfn2.begin_inner_step(wset2);
  REQUIRE(wfn2.stochastic_measure_advances_pool());
  memory::array<MEM, ComplexType, 2> E_a(nwalk, 3), E_b(nwalk, 3);
  measure(wfn2, wset2, E_a);
  measure(wfn2, wset2, E_b);

  // Arm 2: nm = 4, measured once. The same four advances, in one batch.
  StochasticWfnOptions opt4 = opt;
  opt4.inner_n_measure_samples = 4;
  auto& wfn4  = env.push_stochastic_wfn(mpi, "wfn_stoch_nm4_whole", wfn_file, opt4, nwalk);
  auto wset4  = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm4_whole");
  utils::perturb_stochastic_walkers<MEM>(wset4, env.type, env.NMO, env.nup, env.ndown);
  wfn4.begin_inner_step(wset4);
  REQUIRE(wfn4.stochastic_measure_advances_pool());
  memory::array<MEM, ComplexType, 2> E_c(nwalk, 3);
  measure(wfn4, wset4, E_c);

  // Only reassociation separates ((a+b)+c)+d)/4 from ((a+b)/2 + (c+d)/2)/2, so this is tight on purpose:
  // every failure mode above is O(1) in the energy, not O(eps).
  const double tol = 1e-10;
  nda::array<ComplexType, 2> E_a_h(nda::to_host(E_a)), E_b_h(nda::to_host(E_b)), E_c_h(nda::to_host(E_c));
  double dev = 0.0, scale = 1.0;
  for (int w = 0; w < nwalk; ++w)
    for (int k = 0; k < 3; ++k)
    {
      const ComplexType split = 0.5 * (E_a_h(w, k) + E_b_h(w, k));
      dev   = std::max(dev, std::abs(split - E_c_h(w, k)));
      scale = std::max(scale, std::abs(E_c_h(w, k)));
    }
  app_log(0, "  nm split-vs-whole: max|mean(E_nm2 x2) - E_nm4| = {:e} (scale {:.3f}, tol {:e})", dev, scale,
          tol * scale);
  REQUIRE(dev <= tol * scale);
}

TEST_CASE("stochastic_wfn: measure replicas average is mean", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn measure_energy at nm>1 is the exact mean of nm consecutive replicas.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_average_is_true_mean<MEM>(mpi, hamil_file, wfn_file);
    // MOLECULES, not ALL_SYSTEMS: the dynamic (inner_nsteps > 0) trial these replicas need calls
    // energy_fullG, which is implemented ONLY for Real3IndexFactorization. THCOps, KP3IndexFactorization
    // and ModelHamOps all APP_ABORT with "energy_fullG not implemented", so the solid and lattice
    // fixtures cannot run this test at all -- a capability limit, not a defect, and the same one the
    // header comment records for full-G parity. Excluding them keeps CLOSED + COLLINEAR coverage on BH
    // while leaving this gate clean, so a failure here means the nm estimator moved and nothing else.
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

// stochastic_measure_replicas_restore_is_invisible was DELETED with the inner_measure_restore knob it
// tested. It asserted that a measurement leaves the propagation chain bit-identical -- the opposite of
// what the engine now does, and of what hafqmc does: the measurement advance feeds forward. The
// feed-forward itself is pinned by `stochastic_wfn: measure replicas nm1 advances pool`.

// Closed inner_* input surface: unknown / removed keys abort; dynamic trials must name
// inner_sampling_target (never defaulted). validate_stochastic_inputs is static -- no Ham/walker fixture.
//
// Unknown-key and unknown-enum-value rejection is now the JSON schema's job (json_check_keys /
// json_enum_from in from_json), not validate_stochastic_inputs's -- so those checks go through
// nlohmann::json::parse + get<WavefunctionParameters> rather than constructing the struct directly.
// Everything else below is a property of validate_stochastic_inputs itself and is exercised by
// constructing WavefunctionParameters in C++ and calling it directly.
TEST_CASE("stochastic_wfn: input surface closed", "[stochastic_wfn]")
{
  app_log(0, "StochasticWfn input surface: unknown keys abort, sampling target never defaults.");
  using Wfn = StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;

  auto base = [] {
    WavefunctionParameters p{.name = "wfn_surface", .filename = "unused.h5", // never opened here
                             .inner_n_samples = 4, .inner_nsteps = 1};
    utils::mark_stochastic_wfn_input(p);
    return p;
  };
  // Minimal JSON matching base(), for the schema-level (unknown-key / bad-enum) checks.
  auto base_json = [] {
    return nlohmann::ordered_json{{"name", "wfn_surface"},        {"filename", "unused.h5"},
                                  {"type", "stochasticwfn"},      {"inner_n_samples", 4},
                                  {"inner_nsteps", 1}};
  };

  SECTION("every removed key is now simply unknown, and unknown aborts")
  {
    for (auto const& dead : {"inner_conditioning", "inner_leapfrog", "inner_persistence",
                             "inner_equil_steps", "inner_measure_stride", "inner_pool_burn_in",
                             "inner_condition_on_new", "inner_measure_restore", "inner_log_aggregate"})
    {
      auto j                        = base_json();
      j["inner_sampling_target"]    = "walker_overlap";
      j[dead]                       = 1;
      INFO("removed key: " << dead);
      REQUIRE_THROWS_AS(j.get<WavefunctionParameters>(), AppAbortException);
    }
  }

  SECTION("a typo in a live key aborts too -- that is the point of closing the surface")
  {
    auto j                             = base_json();
    j["inner_sampling_target"]         = "walker_overlap";
    j["inner_sample_update_step"]      = 8; // singular: plausible typo of inner_sample_update_steps
    REQUIRE_THROWS_AS(j.get<WavefunctionParameters>(), AppAbortException);
  }

  SECTION("a dynamic trial must name its sampling target -- prior sampling is never reached by omission")
  {
    // The one behaviour that must not default. 'gaussian' is correct but has catastrophic variance, so
    // reaching it by forgetting a key is how a production run silently samples the prior.
    WavefunctionParameters p = base();
    REQUIRE_THROWS_AS(Wfn::validate_stochastic_inputs(p), AppAbortException);
  }

  SECTION("an unknown sampling target is rejected")
  {
    auto j                        = base_json();
    j["inner_sampling_target"]    = "conditioned"; // the old spelling; no longer a value
    REQUIRE_THROWS_AS(j.get<WavefunctionParameters>(), AppAbortException);
  }

  SECTION("target and inner_nsteps must agree in both directions")
  {
    WavefunctionParameters p = base();
    p.inner_sampling_target  = StochasticSamplingTarget::Static; // static IS inner_nsteps == 0, but base() sets 1
    REQUIRE_THROWS_AS(Wfn::validate_stochastic_inputs(p), AppAbortException);

    WavefunctionParameters p2 = base();
    p2.inner_nsteps           = 0;
    p2.inner_sampling_target  = StochasticSamplingTarget::WalkerOverlap; // dynamic target with no path length
    REQUIRE_THROWS_AS(Wfn::validate_stochastic_inputs(p2), AppAbortException);
  }

  SECTION("the static default needs no sampler keys at all")
  {
    WavefunctionParameters p{.name = "wfn_static_default", .filename = "unused.h5", .inner_n_samples = 1};
    utils::mark_stochastic_wfn_input(p);
    REQUIRE_NOTHROW(Wfn::validate_stochastic_inputs(p));
    CHECK(p.inner_sampling_target == StochasticSamplingTarget::Static);
    CHECK(p.inner_nsteps == 0);
  }

  SECTION("both dynamic targets round-trip")
  {
    for (auto target : {StochasticSamplingTarget::WalkerOverlap, StochasticSamplingTarget::Gaussian})
    {
      WavefunctionParameters p = base();
      p.inner_sampling_target  = target;
      REQUIRE_NOTHROW(Wfn::validate_stochastic_inputs(p));
      CHECK(p.inner_sampling_target == target);
    }
  }

  // DELIBERATELY NOT TESTED: the numeric values of inner_burn_in (100) and inner_sample_update_steps
  // (32). A CHECK comparing a literal here to a literal in parameters.hpp covers no logic -- both are
  // plain member initializers, validate_stochastic_inputs does not compute either -- so it can only fail
  // when someone deliberately changes a default, reporting a decision back to the person who made it.
  // The default move 1 -> 32 is exactly that: the assertion went red for a correct change while the real
  // regression beside it (make_stochastic_wfn_params eliding the assignment at 1, silently handing six
  // decks 32) was caught by nothing. Defaults are documented at their definition; their CONSEQUENCES are
  // tested in `constructor rejects` below, which pins the abort boundaries around them.

  SECTION("inner_sample_update_steps = 0 is legal at nm = 1")
  {
    // 0 means "prime the chains, then freeze them" -- a real diagnostic mode.
    WavefunctionParameters p   = base();
    p.inner_sampling_target    = StochasticSamplingTarget::WalkerOverlap;
    p.inner_sample_update_steps = 0;
    REQUIRE_NOTHROW(Wfn::validate_stochastic_inputs(p));
    CHECK(p.inner_sample_update_steps == 0);
  }

  // ⚠️ COVERAGE BOUNDARY, stated so the title of the section above is not read as more than it is.
  // validate_stochastic_inputs holds only THREE validations of its own -- inner_n_samples >= 1,
  // inner_nsteps >= 0, and resolve_sampling_target's target/inner_nsteps agreement -- since the
  // unknown-key sweep now lives in the JSON schema (from_json), not here. Everything else lives in the
  // CONSTRUCTOR and is therefore unreachable from this test case, which calls the static function
  // directly and never builds a wavefunction. Inner-setting constructor guards are covered by
  // `stochastic_wfn: constructor rejects invalid inner settings` below, which builds through the
  // factory and expects AppAbortException. The walker-type gate (dynamic requires CLOSED/COLLINEAR)
  // still needs a GHF fixture and is not covered here.
}

// Every APP_ABORT the StochasticWfn CONSTRUCTOR owns for its inner-sampling settings, reached the only
// way they can be: by building a trial through the factory. The input-surface case above cannot get here
// -- it calls validate_stochastic_inputs directly, and these guards all live past that call.
//
// ⚠️ HOW ATTRIBUTION WORKS HERE, because it is not the obvious way. AppAbortException carries a FIXED
// string ("APP_ABORT triggered (see error log for details)") -- the real message goes to the error log,
// not the exception -- so REQUIRE_THROWS_WITH cannot tell one abort from another, and "something threw"
// is nearly worthless on a fixture family where unsupported input aborts for its own reasons (see
// TestFiles::DYNAMIC_INNER). Attribution therefore comes from the CONTROL: one deck that builds, then one
// field of it broken per leg. Base builds + one-field mutation throws => that field's guard fired. Keep
// the control REQUIRE_NOTHROW first; without it every leg below could be passing vacuously.
template<MEMORY_SPACE MEM>
void stochastic_constructor_rejects_bad_inputs(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                               std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (not utils::dynamic_inner_supports(type))
    return;

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 3;

  // The deck every leg starts from: conditioned, dynamic, all sampler settings left at their defaults.
  auto good_opts = [] {
    StochasticWfnOptions opt;
    opt.inner_n_samples       = 4;
    opt.inner_nsteps          = 1;
    opt.inner_sampling_target = StochasticSamplingTarget::WalkerOverlap;
    return opt;
  };

  // A FRESH factory per leg: a build that throws part-way must not leave a half-registered entry behind
  // for the next one to trip over.
  auto build = [&](std::string const& name, WavefunctionParameters pt) {
    WavefunctionFactory<MEM> WfnFac{};
    utils::apply_wfn_defaults(pt, ham);
    WfnFac.push(name, pt);
    WfnFac.getWavefunction(mpi, name, type, false, &ham, nwalk);
  };

  auto deck = [&](std::string const& name) { return make_stochastic_wfn_params(name, wfn_file, good_opts()); };

  // CONTROL. Everything below is only meaningful relative to this.
  REQUIRE_NOTHROW(build("ctor_control", deck("ctor_control")));

  struct Leg
  {
    std::string what;
    std::function<void(WavefunctionParameters&)> break_it;
  };

  const std::vector<Leg> legs = {
      // The retired determinant-space sampler. The one guard here with production history: inputs
      // naming it predate the field-space kernels and must not be silently reinterpreted as pcn.
      {"inner_sampler = metropolis", [](auto& p) { p.inner_sampler = "metropolis"; }},
      {"inner_sampler = an unimplemented kernel", [](auto& p) { p.inner_sampler = "hmc"; }},
      {"pcn with inner_sampler_step = 0", [](auto& p) { p.inner_sampler_step = 0.0; }},
      {"pcn with inner_sampler_step > 1", [](auto& p) { p.inner_sampler_step = 1.5; }},
      {"gaussian with inner_sampler_step = 0",
       [](auto& p) {
         p.inner_sampler      = "gaussian";
         p.inner_sampler_step = 0.0;
       }},
      {"negative inner_sample_update_steps", [](auto& p) { p.inner_sample_update_steps = -1; }},
      {"negative inner_burn_in", [](auto& p) { p.inner_burn_in = -1; }},
      {"inner_n_measure_samples < 1", [](auto& p) { p.inner_n_measure_samples = 0; }},
      // nm > 1 on a frozen pool is one measurement repeated, not an average of replicas.
      {"nm > 1 with zero sweeps per advance",
       [](auto& p) {
         p.inner_n_measure_samples   = 2;
         p.inner_sample_update_steps = 0;
       }},
      // nm > 1 needs a chain to advance; free projection has none. Sweeps stay at the default so this
      // leg cannot be satisfied by the guard above.
      {"nm > 1 on a free-projection trial",
       [](auto& p) {
         p.inner_n_measure_samples = 2;
         p.inner_sampling_target   = StochasticSamplingTarget::Gaussian;
       }},
      // A dynamic trial's timestep has NO default (0.01 must never be assumed); both the missing-block
      // and present-but-empty spellings are errors.
      {"dynamic trial with an inner_propagator carrying no timestep",
       [](auto& p) { p.inner_propagator = PropagatorParameters{}; }},
      {"dynamic trial with no inner_propagator block at all", [](auto& p) { p.inner_propagator.reset(); }},
  };

  for (std::size_t i = 0; i < legs.size(); ++i)
  {
    auto const& leg          = legs[i];
    std::string name         = "ctor_bad_" + std::to_string(i);
    WavefunctionParameters pt = deck(name);
    leg.break_it(pt);
    INFO("leg: " << leg.what);
    REQUIRE_THROWS_AS(build(name, pt), AppAbortException);
  }

  // Legal edges of the very same knobs, so the guards above are not over-reading: these are the values a
  // diagnostic deck actually uses -- freeze the chains after priming, skip the priming, redraw
  // independently. Deliberately NOT a SECTION: this function runs once per fixture inside
  // run_test_with_files' loop, and a SECTION inside a loop is entered on one iteration only.
  WavefunctionParameters frozen    = deck("ctor_frozen");
  frozen.inner_sample_update_steps = 0; // prime, then hold
  REQUIRE_NOTHROW(build("ctor_frozen", frozen));

  WavefunctionParameters no_burn = deck("ctor_no_burn");
  no_burn.inner_burn_in          = 0; // prime and go straight into the walk
  REQUIRE_NOTHROW(build("ctor_no_burn", no_burn));

  WavefunctionParameters indep = deck("ctor_indep");
  indep.inner_sampler_step     = 1.0; // the pcn independence redraw, the top of the allowed range
  REQUIRE_NOTHROW(build("ctor_indep", indep));
}

TEST_CASE("stochastic_wfn: constructor rejects invalid inner settings", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn constructor guards: sampler kernel, step range, sweep counts, nm, timestep.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_constructor_rejects_bad_inputs<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::DYNAMIC_INNER);
}

} // namespace sfqmc
