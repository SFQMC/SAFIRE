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
#include "IO/ptree/ptree_utilities.hpp"
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

void mark_stochastic_wfn_input(ptree& pt) { pt.put("type", "stochasticwfn"); }
constexpr bool is_ft_walker_type([[maybe_unused]] WALKER_TYPES type) { return false; }

struct StochasticWfnOptions
{
  int inner_nwalkers     = 1;
  int inner_nsteps       = 0;
  bool inner_conditioning  = false;
  bool inner_leapfrog      = false;
  bool inner_persistence   = false;
  int inner_equil_steps    = 1;
  int inner_pool_burn_in   = 0;
  bool inner_log_aggregate = false;
  double inner_prop_timestep = 0.01;
};

ptree make_stochastic_wfn_ptree(std::string const& name, std::string const& wfn_file, StochasticWfnOptions const& opt = {})
{
  ptree pt;
  pt.put("name", name);
  pt.put("system", "info0");
  pt.put("filename", wfn_file);
  mark_stochastic_wfn_input(pt);
  pt.put("inner_nwalkers", opt.inner_nwalkers);
  if (opt.inner_nsteps > 0)
    pt.put("inner_nsteps", opt.inner_nsteps);
  if (opt.inner_conditioning)
    pt.put("inner_conditioning", true);
  if (opt.inner_leapfrog)
    pt.put("inner_leapfrog", true);
  if (opt.inner_persistence)
    pt.put("inner_persistence", true);
  if (opt.inner_persistence || opt.inner_equil_steps != 1)
    pt.put("inner_equil_steps", opt.inner_equil_steps);
  if (opt.inner_pool_burn_in != 0)
    pt.put("inner_pool_burn_in", opt.inner_pool_burn_in);
  if (opt.inner_log_aggregate)
    pt.put("inner_log_aggregate", true);
  if (opt.inner_nsteps > 0)
  {
    ptree inner_prop;
    inner_prop.put("timestep", opt.inner_prop_timestep);
    pt.put_child("inner_propagator", inner_prop);
  }
  return pt;
}

ptree make_nomsd_wfn_ptree(std::string const& name, std::string const& wfn_file)
{
  ptree pt;
  pt.put("name", name);
  pt.put("system", "info0");
  pt.put("filename", wfn_file);
  return pt;
}

template<MEMORY_SPACE MEM>
struct StochasticHamWfnEnv
{
  std::map<std::string, AFQMCInfo> info_map;
  std::unique_ptr<HamiltonianFactory> ham_fac;
  Hamiltonian* ham = nullptr;
  WavefunctionFactory<MEM> wfn_fac;
  ptree wlk_pt;
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

    env.info_map.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", env.NMO, env.nup, env.ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    env.ham_fac = std::make_unique<HamiltonianFactory>(env.info_map);
    env.ham_fac->push("ham0", ham_pt);
    env.ham = &env.ham_fac->getHamiltonian(mpi, "ham0");

    env.wlk_pt.put("name", "wset0");
    env.wlk_pt.put("walker_type", walkerTypeToString(env.type));
    return env;
  }

  Wavefunction<MEM>& push_stochastic_wfn(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                         std::string const& name, std::string const& wfn_file,
                                         StochasticWfnOptions const& opt, int nwalk, bool init_inner_walkers = true)
  {
    ptree pt = make_stochastic_wfn_ptree(name, wfn_file, opt);
    wfn_fac.push(name, pt);
    auto& wfn = wfn_fac.getWavefunction(mpi, name, type, ham, nwalk);
    if (init_inner_walkers)
      wfn_fac.maybe_initialize_stochastic_inner_walkers(wfn, name, type, wlk_pt);
    return wfn;
  }

  Wavefunction<MEM>& push_nomsd_wfn(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                    std::string const& name, std::string const& wfn_file, int nwalk)
  {
    wfn_fac.push(name, make_nomsd_wfn_ptree(name, wfn_file));
    return wfn_fac.getWavefunction(mpi, name, type, ham, nwalk);
  }

  WalkerSet<MEM> make_resized_walker_set(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, int nwalk,
                                         std::string const& wfn_id)
  {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, info_map["info0"], rng);
    wset.resize(nwalk, wfn_fac.getInitialGuess(wfn_id));
    return wset;
  }
};

} // namespace

// ----------------------------------------------------------------------------
// StochasticWfn delegate-limit parity (static inner ensemble, tag [stochastic_wfn]).
//
// At the delegate limit (inner_nwalkers = 1, inner_nsteps = 0) the inner trial ensemble collapses to the
// single trial-determinant anchor, so every stochastic override (Log_Overlap, Energy,
// MixedDensityMatrix_for_vbias -> vbias) must reproduce a plain NOMSD on the same outer walkers, for a
// single-determinant (ndet == 1) trial. A multi-determinant trial diverges by design (a single-det inner
// ensemble cannot reproduce a CI-weighted NOMSD).
//
// This check is HamOp-agnostic: at inner_nsteps = 0 the stochastic vbias uses the *compact* path, so it
// runs on any cholesky/THC NOMSD fixture (including the harness's built-in utils/tests/functional/ files).
// Full-G (inner_nsteps > 0) parity is a separate test that needs the Ne_cc-pvdz DenseFactorized + RHF
// fixture (-> Real3IndexFactorization).
// ----------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
void stochastic_wfn_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                  std::string hamil_file, std::string wfn_file)
{
  // StochasticWfn is only built on the NOMSD path; skip PHMSD inputs.
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info  = read_info_from_wfn(wfn_file, "any");
  const int  NMO   = std::get<0>(info);
  const int  nup   = std::get<1>(info);
  const int  ndown = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  // Finite-temperature trials are out of scope for the stochastic delegate limit.
  if (is_ft_walker_type(type))
    return;
  const int nspin = (type == COLLINEAR) ? 2 : 1;
  const int npol  = (type == NONCOLLINEAR) ? 2 : 1;
  const double dt(0.01);

  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 11; // prime: forces non-trivial splits in shared routines

  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd", wfn_file, nwalk);

  StochasticWfnOptions stoch_opt;
  stoch_opt.inner_nwalkers = 1;
  auto& wfn_stoch          = env.push_stochastic_wfn(mpi, "wfn_stoch", wfn_file, stoch_opt, nwalk);
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());

  // Bisection checkpoints (pinpoint SIGSEGV; trim once the test is stable).
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() > 0);
  REQUIRE(wfn_stoch.number_of_cholesky_vectors() > 0);
  REQUIRE(wfn_nomsd.number_of_cholesky_vectors() == wfn_stoch.number_of_cholesky_vectors());
  const bool delegate_limit = (wfn_nomsd.total_number_of_references() == 1);
  REQUIRE(delegate_limit);

  // Deterministic, identical perturbation of the outer walkers (mirrors wfn_factory_sdet),
  // so the two walker sets are bit-for-bit identical going into the reductions.
  auto perturb = [&](auto& wset) {
    std::array<int,2> nels = {nup, ndown};
    for (int spin = 0; spin < nspin; spin++) {
      nda::array<ComplexType, 1> p_h(long(nwalk) * npol * NMO * nels[spin]);
      for (long k = 0; k < p_h.size(); ++k) {
        double v = 0.1 * (k + 1);
        p_h[k] = {std::cos(v), std::sin(v * v)};
      }
      memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
      auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
      nda::tensor::add(p, "ijk", SM, "ijk");
    }
  };

  // Run Log_Overlap / Energy / vbias and harvest per-walker quantities.
  auto harvest = [&](auto& wfn, auto& wset) {
    wfn.Log_Overlap(wset);
    wfn.runtime_optimization(wset);
    wfn.Energy(wset);
    nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
    wset.getProperty(OVLP, ov);
    wset.getProperty(E1_,  e1);
    wset.getProperty(EXX_, exx);
    wset.getProperty(EJ_,  ej);
    // Discrete (model) propagators must initialize potentials before vbias.
    if (wfn.getHamType() == ModelHamiltonian) {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
    wfn.vbias(wset, X, dt);
    return std::make_tuple(std::move(ov), std::move(e1), std::move(exx), std::move(ej), nda::to_host(X));
  };

  auto wset_nomsd = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd");
  REQUIRE(wset_nomsd.size() == nwalk);
  perturb(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.Log_Overlap(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.runtime_optimization(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  wfn_nomsd.Energy(wset_nomsd);
  REQUIRE(wset_nomsd.size() == nwalk);
  nda::array<ComplexType, 1> ov_n(nwalk), e1_n(nwalk), exx_n(nwalk), ej_n(nwalk);
  wset_nomsd.getProperty(OVLP, ov_n);
  wset_nomsd.getProperty(E1_, e1_n);
  wset_nomsd.getProperty(EXX_, exx_n);
  wset_nomsd.getProperty(EJ_, ej_n);
  REQUIRE(ov_n.size() == nwalk);
  // Discrete (model) propagators must initialize potentials before vbias.
  if (wfn_nomsd.getHamType() == ModelHamiltonian) {
    const long ncv = wfn_nomsd.number_of_cholesky_vectors();
    memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
    memory::host_array<ComplexType, 1> nMF(2 * NMO, ComplexType(0.0, 0.0));
    wfn_nomsd.update_potentials(dt, nMF, vMF_discrete, false);
  }
  memory::array<MEM, ComplexType, 2> X_n(nwalk, wfn_nomsd.number_of_cholesky_vectors());
  wfn_nomsd.vbias(wset_nomsd, X_n, dt);
  REQUIRE(X_n.extent(0) == nwalk);

  auto wset_stoch = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch");
  REQUIRE(wset_stoch.size() == nwalk);
  perturb(wset_stoch);
  auto [ov_s, e1_s, exx_s, ej_s, X_s] = harvest(wfn_stoch, wset_stoch);
  REQUIRE(ov_s.size() == nwalk);

  if (delegate_limit) {
    CHECK_THAT(ov_s, utils::Approx(ov_n));
    CHECK_THAT(e1_s, utils::Approx(e1_n));
    CHECK_THAT(exx_s, utils::Approx(exx_n));
    CHECK_THAT(ej_s, utils::Approx(ej_n));
    CHECK_THAT(X_s, utils::Approx(nda::to_host(X_n)));
  }
}

TEST_CASE("stochastic_wfn_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  app_log(0,"StochasticWfn delegate-limit parity unit test.");

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_wfn_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);

}


// ----------------------------------------------------------------------------
// StochasticWfn build + inner-init smoke test (tag [stochastic_wfn]).
//
// Isolates the StochasticWfn *construction* and inner-walker initialization from
// the reductions and from the plain-NOMSD path: it builds ONLY a stochastic trial
// and initializes its inner ensemble -- no outer walker set, no Log_Overlap/Energy/
// vbias, no second wavefunction. Triangulating the SIGSEGV in stochastic_wfn_matches_nomsd:
//   - If THIS test SIGSEGVs   -> fault is in the stochastic build / inner-walker init.
//   - If THIS test passes but `wfn_factory: sdet` SIGSEGVs on the same fixture
//                              -> fault is in the plain dense-Hamiltonian path (not stochastic).
//   - If both pass            -> fault is specific to running reductions after the stochastic
//                                 build (e.g. shared buffer-manager / global state interaction).
// ----------------------------------------------------------------------------
template<MEMORY_SPACE MEM>
void stochastic_build_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                            std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info  = read_info_from_wfn(wfn_file, "any");
  const int  NMO   = std::get<0>(info);
  const int  nup   = std::get<1>(info);
  const int  ndown = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac{};
  ptree stoch_pt;
  stoch_pt.put("name", "wfn_stoch");
  stoch_pt.put("system", "info0");
  stoch_pt.put("filename", wfn_file);
  mark_stochastic_wfn_input(stoch_pt);
  stoch_pt.put("inner_nwalkers", 1);
  WfnFac.push("wfn_stoch", stoch_pt);

  app_log(0, "[stochastic_build_smoke] building stochastic wavefunction");
  auto& wfn_stoch = WfnFac.getWavefunction(mpi, "wfn_stoch", type, &ham, 11);
  app_log(0, "[stochastic_build_smoke] built; initializing inner walkers");
  REQUIRE(wfn_stoch.is_stochastic_wavefunction());
  WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_stoch, "wfn_stoch", type, wlk_pt);
  REQUIRE(wfn_stoch.stochastic_inner_walkers_initialized());
  app_log(0, "[stochastic_build_smoke] inner walkers initialized OK");
}

TEST_CASE("stochastic_build_smoke", "[stochastic_wfn]")
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
//   (1) inner_nwalkers invariance -- a static replicated ensemble (inner_nsteps = 0) gives observables
//       independent of inner_nwalkers (holds for ANY trial);
//   (2) delegate limit -- for a single-determinant trial the stochastic reduction equals the plain NOMSD
//       result. Multi-determinant trials diverge by design (a single-determinant inner ensemble cannot
//       reproduce a CI-weighted NOMSD), so (2) is gated on ndet == 1.
// ============================================================================

// Deterministic, reproducible perturbation of the outer walker Slater matrices, identical to the
// sequence in stochastic_wfn_matches_nomsd, so independently-built walker sets are bit-for-bit
// identical going into the reductions (a meaningful parity check needs non-trivial overlaps).
template<MEMORY_SPACE MEM>
void perturb_stochastic_walkers(WalkerSet<MEM>& wset, WALKER_TYPES type, int NMO, int nup, int ndown)
{
  const int nspin = (type == COLLINEAR) ? 2 : 1;
  const int npol  = (type == NONCOLLINEAR) ? 2 : 1;
  const int nwalk = wset.size();
  std::array<int, 2> nels = {nup, ndown};
  for (int spin = 0; spin < nspin; spin++) {
    nda::array<ComplexType, 1> p_h(long(nwalk) * npol * NMO * nels[spin]);
    for (long k = 0; k < p_h.size(); ++k) {
      double v = 0.1 * (k + 1);
      p_h[k] = {std::cos(v), std::sin(v * v)};
    }
    memory::array<MEM, ComplexType, 3> p(reshape(p_h, nwalk, npol * NMO, nels[spin]));
    auto SM = wset.SlaterMatrices(static_cast<SpinTypes>(spin));
    nda::tensor::add(p, "ijk", SM, "ijk");
  }
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

// StochasticWfn::Log_Overlap reduces the inner ensemble into an effective trial overlap
// (Eq. 24 of arXiv:2505.18519, static-ensemble limit). Overlap is read WITHOUT a following Energy
// call -- Energy overwrites the OVLP walker property and would otherwise mask the override.
template<MEMORY_SPACE MEM>
void stochastic_overlap_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                      std::string hamil_file, std::string wfn_file)
{
  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 11; // prime: forces non-trivial splits in shared routines
  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_ov", wfn_file, nwalk);

  // Stochastic trials at the static limit (inner_nsteps = 0) with inner_nwalkers = 1 and = 3.
  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    StochasticWfnOptions opt;
    opt.inner_nwalkers = inner_nwalkers;
    auto& w            = env.push_stochastic_wfn(mpi, name, wfn_file, opt, nwalk);
    REQUIRE(w.is_stochastic_wavefunction());
    REQUIRE(w.stochastic_inner_walkers_initialized());
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_ov1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_ov3", 3);

  // Every walker set is initialized from the SAME guess and SAME deterministic perturbation, so the
  // three wavefunctions see bit-for-bit identical outer walkers.
  auto collect_overlaps = [&](Wavefunction<MEM>& wfn) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_ov");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    wfn.Log_Overlap(wset);
    nda::array<ComplexType, 1> ov(nwalk);
    wset.getProperty(OVLP, ov);
    return ov;
  };
  auto ov_ref = collect_overlaps(wfn_nomsd);
  auto ov_s1  = collect_overlaps(wfn_s1);
  auto ov_s3  = collect_overlaps(wfn_s3);

  // (1) inner_nwalkers invariance (holds for any trial).
  CHECK_THAT(linear_overlap(ov_s3), utils::Approx(linear_overlap(ov_s1)));
  // (2) delegate limit: single-determinant trial => stochastic overlap == NOMSD overlap.
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(linear_overlap(ov_s1), utils::Approx(linear_overlap(ov_ref)));
}

TEST_CASE("stochastic_overlap_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn Log_Overlap delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_overlap_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// StochasticWfn::Energy reduces the inner ensemble into an effective local energy
// (E1, EXX, EJ) and overlap per outer walker (Eq. 27 of arXiv:2505.18519, static-ensemble limit).
template<MEMORY_SPACE MEM>
void stochastic_energy_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                     std::string hamil_file, std::string wfn_file)
{
  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 11;
  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_en", wfn_file, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    StochasticWfnOptions opt;
    opt.inner_nwalkers = inner_nwalkers;
    auto& w            = env.push_stochastic_wfn(mpi, name, wfn_file, opt, nwalk);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_en1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_en3", 3);

  struct WalkerEnergies
  {
    nda::array<ComplexType, 1> ov, e1, exx, ej;
  };
  auto collect_energies = [&](Wavefunction<MEM>& wfn) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_en");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
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
  };
  WalkerEnergies ref = collect_energies(wfn_nomsd);
  WalkerEnergies s1  = collect_energies(wfn_s1);
  WalkerEnergies s3  = collect_energies(wfn_s3);

  // (1) inner_nwalkers invariance.
  CHECK_THAT(linear_overlap(s3.ov), utils::Approx(linear_overlap(s1.ov)));
  CHECK_THAT(s3.e1, utils::Approx(s1.e1));
  CHECK_THAT(s3.exx, utils::Approx(s1.exx));
  CHECK_THAT(s3.ej, utils::Approx(s1.ej));

  // (2) delegate limit: single-determinant trial => stochastic energy/overlap == NOMSD.
  if (wfn_nomsd.total_number_of_references() == 1)
  {
    CHECK_THAT(linear_overlap(s1.ov), utils::Approx(linear_overlap(ref.ov)));
    CHECK_THAT(s1.e1, utils::Approx(ref.e1));
    CHECK_THAT(s1.exx, utils::Approx(ref.exx));
    CHECK_THAT(s1.ej, utils::Approx(ref.ej));
  }

  // (3) Overlap/Energy consistency: Energy's Ov is the same reduction as Log_Overlap,
  // computed through a different code path, so the two must agree.
  {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_en");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    wfn_s1.Log_Overlap(wset);
    nda::array<ComplexType, 1> ov_ovlp(nwalk);
    wset.getProperty(OVLP, ov_ovlp);
    CHECK_THAT(linear_overlap(ov_ovlp), utils::Approx(linear_overlap(s1.ov)));
  }

  // (4) Propagator entry point: the 3-arg Energy(wset, E, Ov) -- called directly by the
  // local-energy propagation path in the propagator (not the property-setter form) -- agrees with
  // Energy(wset). Exercises the 3-arg overload with caller-allocated buffers.
  {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_en");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
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
  }
}

TEST_CASE("stochastic_energy_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn Energy delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_energy_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// StochasticWfn::MixedDensityMatrix_for_vbias reduces the inner ensemble into the mixed
// density matrix the force bias contracts against (estimator 3 of arXiv:2505.18519, static limit),
// and vbias contracts it (estimator 4, x_gamma[w] = L_gamma . G[w]) against the True-Ham Cholesky.
// The overhaul vbias(wset, X, dt) drives MixedDensityMatrix_for_vbias internally, so we compare the
// resulting force bias X (= L.G) directly; the intermediate G is not exposed on the variant.
template<MEMORY_SPACE MEM>
void stochastic_vbias_matches_nomsd(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                    std::string hamil_file, std::string wfn_file)
{
  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;
  const double dt(0.01);

  const int nwalk = 11;
  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_vb", wfn_file, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    StochasticWfnOptions opt;
    opt.inner_nwalkers = inner_nwalkers;
    auto& w            = env.push_stochastic_wfn(mpi, name, wfn_file, opt, nwalk);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_vb1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_vb3", 3);

  auto collect_vbias = [&](Wavefunction<MEM>& wfn) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_vb");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    // Discrete (model) propagators must initialize potentials before vbias.
    if (wfn.getHamType() == ModelHamiltonian)
    {
      const long ncv = wfn.number_of_cholesky_vectors();
      memory::array<MEM, ComplexType, 1> vMF_discrete(ncv, ComplexType(0.0, 0.0));
      memory::host_array<ComplexType, 1> nMF(2 * env.NMO, ComplexType(0.0, 0.0));
      wfn.update_potentials(dt, nMF, vMF_discrete, false);
    }
    memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
    wfn.vbias(wset, X, dt);
    return nda::to_host(X);
  };
  auto X_ref = collect_vbias(wfn_nomsd);
  auto X_s1  = collect_vbias(wfn_s1);
  auto X_s3  = collect_vbias(wfn_s3);

  // (1) inner_nwalkers invariance.
  CHECK_THAT(X_s3, utils::Approx(X_s1));
  // (2) delegate limit: single-determinant trial => stochastic force bias == NOMSD.
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(X_s1, utils::Approx(X_ref));
}

TEST_CASE("stochastic_vbias_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vbias delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_vbias_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// StochasticWfn::MixedDensityMatrix reduces the inner ensemble into the observable mixed density matrix (estimator 3 of arXiv:2505.18519, static limit) -- the observable
// analogue of MixedDensityMatrix_for_vbias. Unlike vbias, the observable mixed DM IS exposed on the
// Wavefunction variant, so we compare G directly (both the compact [nel*NMO] and full [NMO*NMO]
// layouts) plus the effective overlap against plain NOMSD: delegate-limit parity (ndet==1) and
// inner_nwalkers invariance via a static replicated ensemble.
template<MEMORY_SPACE MEM>
void stochastic_mixed_density_matrix_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nspin = (env.type == COLLINEAR ? 2 : 1);
  const int npol  = (env.type == NONCOLLINEAR ? 2 : 1);
  const int nel   = (env.type == COLLINEAR ? env.nup + env.ndown : env.nup);

  const int nwalk = 11;
  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_dm", wfn_file, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    StochasticWfnOptions opt;
    opt.inner_nwalkers = inner_nwalkers;
    auto& w            = env.push_stochastic_wfn(mpi, name, wfn_file, opt, nwalk);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_dm1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_dm3", 3);

  auto collect_dm = [&](Wavefunction<MEM>& wfn, bool compact) {
    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_nomsd_dm");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    const int Gsize = compact ? nel * npol * env.NMO : nspin * npol * env.NMO * npol * env.NMO;
    memory::array<MEM, ComplexType, 2> G(nwalk, Gsize);
    memory::array<MEM, ComplexType, 1> Ov(nwalk);
    wfn.MixedDensityMatrix(wset, G, Ov, compact);
    nda::array<ComplexType, 2> Gh = nda::to_host(G);
    nda::array<ComplexType, 1> Ovh = nda::to_host(Ov);
    return std::make_pair(Gh, Ovh);
  };

  // log overlaps differ from NOMSD's only by an integer multiple of 2*pi*i (principal branch); compare
  // exp() to dodge that ambiguity, as stochastic_overlap_matches_nomsd does.
  auto exp_of = [&](nda::array<ComplexType, 1> const& Ov) {
    nda::array<ComplexType, 1> e(Ov.size());
    for (int w = 0; w < int(Ov.size()); ++w)
      e(w) = std::exp(Ov(w));
    return e;
  };

  for (bool compact : {true, false})
  {
    auto [G_ref, Ov_ref] = collect_dm(wfn_nomsd, compact);
    auto [G_s1, Ov_s1]   = collect_dm(wfn_s1, compact);
    auto [G_s3, Ov_s3]   = collect_dm(wfn_s3, compact);

    // (1) inner_nwalkers invariance: a static replicated ensemble gives an inner_nwalkers-independent DM.
    CHECK_THAT(G_s3, utils::Approx(G_s1));
    CHECK_THAT(exp_of(Ov_s3), utils::Approx(exp_of(Ov_s1)));

    // (2) delegate limit: single-determinant trial => stochastic observable DM == NOMSD.
    if (wfn_nomsd.total_number_of_references() == 1)
    {
      CHECK_THAT(G_s1, utils::Approx(G_ref));
      CHECK_THAT(exp_of(Ov_s1), utils::Approx(exp_of(Ov_ref)));
    }
  }
}

TEST_CASE("stochastic_mixed_density_matrix_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn observable MixedDensityMatrix delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mixed_density_matrix_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// StochasticWfn::vMF / G_MF are the trial's OWN mean-field quantities <Psi_T|.|Psi_T>/<Psi_T|Psi_T>,
// built by reducing the inner ensemble against ITSELF (a double sum over inner-walker pairs -- the
// inner-ensemble analogue of NOMSD's multi-determinant mean field). Unlike the propagator hot-path mixed
// estimators there is no outer walker. At the static replicated limit every inner walker
// == the anchor, so both collapse to the anchor mean field == plain NOMSD::vMF / G_MF. We compare the
// mean-field bias vMF (= L . G_MF, a [nCV] vector) and the mean-field DM G_MF directly: inner_nwalkers
// invariance (static replicated 1 vs 3) and delegate-limit equality (ndet==1).
template<MEMORY_SPACE MEM>
void stochastic_mean_field_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  const int  nup    = std::get<1>(info);
  const int  ndown  = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;
  const double dt(0.01);

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac{};

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_mf");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_mf", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_mf", type, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
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
  auto v_ref = collect_vMF(wfn_nomsd);
  auto v_s1  = collect_vMF(wfn_s1);
  auto v_s3  = collect_vMF(wfn_s3);

  // (1) inner_nwalkers invariance of the mean-field bias.
  CHECK_THAT(v_s3, utils::Approx(v_s1));
  // (2) delegate limit: single-determinant trial => stochastic vMF == NOMSD.
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(v_s1, utils::Approx(v_ref));

  // Mean-field one-body Green's function G_MF ([nspin][npol*NMO][npol*NMO]).
  auto collect_GMF = [&](Wavefunction<MEM>& wfn) {
    auto Gshm = wfn.G_MF();
    return nda::to_host(Gshm());
  };
  auto G_ref = collect_GMF(wfn_nomsd);
  auto G_s1  = collect_GMF(wfn_s1);
  auto G_s3  = collect_GMF(wfn_s3);

  CHECK_THAT(G_s3, utils::Approx(G_s1));
  if (wfn_nomsd.total_number_of_references() == 1)
    CHECK_THAT(G_s1, utils::Approx(G_ref));
}

TEST_CASE("stochastic_mean_field_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF mean-field delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Production-order regression: vMF / G_MF are trial-only, but in the real Propagate loop
// begin_inner_step(wset) runs BEFORE generateP1 calls vMF, and in leapfrog mode begin_inner_step
// eagerly resamples -- expanding the inner ensemble to nwalk*P walker-CONDITIONED samples. The mean
// field must NOT then be a 1/(nwalk*P)^2-weighted double sum over that conditioned ensemble; it must
// still be the trial (anchor) mean field == NOMSD. This test reproduces that call order (build a
// leapfrog trial, call begin_inner_step to expand the ensemble, then vMF/G_MF) and asserts equality
// with NOMSD. Without the inner.size()==inner_nwalkers_ gate (mean_field_uses_inner_ensemble) the
// expanded ensemble would be reduced with the wrong normalization and this would fail. CLOSED+CPU
// (leapfrog/conditioning is CPU-only today).
template<MEMORY_SPACE MEM>
void stochastic_mean_field_production_order(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // leapfrog / conditioned inner sampling is CPU-only today.
  else
  {
    const auto info   = read_info_from_wfn(wfn_file, "any");
    const int  NMO    = std::get<0>(info);
    const int  nup    = std::get<1>(info);
    const int  ndown  = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    const double dt(0.01);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    const int inner_nwalkers = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};

    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_mfp");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_mfp", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_mfp", type, &ham, nwalk);

    // Leapfrog stochastic trial: begin_inner_step will conditioned-resample (expand to nwalk*P).
    ptree pt;
    pt.put("name", "wfn_stoch_mfp");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_mfp", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_mfp", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_mfp", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_mfp"));

    // Reproduce the Propagate ordering: begin_inner_step (leapfrog => conditioned resample to nwalk*P)
    // BEFORE the mean-field calls. With the fix, vMF/G_MF detect the non-P-sample ensemble and delegate
    // to the anchor mean field == NOMSD.
    wfn_s.begin_inner_step(wset);

    // The leapfrog begin_inner_step must actually have expanded the ensemble to nwalk*P (otherwise the
    // NOMSD parity below could pass for the wrong reason -- a still-P-sample anchor ensemble would also
    // match). This pins the scenario the gate is meant to handle.
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    memory::array<MEM, ComplexType, 1> v_ref(wfn_nomsd.number_of_cholesky_vectors(), ComplexType(0.0, 0.0));
    memory::array<MEM, ComplexType, 1> v_s(wfn_s.number_of_cholesky_vectors(), ComplexType(0.0, 0.0));
    wfn_nomsd.vMF(v_ref, dt);
    wfn_s.vMF(v_s, dt);
    CHECK_THAT(nda::to_host(v_s), utils::Approx(nda::to_host(v_ref)));

    auto Gmf_ref = wfn_nomsd.G_MF();
    auto Gmf_s   = wfn_s.G_MF();
    CHECK_THAT(nda::to_host(Gmf_s()), utils::Approx(nda::to_host(Gmf_ref())));
  }
}

TEST_CASE("stochastic_mean_field_production_order", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF survive the begin_inner_step-before-generateP1 order.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_production_order<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

namespace
{
// Shared helpers for back-propagation reference tests: max |A-B| over two 2D views, and a finiteness
// finiteness check over a 3D [nref, npol*NMO, nel] reference array (operator()-based, layout-agnostic).
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
} // namespace

// Back-propagation reference set and outer-facing layout queries AT THE STATIC LIMIT (inner_nsteps = 0 -- the default in this test). There bp_uses_inner_ensemble() is false, so
// the stochastic trial DELEGATES its reference set to the outer nomsd_ (= {phi_T}, weight 1, for the
// single-determinant anchor; the full CI expansion for a multi-det outer trial), ignoring the inner
// ensemble. So total_number_of_references / getReferenceWeight / getReferences equal plain NOMSD's
// UNCONDITIONALLY (not just at ndet==1) and INDEPENDENT of inner_nwalkers (1 and 3 both delegate). (For a
// DYNAMIC trial, inner_nsteps > 0 with P > 1, getReferences instead performs a dedicated free-projection
// draw -- exercised by stochastic_back_propagation_inner_refs and _production_order.) This test verifies
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

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  const int  nup    = std::get<1>(info);
  const int  ndown  = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  const int npol = (type == NONCOLLINEAR ? 2 : 1);
  const int nel  = (type == COLLINEAR ? nup + ndown : nup);

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac{};

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_bp");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_bp", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bp", type, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
    return w;
  };
  auto& wfn_s1 = build_stoch("wfn_stoch_bp1", 1);
  auto& wfn_s3 = build_stoch("wfn_stoch_bp3", 3);

  // (1) reference COUNT: anchor-only => matches NOMSD regardless of inner_nwalkers.
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
    const int n = wfn.total_number_of_references();
    memory::array<MEM, ComplexType, 3> Refs(n, npol * NMO, nel);
    Refs() = ComplexType(0.0);
    wfn.getReferences(n, Refs);
    return nda::to_host(Refs);
  };
  auto R_ref = collect_refs(wfn_nomsd);
  CHECK_THAT(collect_refs(wfn_s1), utils::Approx(R_ref));
  CHECK_THAT(collect_refs(wfn_s3), utils::Approx(R_ref));

  // (4) Layout/metadata parity (not merely "by construction"): the outer-facing queries stay on nomsd_ (True Ham) and so equal NOMSD's, independent of inner_nwalkers.
  CHECK(wfn_s1.number_of_cholesky_vectors() == wfn_nomsd.number_of_cholesky_vectors());
  CHECK(wfn_s3.number_of_cholesky_vectors() == wfn_nomsd.number_of_cholesky_vectors());
  CHECK(wfn_s1.getHamType() == wfn_nomsd.getHamType());
  CHECK(wfn_s3.getHamType() == wfn_nomsd.getHamType());
  CHECK(wfn_s1.getWalkerType() == wfn_nomsd.getWalkerType());
  CHECK(wfn_s3.getWalkerType() == wfn_nomsd.getWalkerType());
}

TEST_CASE("stochastic_back_propagation_matches_nomsd", "[stochastic_wfn]")
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
// walker-conditioned forward ensemble. It builds a leapfrog trial (inner_conditioning = inner_leapfrog =
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
  if constexpr (MEM != HOST_MEMORY)
    return; // leapfrog / conditioned inner sampling is CPU-only today.
  else
  {
    const auto info   = read_info_from_wfn(wfn_file, "any");
    const int  NMO    = std::get<0>(info);
    const int  nup    = std::get<1>(info);
    const int  ndown  = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    const int npol = (type == NONCOLLINEAR ? 2 : 1);
    const int nel  = (type == COLLINEAR ? nup + ndown : nup);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    const int inner_nwalkers = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};

    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_bpp");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_bpp", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bpp", type, &ham, nwalk);

    ptree pt;
    pt.put("name", "wfn_stoch_bpp");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_bpp", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_bpp", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_bpp", type, wlk_pt);

    auto get_refs = [&](Wavefunction<MEM>& wfn, int n) {
      memory::array<MEM, ComplexType, 3> Refs(n, npol * NMO, nel);
      Refs() = ComplexType(0.0);
      wfn.getReferences(n, Refs);
      return nda::to_host(Refs);
    };
    auto all = nda::range::all;
    auto R_anchor = get_refs(wfn_nomsd, 1); // NOMSD delegate -> the single anchor reference

    // Drive the inner ensemble: leapfrog begin_inner_step resamples + expands it to nwalk*P.
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_bpp"));
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // The BP references are DECOUPLED from the (walker-conditioned, nwalk*P) forward ensemble: even for a
    // conditioned + leapfrog trial, back-propagation exposes a dedicated walker-INDEPENDENT
    // free-projection draw of the P trial samples at weight 1/P (dedicated reference draw), NOT the anchor.
    CHECK(wfn_s.total_number_of_references() == inner_nwalkers);
    for (int p = 0; p < inner_nwalkers; ++p)
      CHECK_THAT(wfn_s.getReferenceWeight(p), utils::Approx(ComplexType(1.0 / inner_nwalkers, 0.0)));

    auto R_draw1 = get_refs(wfn_s, inner_nwalkers);
    CHECK(all_finite3d(R_draw1));
    for (int p = 0; p < inner_nwalkers; ++p) // each sample is propagated off the anchor (one B_T step)
      CHECK(max_abs_diff2d(R_draw1(p, all, all), R_anchor(0, all, all)) > 1e-6);

    // The dedicated draw left the forward ensemble at size P, but the next begin_inner_step re-expands it
    // to nwalk*P (the forward walk is unharmed by the BP reference draw).
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // Across windows: begin_inner_step opened a new BP window (resetting the idempotency guard), so
    // getReferences now draws a FRESH free-projection ensemble -- different from the previous window's.
    auto R_draw2 = get_refs(wfn_s, inner_nwalkers);
    CHECK(all_finite3d(R_draw2));
    double cross_window = 0.0;
    for (int p = 0; p < inner_nwalkers; ++p)
      cross_window = std::max(cross_window, max_abs_diff2d(R_draw1(p, all, all), R_draw2(p, all, all)));
    CHECK(cross_window > 1e-6);
  }
}

TEST_CASE("stochastic_back_propagation_production_order", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn BP references are a dedicated free-projection draw, decoupled from a "
             "conditioned/leapfrog forward ensemble.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_production_order<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Dedicated free-projection BP references: for a dynamic trial (inner_nsteps > 0, P > 1) back-propagation
// back-propagation draws a FRESH, walker-independent free-projection ensemble {psi_p = B_T(Y^[p])|phi_T>}
// and exposes those P samples as references with weight 1/P, rather than the anchor. This test (CLOSED/
// CPU) builds a non-conditioned dynamic trial and verifies (1) P references each at weight 1/P, (2) the
// references are a finite free-projection draw, each PROPAGATED off the anchor (one B_T step), and (3)
// the draw is IDEMPOTENT within a BP window (a repeated getReferences reuses the same ensemble; cross-
// window freshness after begin_inner_step is checked in _production_order). (At inner_nsteps == 0 / P == 1,
// BP delegates to the anchor == NOMSD -- covered by
// stochastic_back_propagation_matches_nomsd; the conditioned/leapfrog decoupling is covered by
// _production_order.)
template<MEMORY_SPACE MEM>
void stochastic_back_propagation_inner_refs(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // free-projection inner sampling (inner_nsteps > 0) is CPU-only today.
  else
  {
    const auto info   = read_info_from_wfn(wfn_file, "any");
    const int  NMO    = std::get<0>(info);
    const int  nup    = std::get<1>(info);
    const int  ndown  = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;

    const int npol = (type == NONCOLLINEAR ? 2 : 1);
    const int nel  = (type == COLLINEAR ? nup + ndown : nup);
    const int P    = 3;

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};

    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_bpir");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_bpir", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_bpir", type, &ham, nwalk);

    // Non-conditioned, free-projection, dynamic (inner_nsteps = 1) trial -> dedicated reference draw active.
    ptree pt;
    pt.put("name", "wfn_stoch_bpir");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", P);
    pt.put("inner_nsteps", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_bpir", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_bpir", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_bpir", type, wlk_pt);

    // (1) Dedicated reference draw active for a dynamic trial: P references at weight 1/P (vs NOMSD's single anchor).
    CHECK(wfn_s.total_number_of_references() == P);
    CHECK(wfn_nomsd.total_number_of_references() == 1);
    for (int p = 0; p < P; ++p)
      CHECK_THAT(wfn_s.getReferenceWeight(p), utils::Approx(ComplexType(1.0 / P, 0.0)));

    auto get_refs = [&](Wavefunction<MEM>& wfn, int n) {
      memory::array<MEM, ComplexType, 3> Refs(n, npol * NMO, nel);
      Refs() = ComplexType(0.0);
      wfn.getReferences(n, Refs);
      return nda::to_host(Refs);
    };
    auto all = nda::range::all;

    // (2) getReferences performs a fresh free-projection draw {psi_p = B_T(Y^[p])|phi_T>}: P finite
    // references, each PROPAGATED off the anchor (inner_nsteps = 1 bare B_T step).
    auto R_anchor = get_refs(wfn_nomsd, 1); // NOMSD delegate -> the single anchor reference
    auto R_draw1  = get_refs(wfn_s, P);
    CHECK(all_finite3d(R_draw1));
    for (int p = 0; p < P; ++p)
      CHECK(max_abs_diff2d(R_draw1(p, all, all), R_anchor(0, all, all)) > 1e-6);

    // (3) IDEMPOTENT within a BP window: a repeated getReferences with no intervening forward step
    // (begin_inner_step) reuses the SAME draw -- the guard prevents a silent re-draw within a window
    // (cross-window freshness, after begin_inner_step, is checked in _production_order).
    auto R_draw2 = get_refs(wfn_s, P);
    CHECK(all_finite3d(R_draw2));
    CHECK_THAT(R_draw2, utils::Approx(R_draw1));
  }
}

TEST_CASE("stochastic_back_propagation_inner_refs", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn back-propagation draws a fresh free-projection reference ensemble.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_back_propagation_inner_refs<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

namespace
{
template<MEMORY_SPACE MEM>
void require_finite_bp_one_rdm(h5::file const& file, std::string const& avg_path, int iblock)
{
  std::string suffix = std::format("{:09d}", iblock);
  nda::array<ComplexType, 1> read_data;
  ComplexType denom{};
  {
    h5::group root(file);
    utils::h5_read(root, avg_path + "/one_rdm_" + suffix, read_data);
    h5::read(root, avg_path + "/denominator_" + suffix, denom);
  }
  REQUIRE(read_data.size() > 0);
  REQUIRE(std::abs(denom) > 0.0);
  for (auto v : read_data)
  {
    REQUIRE(std::isfinite(real(v)));
    REQUIRE(std::isfinite(imag(v)));
  }
}
} // namespace
template<MEMORY_SPACE MEM>
void stochastic_accumulate_estimators_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  const int  nup    = std::get<1>(info);
  const int  ndown  = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  std::map<std::string, AFQMCInfo> InfoMap;
  AFQMCInfo info0{"info0", NMO, nup, ndown, 0};
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", info0));

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("system", "info0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  const int nwalk = 11;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac{};

  ptree nomsd_pt;
  nomsd_pt.put("name", "wfn_nomsd_ae");
  nomsd_pt.put("system", "info0");
  nomsd_pt.put("filename", wfn_file);
  WfnFac.push("wfn_nomsd_ae", nomsd_pt);
  auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_ae", type, &ham, nwalk);

  auto build_stoch = [&](const std::string& name, int inner_nwalkers) -> Wavefunction<MEM>& {
    ptree pt;
    pt.put("name", name);
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    WfnFac.push(name, pt);
    auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
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
    memory::array<MEM, ComplexType, 4> A(nwalk, nspin, npol * NMO, npol * NMO);
    for (int w = 0; w < nwalk; ++w)
      for (int s = 0; s < nspin; ++s)
        for (int i = 0; i < npol * NMO; ++i)
          for (int j = 0; j < npol * NMO; ++j)
            A(w, s, i, j) = ComplexType(i == j ? diag : off * double((i + 3 * j) % 5), 0.0);
    return A;
  };
  auto Xop  = make_op(1.0, 0.05);
  auto Ycop = make_op(1.0, 0.03);
  auto Mop  = make_op(0.0, 0.01);

  // Accumulate the one-body RDM for one walker block, then read back the printed one_rdm. full1rdm is
  // used directly as the Observable template type (same v.accumulate(...) path as the variant). With
  // time_evolved the operators above transform the (stochastic) mixed DM as in BPWithTimeEvolvedOperators.
  auto collect_one_rdm = [&](Wavefunction<MEM>& wfn, const std::string& tag, bool time_evolved) {
    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_ae"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    nda::array<ComplexType, 1> wgt(nwalk);
    wgt() = ComplexType(1.0, 0.0);

    std::vector<full1rdm> props1;
    props1.emplace_back(mpi, info0, ptree{}, type, 1);
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
    // (1) inner_nwalkers invariance: a static replicated ensemble gives an inner_nwalkers-independent 1RDM.
    CHECK_THAT(rdm_s3, utils::Approx(rdm_s1));
    // (2) delegate limit: single-determinant trial => stochastic accumulated 1RDM == NOMSD.
    if (single_det)
      CHECK_THAT(rdm_s1, utils::Approx(rdm_ref));
    CHECK_THAT(trdm_s3, utils::Approx(trdm_s1));
    if (single_det)
      CHECK_THAT(trdm_s1, utils::Approx(trdm_ref));
  }
}

TEST_CASE("stochastic_accumulate_estimators_matches_nomsd", "[stochastic_wfn]")
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
  if constexpr (MEM != HOST_MEMORY)
    return; // Un-rotated full-G kernels are CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return; // Un-rotated full-G kernels support CLOSED (RHF) trials only.
    const double dt(0.01);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk = 11;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};

    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_fg");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_fg", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_fg", type, &ham, nwalk);

    auto build_stoch = [&](const std::string& name, int inner_nwalkers, int inner_nsteps) -> Wavefunction<MEM>& {
      ptree pt;
      pt.put("name", name);
      pt.put("system", "info0");
      pt.put("filename", wfn_file);
      mark_stochastic_wfn_input(pt);
      pt.put("inner_nwalkers", inner_nwalkers);
      pt.put("inner_nsteps", inner_nsteps);
      WfnFac.push(name, pt);
      auto& w = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(w, name, type, wlk_pt);
      return w;
    };
    auto& wfn_compact = build_stoch("wfn_stoch_fg0", 1, 0); // compact nd = 0 (delegates to NOMSD)
    auto& wfn_full    = build_stoch("wfn_stoch_fg1", 1, 1); // un-rotated full-G; NOT resampled

    struct WalkerEnergies
    {
      nda::array<ComplexType, 1> ov, e1, exx, ej;
    };
    auto collect_energies = [&](Wavefunction<MEM>& wfn) {
      auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
      wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_fg"));
      perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
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
      auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
      wset.resize(nwalk, WfnFac.getInitialGuess("wfn_nomsd_fg"));
      perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt);
      return nda::to_host(X);
    };
    auto X_comp = collect_vbias(wfn_compact);
    auto X_full = collect_vbias(wfn_full);
    CHECK_THAT(X_full, utils::Approx(X_comp));
  }
}

TEST_CASE("stochastic_full_g_matches_compact", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn un-rotated full-G vs compact at the anchor.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_full_g_matches_compact<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
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
  if constexpr (MEM != HOST_MEMORY)
    return; // Un-rotated full-G kernels are CPU-only today.
  else
  {
    const double dt(0.01);

    auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                   [](WALKER_TYPES t) { return t == CLOSED; });
    if (not env_opt)
      return;
    auto& env = *env_opt;

    const int nwalk = 11;

    StochasticWfnOptions stoch_opt;
    stoch_opt.inner_nwalkers = 4;
    stoch_opt.inner_nsteps   = 1;
    auto& wfn                = env.push_stochastic_wfn(mpi, "wfn_stoch_dyn", wfn_file, stoch_opt, nwalk);

    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_dyn");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);

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
      for (int w = 0; w < nwalk; ++w)
      {
        REQUIRE(std::isfinite(real(ov(w))));
        REQUIRE(std::isfinite(imag(ov(w))));
        REQUIRE(std::isfinite(real(e1(w))));
        REQUIRE(std::isfinite(real(exx(w))));
        REQUIRE(std::isfinite(real(ej(w))));
      }
      for (int w = 0; w < nwalk; ++w)
        for (int g = 0; g < X_h.extent(1); ++g)
          REQUIRE(std::isfinite(real(X_h(w, g))));
    }
  }
}

TEST_CASE("stochastic_dynamic_ensemble_smoke", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn dynamic free-projection ensemble smoke.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_dynamic_ensemble_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Persistent (tethered) inner pool under production conditioning + leapfrog. Instead of the
// reset-then-redraw of the Gaussian-shift conditioned path, the pool is kept across outer steps and
// re-equilibrated by a short independence-Metropolis MCMC (proposal ~ bare free projection; accept on
// |<psi|phi_w>|, targeting the exact conditioned distribution p_T(Y)*|<psi|phi_w>|). Drives several outer
// steps through all four hot-path overrides and asserts finiteness (values are stochastic, not fixed),
// that the slot-major nw*P pool layout is preserved across steps (persistence, not a resize to P or a
// collapse), and -- via the equil_steps = 0 leg -- that the zero-equilibration prime path is well-behaved.
// Regression for the default-off path is byte-identical inner_persistence = false (every other
// [stochastic_wfn] case).
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                      std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // conditioned inner sampling + un-rotated full-G kernels are CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    const double dt(0.01);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 7;
    const int inner_nwalkers = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    // Two persistent decks: (1) equil_steps = 1 + burn_in = 1 (a genuine short MCMC), and (2) equil_steps
    // = 0 (zero equilibration after prime -- pool reset to the anchor once, then frozen).
    auto run_persistent = [&](const std::string& name, int equil_steps, int burn_in) {
      WavefunctionFactory<MEM> WfnFac{};
      ptree pt;
      pt.put("name", name);
      pt.put("system", "info0");
      pt.put("filename", wfn_file);
      mark_stochastic_wfn_input(pt);
      pt.put("inner_nwalkers", inner_nwalkers);
      pt.put("inner_nsteps", 1);
      pt.put("inner_conditioning", true);
      pt.put("inner_leapfrog", true);
      pt.put("inner_persistence", true);
      pt.put("inner_equil_steps", equil_steps);
      pt.put("inner_pool_burn_in", burn_in);
      ptree inner_prop;
      inner_prop.put("timestep", 0.01);
      pt.put_child("inner_propagator", inner_prop);
      WfnFac.push(name, pt);
      auto& wfn = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, name, type, wlk_pt);

      auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
      wset.resize(nwalk, WfnFac.getInitialGuess(name));
      perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

      for (int step = 0; step < 4; ++step)
      {
        wfn.begin_inner_step(wset); // leapfrog: equilibrates the persistent pool + stores the old overlap
        memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
        wfn.vbias(wset, X, dt);
        wfn.Energy(wset);
        wfn.Log_Overlap(wset);
        // Persistence: the pool must stay in the slot-major nw*P conditioned layout across steps (it is
        // re-equilibrated in place, never resized to P or collapsed).
        REQUIRE(wfn.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
        nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
        wset.getProperty(OVLP, ov);
        wset.getProperty(E1_, e1);
        wset.getProperty(EXX_, exx);
        wset.getProperty(EJ_, ej);
        auto X_h = nda::to_host(X);
        for (int w = 0; w < nwalk; ++w)
        {
          REQUIRE(std::isfinite(real(ov(w))));
          REQUIRE(std::isfinite(imag(ov(w))));
          REQUIRE(std::isfinite(real(e1(w))));
          REQUIRE(std::isfinite(real(exx(w))));
          REQUIRE(std::isfinite(real(ej(w))));
        }
        for (int w = 0; w < nwalk; ++w)
          for (int g = 0; g < X_h.extent(1); ++g)
            REQUIRE(std::isfinite(real(X_h(w, g))));
      }
    };

    run_persistent("wfn_stoch_persist", 1, 1);
    run_persistent("wfn_stoch_persist_noequil", 0, 0);
  }
}

// Log-domain P-sample aggregation (inner_log_aggregate=true) on the persistent leapfrog path.
template<MEMORY_SPACE MEM>
void stochastic_log_aggregate_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                  std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return;
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    const double dt(0.01);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 7;
    const int inner_nwalkers = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_logagg");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    pt.put("inner_persistence", true);
    pt.put("inner_log_aggregate", true);
    pt.put("inner_equil_steps", 1);
    pt.put("inner_pool_burn_in", 0);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_logagg", pt);
    auto& wfn = WfnFac.getWavefunction(mpi, "wfn_stoch_logagg", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, "wfn_stoch_logagg", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_logagg"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    for (int step = 0; step < 4; ++step)
    {
      wfn.begin_inner_step(wset);
      memory::array<MEM, ComplexType, 2> X(nwalk, wfn.number_of_cholesky_vectors());
      wfn.vbias(wset, X, dt);
      wfn.Energy(wset);
      wfn.Log_Overlap(wset);
      REQUIRE(wfn.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
      nda::array<ComplexType, 1> ov(nwalk), e1(nwalk), exx(nwalk), ej(nwalk);
      wset.getProperty(OVLP, ov);
      wset.getProperty(E1_, e1);
      wset.getProperty(EXX_, exx);
      wset.getProperty(EJ_, ej);
      for (int w = 0; w < nwalk; ++w)
      {
        REQUIRE(std::isfinite(real(ov(w))));
        REQUIRE(std::isfinite(imag(ov(w))));
        REQUIRE(std::isfinite(real(e1(w))));
        REQUIRE(std::isfinite(real(exx(w))));
        REQUIRE(std::isfinite(real(ej(w))));
      }
    }
  }
}

TEST_CASE("stochastic_log_aggregate_smoke", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn inner_log_aggregate smoke.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_log_aggregate_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

TEST_CASE("stochastic_persistent_pool_smoke", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent conditioned/leapfrog inner pool smoke.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_smoke<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Regression: a driver's pre-loop initial-energy call (Energy/Log_Overlap called BEFORE the first
// begin_inner_step ever arms inner_step_pending_) with a single outer walker per rank (nwalk == 1). The
// freshly-initialized inner ensemble has size == inner_nwalkers_ (= P), which coincidentally already
// equals nwalk*P at nwalk == 1, so the conditioned_resample bootstrap size-check alone cannot tell an
// un-conditioned ensemble from a properly resampled one -- it must fall through to inner_pool_primed_.
// Without that guard this aborts ("leapfrog inner_cond_mag_ not sized") the first time SAFIRE reports the
// starting local energy on a single-walker-per-rank run (e.g. n_walkers_per_mpi_task=1).
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_nwalk1_bootstrap(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                 std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return;
  else
  {
    auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file,
                                                   [](WALKER_TYPES t) { return t == CLOSED; });
    if (not env_opt)
      return;
    auto& env = *env_opt;

    const int nwalk          = 1; // the bug's trigger: nwalk*P aliases the pre-resample ensemble size at P.
    const int inner_nwalkers = 4;

    StochasticWfnOptions stoch_opt;
    stoch_opt.inner_nwalkers     = inner_nwalkers;
    stoch_opt.inner_nsteps       = 1;
    stoch_opt.inner_conditioning = true;
    stoch_opt.inner_leapfrog     = true;
    stoch_opt.inner_persistence  = true;
    stoch_opt.inner_equil_steps  = 1;
    stoch_opt.inner_pool_burn_in = 0;
    auto& wfn                    = env.push_stochastic_wfn(mpi, "wfn_stoch_nwalk1", wfn_file, stoch_opt, nwalk);

    auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nwalk1");
    perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);

    // The driver's initial-energy report: Energy() called directly, with NO prior begin_inner_step.
    REQUIRE(wfn.stochastic_inner_ensemble_size() == inner_nwalkers); // pre-resample: still the P-sized anchor.
    wfn.Energy(wset);
    REQUIRE(wfn.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers); // now properly conditioned.
    nda::array<ComplexType, 1> e1(nwalk), exx(nwalk), ej(nwalk);
    wset.getProperty(E1_, e1);
    wset.getProperty(EXX_, exx);
    wset.getProperty(EJ_, ej);
    for (int w = 0; w < nwalk; ++w)
    {
      REQUIRE(std::isfinite(real(e1(w))));
      REQUIRE(std::isfinite(real(exx(w))));
      REQUIRE(std::isfinite(real(ej(w))));
    }
  }
}

TEST_CASE("stochastic_persistent_pool_nwalk1_bootstrap", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent/leapfrog pool: nwalk=1 pre-begin_inner_step bootstrap regression.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_nwalk1_bootstrap<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Outer population-control / inner-slot realignment by lineage permutation. A conditioned dynamic trial
// keeps a slot-major inner ensemble (q = ip*nwalk + w, block w conditioned on outer walker phi_w) OUTSIDE
// the outer walker buffer. An outer popControl that clones/shuffles walkers WITHOUT changing the per-rank
// count leaves block w attached to the OLD phi_w, so a reduction run after popControl but before the next
// begin_inner_step (e.g. accumulate_estimators when measure_interval == population_control_interval) would
// pair post-branch walkers with pre-branch blocks. The driver calls
// Wavefunction::permute_inner_blocks_after_pop right after popControl, which permutes the slot-major inner
// blocks (and inner_cond_mag_) by the SLOT_LINEAGE parent map so block w follows the walker now at w.
//
// EXACT check: condition a leapfrog ensemble on distinct walkers and record per-walker overlaps Ov_before.
// Then simulate a pop-control CLONE -- outer slot 0 becomes a copy of slot 3 (phi_0 := phi_3) with
// SLOT_LINEAGE(0) = 3, all other slots identity -- and permute. Because permutation REUSES the samples
// (no resample) and the next Log_Overlap does not resample (latch consumed, size unchanged), the overlaps
// are deterministic functions of (inner block, outer walker):
//   Ov_after[0] == Ov_before[3]  (block 3 moved to slot 0, paired with phi_0 == phi_3)
//   Ov_after[3] == Ov_before[3]  (slot 3 unchanged: block 3 vs phi_3)
//   Ov_after[w] == Ov_before[w]  (untouched slots: block + walker unchanged)
// This FAILS if the permutation is a no-op/incorrect (Ov_after[0] would be the stale block 0 vs phi_3).
// Linear overlaps are compared to avoid the log-branch ambiguity (see linear_overlap). Also pins the
// no-op contract for plain NOMSD.
template<MEMORY_SPACE MEM>
void stochastic_inner_permute_after_pop_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // conditioned inner sampling is CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    // The permutation is per-rank-local and rank-count-independent, so single-rank fully validates it.
    // The exact cross-slot overlap equalities below assume no cross-rank reduction mixing; cross-rank
    // realignment (the sentinel fallback path) is covered by the multi-rank suite run, not this equality.
    if (mpi->comm.size() != 1)
      return;
    const double dt(0.01);

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_nwalkers = 3;
    const int clone_src      = 3; // outer slot 3 is cloned into slot 0 below
    const int clone_dst      = 0;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};

    // Plain NOMSD -- the no-op target for the visitor.
    ptree nomsd_pt;
    nomsd_pt.put("name", "wfn_nomsd_pp");
    nomsd_pt.put("system", "info0");
    nomsd_pt.put("filename", wfn_file);
    WfnFac.push("wfn_nomsd_pp", nomsd_pt);
    auto& wfn_nomsd = WfnFac.getWavefunction(mpi, "wfn_nomsd_pp", type, &ham, nwalk);

    // Conditioned + leapfrog dynamic stochastic trial -> slot-major nwalk*P inner ensemble (exercises the
    // inner_cond_mag_ permutation too).
    ptree pt;
    pt.put("name", "wfn_stoch_pp");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_pp", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_pp", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_pp", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_pp"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown); // distinct outer walkers

    // Contract: the hook is a no-op for a non-stochastic wavefunction (callable, no crash/side effect).
    wfn_nomsd.permute_inner_blocks_after_pop(wset);

    // Condition the inner ensemble on the current distinct walkers (leapfrog begin_inner_step resamples
    // to nwalk*P), then record the per-walker effective overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before); // latch consumed by begin_inner_step -> reuses the conditioned ensemble

    // Simulate an outer popControl clone that preserves the per-rank count (the case the size-mismatch
    // guard in conditioned_resample does NOT catch): outer slot clone_dst becomes a copy of clone_src.
    {
      auto all = nda::range::all;
      auto SM  = wset.SlaterMatrices(Alpha);
      SM(clone_dst, all, all) = SM(clone_src, all, all);
    }
    // SLOT_LINEAGE parent map: clone_dst <- clone_src, every other slot identity.
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      lin(clone_dst) = ComplexType(double(clone_src), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    }

    // permute the inner blocks by the lineage map (no resample). Block clone_dst now holds clone_src's.
    wfn_s.permute_inner_blocks_after_pop(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // Reduce again -- latch consumed + size unchanged => NO resample, so this scores the PERMUTED ensemble
    // against the cloned walkers.
    nda::array<ComplexType, 1> ov_after(nwalk);
    wfn_s.Log_Overlap(wset, ov_after);

    // Exact block identity via linear overlaps (branch-independent).
    auto lin_before = linear_overlap(ov_before);
    auto lin_after  = linear_overlap(ov_after);
    // Block clone_src moved to slot clone_dst, paired with phi_clone_dst == phi_clone_src.
    CHECK_THAT(lin_after(clone_dst), utils::Approx(lin_before(clone_src)));
    // Slot clone_src itself is unchanged (identity parent, walker untouched).
    CHECK_THAT(lin_after(clone_src), utils::Approx(lin_before(clone_src)));
    // All untouched slots are unchanged (block + walker identical).
    for (int w = 0; w < nwalk; ++w)
      if (w != clone_dst)
        CHECK_THAT(lin_after(w), utils::Approx(lin_before(w)));
    (void)dt;
  }
}

TEST_CASE("stochastic_inner_permute_after_pop_control", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn permute_inner_blocks_after_pop realigns inner blocks after population control.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_inner_permute_after_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Cross-rank fallback for permute_inner_blocks_after_pop. A walker received from another rank during load
// balancing carries the SLOT_LINEAGE sentinel -1 (this rank holds no inner block for the slot it came
// from), so the realignment must REBUILD the conditioned ensemble with a fresh resample rather than
// permute. Validated synthetically at single rank (no MPI needed): contrast an identity lineage (permute
// reuses the ensemble -> overlaps unchanged) against a lineage with one sentinel (fallback resamples ->
// overlaps change). This pins the fallback branch deterministically.
template<MEMORY_SPACE MEM>
void stochastic_inner_permute_cross_rank_fallback(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // conditioned inner sampling is CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    if (mpi->comm.size() != 1)
      return; // synthetic single-rank sentinel check (real cross-rank moves need no special harness)

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_nwalkers = 3;
    const int sentinel_slot  = 2; // pretend outer slot 2 was imported from another rank
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_xr");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_xr", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_xr", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_xr", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_xr"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Condition on the current walkers and record the baseline overlaps (first reduction resamples to
    // the slot-major nwalk*P form and consumes the latch).
    wfn_s.begin_inner_step(wset);
    nda::array<ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    auto lin_before = linear_overlap(ov_before);

    auto set_lineage = [&](int sentinel) {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0); // identity
      if (sentinel >= 0)
        lin(sentinel) = ComplexType(-1.0, 0.0); // mark as foreign-rank arrival
      wset.setProperty(SLOT_LINEAGE, lin);
    };

    // (A) Identity lineage: permute is a no-op reuse, so the next reduction does NOT resample and the
    // overlaps are unchanged.
    set_lineage(-1);
    wfn_s.permute_inner_blocks_after_pop(wset);
    nda::array<ComplexType, 1> ov_id(nwalk);
    wfn_s.Log_Overlap(wset, ov_id);
    auto lin_id = linear_overlap(ov_id);
    for (int w = 0; w < nwalk; ++w)
      CHECK_THAT(lin_id(w), utils::Approx(lin_before(w)));

    // (B) One sentinel: the realignment must fall back to a full conditioned resample, so the next
    // reduction draws a fresh ensemble and the overlaps change.
    set_lineage(sentinel_slot);
    wfn_s.permute_inner_blocks_after_pop(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ov_fb(nwalk);
    wfn_s.Log_Overlap(wset, ov_fb);
    auto lin_fb = linear_overlap(ov_fb);
    double max_diff = 0.0;
    for (int w = 0; w < nwalk; ++w)
      max_diff = std::max(max_diff, std::abs(lin_fb(w) - lin_before(w)));
    REQUIRE(max_diff > 1e-6); // a fresh resample drew new fields -> ensemble (and overlaps) changed
  }
}

TEST_CASE("stochastic_inner_permute_cross_rank_fallback", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn permute_inner_blocks_after_pop resamples on a foreign-rank (sentinel) slot.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_inner_permute_cross_rank_fallback<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// -----------------------------------------------------------------------------------------------------
// Persistent inner pool x population-control permute coupling.
//
// Both features touch the same slot-major inner ensemble but were validated separately:
// permute_inner_blocks_after_pop was written for the reset-then-redraw path (where the pool is thrown
// away at the next begin_inner_step anyway), and the persistent pool was validated without an
// intervening popControl. In production they run back to back every pop step (AFQMCDriver: popControl ->
// permute_inner_blocks_after_pop -> accumulate_step), and the persistent pool must survive that permute.
// The two cases below pin the contract at both ends of the branch.
// -----------------------------------------------------------------------------------------------------

// Case 1 -- a SUCCESSFUL (local) permute realigns the PERSISTENT pool exactly, like the reset-then-redraw
// case, and the post-permute reduction reuses it (no re-prime, no resample). Same clone-3-into-0 exact
// overlap-identity check as stochastic_inner_permute_after_pop_control, but with inner_persistence = true
// so the block being moved is a tethered MCMC sample (and its leapfrog magnitude inner_cond_mag_ must move
// with it). If the persistent branch failed to permute the pool -- or spuriously re-primed/re-equilibrated
// it -- Ov_after[0] would not equal Ov_before[3].
template<MEMORY_SPACE MEM>
void stochastic_persistent_permute_after_pop_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // conditioned inner sampling is CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    if (mpi->comm.size() != 1)
      return; // per-rank-local permutation; the exact cross-slot equalities assume no cross-rank mixing.

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_nwalkers = 3;
    const int clone_src      = 3;
    const int clone_dst      = 0;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_pp_persist");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    pt.put("inner_persistence", true); // <-- the only difference from stochastic_inner_permute_after_pop_control
    pt.put("inner_equil_steps", 1);
    pt.put("inner_pool_burn_in", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_pp_persist", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_pp_persist", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_pp_persist", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_pp_persist"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Prime + equilibrate the persistent pool conditioned on the distinct walkers, then record overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before); // latch consumed by begin_inner_step -> reuses the primed pool

    // Simulate a count-preserving popControl clone: outer slot clone_dst becomes a copy of clone_src.
    {
      auto all = nda::range::all;
      auto SM  = wset.SlaterMatrices(Alpha);
      SM(clone_dst, all, all) = SM(clone_src, all, all);
    }
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      lin(clone_dst) = ComplexType(double(clone_src), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    }

    // Permute the PERSISTENT pool by the lineage map (no resample, no re-prime).
    wfn_s.permute_inner_blocks_after_pop(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // Score again -- latch consumed + size unchanged + pool still primed => NO resample, so this reads the
    // permuted persistent pool against the cloned walkers.
    nda::array<ComplexType, 1> ov_after(nwalk);
    wfn_s.Log_Overlap(wset, ov_after);

    auto lin_before = linear_overlap(ov_before);
    auto lin_after  = linear_overlap(ov_after);
    // Tethered block clone_src moved to slot clone_dst, paired with phi_clone_dst == phi_clone_src.
    CHECK_THAT(lin_after(clone_dst), utils::Approx(lin_before(clone_src)));
    CHECK_THAT(lin_after(clone_src), utils::Approx(lin_before(clone_src)));
    for (int w = 0; w < nwalk; ++w)
      if (w != clone_dst)
        CHECK_THAT(lin_after(w), utils::Approx(lin_before(w)));
  }
}

TEST_CASE("stochastic_persistent_permute_after_pop_control", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn permute_inner_blocks_after_pop realigns the PERSISTENT inner pool after pop control.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_permute_after_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// Case 2 -- the prime-flag semantics of the coupling: whether a successful local permute correctly KEEPS
// the pool primed (persist across the pop event) while
// the cross-rank fallback correctly INVALIDATES it (re-prime from the anchor). The two paths are made
// observable with inner_equil_steps = 0: a PERSIST does zero MCMC sweeps and leaves the pool byte-identical,
// whereas a RE-PRIME resets every slot to the anchor and runs inner_pool_burn_in sweeps -> a different pool.
//   (A) identity lineage  -> successful permute -> primed stays true -> next step persists -> overlaps UNCHANGED
//   (B) sentinel lineage  -> fallback          -> primed cleared     -> next step re-primes -> overlaps CHANGE
// This pins that persistence and population control are not silently resetting or freezing each other: the
// good samples survive a local branch, and only a genuinely foreign block forces the expensive re-prime.
template<MEMORY_SPACE MEM>
void stochastic_persistent_pool_survives_pop_control(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  if constexpr (MEM != HOST_MEMORY)
    return; // conditioned inner sampling is CPU-only today.
  else
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    if (type != CLOSED)
      return;
    if (mpi->comm.size() != 1)
      return; // synthetic single-rank sentinel check (real cross-rank moves need no special harness).

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown, 0}));

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("system", "info0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_nwalkers = 3;
    const int sentinel_slot  = 2;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_persist_pop");
    pt.put("system", "info0");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    pt.put("inner_persistence", true);
    // equil_steps = 0 makes the persist path a no-op on the pool (0 sweeps), so a PERSIST is observably
    // byte-identical and a RE-PRIME (reset-to-anchor + burn_in sweeps) is observably different.
    pt.put("inner_equil_steps", 0);
    pt.put("inner_pool_burn_in", 2);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_persist_pop", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_persist_pop", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_persist_pop", type, wlk_pt);

    auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
    wset.resize(nwalk, WfnFac.getInitialGuess("wfn_stoch_persist_pop"));
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

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
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ov0(nwalk);
    wfn_s.Log_Overlap(wset, ov0);
    auto lin0 = linear_overlap(ov0);

    // (A) A local branch (identity lineage): the permute keeps the pool primed. The next begin_inner_step
    // runs the persist path (0 equil sweeps), so the tethered samples carry across the pop event UNCHANGED.
    identity_lineage();
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ovA(nwalk);
    wfn_s.Log_Overlap(wset, ovA);
    auto linA = linear_overlap(ovA);
    for (int w = 0; w < nwalk; ++w)
      CHECK_THAT(linA(w), utils::Approx(lin0(w))); // persistence held: no re-prime, no re-equilibration

    // (B) A foreign-rank arrival (one sentinel): the permute clears the prime latch, so the next
    // begin_inner_step re-primes from the anchor (+ burn_in sweeps) -> a fresh pool -> overlaps change.
    sentinel_lineage();
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    nda::array<ComplexType, 1> ovB(nwalk);
    wfn_s.Log_Overlap(wset, ovB);
    auto linB = linear_overlap(ovB);
    double max_diff = 0.0;
    for (int w = 0; w < nwalk; ++w)
      max_diff = std::max(max_diff, std::abs(linB(w) - lin0(w)));
    REQUIRE(max_diff > 1e-6); // fallback re-primed the pool from the anchor -> ensemble (and overlaps) changed
  }
}

TEST_CASE("stochastic_persistent_pool_survives_pop_control", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent inner pool survives a local pop-control permute and re-primes on a foreign slot.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_survives_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// End-to-end propagator integration: build a real OUTER AFQMCBasePropagator (default hybrid) bound to
// the dynamic stochastic trial (inner_nsteps = 1) and run Propagate() steps. Drives the full hot path
// THROUGH the propagator (vbias -> vHS -> apply -> Log_Overlap), validating that the stochastic
// overrides plug into a real propagation step. Asserts the walkers stay finite.

} // namespace sfqmc
