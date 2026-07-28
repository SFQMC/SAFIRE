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
  std::string inner_mcmc   = ""; // empty = input default ("pcn")
  double inner_mcmc_step   = 0.0; // <= 0 = kernel default
  bool inner_log_aggregate = false;
  double inner_prop_timestep = 0.01;
  int inner_measure_replicas = 1;  // nm; 1 => measure_energy IS Energy, byte-for-byte
  int inner_measure_stride   = 0;  // <= 0 = input default (inner_equil_steps)
  bool inner_measure_restore = true; // false => replica sweeps are LEFT in the propagation chain
};

ptree make_stochastic_wfn_ptree(std::string const& name, std::string const& wfn_file, StochasticWfnOptions const& opt = {})
{
  ptree pt;
  pt.put("name", name);
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
  if (not opt.inner_mcmc.empty())
    pt.put("inner_mcmc", opt.inner_mcmc);
  if (opt.inner_mcmc_step > 0.0)
    pt.put("inner_mcmc_step", opt.inner_mcmc_step);
  if (opt.inner_log_aggregate)
    pt.put("inner_log_aggregate", true);
  // Only emitted when non-default, so every pre-existing test's ptree is unchanged.
  if (opt.inner_measure_replicas != 1)
    pt.put("inner_measure_replicas", opt.inner_measure_replicas);
  if (opt.inner_measure_stride > 0)
    pt.put("inner_measure_stride", opt.inner_measure_stride);
  if (not opt.inner_measure_restore)
    pt.put("inner_measure_restore", false);
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
  pt.put("filename", wfn_file);
  return pt;
}

template<MEMORY_SPACE MEM>
struct StochasticHamWfnEnv
{
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

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    env.ham_fac = std::make_unique<HamiltonianFactory>();
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
    auto const& initial_guess = wfn_fac.getInitialGuess(wfn_id);
    return WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
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

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac;
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  ptree wlk_pt;
  wlk_pt.put("name", "wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  WavefunctionFactory<MEM> WfnFac{};
  ptree stoch_pt;
  stoch_pt.put("name", "wfn_stoch");
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
  memory::array<MEM, ComplexType, 1> Ov(nwalk);
  wfn_nomsd.MixedDensityMatrix(wset_at_guess, G, Ov, false);
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
  // PR-3: this off-anchor path is device-ported; runs on DEVICE_MEMORY too.
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
  // PR-3: the off-anchor Energy reduction is device-ported (row_accumulate / row_divide /
  // inner_scalar_reduce / elementwise_log kernels), so this runs on DEVICE_MEMORY too.
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
  // PR-3: this off-anchor path is device-ported; runs on DEVICE_MEMORY too.
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
  // PR-3: this off-anchor path is device-ported; runs on DEVICE_MEMORY too.
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
  // PR-3: this off-anchor path is device-ported; runs on DEVICE_MEMORY too.

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;
  const double dt(0.01);

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac;
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

  // Localize before asserting. inner_nsteps is unset here => 0 => the STATIC limit, where the inner
  // ensemble is P exact replicas of the anchor. So these are deterministic identities, not stochastic
  // estimates: any nonzero deviation is a real inner_nwalkers dependence, not sampling noise.
  auto amax = [](auto const& a, auto const& b) {
    double d = 0.0, s = 0.0;
    for (long i = 0; i < a.size(); ++i)
    {
      d = std::max(d, std::abs(a(i) - b(i)));
      s = std::max(s, std::abs(b(i)));
    }
    return std::make_pair(d, s);
  };
  {
    auto [d31, s31] = amax(v_s3, v_s1);
    auto [d1r, s1r] = amax(v_s1, v_ref);
    app_log(0, "  vMF static-limit: max|v_s3-v_s1| = {:.6e} (rel {:.3e})   max|v_s1-v_ref| = {:.6e} (rel {:.3e})",
            d31, s31 > 0 ? d31 / s31 : 0.0, d1r, s1r > 0 ? d1r / s1r : 0.0);
  }
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

TEST_CASE("stochastic_mean_field_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF mean-field delegate-limit parity.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

namespace
{
// Shared helpers for the mean-field + back-propagation reference tests: max |A-B| over two views (1D/2D),
// finiteness checks (1D/3D), and tr(G) (electron count -- a physical invariant, exact and noise-free
// regardless of the stochastic sampling). Defined before their first use (stochastic_mean_field_production_order).
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
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    const double dt(0.01);
    auto all = nda::range::all;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_mfp");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);

    // Reproduce the Propagate ordering: begin_inner_step (leapfrog => conditioned resample to nwalk*P)
    // BEFORE the mean-field calls. With the fix, vMF/G_MF see the non-P-sample (conditioned) ensemble and
    // draw a dedicated walker-independent free-projection ensemble to reduce the FULL trial's mean field.
    wfn_s.begin_inner_step(wset);

    // The leapfrog begin_inner_step must actually have expanded the ensemble to nwalk*P (otherwise the
    // scenario the fix handles is not exercised). This pins that the mean-field call happens while the
    // forward ensemble is in the conditioned form.
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

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
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

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

TEST_CASE("stochastic_mean_field_production_order", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn vMF / G_MF survive the begin_inner_step-before-generateP1 order.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_mean_field_production_order<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

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
  // PR-3: free-projection BP / accumulate is device-ported (getReferences + accumulate_estimators); runs on DEVICE_MEMORY.

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac;
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
    memory::array<MEM, ComplexType, 3> Refs;
    wfn.getReferences(Refs);
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
  // PR-3: conditioned + leapfrog inner sampling is device-ported; runs on DEVICE_MEMORY too.
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto get_refs = [&](Wavefunction<MEM>& wfn) {
      memory::array<MEM, ComplexType, 3> Refs;
      wfn.getReferences(Refs);
      return nda::to_host(Refs);
    };
    auto all = nda::range::all;
    auto R_anchor = get_refs(wfn_nomsd); // NOMSD delegate -> the single anchor reference

    // Drive the inner ensemble: leapfrog begin_inner_step resamples + expands it to nwalk*P.
    auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_bpp");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // The BP references are DECOUPLED from the (walker-conditioned, nwalk*P) forward ensemble: even for a
    // conditioned + leapfrog trial, back-propagation exposes a dedicated walker-INDEPENDENT
    // free-projection draw of the P trial samples at weight 1/P (dedicated reference draw), NOT the anchor.
    CHECK(wfn_s.total_number_of_references() == inner_nwalkers);
    for (int p = 0; p < inner_nwalkers; ++p)
      CHECK_THAT(wfn_s.getReferenceWeight(p), utils::Approx(ComplexType(1.0 / inner_nwalkers, 0.0)));

    auto R_draw1 = get_refs(wfn_s);
    CHECK(all_finite3d(R_draw1));
    for (int p = 0; p < inner_nwalkers; ++p) // each sample is propagated off the anchor (one B_T step)
      CHECK(max_abs_diff2d(R_draw1(p, all, all), R_anchor(0, all, all)) > 1e-6);

    // The dedicated draw left the forward ensemble at size P, but the next begin_inner_step re-expands it
    // to nwalk*P (the forward walk is unharmed by the BP reference draw).
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

    // Across windows: begin_inner_step opened a new BP window (resetting the idempotency guard), so
    // getReferences now draws a FRESH free-projection ensemble -- different from the previous window's.
    auto R_draw2 = get_refs(wfn_s);
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
  // PR-3: free-projection BP references are device-ported (getReferences); runs on DEVICE_MEMORY.
  {
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;

    const int P    = 3;

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto get_refs = [&](Wavefunction<MEM>& wfn) {
      memory::array<MEM, ComplexType, 3> Refs;
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
    // (cross-window freshness, after begin_inner_step, is checked in _production_order).
    auto R_draw2 = get_refs(wfn_s);
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
  // PR-3: free-projection BP / accumulate is device-ported (getReferences + accumulate_estimators); runs on DEVICE_MEMORY.

  const auto info   = read_info_from_wfn(wfn_file, "any");
  const int  NMO    = std::get<0>(info);
  const int  nup    = std::get<1>(info);
  const int  ndown  = std::get<2>(info);
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
  if (is_ft_walker_type(type))
    return;

  ptree ham_pt;
  ham_pt.put("name", "ham0");
  ham_pt.put("filename", hamil_file);
  HamiltonianFactory HamFac;
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
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    nda::array<ComplexType, 1> wgt(nwalk);
    wgt() = ComplexType(1.0, 0.0);

    std::vector<full1rdm> props1;
    props1.emplace_back(mpi, ptree{}, type, NMO, 1);
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
  // PR-3: the un-rotated full-G energy is device-ported; this runs on DEVICE_MEMORY too (free-projection,
  // ensemble held at the anchor, so full-G must reproduce the compact/NOMSD result).
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // Was `type != CLOSED` with the comment "full-G kernels support CLOSED (RHF) trials only" -- that
    // predated full_g::energy_collinear landing, and it meant this comparison (the ONLY thing that
    // validates an un-rotated full-G energy kernel against the compact/NOMSD reference) silently
    // skipped COLLINEAR. So energy_collinear shipped unvalidated, and it is the kernel every
    // broken-symmetry production panel runs through (N2-stretched, C2, Fe2S2). NONCOLLINEAR full-G
    // really is unimplemented -- Real3IndexFactorization::energy_fullG APP_ABORTs on it -- so that is
    // the only type still excluded.
    if (type == NONCOLLINEAR)
      return;
    const double dt(0.01);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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
      auto const& initial_guess = WfnFac.getInitialGuess("wfn_nomsd_fg");
      auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
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

// Isolates G from the energy kernels. stochastic_full_g_matches_compact shows full-G energies drifting
// from the compact reference on COLLINEAR fixtures (~2e-5 relative) while CLOSED holds to 1e-8, and the
// error scales 1 : 2 : 2 across E1 : EXX : EJ -- E1 is LINEAR in G and involves no Cholesky at all, EXX
// and EJ are QUADRATIC in G. That is the signature of G itself being off, not of the Cholesky or the
// one-body. This checks G directly.
//
// The gap this fills: stochastic_mixed_density_matrix_matches_nomsd already compares the non-compact DM
// against NOMSD for every walker type -- but only at inner_nsteps = 0 (the STATIC replicated ensemble,
// which delegates to NOMSD). The full-G energy path requires inner_nsteps > 0, whose DM is assembled by
// a different route (reduce_inner_cross_dm, [nwalk][nspin*NMO*NMO]). That assembly has never been
// compared to anything. Held at the anchor (begin_inner_step never called, inner_nwalkers = 1) the
// dynamic DM must equal NOMSD's exactly, so any deviation localizes the defect to the assembly.
template<MEMORY_SPACE MEM>
void stochastic_dynamic_full_g_dm_matches_nomsd(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi, std::string hamil_file,
    std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  auto env_opt = StochasticHamWfnEnv<MEM>::build(
      mpi, hamil_file, wfn_file, [](WALKER_TYPES t) { return !is_ft_walker_type(t); });
  if (not env_opt)
    return;
  auto& env = *env_opt;
  if (env.type == NONCOLLINEAR)
    return; // full-G aborts on NONCOLLINEAR by design

  const int nspin = (env.type == COLLINEAR ? 2 : 1);
  const int npol  = (env.type == NONCOLLINEAR ? 2 : 1);
  const int nwalk = 11;

  auto& wfn_nomsd = env.push_nomsd_wfn(mpi, "wfn_nomsd_dyndm", wfn_file, nwalk);

  StochasticWfnOptions opt;
  opt.inner_nwalkers = 1;
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
      perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
    memory::array<MEM, ComplexType, 2> G(nwalk, Gsize);
    memory::array<MEM, ComplexType, 1> Ov(nwalk);
    wfn.MixedDensityMatrix(wset, G, Ov, false); // never begin_inner_step => ensemble at the anchor
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

TEST_CASE("stochastic_dynamic_full_g_dm_matches_nomsd", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn dynamic (inner_nsteps>0) full-G mixed DM vs NOMSD at the anchor.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_dynamic_full_g_dm_matches_nomsd<MEM>(mpi, hamil_file, wfn_file);
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
// reset-then-redraw of the Gaussian-shift conditioned path, each (walker, p) slot owns a FIELD-space
// Markov chain (state = the auxiliary-field configuration Y behind psi = B_T(Y)|phi_T>, stored in the
// outer walker buffer's TrialFields block) re-equilibrated each step by Metropolis-Hastings sweeps
// targeting the conditioned distribution p_T(Y)*|<psi(Y)|phi_w>|. Drives several outer steps through
// all four hot-path overrides for BOTH proposal kernels (pcn and gaussian) and asserts finiteness
// (values are stochastic, not fixed), that the slot-major nw*P pool layout is preserved across steps
// (persistence, not a resize to P or a collapse), that the acceptance counters advance sanely, and --
// via the equil_steps = 0 leg -- that the zero-equilibration prime path is well-behaved. Regression for
// the default-off path is byte-identical inner_persistence = false (every other [stochastic_wfn] case).
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
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    const double dt(0.01);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 7;
    const int inner_nwalkers = 4;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    // Persistent decks: for each proposal kernel, (1) equil_steps = 1 + burn_in = 1 (a genuine short
    // MCMC), and (2) equil_steps = 0 (zero equilibration after prime -- chains drawn once, then frozen).
    auto run_persistent = [&](const std::string& name, int equil_steps, int burn_in,
                              const std::string& mcmc, double mcmc_step) {
      WavefunctionFactory<MEM> WfnFac{};
      ptree pt;
      pt.put("name", name);
      pt.put("filename", wfn_file);
      mark_stochastic_wfn_input(pt);
      pt.put("inner_nwalkers", inner_nwalkers);
      pt.put("inner_nsteps", 1);
      pt.put("inner_conditioning", true);
      pt.put("inner_leapfrog", true);
      pt.put("inner_persistence", true);
      pt.put("inner_equil_steps", equil_steps);
      pt.put("inner_pool_burn_in", burn_in);
      pt.put("inner_mcmc", mcmc);
      if (mcmc_step > 0.0)
        pt.put("inner_mcmc_step", mcmc_step);
      ptree inner_prop;
      inner_prop.put("timestep", 0.01);
      pt.put_child("inner_propagator", inner_prop);
      WfnFac.push(name, pt);
      auto& wfn = WfnFac.getWavefunction(mpi, name, type, &ham, nwalk);
      WfnFac.maybe_initialize_stochastic_inner_walkers(wfn, name, type, wlk_pt);

      auto const& initial_guess = WfnFac.getInitialGuess(name);
      auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
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
      // Acceptance bookkeeping: with equil/burn-in sweeps the chains proposed at least once and the
      // cumulative acceptance is a valid fraction; the frozen (equil = 0, burn-in = 0) leg makes no
      // proposals and reports the 1.0 sentinel.
      const double acc = wfn.stochastic_inner_chain_acceptance();
      REQUIRE(acc >= 0.0);
      REQUIRE(acc <= 1.0);
    };

    run_persistent("wfn_stoch_persist_pcn", 1, 1, "pcn", 0.5);
    run_persistent("wfn_stoch_persist_pcn_indep", 1, 0, "pcn", 1.0); // s = 1: independence redraw limit
    run_persistent("wfn_stoch_persist_gauss", 1, 1, "gaussian", 0.05);
    run_persistent("wfn_stoch_persist_noequil", 0, 0, "pcn", 0.5);
  }
}

// Log-domain P-sample aggregation (inner_log_aggregate=true) on the persistent leapfrog path.
template<MEMORY_SPACE MEM>
void stochastic_log_aggregate_smoke(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                  std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // PR-3: persistent / log-aggregate device-ported (host-RNG hybrid MCMC); runs on DEVICE_MEMORY.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    const double dt(0.01);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_logagg");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
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
// begin_inner_step, i.e. before the persistent chains exist) with a single outer walker per rank
// (nwalk == 1). The freshly-initialized inner ensemble has size == inner_nwalkers_ (= P), which
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
  // PR-3: conditioned + leapfrog inner sampling is device-ported; runs on DEVICE_MEMORY too.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    // The permutation is per-rank-local and rank-count-independent, so single-rank fully validates it.
    // The exact cross-slot overlap equalities below assume no cross-rank reduction mixing; cross-rank
    // realignment (the sentinel fallback path) is covered by the multi-rank suite run, not this equality.
    if (mpi->comm.size() != 1)
      return;
    const double dt(0.01);

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_pp");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown); // distinct outer walkers

    // Contract: the hook is a no-op for a non-stochastic wavefunction (callable, no crash/side effect).
    wfn_nomsd.permute_inner_blocks_after_pop(wset);

    // Condition the inner ensemble on the current distinct walkers (leapfrog begin_inner_step resamples
    // to nwalk*P), then record the per-walker effective overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    memory::buffered_array<MEM, ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before); // latch consumed by begin_inner_step -> reuses the conditioned ensemble

    // Simulate an outer popControl clone that preserves the per-rank count (the case the size-mismatch
    // guard in conditioned_resample does NOT catch): outer slot clone_dst becomes a copy of clone_src.
    // BOTH spin blocks: a real popControl clone copies the whole walker, and the exact block-identity
    // assertions below require phi_clone_dst == phi_clone_src. Copying Alpha alone leaves clone_dst with
    // its ORIGINAL beta, so for COLLINEAR the overlaps legitimately differ and the test fails for a reason
    // that has nothing to do with the permutation under test.
    {
      auto all = nda::range::all;
      auto SM  = wset.SlaterMatrices(Alpha);
      SM(clone_dst, all, all) = SM(clone_src, all, all);
      if (type == COLLINEAR)
      {
        auto SMb = wset.SlaterMatrices(Beta);
        SMb(clone_dst, all, all) = SMb(clone_src, all, all);
      }
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
    memory::buffered_array<MEM, ComplexType, 1> ov_after(nwalk);
    wfn_s.Log_Overlap(wset, ov_after);

    // Exact block identity via linear overlaps (branch-independent). ov_* are MEM arrays (Log_Overlap runs
    // on device); pull to host for the comparison.
    auto lin_before = linear_overlap(nda::to_host(ov_before));
    auto lin_after  = linear_overlap(nda::to_host(ov_after));
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
  // PR-3: conditioned + leapfrog inner sampling is device-ported; runs on DEVICE_MEMORY too.
  {
    const auto info  = read_info_from_wfn(wfn_file, "any");
    const int  NMO   = std::get<0>(info);
    const int  nup   = std::get<1>(info);
    const int  ndown = std::get<2>(info);
    WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "any");
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    if (mpi->comm.size() != 1)
      return; // synthetic single-rank sentinel check (real cross-rank moves need no special harness)

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_xr");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Condition on the current walkers and record the baseline overlaps (first reduction resamples to
    // the slot-major nwalk*P form and consumes the latch).
    wfn_s.begin_inner_step(wset);
    memory::buffered_array<MEM, ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    auto lin_before = linear_overlap(nda::to_host(ov_before));

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
    memory::buffered_array<MEM, ComplexType, 1> ov_id(nwalk);
    wfn_s.Log_Overlap(wset, ov_id);
    auto lin_id = linear_overlap(nda::to_host(ov_id));
    for (int w = 0; w < nwalk; ++w)
      CHECK_THAT(lin_id(w), utils::Approx(lin_before(w)));

    // (B) One sentinel: the realignment must fall back to a full conditioned resample, so the next
    // reduction draws a fresh ensemble and the overlaps change.
    set_lineage(sentinel_slot);
    wfn_s.permute_inner_blocks_after_pop(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    memory::buffered_array<MEM, ComplexType, 1> ov_fb(nwalk);
    wfn_s.Log_Overlap(wset, ov_fb);
    auto lin_fb = linear_overlap(nda::to_host(ov_fb));
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

// Case 1 -- pop control transports the PERSISTENT chains with their walkers, and the post-pop hook
// rebuilds the pool determinants from the transported fields exactly. Same clone-3-into-0 exact
// overlap-identity check as stochastic_inner_permute_after_pop_control, but with inner_persistence =
// true: the chain state (fields) lives inside the outer walker buffer, so the clone is simulated by
// copying the walker's Slater matrix AND its TrialFields row (production branch() copies the whole
// buffer row -- see stochastic_branch_lineage_metadata for the row-copy contract). The rebuild is
// deterministic, so Ov_after[0] must equal Ov_before[3] exactly; a spurious re-prime, a chain restart,
// or dets rebuilt from the wrong slot's fields all break the identity.
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
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    if (mpi->comm.size() != 1)
      return; // per-rank-local permutation; the exact cross-slot equalities assume no cross-rank mixing.

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_pp_persist");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // Prime + equilibrate the persistent pool conditioned on the distinct walkers, then record overlaps.
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    memory::buffered_array<MEM, ComplexType, 1> ov_before(nwalk);
    wfn_s.Log_Overlap(wset, ov_before); // latch consumed by begin_inner_step -> reuses the primed pool

    // Simulate a count-preserving popControl clone: outer slot clone_dst becomes a copy of clone_src.
    // Production branch() clones the ENTIRE walker_buffer row, so the copy includes the walker's chain
    // fields (TrialFields row) alongside its Slater matrix -- and, for COLLINEAR, BOTH spin blocks (see
    // the same note in stochastic_inner_permute_after_pop_control); the lineage scalar is set for
    // completeness (the persistent path rebuilds from the fields and does not consume it).
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
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);

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

// Case 2 -- chain-transport semantics of the coupling: the pool determinants always follow the chain
// FIELDS stored in the outer walker buffer, with no chain restart in either direction. Made observable
// with inner_equil_steps = 0 (zero sweeps per step, so the only thing that can change the pool is the
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
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    if (mpi->comm.size() != 1)
      return; // synthetic single-rank sentinel check (real cross-rank moves need no special harness).

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
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
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    pt.put("inner_persistence", true);
    // equil_steps = 0 freezes the chains after the prime (+ burn-in), so the only thing that can change
    // the pool across the pop event is the post-pop rebuild itself -- transported fields => identical
    // determinants; replaced fields => different determinants at exactly that slot.
    pt.put("inner_equil_steps", 0);
    pt.put("inner_pool_burn_in", 2);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_persist_pop", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_persist_pop", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_persist_pop", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_persist_pop");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
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
    memory::buffered_array<MEM, ComplexType, 1> ov0(nwalk);
    wfn_s.Log_Overlap(wset, ov0);
    auto lin0 = linear_overlap(nda::to_host(ov0));

    // (A) A local branch (identity lineage, fields untouched): the post-pop rebuild reproduces the same
    // pool from the same fields, and the next begin_inner_step (0 equil sweeps) leaves it alone -- the
    // tethered samples carry across the pop event UNCHANGED.
    identity_lineage();
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
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
    wfn_s.permute_inner_blocks_after_pop(wset);
    wfn_s.begin_inner_step(wset);
    REQUIRE(wfn_s.stochastic_inner_ensemble_size() == long(nwalk) * inner_nwalkers);
    memory::buffered_array<MEM, ComplexType, 1> ovB(nwalk);
    wfn_s.Log_Overlap(wset, ovB);
    auto linB = linear_overlap(nda::to_host(ovB));
    REQUIRE(std::abs(linB(sentinel_slot) - lin0(sentinel_slot)) > 1e-6); // dets follow the new fields
    for (int w = 0; w < nwalk; ++w)
      if (w != sentinel_slot)
        CHECK_THAT(linB(w), utils::Approx(lin0(w))); // every other chain carried exactly
  }
}

TEST_CASE("stochastic_persistent_pool_survives_pop_control", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn persistent chains survive pop control; a replaced chain rebuilds its own slot only.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_pool_survives_pop_control<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
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
    // See the COLLINEAR-coverage note in stochastic_full_g_matches_compact: the dynamic inner path
    // runs the full-G kernels, which implement CLOSED and COLLINEAR and abort only on NONCOLLINEAR.
    // This read `type != CLOSED` with no stated reason, leaving COLLINEAR unexercised.
    if (type == NONCOLLINEAR)
      return;
    if (mpi->comm.size() != 1)
      return; // manipulates SLOT_LINEAGE directly; the identity-lineage permute assumes no cross-rank mixing.

    ptree ham_pt;
    ham_pt.put("name", "ham0");
    ham_pt.put("filename", hamil_file);
    HamiltonianFactory HamFac;
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

    const int nwalk          = 6;
    const int inner_nwalkers = 3;
    std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
    ptree wlk_pt;
    wlk_pt.put("name", "wset0");
    wlk_pt.put("walker_type", walkerTypeToString(type));

    WavefunctionFactory<MEM> WfnFac{};
    ptree pt;
    pt.put("name", "wfn_stoch_condmag");
    pt.put("filename", wfn_file);
    mark_stochastic_wfn_input(pt);
    pt.put("inner_nwalkers", inner_nwalkers);
    pt.put("inner_nsteps", 1);
    pt.put("inner_conditioning", true);
    pt.put("inner_leapfrog", true);
    pt.put("inner_persistence", true);
    pt.put("inner_equil_steps", 1);
    pt.put("inner_pool_burn_in", 1);
    ptree inner_prop;
    inner_prop.put("timestep", 0.01);
    pt.put_child("inner_propagator", inner_prop);
    WfnFac.push("wfn_stoch_condmag", pt);
    auto& wfn_s = WfnFac.getWavefunction(mpi, "wfn_stoch_condmag", type, &ham, nwalk);
    WfnFac.maybe_initialize_stochastic_inner_walkers(wfn_s, "wfn_stoch_condmag", type, wlk_pt);

    auto const& initial_guess = WfnFac.getInitialGuess("wfn_stoch_condmag");
    auto wset = WalkerSet<MEM>(mpi, wlk_pt, rng, type, initial_guess, nwalk);
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown); // distinct walkers => distinct cond_mag

    // Equilibrate the chains against the CURRENT walkers phi_cond and set inner_cond_mag_ (leapfrog).
    wfn_s.begin_inner_step(wset);
    const double sum_cond = wfn_s.stochastic_inner_cond_mag_sum();
    REQUIRE(sum_cond > 0.0); // magnitudes were actually set

    // Move the walkers (mimic the propagation that runs before pop control) WITHOUT re-equilibrating
    // the chains -- the leapfrog ensemble and its phi_cond-referenced magnitudes must stay fixed.
    perturb_stochastic_walkers<MEM>(wset, type, NMO, nup, ndown);

    // A count-preserving local pop event: identity lineage (no clone, no cross-rank arrival).
    {
      nda::array<ComplexType, 1> lin(nwalk);
      for (int w = 0; w < nwalk; ++w)
        lin(w) = ComplexType(double(w), 0.0);
      wset.setProperty(SLOT_LINEAGE, lin);
    }
    wfn_s.permute_inner_blocks_after_pop(wset);

    // The magnitudes still reference phi_cond, so an identity re-indexing leaves their sum EXACTLY
    // unchanged. A recompute against the moved walker would change it (the fixed bug).
    CHECK_THAT(wfn_s.stochastic_inner_cond_mag_sum(), utils::Approx(sum_cond));
  }
}

TEST_CASE("stochastic_persistent_cond_mag_invariant_under_permute", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn leapfrog inner_cond_mag_ is permuted (not recomputed) across a pop-control realignment.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_persistent_cond_mag_invariant_under_permute<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// End-to-end propagator integration: build a real OUTER AFQMCBasePropagator (default hybrid) bound to
// the dynamic stochastic trial (inner_nsteps = 1) and run Propagate() steps. Drives the full hot path
// THROUGH the propagator (vbias -> vHS -> apply -> Log_Overlap), validating that the stochastic
// overrides plug into a real propagation step. Asserts the walkers stay finite.

// At inner_measure_replicas == 1, measure_energy must be exactly Energy (same code path, exact equality).
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_nm1_matches_energy(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
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
  const int inner_nwalkers = 4;

  StochasticWfnOptions opt;
  opt.inner_nwalkers     = inner_nwalkers;
  opt.inner_nsteps       = 1;
  opt.inner_conditioning = true;
  opt.inner_leapfrog     = true;
  opt.inner_persistence  = true;
  opt.inner_equil_steps  = 1;
  opt.inner_pool_burn_in = 0;
  // nm left at its default of 1 -- that IS the case under test.
  auto& wfn = env.push_stochastic_wfn(mpi, "wfn_stoch_nm1", wfn_file, opt, nwalk);

  auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm1");
  perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
  wfn.begin_inner_step(wset); // prime the chains, so only nm == 1 keeps the replica path off
  REQUIRE_FALSE(wfn.stochastic_measure_replicas_active());

  nda::array<ComplexType, 2> E_m(nwalk, 3), E_e(nwalk, 3);
  nda::array<ComplexType, 1> Ov_m(nwalk), Ov_e(nwalk);
  wfn.measure_energy(wset, E_m, Ov_m);
  wfn.Energy(wset, E_e, Ov_e);

  for (int w = 0; w < nwalk; ++w)
  {
    REQUIRE(real(Ov_m(w)) == real(Ov_e(w)));
    REQUIRE(imag(Ov_m(w)) == imag(Ov_e(w)));
    for (int k = 0; k < 3; ++k)
    {
      REQUIRE(real(E_m(w, k)) == real(E_e(w, k)));
      REQUIRE(imag(E_m(w, k)) == imag(E_e(w, k)));
    }
  }
}

TEST_CASE("stochastic_measure_replicas_nm1_matches_energy", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn measure_energy at nm=1 is exactly Energy (no existing result perturbed).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_nm1_matches_energy<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// At inner_measure_replicas > 1, measure_energy must return the walker's stored log overlap so
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
  const int inner_nwalkers = 4;

  StochasticWfnOptions opt;
  opt.inner_nwalkers         = inner_nwalkers;
  opt.inner_nsteps           = 1;
  opt.inner_conditioning     = true;
  opt.inner_leapfrog         = true;
  opt.inner_persistence      = true;
  opt.inner_equil_steps      = 1;
  opt.inner_pool_burn_in     = 0;
  opt.inner_measure_replicas = 4;
  opt.inner_measure_stride   = 1;
  auto& wfn = env.push_stochastic_wfn(mpi, "wfn_stoch_nm4", wfn_file, opt, nwalk);

  auto wset = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm4");
  perturb_stochastic_walkers<MEM>(wset, env.type, env.NMO, env.nup, env.ndown);
  wfn.begin_inner_step(wset);
  // Without this the test would pass vacuously through the plain-Energy fallback, where the invariant
  // holds trivially and the fixed bug could return unnoticed.
  REQUIRE(wfn.stochastic_measure_replicas_active());

  nda::array<ComplexType, 1> ovlp_before(nwalk);
  wset.getProperty(OVLP, ovlp_before);

  nda::array<ComplexType, 2> E(nwalk, 3);
  nda::array<ComplexType, 1> Ov(nwalk);
  wfn.measure_energy(wset, E, Ov);

  nda::array<ComplexType, 1> ovlp_after(nwalk);
  wset.getProperty(OVLP, ovlp_after);

  for (int w = 0; w < nwalk; ++w)
  {
    // The returned Ov IS the stored OVLP: this is deno_real == 1 exactly, by construction.
    REQUIRE(real(Ov(w)) == real(ovlp_before(w)));
    REQUIRE(imag(Ov(w)) == imag(ovlp_before(w)));
    // ...and measuring left the stored property alone, so the next step's reweight is inert too.
    REQUIRE(real(ovlp_after(w)) == real(ovlp_before(w)));
    REQUIRE(imag(ovlp_after(w)) == imag(ovlp_before(w)));
    // The replica-averaged energy is still a usable number (the averaging ran, nothing overflowed).
    for (int k = 0; k < 3; ++k)
      REQUIRE(std::isfinite(real(E(w, k))));
  }
}

TEST_CASE("stochastic_measure_replicas_ovlp_is_stored", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn measure_energy at nm>1 returns the STORED OVLP (deno_real == 1 invariant).");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_ovlp_is_stored<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::ALL_SYSTEMS);
}

// ----------------------------------------------------------------------------
// nm ENERGY correctness (the two tests above cover the nm == 1 identity and the OVLP invariant, both of
// which hold even if the returned energy is wrong). nm is documented as a pure VARIANCE knob, so the
// claim under test is arithmetic, not statistical: measure_energy must return exactly the unweighted mean
// of nm consecutive replica measurements, and (with restore on) must hand the propagation chain back
// untouched. Both are checkable deterministically, which is worth more here than a 2-sigma band on a
// 3-walker fixture -- a statistical check on this fixture would be too loose to catch a 1/nm scale error
// and too tight to run without flaking.
//
// What these two do NOT cover, stated plainly so it is not mistaken for full coverage: that each replica's
// own Energy is an unbiased estimate of the trial energy (that is Energy(), covered upstream by
// stochastic_mean_field_matches_nomsd), and the nm error-scaling claim (~1/sqrt(nm)), which is a
// production-scale measurement, not a unit test.
// ----------------------------------------------------------------------------

// The nm-replica average must be the true unweighted mean of nm CONSECUTIVE replicas. Verified without a
// reference energy: with restore OFF the pool motion is cumulative across calls, so two nm=2 measurements
// visit exactly the same four pool states, in the same order, as one nm=4 measurement -- every
// StochasticWfn seeds its own inner RNG from inner_seed (default 777), so the two wavefunctions get
// identical, non-interleaved streams, and Energy() itself draws nothing. Hence
// (E_call1 + E_call2) / 2 == E_nm4 up to floating-point reassociation alone.
//
// Catches, as a hard failure: a wrong divisor (1/nm), an accumulator not zeroed on entry, an off-by-one in
// the advance-per-replica loop (2+2 advances would stop lining up with 4), and any host/device
// accumulation mismatch (row_accumulate/row_divide vs the host loop), since each MEM runs separately.
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_average_is_true_mean(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                     std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  // CLOSED *and* COLLINEAR: both checks are reference-free (SAFIRE against itself), so neither is
  // affected by the Psi0 != PsiT fixtures that tripped the retracted COLLINEAR full-G report.
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file, [](WALKER_TYPES t) {
    return t == CLOSED || t == COLLINEAR;
  });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 3;

  StochasticWfnOptions opt;
  opt.inner_nwalkers         = 4;
  opt.inner_nsteps           = 1;
  opt.inner_conditioning     = true;
  opt.inner_leapfrog         = true;
  opt.inner_persistence      = true;
  opt.inner_equil_steps      = 1;
  opt.inner_pool_burn_in     = 0;
  opt.inner_measure_stride   = 1;
  // Restore OFF is what makes the two arms comparable: the pool must carry over between the split calls.
  opt.inner_measure_restore  = false;

  auto measure = [&](Wavefunction<MEM>& wfn, WalkerSet<MEM>& wset, nda::array<ComplexType, 2>& E) {
    nda::array<ComplexType, 1> Ov(nwalk);
    E() = ComplexType(0.0); // a caller-side zero, so a missing zero INSIDE measure_energy still shows up
    wfn.measure_energy(wset, E, Ov);
  };

  // Arm 1: nm = 2, measured twice. Four advances total, in two batches.
  StochasticWfnOptions opt2 = opt;
  opt2.inner_measure_replicas = 2;
  auto& wfn2  = env.push_stochastic_wfn(mpi, "wfn_stoch_nm2_split", wfn_file, opt2, nwalk);
  auto wset2  = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm2_split");
  perturb_stochastic_walkers<MEM>(wset2, env.type, env.NMO, env.nup, env.ndown);
  wfn2.begin_inner_step(wset2);
  REQUIRE(wfn2.stochastic_measure_replicas_active());
  nda::array<ComplexType, 2> E_a(nwalk, 3), E_b(nwalk, 3);
  measure(wfn2, wset2, E_a);
  measure(wfn2, wset2, E_b);

  // Arm 2: nm = 4, measured once. The same four advances, in one batch.
  StochasticWfnOptions opt4 = opt;
  opt4.inner_measure_replicas = 4;
  auto& wfn4  = env.push_stochastic_wfn(mpi, "wfn_stoch_nm4_whole", wfn_file, opt4, nwalk);
  auto wset4  = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_nm4_whole");
  perturb_stochastic_walkers<MEM>(wset4, env.type, env.NMO, env.nup, env.ndown);
  wfn4.begin_inner_step(wset4);
  REQUIRE(wfn4.stochastic_measure_replicas_active());
  nda::array<ComplexType, 2> E_c(nwalk, 3);
  measure(wfn4, wset4, E_c);

  // Only reassociation separates ((a+b)+c)+d)/4 from ((a+b)/2 + (c+d)/2)/2, so this is tight on purpose:
  // every failure mode above is O(1) in the energy, not O(eps).
  const double tol = 1e-10;
  double dev = 0.0, scale = 1.0;
  for (int w = 0; w < nwalk; ++w)
    for (int k = 0; k < 3; ++k)
    {
      const ComplexType split = 0.5 * (E_a(w, k) + E_b(w, k));
      dev   = std::max(dev, std::abs(split - E_c(w, k)));
      scale = std::max(scale, std::abs(E_c(w, k)));
    }
  app_log(0, "  nm split-vs-whole: max|mean(E_nm2 x2) - E_nm4| = {:e} (scale {:.3f}, tol {:e})", dev, scale,
          tol * scale);
  REQUIRE(dev <= tol * scale);
}

TEST_CASE("stochastic_measure_replicas_average_is_true_mean", "[stochastic_wfn]")
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
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::MOLECULES);
}

// With inner_measure_restore on, a measurement must be invisible to the propagation chain -- otherwise nm
// is a BIAS knob (it would silently add nm*stride sweeps of pool relaxation per measured step) rather than
// the variance knob it is documented as, and every nm curve would be measuring a different trial than the
// nm=1 curve it is compared against. restore_chain_fields claims to put the pool back "bit for bit"
// (fields restored, determinants rebuilt deterministically from them), so this is asserted EXACTLY: two
// wavefunctions identical but for nm, same inner_seed, same walkers. One measures nothing, the other burns
// four replicas and restores. A subsequent plain Energy() must agree to the last bit.
template<MEMORY_SPACE MEM>
void stochastic_measure_replicas_restore_is_invisible(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                                     std::string hamil_file, std::string wfn_file)
{
  if (getWavefunctionType(wfn_file) != NOMSD_WFN)
    return;
  auto env_opt = StochasticHamWfnEnv<MEM>::build(mpi, hamil_file, wfn_file, [](WALKER_TYPES t) {
    return t == CLOSED || t == COLLINEAR;
  });
  if (not env_opt)
    return;
  auto& env = *env_opt;

  const int nwalk = 3;

  StochasticWfnOptions opt;
  opt.inner_nwalkers       = 4;
  opt.inner_nsteps         = 1;
  opt.inner_conditioning   = true;
  opt.inner_leapfrog       = true;
  opt.inner_persistence    = true;
  opt.inner_equil_steps    = 1;
  opt.inner_pool_burn_in   = 0;
  opt.inner_measure_stride = 1;

  // Control: nm = 1, so measure_replicas_active() is false and the pool is never advanced.
  auto& wfn_ref = env.push_stochastic_wfn(mpi, "wfn_stoch_restore_ref", wfn_file, opt, nwalk);
  auto wset_ref = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_restore_ref");
  perturb_stochastic_walkers<MEM>(wset_ref, env.type, env.NMO, env.nup, env.ndown);
  wfn_ref.begin_inner_step(wset_ref);
  REQUIRE_FALSE(wfn_ref.stochastic_measure_replicas_active());
  nda::array<ComplexType, 2> E_ref(nwalk, 3);
  nda::array<ComplexType, 1> Ov_ref(nwalk);
  wfn_ref.Energy(wset_ref, E_ref, Ov_ref);

  // Test: nm = 4 with restore on. Four replica advances, then the pool is put back.
  StochasticWfnOptions opt4  = opt;
  opt4.inner_measure_replicas = 4;
  opt4.inner_measure_restore  = true; // the default; stated because it IS the property under test
  auto& wfn_rst = env.push_stochastic_wfn(mpi, "wfn_stoch_restore_nm4", wfn_file, opt4, nwalk);
  auto wset_rst = env.make_resized_walker_set(mpi, nwalk, "wfn_stoch_restore_nm4");
  perturb_stochastic_walkers<MEM>(wset_rst, env.type, env.NMO, env.nup, env.ndown);
  wfn_rst.begin_inner_step(wset_rst);
  REQUIRE(wfn_rst.stochastic_measure_replicas_active());

  nda::array<ComplexType, 2> E_burn(nwalk, 3);
  nda::array<ComplexType, 1> Ov_burn(nwalk);
  wfn_rst.measure_energy(wset_rst, E_burn, Ov_burn); // result discarded; the point is the side effect

  nda::array<ComplexType, 2> E_after(nwalk, 3);
  nda::array<ComplexType, 1> Ov_after(nwalk);
  wfn_rst.Energy(wset_rst, E_after, Ov_after);

  double dev = 0.0;
  for (int w = 0; w < nwalk; ++w)
  {
    dev = std::max(dev, std::abs(Ov_after(w) - Ov_ref(w)));
    for (int k = 0; k < 3; ++k)
      dev = std::max(dev, std::abs(E_after(w, k) - E_ref(w, k)));
  }
  app_log(0, "  nm restore invisibility: max|E_after_restore - E_never_measured| = {:e} (expect 0)", dev);
  REQUIRE(dev == 0.0);
}

TEST_CASE("stochastic_measure_replicas_restore_is_invisible", "[stochastic_wfn]")
{
  auto& mpi = utils::make_unit_test_mpi_context();
  app_log(0, "StochasticWfn nm>1 with inner_measure_restore leaves the propagation chain bit-identical.");
  using namespace utils;
  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool finiteT) {
    stochastic_measure_replicas_restore_is_invisible<MEM>(mpi, hamil_file, wfn_file);
    // MOLECULES, not ALL_SYSTEMS: the dynamic (inner_nsteps > 0) trial these replicas need calls
    // energy_fullG, which is implemented ONLY for Real3IndexFactorization. THCOps, KP3IndexFactorization
    // and ModelHamOps all APP_ABORT with "energy_fullG not implemented", so the solid and lattice
    // fixtures cannot run this test at all -- a capability limit, not a defect, and the same one the
    // header comment records for full-G parity. Excluding them keeps CLOSED + COLLINEAR coverage on BH
    // while leaving this gate clean, so a failure here means the nm estimator moved and nothing else.
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::NOMSD | TestFiles::MOLECULES);
}

} // namespace sfqmc
