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

// Consistency test for spin-polarized trials.
//
// A fully spin-polarized state (all electrons in the alpha sector, beta empty)
// can be represented in several ways. This test derives, from a collinear NOMSD
// trial, an "up-only NONCOLLINEAR" trial (used as the trusted reference) and one
// or more candidate polarized trials, propagates each a few steps with an
// identical random seed, and checks that energy and the mixed 1RDM agree.
//
// The reference and candidate describe the same alpha determinant and the
// auxiliary-field draws are dimensioned by the (shared) Hamiltonian, so with the
// same seed the alpha sector evolves identically across representations.
//
// The candidate representation is COLLINEAR with ndown=0 (empty beta sector),
// the target that is meant to replace the legacy FULLYPOLARIZED walker type.

#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"

#include "AFQMC/parameters.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "test_common.hpp"
#include "IO/app_loggers.h"

#include <nda/nda.hpp>
#include <nda/tensor.hpp>
#include <nda/h5.hpp>

#include <string>
#include <vector>
#include <complex>
#include <format>
#include <filesystem>

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Utilities/readWfn.h"
#include "numerics/sparse/sparse.hpp"
#include "test_utils.hpp"

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

namespace
{
using nda::range;

// Copy a CSR into a matrix with more columns, keeping (row, column) indices
// unchanged. Used to embed a collinear up block (nup x NMO) into the alpha
// columns [0,NMO) of a noncollinear block (nup x 2*NMO). combine_csr cannot be
// used here: with an empty down block it short-circuits and returns the up block
// verbatim with ncols = NMO (see src/numerics/sparse/csr_utils.hpp).
inline PsiT_Matrix<HOST_MEMORY> widen_csr(PsiT_Matrix<HOST_MEMORY> const& up, int ncols_out)
{
  long nrows = up.extent(0);
  utils::check(ncols_out >= up.extent(1), "widen_csr: ncols_out smaller than source");
  nda::array<int, 1> nnzpr(nrows);
  for(long r = 0; r < nrows; ++r) {
    nnzpr(r) = int(up.nnz(r));
  }
  PsiT_Matrix<HOST_MEMORY> out({nrows, long(ncols_out)}, nnzpr);
  auto ptrb = up.row_begin();
  auto ptre = up.row_end();
  auto cols = up.columns();
  auto vals = up.values();
  for(long r = 0; r < nrows; ++r) {
    for(long i = ptrb(r); i < ptre(r); ++i) {
      out.emplace_back({int(r), int(cols(i))}, vals(i));
    }
  }
  return out;
}

// Path for a derived trial file in the scratch dir.
inline std::string polarized_tmp_path(std::string const& tag, WALKER_TYPES target)
{
  auto p = std::filesystem::temp_directory_path() /
           std::format("polarized_{}_{}_{}.h5", tag, walkerTypeToString(target), getpid());
  return p.string();
}

// Derive a polarized trial file (dropping the down block) from a COLLINEAR NOMSD
// source and write it to out_path. Only the alpha (up) block is used.
//   NONCOLLINEAR -> up-only spinor: PsiT (nup, 2*NMO), beta columns empty.
//   COLLINEAR    -> up block + empty down block: ndown=0 (migration target).
inline void derive_polarized_wfn(std::string const& src_file, WALKER_TYPES target,
                                 std::string const& out_path)
{
  h5::file fin(src_file, 'r');
  h5::group nin = h5::group(fin).open_group("Wavefunction").open_group("NOMSD");

  std::vector<int> dims(5);
  h5::h5_read(nin, "dims", dims);
  int NMO = dims[0];
  int nup = dims[1];
  utils::check(initWALKER_TYPES(dims[3]) == COLLINEAR,
               "derive_polarized_wfn expects a COLLINEAR source, got {}",
               walkerTypeToString(initWALKER_TYPES(dims[3])));

  auto up = math::sparse::HDF2CSR<ComplexType, HOST_MEMORY, int, int>(nin.open_group("PsiT_0"));
  utils::check(up.extent(0) == nup && up.extent(1) == NMO,
               "unexpected up block shape ({}, {}); expected ({}, {})",
               up.extent(0), up.extent(1), nup, NMO);

  nda::matrix<ComplexType> psi0a(NMO, nup);
  psi0a() = ComplexType(0.0);
  utils::h5_read(nin, "Psi0_alpha", psi0a);

  h5::file fout(out_path, 'w');
  h5::group nout = h5::group(fout).create_group("Wavefunction").create_group("NOMSD");

  nda::array<ComplexType, 1> ci(1);
  ci(0) = ComplexType(1.0);

  nda::vector<int> d = {NMO, nup, 0, int(target), 1};
  nda::h5_write(nout, "dims", d);
  nda::h5_write(nout, "ci_coeffs", ci);

  if(target == NONCOLLINEAR) {
    // Alpha orbitals occupy the top NMO rows of the 2*NMO spinor.
    nda::matrix<ComplexType> psi0_nc(2 * NMO, nup);
    psi0_nc() = ComplexType(0.0);
    psi0_nc(range(NMO), range::all) = psi0a;
    nda::h5_write(nout, "Psi0_alpha", psi0_nc);

    auto up2 = widen_csr(up, 2 * NMO);
    h5::group g0 = nout.create_group("PsiT_0");
    math::sparse::CSR2HDF(g0, up2);
  } else if(target == COLLINEAR) {
    // ndown == 0: alpha as-is, empty beta block.
    nda::h5_write(nout, "Psi0_alpha", psi0a);
    nda::matrix<ComplexType> psi0b(NMO, 0);
    nda::h5_write(nout, "Psi0_beta", psi0b);

    h5::group g0 = nout.create_group("PsiT_0");
    math::sparse::CSR2HDF(g0, up);

    PsiT_Matrix<HOST_MEMORY> dn({0, long(NMO)}, nda::array<int, 1>(0));
    h5::group g1 = nout.create_group("PsiT_1");
    math::sparse::CSR2HDF(g1, dn);
  } else {
    utils::check(false, "derive_polarized_wfn: unsupported target {}", walkerTypeToString(target));
  }
}

// Result of a short propagation: per-walker energy components and the mixed 1RDM
// of walker 0, reshaped to (nspin, npol*NMO, npol*NMO) on the host.
struct run_result
{
  nda::array<ComplexType, 1> e1, ej, exx;
  nda::array<ComplexType, 3> G;
  int nspin = 0, npol = 0, NMO = 0;
};

template<MEMORY_SPACE MEM>
run_result run_polarized(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                         std::string const& hamil_file, std::string const& wfn_file,
                         int nsteps, int nStab)
{
  WALKER_TYPES type = afqmc::getWalkerType(wfn_file);
  auto [NMO, nup, ndown] = read_info_from_wfn(wfn_file, "any");
  int nspin = (type == COLLINEAR) ? 2 : 1;
  int npol  = (type == NONCOLLINEAR) ? 2 : 1;

  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();
  // Same seed for every representation so the auxiliary fields are identical.
  std::shared_ptr<utils::RandomGenerator_t<MEM>> rng_dev =
      std::make_shared<utils::RandomGenerator_t<MEM>>(777);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  auto& ham = HamFac.getHamiltonian(mpi, "ham0");

  int nwalk = 2;
  WavefunctionFactory<MEM> WfnFac{};
  WavefunctionParameters wfn_params{.name = "wfn0", .filename = wfn_file};
  apply_defaults(wfn_params, ham.getHamType());
  WfnFac.push("wfn0", wfn_params);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, false, &ham, nwalk);

  const WalkerSetParameters wlk_params{.name = "wset0", .walker_type = type};
  auto const& initial_guess = WfnFac.getInitialGuess("wfn0");
  auto wset = WalkerSet<MEM>(mpi, wlk_params, rng, type, initial_guess, nwalk);

  PropagatorFactory<MEM> PropFac;
  PropagatorParameters prop_params{.name = "prop0"};
  apply_defaults(prop_params, ham.getHamType());
  PropFac.push("prop0", prop_params);
  auto& prop = PropFac.getPropagator(mpi, "prop0", wfn, rng_dev);

  wfn.Log_Overlap(wset);
  wfn.runtime_optimization(wset);

  // No population control / weight reset: keep the walker mapping fixed so the
  // two representations remain directly comparable walker-by-walker.
  RealType Eshift = 0.0;
  RealType dt     = 0.01;
  for(int s = 0; s < nsteps; ++s) {
    prop.Propagate(wset, Eshift, dt);
    if((s + 1) % nStab == 0) {
      prop.Orthogonalize(wset);
    }
  }

  wfn.Energy(wset);

  run_result res;
  res.e1.resize(nwalk);
  res.ej.resize(nwalk);
  res.exx.resize(nwalk);
  wset.getProperty(E1_, res.e1);
  wset.getProperty(EJ_, res.ej);
  wset.getProperty(EXX_, res.exx);

  memory::array<MEM, ComplexType, 2> Gw(wset.size(), nspin * npol * NMO * npol * NMO);
  wfn.MixedDensityMatrix(wset, Gw, false);
  auto G0 = nda::reshape(Gw(0, nda::ellipsis{}),
                         std::array<long, 3>{nspin, npol * NMO, npol * NMO});
  res.G     = nda::make_regular(nda::to_host(G0));
  res.nspin = nspin;
  res.npol  = npol;
  res.NMO   = NMO;
  return res;
}

// Compare a candidate polarized run against the up-only noncollinear reference.
inline void compare_to_reference(std::string const& label,
                                 run_result const& cand, run_result const& ref)
{
  INFO("candidate: " << label);
  int NMO = ref.NMO;

  nda::array<ComplexType, 1> e_cand = cand.e1 + cand.ej + cand.exx;
  nda::array<ComplexType, 1> e_ref  = ref.e1 + ref.ej + ref.exx;
  CHECK_THAT(e_cand, utils::Approx(e_ref, 1e-7, 1e-7));
  CHECK_THAT(cand.e1, utils::Approx(ref.e1, 1e-7, 1e-7));
  CHECK_THAT(cand.ej, utils::Approx(ref.ej, 1e-7, 1e-7));
  CHECK_THAT(cand.exx, utils::Approx(ref.exx, 1e-7, 1e-7));

  // alpha-alpha block of the mixed 1RDM (spin 0, first NMO x NMO).
  auto cand_aa = nda::make_regular(cand.G(0, range(NMO), range(NMO)));
  auto ref_aa  = nda::make_regular(ref.G(0, range(NMO), range(NMO)));
  CHECK_THAT(cand_aa, utils::Approx(ref_aa, 1e-7, 1e-7));

  // The reference's beta-beta block must be (numerically) empty.
  auto ref_bb = nda::make_regular(ref.G(0, range(NMO, 2 * NMO), range(NMO, 2 * NMO)));
  nda::array<ComplexType, 2> zero(NMO, NMO);
  zero() = ComplexType(0.0);
  CHECK_THAT(ref_bb, utils::Approx(zero, 1e-7, 1e-7));
}

template<MEMORY_SPACE MEM>
void polarized_consistency(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                           std::string hamil_file, std::string src_wfn_file)
{
  // Only collinear NOMSD sources are meaningful here (we drop their down block).
  if(afqmc::getWalkerType(src_wfn_file, "any") != COLLINEAR) {
    return;
  }

  constexpr int nsteps = 6;
  constexpr int nStab  = 3;

  auto stem = [](std::string const& p) {
    auto s = p.find_last_of("\\/");
    auto e = p.find_last_of(".");
    return p.substr(s + 1, e - s - 1);
  }(src_wfn_file);

  // Reference: up-only noncollinear (well-supported path).
  std::string ref_file = polarized_tmp_path(stem, NONCOLLINEAR);
  if(mpi->comm.root()) {
    derive_polarized_wfn(src_wfn_file, NONCOLLINEAR, ref_file);
  }
  mpi->comm.barrier();
  auto ref = run_polarized<MEM>(mpi, hamil_file, ref_file, nsteps, nStab);

  // Candidate polarized representations to check against the reference.
  // COLLINEAR with ndown=0 is the target that replaces the legacy FULLYPOLARIZED.
  std::vector<WALKER_TYPES> candidates = {COLLINEAR};

  for(auto cand_type : candidates) {
    std::string cand_file = polarized_tmp_path(stem, cand_type);
    if(mpi->comm.root()) {
      derive_polarized_wfn(src_wfn_file, cand_type, cand_file);
    }
    mpi->comm.barrier();
    auto cand = run_polarized<MEM>(mpi, hamil_file, cand_file, nsteps, nStab);
    compare_to_reference(walkerTypeToString(cand_type), cand, ref);
    std::filesystem::remove(cand_file);
  }
}

} // namespace

TEST_CASE("polarized: consistency with up-only noncollinear", "[polarized]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool) {
    polarized_consistency<MEM>(mpi, hamil_file, wfn_file);
  }, UTEST_HAMIL, UTEST_WFN,
     TestFiles::UHF | TestFiles::NOMSD | TestFiles::MOLECULES | TestFiles::SOLIDS | TestFiles::LATTICES);
}

} // namespace sfqmc
