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

//#undef NDEBUG

#include "catch2/catch_test_macros.hpp"

#include "config.h"
#include "AFQMC/config.h"

#include <string>
#include <complex>

#include "IO/app_loggers.h"
#include "test_common.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "test_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Utilities/readWfn.h"

using std::complex;
using std::ifstream;
using std::string;


extern std::string UTEST_HAMIL, UTEST_WFN;
namespace sfqmc
{
using namespace afqmc;

// Fill a physical mixed density matrix G (walker == trial determinant) from PsiT,
// laid out as [1][ nel * npol * NMO ] to match HamiltonianOperations::energy. G must be
// pre-sized by the caller; this zeroes and fills it.
template<class PsiTArray>
void fill_physical_G(nda::array<ComplexType, 2>& G, PsiTArray const& PsiT,
                     WALKER_TYPES walker_type, int NMO)
{
  long const nspin = (walker_type == COLLINEAR) ? 2 : 1;
  int const  npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
  G() = ComplexType(0.0);
  int offset = 0;
  for(int is = 0; is < nspin; ++is) {
    const auto& pt = PsiT(0, is);
    for(long a = 0; a < pt.shape(0); ++a) {
      auto row  = pt[a];
      auto cols = row.columns();
      auto vals = row.values();
      for(long k = 0; k < row.nnz(); ++k) {
        G(0, (offset + a) * NMO * npol + cols(k)) = std::conj(vals(k));
      }
    }
    offset += static_cast<int>(pt.shape(0));
  }
}

void thc_vs_chol_energy_agreement(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
    std::string chol_file, std::string thc_file, std::string wfn_file)
{
  using nda::range;
  auto all = range::all;

  utils::check(utils::file_exists(chol_file), "Cholesky file not found: {}", chol_file);
  utils::check(utils::file_exists(thc_file),  "THC file not found: {}",      thc_file);
  utils::check(utils::file_exists(wfn_file),  "Wavefunction file not found: {}", wfn_file);

  auto [NMO, nup, ndown] = read_info_from_wfn(wfn_file, "NOMSD");
  WALKER_TYPES walker_type = getWalkerType(wfn_file, "NOMSD");
  int npol = (walker_type == NONCOLLINEAR) ? 2 : 1;
  int nel  = (walker_type == COLLINEAR)    ? nup + ndown : nup;

  // Read the trial wavefunction PsiT from file
  h5::file  wfn_f(wfn_file, 'r');
  h5::group wfn_grp(wfn_f);
  h5::group nomsd_grp = wfn_grp.open_group("Wavefunction").open_group("NOMSD");
  auto PsiT = read_nomsd_wavefunction<HOST_MEMORY>(nomsd_grp, 1, walker_type, NMO, nup, ndown);

  // Build HamiltonianOperations from a file-backed Hamiltonian
  auto make_ham_ops = [&](std::string hamil_file) {
    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
    auto& ham = HamFac.getHamiltonian(mpi, "ham0");
    return ham.template getHamiltonianOperations<HOST_MEMORY>(walker_type, mpi, PsiT);
  };

  auto H_chol = make_ham_ops(chol_file);
  auto H_thc  = make_ham_ops(thc_file);

  // Fixed random G (seed 0) — identical for both energy evaluations
  nda::array<ComplexType, 2> G(1, nel * npol * NMO);
  sfqmc::utils::fillRandomArray(G);

  auto eval_energy = [&](auto& H) -> nda::array<ComplexType, 2> {
    nda::array<ComplexType, 2> E(1, 3);
    H.energy(E, G, 0, true, true, true);
    return E;
  };

  auto E_chol = eval_energy(H_chol);
  auto E_thc  = eval_energy(H_thc);

  app_log(0, "  THC vs Chol (NMO={} nup={} ndown={} walker={}):",
          NMO, nup, ndown, walkerTypeToString(walker_type));
  app_log(0, "    Chol: E1={:+.8e}  EXX={:+.8e}  EJ={:+.8e}",
          std::real(E_chol(0,0)), std::real(E_chol(0,1)), std::real(E_chol(0,2)));
  app_log(0, "    THC:  E1={:+.8e}  EXX={:+.8e}  EJ={:+.8e}",
          std::real(E_thc(0,0)),  std::real(E_thc(0,1)),  std::real(E_thc(0,2)));

  // E1 must match exactly: both files use the same one-body H0 and nuclear data.
  CHECK_THAT(E_thc(all, 0), utils::Approx(E_chol(all, 0)));
  // Two-body terms are allowed to differ by the factorization approximation error.
  CHECK_THAT(E_thc(all, 1), utils::Approx(E_chol(all, 1), 1e-2, 1e-2));
  CHECK_THAT(E_thc(all, 2), utils::Approx(E_chol(all, 2), 1e-2, 1e-2));

  nda::array<ComplexType, 2> G_phys(1, nel * npol * NMO);
  fill_physical_G(G_phys, PsiT, walker_type, NMO);

  auto eval_phys_energy = [&](auto& H) -> nda::array<ComplexType, 2> {
    nda::array<ComplexType, 2> E(1, 3);
    H.energy(E, G_phys, 0, true, true, true);
    return E;
  };

  auto E_chol_phys = eval_phys_energy(H_chol);
  auto E_thc_phys  = eval_phys_energy(H_thc);

  app_log(0, "  Physical G (initial walker = PsiT):");
  app_log(0, "    Chol: E1={:+.8e}  EXX={:+.8e}  EJ={:+.8e}",
          std::real(E_chol_phys(0,0)), std::real(E_chol_phys(0,1)), std::real(E_chol_phys(0,2)));
  app_log(0, "    THC:  E1={:+.8e}  EXX={:+.8e}  EJ={:+.8e}",
          std::real(E_thc_phys(0,0)),  std::real(E_thc_phys(0,1)),  std::real(E_thc_phys(0,2)));

  // For a physical density matrix the direct Coulomb energy must be > 0.
  CHECK(std::real(E_chol_phys(0,2)) > 0.0);
  CHECK(std::real(E_thc_phys(0,2))  > 0.0);
  // THC and Cholesky must agree on all energy components
  CHECK_THAT(E_thc_phys(all, 0), utils::Approx(E_chol_phys(all, 0)));
  CHECK_THAT(E_thc_phys(all, 1), utils::Approx(E_chol_phys(all, 1), 1e-2, 1e-2));
  CHECK_THAT(E_thc_phys(all, 2), utils::Approx(E_chol_phys(all, 2), 1e-2, 1e-2));

  RealType dt = 1.0;
  auto run_h1 = [&](auto& H) {
    int nCV = H.number_of_cholesky_vectors();
    nda::array<ComplexType, 1> vMF(nCV);
    vMF() = ComplexType(0);
    return H.getOneBodyPropagatorMatrix(dt, vMF);
  };
  auto H1_chol = run_h1(H_chol);
  auto H1_thc  = run_h1(H_thc);

  CHECK_THAT(H1_thc, utils::Approx(H1_chol, 1e-2, 1e-2));
}

template<MEMORY_SPACE MEM>
void hamiltonian_factory_build(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                 std::string hamil_file) 
{
  utils::check(utils::file_exists(hamil_file), 
               " Hamiltonian file not found: {}. \n Run unit test with --hamil /path/to/hamil.h5 ", hamil_file);

  int NMO = read_nmo_from_hdf(hamil_file);
  CHECK(NMO > 0);

  HamiltonianFactory HamFac;
  HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = hamil_file});
  [[maybe_unused]] Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");
}

TEST_CASE("hamiltonian_factory: build", "[hamiltonian_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES, bool) {
    hamiltonian_factory_build<MEM>(mpi, hamil_file);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::RHF | TestFiles::UHF | TestFiles::GHF | TestFiles::NOMSD | TestFiles::PHMSD | TestFiles::FINITE_T | TestFiles::ALL_SYSTEMS);
}


TEST_CASE("hamiltonian_factory: thc_vs_chol_energy", "[hamiltonian_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/tests/unit_test_files/C_1x1x1_ks_basis/";
    thc_vs_chol_energy_agreement(mpi,
      pre + "ham_chol_1e-5.h5",
      pre + "ham_thc_1e-6.h5",
      pre + "wfn_mf_pbe.h5");
}

// Regression test for the Madelung electron self-interaction offset (see the constant
// energy offset E0 in read_energy_offset / hdf5_helpers.hpp). E0 lands in the one-body
// energy slot E(:,0). For a periodic (coqui) system with a madelung_constant the offset
// scales with the TOTAL electron count, so the one-body energy of the trial determinant
// must be identical whether the same physical system is built as CLOSED or COLLINEAR.
TEST_CASE("hamiltonian_factory: closed_vs_collinear_energy_offset", "[hamiltonian_factory]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/tests/unit_test_files/C_1x1x1_ks_basis/";
  std::string chol_file = pre + "ham_chol_1e-5.h5";
  std::string wfn_file  = pre + "wfn_mf_pbe.h5";
  utils::check(utils::file_exists(chol_file), "Cholesky file not found: {}", chol_file);
  utils::check(utils::file_exists(wfn_file),  "Wavefunction file not found: {}", wfn_file);

  auto [NMO, nup, ndown] = read_info_from_wfn(wfn_file, "NOMSD");

  // Build the coqui Cholesky Hamiltonian with the given walker type and return the
  // one-body energy E(:,0) (which includes the constant offset E0) of the trial det.
  auto one_body_energy = [&](WALKER_TYPES wt) -> ComplexType {
    int npol = (wt == NONCOLLINEAR) ? 2 : 1;
    int nel  = (wt == COLLINEAR)    ? nup + ndown : nup;

    h5::file  wfn_f(wfn_file, 'r');
    h5::group wfn_grp(wfn_f);
    h5::group nomsd_grp = wfn_grp.open_group("Wavefunction").open_group("NOMSD");
    auto PsiT = read_nomsd_wavefunction<HOST_MEMORY>(nomsd_grp, 1, wt, NMO, nup, ndown);

    HamiltonianFactory HamFac;
    HamFac.push("ham0", HamiltonianParameters{.name = "ham0", .filename = chol_file});
    auto& ham = HamFac.getHamiltonian(mpi, "ham0");
    auto H = ham.template getHamiltonianOperations<HOST_MEMORY>(wt, mpi, PsiT);

    nda::array<ComplexType, 2> G(1, nel * npol * NMO);
    fill_physical_G(G, PsiT, wt, NMO);
    nda::array<ComplexType, 2> E(1, 3);
    H.energy(E, G, 0, true, true, true);
    return E(0, 0);
  };

  auto e1_closed    = one_body_energy(CLOSED);
  auto e1_collinear = one_body_energy(COLLINEAR);

  app_log(0, "  One-body energy (incl. offset): CLOSED={:+.8e}  COLLINEAR={:+.8e}",
          std::real(e1_closed), std::real(e1_collinear));

  // Pre-fix these differ by half the Madelung correction (~2.72 Ha for this system).
  nda::array<ComplexType, 1> a{e1_closed}, b{e1_collinear};
  CHECK_THAT(a, utils::Approx(b));
}

} // namespace sfqmc
