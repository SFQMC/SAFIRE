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
#include "IO/AppAbort.hpp"

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Random.hpp"
#include "utilities/check.hpp"
#include "utilities/h5_utils.hpp"
#include "test_common.hpp"
#include "IO/app_loggers.h"

#include <string>
#include <complex>
#include <algorithm>

#include "AFQMC/Wavefunctions/Excitations.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "test_utils.hpp"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "numerics/sparse/sparse.hpp"

extern bool WRITE_REFERENCE;
extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<MEMORY_SPACE MEM>
void phmsd_read(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file)
{
  using nda::range;

  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "PHMSD");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  WALKER_TYPES type    = afqmc::getWalkerType(wfn_file, "PHMSD");

  h5::file file(wfn_file,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");
  h5::group ngrp = wgrp.open_group("PHMSD");
  int ndets_to_read = -1;
  std::string wfn_type;

  nda::array<int,2> occs;
  nda::array<ComplexType,1> coeffs;
  nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
  read_ph_wavefunction_hdf(ngrp, coeffs, occs, ndets_to_read, type, NMO, nup, ndown, PsiT_MO, wfn_type); 

  auto abij = build_ph_struct<MEM>(coeffs, occs, ndets_to_read, NMO, nup, ndown);
  using std::get;
  auto cit = abij.configurations_begin();
  nda::array<int,1> configa(nup), configb(ndown);
  // Is it fortuitous that the order of determinants is the same?
  for (int nd = 0; nd < ndets_to_read; nd++, ++cit)
  {
    int alpha_ix = get<0>(*cit);
    int beta_ix  = get<1>(*cit);
    auto ci      = get<2>(*cit);
    abij.get_configuration(0, alpha_ix, configa);
    abij.get_configuration(1, beta_ix, configb);
    std::sort(configa.begin(), configa.end());
    std::sort(configb.begin(), configb.end());
    for (int i = 0; i < nup; i++)
    {
      REQUIRE(configa[i] == occs(nd,i));
    }
    for (int i = 0; i < ndown; i++)
    {
      REQUIRE(configb[i] == occs(nd,i + nup));
    }
    REQUIRE(std::abs(coeffs[nd]) == std::abs(ci));
  }
  // Check sign of permutation.
  REQUIRE(abij.number_of_configurations() == ndets_to_read);
}

template<class Mat>
void getSlaterMatrix_mixed(Mat&& SM, math::sparse::CSRMatrix auto&& Orbs, nda::MemoryVector auto&& occs)
{
  using nda::range;
  SM() = ComplexType(0.0);
  auto row_begin = Orbs.row_begin();
  auto row_end = Orbs.row_end();
  auto vals = Orbs.values();
  auto cols = Orbs.columns();
  // array copies, which work on GPU!
  for (int r = 0; r < occs.extent(0); r++)
    for(int j=row_begin(occs(r)); j<row_end(occs(r)); ++j)
      SM(r,range(cols(j),cols(j)+1)) = vals(range(j,j+1)); 
}

template<class Mat>
void getSlaterMatrix_occ(Mat&& SM, nda::MemoryVector auto&& occs)
{
  using nda::range;
  SM() = ComplexType(0.0);
  // array copies, which work on GPU!
  for (int r = 0; r < occs.extent(0); r++)
    SM(r,range(occs(r),occs(r)+1)) = ComplexType(1.0);
}

template<MEMORY_SPACE MEM>
void phmsd_compute(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
             std::string hamil_file, std::string wfn_file, bool write_reference)
{
  using nda::range;
  auto all = range::all;
  app_log(1, "Running estimators unit test "
    "with files:\n --hamil {} \\\n --wfn {}", 
    hamil_file, wfn_file
  );

  // First strip path of filename.
  std::string base_name = wfn_file.substr(wfn_file.find_last_of("\\/") + 1);
  // Remove file extension.
  std::string test_wfn = base_name.substr(0, base_name.find_last_of("."));
  test_wfn = test_wfn.substr(test_wfn.find('_') + 1);
  auto file_data       = read_test_results_from_hdf<ComplexType>(hamil_file, test_wfn);
  auto [NMO,nup,ndown] = read_info_from_wfn(wfn_file, "PHMSD");
  utils::check(NMO == file_data.NMO, "Incompatible NMO.");

  WALKER_TYPES type = afqmc::getWalkerType(wfn_file, "PHMSD");
  int nspin         = (type == COLLINEAR) ? 2 : 1;
  int npol          = (type == NONCOLLINEAR) ? 2 : 1;
  int nwalk         = 1;
  int ndets         = -1; 
  double dt         = 0.01;
  std::shared_ptr<utils::RandomGenerator_t<>> rng = std::make_shared<utils::RandomGenerator_t<>>();

  std::map<std::string, AFQMCInfo> InfoMap;
  InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, nup, ndown}));

  ptree ham_pt;
  ham_pt.put("name","ham0");
  ham_pt.put("system","info0");
  ham_pt.put("filename",hamil_file);

  HamiltonianFactory HamFac(InfoMap);
  HamFac.push("ham0", ham_pt);
  Hamiltonian& ham = HamFac.getHamiltonian(mpi, "ham0");

  ptree wfn_pt;
  wfn_pt.put("name","wfn0");
  wfn_pt.put("system","info0");
  wfn_pt.put("filename",wfn_file);
  wfn_pt.put("rediag","no");
  wfn_pt.put("ndets_to_read",-1);
  wfn_pt.put("algorithm",0);

  WavefunctionFactory<MEM> WfnFac{};
  WfnFac.push("wfn0", wfn_pt);
  auto& wfn = WfnFac.getWavefunction(mpi, "wfn0", type, &ham, nwalk);

  ptree wlk_pt;
  wlk_pt.put("name","wset0");
  wlk_pt.put("walker_type", walkerTypeToString(type));

  auto wset = make_WalkerSet<MEM>(mpi, wlk_pt, InfoMap["info0"], rng);
  auto initial_guess = WfnFac.getInitialGuess("wfn0");
  REQUIRE(initial_guess.shape() == std::array<long,3>{nspin,npol*NMO,nup});

  // apply small unitary rotation to initial_guess
  // add different rotations to every walker to test routines
  {
    nda::array<ComplexType,3> rotated_initial_guess(nspin,npol*NMO,nup);
    nda::array<ComplexType,2> R = nda::rand(std::array<long,2>{npol*NMO,npol*NMO});
    nda::array<ComplexType,1> tau(npol*NMO);
    nda::lapack::geqrf(nda::transpose(R),tau);
    nda::lapack::gqr(nda::transpose(R),tau);
    for(int is=0; is<nspin; ++is)
      nda::blas::gemm(R,initial_guess(is,all,all),rotated_initial_guess(is,all,all));
    wset.resize(nwalk, rotated_initial_guess);
  }

  // 0. Get raw occupancies and coefficients from file.
  nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
  memory::array<MEM,ComplexType,2> PsiA(nup,npol*NMO); 
  memory::array<MEM,ComplexType,2> PsiB(ndown,npol*NMO); 
  nda::array<ComplexType,1> coeffs;
  nda::array<int,2> occs;
  h5::file f(wfn_file,'r');
  h5::group g = h5::group(f).open_group("Wavefunction").open_group("PHMSD");
  std::string orb_type;
  read_ph_wavefunction_hdf(g, coeffs, occs, ndets, type, 
                                NMO, nup, ndown, PsiT_MO, orb_type);

  if(orb_type=="occ") {
    PsiT_MO.resize(1);
    PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({npol*NMO,npol*NMO},1);
    for(int i=0; i<npol*NMO; ++i)
      PsiT_MO(0).emplace_back({i,i},ComplexType(1.0));
  }

  // Using NOMSD as reference. Making h5 from input phmsd wfn
  std::string nomsd_file = "_nomsd_dummy_.h5";
  if(mpi->comm.root()) {
    // 
    h5::file f_(nomsd_file,'w');
    h5::group g_(f_);
    h5::group wg = g_.create_group("Wavefunction");
    h5::group ng = wg.create_group("NOMSD");
 
    nda::vector<int> dims = {NMO,nup,ndown,int(type),int(coeffs.size())};
    nda::h5_write(ng,"dims",dims);
    nda::h5_write(ng,"ci_coeffs",coeffs);

    {
      auto Psi0 = initial_guess(0,all,all);
      nda::h5_write(ng,"Psi0_alpha",Psi0);
    }
    if(type == COLLINEAR) {
      auto Psi0 = initial_guess(1,all,range(ndown));
      nda::h5_write(ng,"Psi0_beta",Psi0);
    }

    for(int idet=0, n=0; idet<ndets; ++idet) {
      {
        h5::group gi = ng.create_group(std::string("PsiT_")+std::to_string(n));
        math::sparse::CSR2HDF(gi,PsiT_MO(0),occs(idet,range(nup)));
        n++;
      }
      if(type == COLLINEAR) {
        h5::group gi = ng.create_group(std::string("PsiT_")+std::to_string(n));
        nda::vector<int> ob = occs(idet,range(nup,nup+ndown))-NMO;
        math::sparse::CSR2HDF(gi,PsiT_MO( PsiT_MO.extent(0)-1 ),ob);
        n++;
      }
    }
  }
  mpi->comm.barrier();

  ptree nomsd_pt;
  nomsd_pt.put("name","nomsd");
  nomsd_pt.put("system","info0");
  nomsd_pt.put("filename",nomsd_file);

  WfnFac.push("nomsd", nomsd_pt);
  auto& nomsd = WfnFac.getWavefunction(mpi, "nomsd", type, &ham, nwalk);

  // 1. Overlap 
  ComplexType ovlp_sum = ComplexType(0.0);
  for (int idet = 0; idet < ndets; idet++)
  {
    // Construct slater matrix from given set of occupied orbitals.
    memory::array<MEM,ComplexType,1> ov(nwalk,ComplexType(0.0));
    if(orb_type == "mixed")
      getSlaterMatrix_mixed(PsiA, PsiT_MO(0), occs(idet,range(nup)));
    else
      getSlaterMatrix_occ(PsiA, occs(idet,range(nup)));
    det_ops::Log_Overlap(PsiA,wset.SlaterMatrices(Alpha),ov);
    if(type == COLLINEAR) {
      nda::array<int, 1> ob = occs(idet,range(nup,nup+ndown)) - NMO; 
      if(orb_type == "mixed")
        getSlaterMatrix_mixed(PsiB, PsiT_MO(PsiT_MO.size()-1), ob);
      else
        getSlaterMatrix_occ(PsiB, ob);
      det_ops::Log_Overlap(PsiB,wset.SlaterMatrices(Beta),ov);
    }
    ovlp_sum += std::conj(coeffs[idet]) * std::exp(nda::to_host(ov)(0));
  }
  wfn.Log_Overlap(wset);
  
  // log(ovlp_sum)
  ComplexType log_ovlp_sum = std::log(ovlp_sum);

  // the phase can be off by 2*pi due to small round-off errors around 0, what to do???
  for (const auto &w : wset)
    CHECK_THAT(std::exp(w.get_property(OVLP)), utils::Approx(ovlp_sum));

  {
    memory::array<MEM,ComplexType,1> log_ov(nwalk);
    nomsd.Log_Overlap(wset,log_ov); 
    auto ov_h = nda::to_host(log_ov);
    ov_h() = nda::exp(ov_h());
    for(int i=0; i<nwalk; ++i) {
      CHECK_THAT(std::exp(wset[i].get_property(OVLP)), utils::Approx(ov_h(i))); 
    }
  }

  // 2. Green function
  {
    nda::array<ComplexType,3> G(nwalk,nspin*npol*NMO,npol*NMO);
    memory::array<MEM,ComplexType,3> Gt(nwalk,nspin*npol*NMO,npol*NMO);
    G() = ComplexType(0.0);
    for (int idet = 0; idet < ndets; idet++)
    {
      memory::array<MEM,ComplexType,1> ov(nwalk,ComplexType(0.0));
      Gt() = ComplexType(0.0);
      if(orb_type == "mixed")
        getSlaterMatrix_mixed(PsiA, PsiT_MO(0), occs(idet,range(nup)));
      else
        getSlaterMatrix_occ(PsiA, occs(idet,range(nup)));
      det_ops::MixedDensityMatrix(PsiA,wset.SlaterMatrices(Alpha),Gt(all,range(npol*NMO),all),ov,false);
      if(type == COLLINEAR) {
        nda::array<int, 1> ob = occs(idet,range(nup,nup+ndown)) - NMO;
        if(orb_type == "mixed")
          getSlaterMatrix_mixed(PsiB, PsiT_MO(PsiT_MO.size()-1), ob);
        else
          getSlaterMatrix_occ(PsiB, ob);
        det_ops::MixedDensityMatrix(PsiB,wset.SlaterMatrices(Beta),Gt(all,range(npol*NMO,2*npol*NMO),all),ov,false);
      }
      auto Gt_h = nda::to_host(Gt());
      auto ov_h = nda::to_host(ov());
      for(int iw=0; iw<nwalk; ++iw)
        G(iw,all,all) += std::conj(coeffs[idet]) * std::exp(ov_h(iw) - log_ovlp_sum) * Gt_h(iw,all,all); 
    }
    memory::array<MEM,ComplexType,3> Gd(nwalk,nspin*npol*NMO,npol*NMO);
    auto Gt2d = nda::reshape(Gd,std::array<long,2>{nwalk,nspin*npol*NMO*npol*NMO});
    Gt2d() = ComplexType(0.0);
    wfn.MixedDensityMatrix(wset,Gt2d,false);
    CHECK_THAT(G, utils::Approx(Gd));
  }

  memory::array<MEM,ComplexType,2> eloc_ph0(nwalk,3);
  memory::array<MEM,ComplexType,1> ov_ph0(nwalk);
  wfn.Energy(wset,eloc_ph0,ov_ph0);
  nda::apply(ComplexType(1.0),ov_ph0,nda::tensor::op::EXP);

  {
    auto eloc_h = nda::to_host(eloc_ph0);
    nda::array<ComplexType,1> e1_w  = eloc_h(all,0);
    nda::array<ComplexType,1> exx_w = eloc_h(all,1);
    nda::array<ComplexType,1> ej_w  = eloc_h(all,2);
    if (!write_reference) {
      if (file_data.available) {
        CHECK_THAT(e1_w, utils::Approx(file_data.E1));
        CHECK_THAT(ej_w, utils::Approx(file_data.EJ));
        CHECK_THAT(exx_w, utils::Approx(file_data.EXX));
      }
    } else {
      file_data.E1  = e1_w;
      file_data.EJ  = ej_w;
      file_data.EXX = exx_w;
    }
  }

  {
    memory::array<MEM,ComplexType,2> eloc(nwalk,3);
    memory::array<MEM,ComplexType,1> ov(nwalk);
    nomsd.Energy(wset,eloc,ov);
    nda::apply(ComplexType(1.0),ov,nda::tensor::op::EXP);
    CHECK_THAT(ov_ph0, utils::Approx(ov));
    CHECK_THAT(eloc_ph0, utils::Approx(eloc));
  }

  if(wfn.getHamType() == RealDenseFactorized)  // add THC
  {
    ptree wfn1_pt;
    wfn1_pt.put("name","wfn1");
    wfn1_pt.put("system","info0");
    wfn1_pt.put("filename",wfn_file);
    wfn1_pt.put("rediag","no");
    wfn1_pt.put("ndets_to_read",-1);
    wfn1_pt.put("algorithm",1);

    WfnFac.push("wfn1", wfn1_pt);
    auto& wfn1 = WfnFac.getWavefunction(mpi, "wfn1", type, &ham, nwalk);

    memory::array<MEM,ComplexType,2> eloc(nwalk,3);
    memory::array<MEM,ComplexType,1> ov(nwalk);
    wfn1.Energy(wset,eloc,ov);
    nda::apply(ComplexType(1.0),ov,nda::tensor::op::EXP);

    CHECK_THAT(ov_ph0, utils::Approx(ov));
    CHECK_THAT(eloc_ph0, utils::Approx(eloc));
  }

  // vMF
  {
    memory::array<MEM,ComplexType,1> v(wfn.number_of_cholesky_vectors());
    wfn.vMF(v,dt);
  }

  // G_MF
  {
    auto gMF = wfn.G_MF();
  }

  // vbias
  memory::array<MEM,ComplexType,2> X(nwalk,wfn.number_of_cholesky_vectors());
  wfn.vbias(wset, X, dt);
  {
    auto X_h = nda::to_host(X);
    if (!write_reference) {
      if (file_data.available) {
        CHECK_THAT(X_h, utils::Approx(file_data.vbias));
      }
    } else {
      file_data.vbias = X_h;
    }

    memory::array<MEM,ComplexType,2> X2(nwalk,nomsd.number_of_cholesky_vectors());
    nomsd.vbias(wset, X2, dt);
    CHECK_THAT(X, utils::Approx(X2));
  }

  // vHS
  auto vHS_d = wfn.vHS(X, dt);
  auto vHS = nda::to_host(vHS_d);
  if (!write_reference) {
    if (file_data.available) {
      CHECK_THAT(vHS, utils::Approx(file_data.VHS));
    }
  } else {
    file_data.VHS = vHS;
  }

  if (write_reference) {
    write_test_results_to_hdf(hamil_file, test_wfn, file_data);
  }

  mpi->comm.barrier();
  if(mpi->comm.root()) remove(nomsd_file.c_str());
  mpi->comm.barrier();
}

TEST_CASE("phmsd: read", "[phmsd]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    if constexpr (MEM == HOST_MEMORY) {
      phmsd_read<MEM>(mpi, hamil_file, wfn_file);
    }
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::UHF | TestFiles::PHMSD | TestFiles::ALL_SYSTEMS);
}

TEST_CASE("phmsd: compute", "[phmsd]")
{
  auto& mpi = utils::make_unit_test_mpi_context();

  bool write_reference = WRITE_REFERENCE;
  
  using namespace utils;

  run_test_with_files([&]<auto MEM>(std::string hamil_file, std::string wfn_file, WALKER_TYPES) {
    phmsd_compute<MEM>(mpi, hamil_file, wfn_file, write_reference && MEM == HOST_MEMORY);
  }, UTEST_HAMIL, UTEST_WFN, TestFiles::UHF | TestFiles::PHMSD | TestFiles::ALL_SYSTEMS);
}

} // namespace sfqmc
