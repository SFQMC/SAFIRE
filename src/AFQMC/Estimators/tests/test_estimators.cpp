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

#include "catch_amalgamated.hpp"

#include "config.h"
#include "Utilities/AppAbort.hpp"

#include "io/ptree/ptree_utilities.hpp"
#include "hdf/hdf_archive.h"

#include <stdio.h>
#include <string>

#include "AFQMC/config.h"
#include "SparseMatrix/tests/matrix_helpers.h"
#include "AFQMC/Hamiltonians/HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Wavefunctions/WavefunctionFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/EstimatorBase.h"
#include "AFQMC/Propagators/PropagatorFactory.h"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Estimators/BackPropagatedEstimator.hpp"
#include "AFQMC/Utilities/test_utils.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "Memory/buffer_managers.h"
#include "Memory/device_rng.hpp"

using std::cerr;
using std::complex;
using std::cout;
using std::endl;
using std::ifstream;
using std::setprecision;
using std::string;

extern std::string UTEST_HAMIL, UTEST_WFN;

namespace sfqmc
{
using namespace afqmc;

template<bool MP, class Allocator>
void reduced_density_matrix(boost::mpi3::communicator& world)
{
  using pointer = typename std::allocator_traits<Allocator>::pointer;

  if (not file_exists(UTEST_HAMIL) || not file_exists(UTEST_WFN))
  {
    APP_ABORT(" Hamiltonian or wavefunction file not found. Run unit test with --hamil /path/to/hamil.h5 and --wfn /path/to/wfn.h5.");
  }
  else
  {
    // Global Task Group
    afqmc::GlobalTaskGroup gTG(world);

    hdf_archive dump;
    if (!dump.open(UTEST_HAMIL, H5F_ACC_RDONLY))
    {
      APP_ABORT(" Error opening integral file in SparseGeneralHamiltonian. ");
    }
    auto format = get_hamiltonian_format(dump,gTG.Global());
    dump.close();

    int NMO, wfn_NMO, NAEA, NAEB;
    NMO = read_nmo_from_hdf(UTEST_HAMIL,format);
    std::tie(wfn_NMO,NAEA, NAEB) = read_info_from_wfn(UTEST_WFN, "any");
    CHECK(NMO == wfn_NMO);

    WALKER_TYPES type = afqmc::getWalkerType(UTEST_WFN);
    int npol = (type == NONCOLLINEAR) ? 2 : 1;

    utils::RandomGenerator_t rng;
    auto rng_dev = utils::make_device_rng(777);
    auto TG = TaskGroup_(gTG, std::string("WfnTG"), 1, gTG.getTotalCores());

    std::map<std::string, AFQMCInfo> InfoMap;
    InfoMap.insert(std::pair<std::string, AFQMCInfo>("info0", AFQMCInfo{"info0", NMO, NAEA, NAEB}));

    ptree ham_pt;
    ham_pt.put("name","ham0");
    ham_pt.put("system","info0");
    ham_pt.put("filename",UTEST_HAMIL);

    HamiltonianFactory HamFac(InfoMap);
    HamFac.push("ham0", ham_pt);
    Hamiltonian& ham = HamFac.getHamiltonian(gTG, "ham0");

    ptree wlk_pt;
    wlk_pt.put("name","wset0");
    wlk_pt.put("system","info0");
    if(type == CLOSED) wlk_pt.put("walker_type","closed");
    else if(type == COLLINEAR) wlk_pt.put("walker_type","collinear");
    else if(type == NONCOLLINEAR) wlk_pt.put("walker_type","noncollinear");
    else if(type == FULLYPOLARIZED) wlk_pt.put("walker_type","fullypolarized");
    WalkerSet wset(TG, wlk_pt, InfoMap["info0"], &rng);

    ptree wfn_pt;
    wfn_pt.put("name","wfn0");
    wfn_pt.put("system","info0");
    wfn_pt.put("filename",UTEST_WFN);
    wfn_pt.put("dense_trial","yes");

    Allocator alloc_(make_localTG_allocator<ComplexType>(TG));
    int nwalk = 1; // choose prime number to force non-trivial splits in shared routines
    WavefunctionFactory WfnFac(InfoMap, MP);
    WfnFac.push("wfn0", wfn_pt);
    Wavefunction& wfn = WfnFac.getWavefunction(TG, TG, "wfn0", type, &ham, 1e-6, nwalk);

    ptree prop_pt;
    prop_pt.put("name","prop0");
    prop_pt.put("system","info0");

    PropagatorFactory PropgFac(InfoMap, MP);
    PropgFac.push("prop0", prop_pt);
    Propagator& prop = PropgFac.getPropagator(TG, "prop0", wfn, &rng_dev);

    auto initial_guess = WfnFac.getInitialGuess("wfn0");
    REQUIRE(initial_guess.size(0) == 2);
    REQUIRE(initial_guess.size(1) == npol * NMO);
    REQUIRE(initial_guess.size(2) == NAEA);
    wset.resize(nwalk, initial_guess[0], initial_guess[0]);
    using EstimPtr = std::shared_ptr<EstimatorBase>;
    std::vector<EstimPtr> estimators;

    ptree est_pt;
    est_pt.put("name","back_propagation");
    est_pt.put("nsteps",1); // deprecated
    est_pt.put("block_size",1); // deprecated
    est_pt.put("measure_interval_multiplier", 2); // test with single value
    est_pt.put("path_restoration","no");
    est_pt.put("OneRDM.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt));
    bool impsamp = true;
    estimators.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BackPropagatedEstimator>(TG, InfoMap["info0"], "none", est_pt,
                                                                        type, wset, wfn, prop, impsamp)));

    // generate P1 with dt=0
    //   Discrete HS transformation requires finite dt to generate P1, 
    //   but for continuous HS transformation we can safely use dt=0.0
    double dt = ( wfn.getHamType() == ModelHamiltonian ) ? 1.0e-12 : 0.0;                                                                      
    prop.generateP1(dt, wset.getWalkerType());

    std::string file = create_test_hdf(UTEST_WFN, UTEST_HAMIL);
    std::ofstream out;
    dump.create(file);
    dump.open(file);
    for (int iblock = 0; iblock < 10*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; iblock++)
    {
      wset.advanceBPPos();
      estimators[0]->accumulate_block(iblock*0.01,wset);
      estimators[0]->print(out, dump, wset);
    }
    dump.close();
    boost::multi::array<ComplexType, 1> read_data(boost::multi::iextensions<1u>{2 * NMO * NMO});

    ComplexType denom;
    hdf_archive reader;
    // Read from a particular block.
    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_0/one_rdm_000000004");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_0/denominator_000000004");
    // Test EstimatorHandler eventually.
    //int NAEA_READ, NAEB_READ, NMO_READ, WALKER_TYPE_READ;
    //reader.read(NAEA_READ, "Metadata/NAEA");
    //REQUIRE(NAEA_READ==NAEA);
    //reader.read(NAEB_READ, "Metadata/NAEB");
    //REQUIRE(NAEB_READ==NAEB);
    //reader.read(NMO_READ, "Metadata/NMO");
    //REQUIRE(NMO_READ==NMO);
    //reader.read(WALKER_TYPE_READ, "Metadata/WALKER_TYPE");
    //REQUIRE(WALKER_TYPE_READ==type);
    reader.close();
    // Test the RDM. Since no back propagation has been performed the RDM should be
    // identical to the mixed estimate.
    if (type == CLOSED)
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == COLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= 2 * NMO * NMO);
      boost::multi::array_ref<ComplexType, 3> BPRDM(read_data.origin(), {2, NMO, NMO});
      ma::scal(1.0 / denom, BPRDM[0]);
      ma::scal(1.0 / denom, BPRDM[1]);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[0][i][i] + BPRDM[1][i][i];
      REQUIRE(trace.real() == Approx(NAEA + NAEB));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, 2 * NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 3, pointer> G(Gw.origin(), {2, NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == FULLYPOLARIZED) // Kyle: be careful here; I'm not sure that I have a reference!
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == NONCOLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= npol * NMO * npol * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {npol * NMO, npol * NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < npol * NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA + NAEB));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, npol * NMO * npol * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {npol * NMO, npol * NMO});
      verify_approx(G, BPRDM);
    }
    else
    {
      APP_ABORT(" Unknown wavefunction type.");
    }

    ptree est_pt2;
    est_pt2.put("name","back_propagation");
    est_pt2.put("nsteps",1); // deprecated
    est_pt2.put("block_size",1); // deprecated
    
    std::vector<int> nback_prop_interval_multipliers = {1, 2, 3};
    ptree temp_tree;
    for (const auto& value : nback_prop_interval_multipliers) {
        ptree item;
        item.put("", value); // empty key for the value
        temp_tree.push_back(std::make_pair("", item));
    }
    est_pt2.add_child("measure_interval_multiplier", temp_tree); // test with a vector
    est_pt2.put("path_restoration","no");
    est_pt2.put("OneRDM.nskip_output",0);
    app_log(1,"\nEstimator input:\n{}\n",io::to_string(est_pt2));

    std::vector<EstimPtr> estimators2;
    estimators2.emplace_back(
        static_cast<EstimPtr>(std::make_shared<BackPropagatedEstimator>(TG, InfoMap["info0"], "none", est_pt2,
                                                                        type, wset, wfn, prop, impsamp)));

    dump.open(file);
    for (int iblock = 0; iblock < 6*afqmc::DEFAULT_POPULATION_CONTROL_INTERVAL; iblock++)
    {
      wset.advanceBPPos();
      estimators2[0]->accumulate_block(iblock*0.01,wset);
      estimators2[0]->print(out, dump, wset);
    }
    dump.close();

    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    else
      app_log(1," Successfully opened {}", file);
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_1/one_rdm_000000002");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_1/denominator_000000002");
    reader.close();

    if (type == CLOSED)
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == COLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= 2 * NMO * NMO);
      boost::multi::array_ref<ComplexType, 3> BPRDM(read_data.origin(), {2, NMO, NMO});
      ma::scal(1.0 / denom, BPRDM[0]);
      ma::scal(1.0 / denom, BPRDM[1]);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[0][i][i] + BPRDM[1][i][i];
      REQUIRE(trace.real() == Approx(NAEA + NAEB));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, 2 * NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 3, pointer> G(Gw.origin(), {2, NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == FULLYPOLARIZED) // Kyle: be careful here; I'm not sure that I have a reference!
    {
      REQUIRE(read_data.num_elements() >= NMO * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {NMO, NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, NMO * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {NMO, NMO});
      verify_approx(G, BPRDM);
    }
    else if (type == NONCOLLINEAR)
    {
      REQUIRE(read_data.num_elements() >= npol * NMO * npol * NMO);
      boost::multi::array_ref<ComplexType, 2> BPRDM(read_data.origin(), {npol * NMO, npol * NMO});
      ma::scal(1.0 / denom, BPRDM);
      ComplexType trace = ComplexType(0.0);
      for (int i = 0; i < npol * NMO; i++)
        trace += BPRDM[i][i];
      REQUIRE(trace.real() == Approx(NAEA + NAEB));
      boost::multi::array<ComplexType, 2, Allocator> Gw({1, npol * NMO * npol * NMO}, alloc_);
      wfn.MixedDensityMatrix(wset, Gw, false, true);
      boost::multi::array_ref<ComplexType, 2, pointer> G(Gw.origin(), {npol * NMO, npol * NMO});
      verify_approx(G, BPRDM);
    }
    else
    {
      APP_ABORT(" Unknown wavefunction type.");
    }

    if (!reader.open(file, H5F_ACC_RDONLY))
      APP_ABORT(" Error opening estimates.h5. ");
    reader.read(read_data, "Observables/BackPropagated/FullOneRDM/Average_2/one_rdm_000000002");
    reader.read(denom, "Observables/BackPropagated/FullOneRDM/Average_2/denominator_000000002");
    reader.close();
  }
}

TEST_CASE("reduced_density_matrix", "[estimators]")
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());
  setup_loggers(world.root(),2,0);

#if defined(ENABLE_CUDA) || defined(ENABLE_HIP)
  arch::INIT(node);
  using Alloc = device::device_allocator<ComplexType>;
#else
  using Alloc = shared_allocator<ComplexType>;
#endif
  setup_AFQMC_timer(); 
  setup_memory_managers(node, 10uL * 1024uL * 1024uL);
  reduced_density_matrix<false,Alloc>(world);
  reduced_density_matrix<true,Alloc>(world);
  release_memory_managers();
}

} // namespace sfqmc
