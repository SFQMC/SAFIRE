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

#include "AFQMC/config.h"
#include "utilities/h5_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "AFQMC/Utilities/readWfn.h"
#include "WavefunctionFactory.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
//#include "AFQMC/Wavefunctions/Excitations.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace {

// Convert an array of numbers of electrons per flavor ({nup, ndown}) from one walker type to another.
// Most of the time, this is is an identity. However, in the noncollinear case, only
// the first component contains all electrons and the rest is supposed to be zero.
template<std::size_t N = 2>
auto broadcast_number_of_electrons(const std::array<int, N> &nel, WALKER_TYPES from, WALKER_TYPES to) {
  static_assert(N > 0, "Cannot have no electron flavors");
  utils::check(walkerTypeIsConvertible(from, to), "Cannot convert {} wavefunction to {} walker type", walkerTypeToString(from), walkerTypeToString(to));
  if(from == CLOSED) {
    utils::check(
      !nel.empty() && std::all_of(nel.begin(), nel.end(), [&](auto n) { return n == nel.front(); }),
      "Closed wavefunction does not have uniform number of electrons: {}", nel);
  }
  
  if(to == NONCOLLINEAR) {
    std::array<int, N> result{};
    result[0] = std::accumulate(nel.begin(), nel.end(), 0);
    return result;
  }    
  return nel;
}

}

template<MEMORY_SPACE MEM>
Wavefunction<MEM> WavefunctionFactory<MEM>::fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                           const WavefunctionParameters& params,
                                           WALKER_TYPES walker_type,
                                           bool finiteT,
                                           Hamiltonian& h,
                                           int targetNW)
{
  bool dense_trial;
  const std::string& name     = params.name;
  const std::string& filename = params.filename;
  utils::check(not name.empty(), "Error in WavefunctionFactory: missing required input: name");
  utils::check(not filename.empty(), "Error in WavefunctionFactory: missing required input: filename");
  bool recompute_ci  = params.rediag;
  int ndets_to_read  = params.ndets_to_read;

  const auto [NMO, nup_in_wfn, ndown_in_wfn] = read_info_from_wfn(filename,"any");
  utils::check(ndown_in_wfn <= nup_in_wfn,"Error nup < ndown: Up spin must be the majority spin. nup: {}, ndown: {}",nup_in_wfn,ndown_in_wfn);

  int nspin = walker_type == COLLINEAR ? 2 : 1;
  int npol = walker_type == NONCOLLINEAR ? 2 : 1;

  WAVEFUNCTION_TYPES wfn_type; 
  if(mpi->comm.root()) { 
    wfn_type = afqmc::getWavefunctionType(filename); 
    int itype(wfn_type);
    mpi->comm.broadcast_n(&itype,1,0); 
  } else {
    int itype;
    mpi->comm.broadcast_n(&itype,1,0); 
    wfn_type = WAVEFUNCTION_TYPES(itype);
  }

  // everyone reading for now, change it problematic
  h5::file file(filename,'r');
  h5::group grp(file);
  h5::group wgrp = grp.open_group("Wavefunction");

  
  if (wfn_type == NOMSD_WFN)
  {
    app_log(1,"Wavefunction type: NOMSD");
    nda::array<ComplexType,1> ci;
    h5::group ngrp = wgrp.open_group("NOMSD");
    // Read common trial wavefunction input options.
    WALKER_TYPES input_wtype{};
    getCommonInput(ngrp, ndets_to_read, ci, input_wtype);

    if (!finiteT) {      
      // validation blocks
      utils::check(input_wtype != NONCOLLINEAR or walker_type == NONCOLLINEAR,
          "Error: Trial wavefunction is NONCOLLINEAR and requires NONCOLLINEAR walkers. walker_type: {}", walkerTypeToString(walker_type));
      
      auto [nup, ndown] = broadcast_number_of_electrons({nup_in_wfn, ndown_in_wfn}, input_wtype, walker_type);

      //mpi->comm.broadcast_n(ci.data(), ci.size());

      // Create Trial wavefunction.
      auto PsiT = read_nomsd_wavefunction<MEM>(ngrp,ndets_to_read,walker_type,NMO,nup,ndown);

      // Set initial walker's Slater matrix.
      getInitialGuess(ngrp, name, NMO, nup, ndown, walker_type);

      dense_trial = resolved(params.dense_trial, "dense_trial");

      auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, PsiT);

      if (dense_trial)
      {
        using MType = memory::const_shared_array<MEM,ComplexType,2>;
        nda::array<MType,2> PsiT_dense(ndets_to_read,nspin);
        for(int id=0; id<ndets_to_read; ++id) {
          for(int is=0; is<nspin; ++is) {
            PsiT_dense(id,is) = memory::share_from_root(*mpi, [&] {
              return memory::to_memory_space<MEM>(math::sparse::to_array<'N'>(PsiT(id,is)));
            });
          }
        }
        return Wavefunction<MEM>(NOMSD<MEM,MType>(params, NMO, nup, ndown, walker_type, mpi, std::move(HOps), 
                                      std::move(ci), std::move(PsiT_dense),targetNW));
      }
      else
      {
        return Wavefunction<MEM>(NOMSD<MEM,PsiT_Matrix<MEM>>(params, NMO, nup, ndown, walker_type, mpi, std::move(HOps), 
                                      std::move(ci), std::move(PsiT),targetNW)); 
      }
    }
    else
    {
      // validation blocks
      utils::check(input_wtype != NONCOLLINEAR or walker_type == NONCOLLINEAR,
          "Error: Trial wavefunction is NONCOLLINEAR and requires NONCOLLINEAR walkers. walker_type: {}", walkerTypeToString(walker_type));
      
      int ntau = nup_in_wfn;
      utils::check(ndown_in_wfn == 0, "expected ndown dimension to be 0 at finite temperature");

      //mpi->comm.broadcast_n(ci.data(), ci.size());

      // Create Trial wavefunction.
      auto PsiT = read_nomsd_wavefunction<MEM>(ngrp,ndets_to_read,walker_type,NMO,ntau);

      // Set initial walker's Slater matrix.
      getInitialGuess_ft(ngrp, *mpi, name, NMO, walker_type, finiteT);

      dense_trial = resolved(params.dense_trial, "dense_trial");

      nda::array<PsiT_Matrix<MEM>, 2> IMat(ndets_to_read,nspin);
      // dim = NMO
      int dim = PsiT(0,0,0).extent(1);
      for(int i = 0; i < ndets_to_read; ++i)
        for(int s = 0; s < nspin; ++s)
          IMat(i,s) = math::sparse::identity<ComplexType>(dim);

      auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, IMat);

      if (dense_trial)
      {
        using MType = memory::const_shared_array<MEM,ComplexType,2>;
        nda::array<MType,3> PsiT_dense(ndets_to_read,nspin,3);
        for(int id=0; id<ndets_to_read; ++id) {
          for(int is=0; is<nspin; ++is) {
            for(int m=0; m<3; ++m) {
              PsiT_dense(id,is,m) = memory::share_from_root(*mpi, [&] {
                return memory::to_memory_space<MEM>(math::sparse::to_array<'N'>(PsiT(id,is,m)));
              });
            }
          }
        }
        return Wavefunction<MEM>(NOMSD_FT<MEM,MType>(params, NMO, ntau, walker_type, mpi, std::move(HOps), 
                                      std::move(ci), std::move(PsiT_dense),targetNW));
      }
      else
      {
        return Wavefunction<MEM>(NOMSD_FT<MEM,PsiT_Matrix<MEM>>(params, NMO, ntau, walker_type, mpi, std::move(HOps), 
                                      std::move(ci), std::move(PsiT),targetNW));
      }

    }

  }
  else if (wfn_type == PHMSD_WFN)
  {

    app_log(1,"Wavefunction type: PHMSD");

    // Implementation notes:
    //  - PsiT: [Nact, NMO] where Nact is the number of active space orbitals,
    //                     those that participate in the ci expansion
    //  - The half rotation is done with respect to the supermatrix PsiT
    //  - Need to calculate Nact and create a mapping from orbital index to actice space index.
    //    Those orbitals in the corresponding virtual space (not in active) map to -1 as a precaution.
    //

    nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
    
    // phmsd does not support conversion to noncollinear
    int nup = nup_in_wfn;
    int ndown = ndown_in_wfn;

    std::string orb_type;
    h5::group ngrp = wgrp.open_group("PHMSD");

    nda::array<int,2> occs;
    nda::array<ComplexType,1> coeffs;
    // 1. Read occupancies and coefficients.
    app_log(1,"Reading PHMSD wavefunction from {}", filename);
    read_ph_wavefunction_hdf(ngrp, coeffs, occs, ndets_to_read, walker_type, 
				NMO, nup, ndown, PsiT_MO, orb_type);
    utils::check(occs.shape() == std::array<long,2>{ndets_to_read, nup + ndown}, "Size mismatch");
    app_log(1,"Finished reading PHMSD wavefunction ");
    if(recompute_ci) {
      utils::check(false, "finish");
      // 2. Compute Variational Energy / update coefficients
      app_log(1,"Computing variational energy of trial wavefunction.");
//      computeVariationalEnergyPHMSD(TGwfn, h, occs, coeffs, ndets_to_read, nup, ndown, NMO, recompute_ci);
      app_log(1,"Finished computing variational energy of trial wavefunction.");
    }

    // build reference MOs (PsiT_MO) if needed...
    utils::check((orb_type == "occ") or (orb_type == "mixed"), "Invalid wavefunction type:{}",orb_type);
    if (orb_type == "occ")
      build_PsiT_MO_phmsd(walker_type,npol,NMO,nup,ndown,ndets_to_read,coeffs,occs,PsiT_MO);

    // 3. Construct Structures.
    ph_excitations<int, ComplexType, MEM> abij = build_ph_struct<MEM>(coeffs, occs, ndets_to_read, npol*NMO, nup, ndown);

    // Final Psi matrix, where we will remove orbitals that do not appear in any configuration 
    // and relabel occupation indexes
    nda::array<PsiT_Matrix<HOST_MEMORY>,1> PsiT(PsiT_MO.extent(0));

    // returns the number of times a given orbital appears in the ci expansion
    auto orb_counts = find_active_space(walker_type, abij, NMO, nup, ndown);

    // mapping from old to new occupation indexes 
    std::map<int, int> mo2active;
    for (int i = 0; i < 2 * NMO; i++) mo2active[i] = -1;

    if (PsiT_MO.extent(0) == 1)
    {

      std::vector<int> active_combined;
      for (int i = 0; i < npol*NMO; i++)
      {
        if (walker_type == COLLINEAR) {
          if (orb_counts[i] >= 0 || orb_counts[i + NMO] >= 0) {
            if(orb_counts[i] >= 0) mo2active[i] = active_combined.size();
            if(orb_counts[i+NMO] >= 0) mo2active[i+NMO] = active_combined.size();
            active_combined.push_back(i);
          }
        } else {
          if (orb_counts[i] >= 0) { 
            mo2active[i] = active_combined.size();
            active_combined.push_back(i);
          } 
        } 
      }

      // RHF/GHF reference
      auto nnzpr = get_nnz(PsiT_MO(0), active_combined.data(), active_combined.size(), 0);
      PsiT(0) = PsiT_Matrix<HOST_MEMORY>({active_combined.size(),npol*NMO},nnzpr);
      {
        auto vals = PsiT_MO(0).values();
        auto cols = PsiT_MO(0).columns();
        auto row_begin = PsiT_MO(0).row_begin();
        auto row_end = PsiT_MO(0).row_end();
        for (int k = 0; k < active_combined.size(); k++)
        {
          size_t ki = active_combined[k]; // occupied state #k
          // change alpha occupation from ki to k
          for (long ic = row_begin(ki); ic < row_end(ki); ic++)
            PsiT(0).emplace_back({k, cols(ic)}, vals(ic));
        }
      }

    }
    else
    {
      // UHF reference
      std::vector<int> active_alpha;
      std::vector<int> active_beta;
      for (int i = 0; i < npol*NMO; i++)
      {
        if(orb_counts[i] >= 0) active_alpha.push_back(i);
        if(orb_counts[i + NMO] >= 0) active_beta.push_back(i);
      }

      {
        auto nnzpr = get_nnz(PsiT_MO[0], active_alpha.data(), active_alpha.size(), 0);
        PsiT(0) = PsiT_Matrix<HOST_MEMORY>({active_alpha.size(),npol*NMO},nnzpr);
        auto vals = PsiT_MO(0).values();
        auto cols = PsiT_MO(0).columns();
        auto row_begin = PsiT_MO(0).row_begin();
        auto row_end = PsiT_MO(0).row_end();
        for (int k = 0; k < active_alpha.size(); k++)
        {
          size_t ki = active_alpha[k]; // occupied state #k
          // change alpha occupation from ki to k
          mo2active[ki] = k;
          for (long ic = row_begin(ki); ic < row_end(ki); ic++)
            PsiT(0).emplace_back({k, cols(ic)}, vals(ic));
        }
      }
      {
        auto nnzpr = get_nnz(PsiT_MO[1], active_beta.data(), active_beta.size(), 0);
        PsiT(1) = PsiT_Matrix<HOST_MEMORY>({active_beta.size(),npol*NMO},nnzpr);
        auto vals = PsiT_MO(1).values();
        auto cols = PsiT_MO(1).columns();
        auto row_begin = PsiT_MO(1).row_begin();
        auto row_end = PsiT_MO(1).row_end();
        for (int k = 0; k < active_beta.size(); k++)
        { 
          // change beta occupation from ki to k
          size_t ki = active_beta[k]; // occupied state #k
          mo2active[ki+NMO] = k;
          for (long ic = row_begin(ki); ic < row_end(ki); ic++)
            PsiT(1).emplace_back({k, cols(ic)}, vals(ic));
        }
      }
    }
    // now that mappings have been constructed, map indexes of excited state orbitals
    // to the corresponding active space indexes
    {
      // map reference
      auto refc = abij.reference_configuration();
      for (int i = 0; i < nup + ndown; i++, ++refc)
        *refc = mo2active[*refc];
      for (int n = 1; n < abij.maximum_excitation_number()[0]; n++)
      {
        auto it  = abij.alpha_begin(n);
        auto ite = abij.alpha_end(n);
        for (; it < ite; ++it)
        {
          auto exct = (*it) + n; // only need to map excited state indexes
          for (int np = 0; np < n; ++np, ++exct)
            *exct = mo2active[*exct];
        }
      }
      for (int n = 1; n < abij.maximum_excitation_number()[1]; n++)
      {
        auto it  = abij.beta_begin(n);
        auto ite = abij.beta_end(n);
        for (; it < ite; ++it)
        {
          auto exct = (*it) + n; // only need to map excited state indexes
          for (int np = 0; np < n; ++np, ++exct) 
            *exct = mo2active[*exct];
        }
      }
    }

    getInitialGuess(ngrp, name, NMO, nup, ndown, walker_type);

    auto n_unique(abij.number_of_unique_excitations());
    app_log(1,"Number of unique determinants per spin channel: {} {} ",
                n_unique[0],n_unique[1]);
    nda::array<int,1> counts_alpha(n_unique[0],0);
    nda::array<int,1> counts_beta(n_unique[1],0);
    {
      for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
      {
        ++counts_alpha(std::get<0>(*it));
        ++counts_beta(std::get<1>(*it));
      }
    }

    using ucsr_mat_t = math::sparse::ucsr_matrix<ComplexType, HOST_MEMORY, int, int>;
    std::vector<ucsr_mat_t> unsorted_det_coupling;
    unsorted_det_coupling.reserve(2);
    unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[0],n_unique[1]}, counts_alpha));
    unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[1],n_unique[0]}, counts_beta));

    {
      int ni = 0;
      for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it, ++ni)
      {
        // sparse matrix
        unsorted_det_coupling[0].emplace({std::get<0>(*it),std::get<1>(*it)},
                                       std::conj(std::get<2>(*it)));
        unsorted_det_coupling[1].emplace({std::get<1>(*it),std::get<0>(*it)},
                                       std::conj(std::get<2>(*it)));
      }
    }

    // csr matrix with determinant couplings in COLLINEAR case
    nda::array<PsiT_Matrix<MEM>,1> det_coupling_matrix;
    det_coupling_matrix.resize(2);
    det_coupling_matrix(0) = unsorted_det_coupling[0];
    det_coupling_matrix(1) = unsorted_det_coupling[1];

    // PsiT carries n_ref reference(s): 1 for RHF/GHF (combined), 2 for UHF.
    // The Hamiltonian expects one entry per spin channel (nspin), so for a single
    // combined reference under COLLINEAR we duplicate it across both spins -- the
    // same walker-type conversion every other wavefunction path performs on read.
    int const n_ref = PsiT_MO.extent(0);
    int const nspin = (walker_type == COLLINEAR ? 2 : 1);

    // 2d version just for getHamiltonianOperations (copy, so PsiT_MO survives)
    nda::array<PsiT_Matrix<MEM>, 2> PsiT_2d(1, nspin);
    for(int i=0; i<nspin; i++) {
      PsiT_2d(0,i) = PsiT_MO(i % n_ref);
    }
    auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, PsiT_2d);

    // 1-d array for PHMSD keeps the original reference count (1 or 2)
    nda::array<PsiT_Matrix<MEM>, 1> PsiT_1d(n_ref);
    for(int i=0; i<n_ref; i++) {
      PsiT_1d(i) = std::move(PsiT_MO(i));
    }

    return Wavefunction<MEM>(PHMSD<MEM>(params, walker_type, NMO, nup, ndown, mpi, std::move(HOps),
                    std::move(abij), std::move(det_coupling_matrix),
                    std::move(PsiT_1d), targetNW));
  }
  else
  {
    APP_ABORT("Error: Unknown wave-function wfn_type: {}", wfn_type);
  }
}

/*
 * Read Initial walker from file. Needs all mpi tasks, since it allocates on shared memory.
*/
template<MEMORY_SPACE MEM>
void WavefunctionFactory<MEM>::getInitialGuess_ft(h5::group grp,
         utils::mpi_context_t<boost::mpi3::communicator>& mpi,
         const std::string& name, int NMO, WALKER_TYPES walker_type, bool finiteT)
{

    utils::check(finiteT, "Error: attempting to read finite-T wfn with finiteT flag set to false");

  using nda::range;
  int nspin = walker_type == COLLINEAR ? 2 : 1;
  int npol = walker_type == NONCOLLINEAR ? 2 : 1;
  nda::array<int,1> dims(5);
  nda::h5_read(grp,"dims",dims);
  
  WALKER_TYPES wtype(initWALKER_TYPES(dims[3]));
  utils::check(walkerTypeIsConvertible(wtype, walker_type), "Initial guess ({}) not convertible to walker_type {}", walkerTypeToString(wtype), walkerTypeToString(walker_type));
  
  auto guess = initial_guess_ft.find(name);
  utils::check(guess == initial_guess_ft.end(), 
             "Error: Problems adding new initial guess, already exists.");
  auto newg = initial_guess_ft.insert(std::make_pair(name, memory::share_from_root(mpi, [&] {
    nda::array<ComplexType,4> M(3, nspin, npol * NMO, NMO);
    M() = ComplexType(0.0, 0.0);
    auto URup = M(0,0,nda::ellipsis{});
    utils::h5_read(grp,"UR_alpha",URup);
    auto DRup = M(1,0,nda::ellipsis{});
    utils::h5_read(grp,"DR_alpha",DRup);
    auto VRup = M(2,0,nda::ellipsis{});
    utils::h5_read(grp,"VR_alpha",VRup);
    if (walker_type == COLLINEAR)
    {
      if (wtype == COLLINEAR)
      {
        auto URdn = M(0,1,nda::range::all,nda::range(NMO));
        utils::h5_read(grp,"UR_beta",URdn);
        auto DRdn = M(1,1,nda::range::all,nda::range(NMO));
        utils::h5_read(grp,"DR_beta",DRdn);
        auto VRdn = M(2,1,nda::range::all,nda::range(NMO));
        utils::h5_read(grp,"VR_beta",VRdn);
      }
      else if (wtype == CLOSED)
      {
        M(0,1,nda::ellipsis{}) = URup();
        M(1,1,nda::ellipsis{}) = DRup();
        M(2,1,nda::ellipsis{}) = VRup();
      }
      else
        utils::check(false,"Error: Unknown wtype. ");
    }
    return M;
  })));
  utils::check(newg.second, "Error: Problems adding new initial guess. ");
}

/*
 * Read Initial walker from file. Needs all mpi tasks, since it allocates on shared memory.
*/
template<MEMORY_SPACE MEM>
void WavefunctionFactory<MEM>::getInitialGuess(h5::group grp,
         const std::string& name, int NMO, int nup, int ndown, WALKER_TYPES walker_type)
{
  using nda::range;
  auto all = range::all;

  nda::array<int,1> dims(5);
  nda::h5_read(grp,"dims",dims);
  
  auto nel_in_guess = std::to_array({dims[1],dims[2]});

  WALKER_TYPES wtype(initWALKER_TYPES(dims[3]));
  utils::check(walkerTypeIsConvertible(wtype, walker_type), "Initial guess ({}) not convertible to walker_type {}", walkerTypeToString(wtype), walkerTypeToString(walker_type));
  auto guess = initial_guess.find(name);
  utils::check(guess == initial_guess.end(), 
             "Error: Problems adding new initial guess, already exists.");
  auto [nspin_in_guess, npol_in_guess] = walkerTypeToDims(wtype);

  // Read the trial's per-spin orbital matrices at their true (in-file) widths.
  std::array<std::string,2> dataset_names{{"Psi0_alpha", "Psi0_beta"}};
  std::vector<nda::matrix<ComplexType>> Min;
  Min.reserve(nspin_in_guess);
  for(int is = 0; is < nspin_in_guess; is++) {
    utils::check(nup >= nel_in_guess[is], "initial guess contains more electrons of spin {} than walker nup ({})", nel_in_guess[is], nup);
    nda::matrix<ComplexType> m(npol_in_guess * NMO, nel_in_guess[is]);
    m() = ComplexType(0.0);
    utils::h5_read(grp, dataset_names[is], m);
    Min.push_back(std::move(m));
  }

  auto [nspin, npol] = walkerTypeToDims(walker_type);
  // Walker-sized per-spin widths: alpha=nup, beta=ndown (collinear). Kept exact
  // (no max-padding) so naeb is recoverable from the beta matrix's width.
  std::array<int,2> out_width{{nup, ndown}};

  std::vector<nda::matrix<ComplexType>> M;
  M.reserve(nspin);
  if(walker_type == NONCOLLINEAR and wtype != NONCOLLINEAR) {
    // Interleave the (NMO-row) spin channels into one 2*NMO-row matrix.
    nda::matrix<ComplexType> a(npol * NMO, nup);
    a() = ComplexType(0.0);
    auto a3 = reshape(a, npol, NMO, nup);
    int offset = 0;
    for(int ip = 0; ip < npol; ip++) {
      a3(ip, all, range(offset, offset + nel_in_guess[ip])) =
          Min[ip % nspin_in_guess](all, range(nel_in_guess[ip]));
      offset += nel_in_guess[ip];
    }
    M.push_back(std::move(a));
  } else {
    for(int is = 0; is < nspin; is++) {
      nda::matrix<ComplexType> m(npol * NMO, out_width[is]);
      m() = ComplexType(0.0);
      int src = is % nspin_in_guess;
      int nc  = std::min<int>(out_width[is], nel_in_guess[src]);
      m(all, range(nc)) = Min[src](all, range(nc));
      M.push_back(std::move(m));
    }
  }

  auto newg = initial_guess.insert(std::make_pair(name, std::move(M)));
  utils::check(newg.second, "Error: Problems adding new initial guess.");
}

/*
void WavefunctionFactory::computeVariationalEnergyPHMSD(TaskGroup_& TG,
                                                        Hamiltonian& ham,
                                                        boost::multi::array_ref<int, 2>& occs,
                                                        std::vector<ComplexType>& coeff,
                                                        int ndets,
                                                        int nup,
                                                        int ndown,
                                                        int NMO,
                                                        bool recompute_ci)
{
  // CI coefficients can in general be complex and want to avoid two mpi communications so
  // keep everything complex even if Hamiltonian matrix elements are real.
  // Allocate H in Node's shared memory, but use as a raw array with proper synchronization
  int dim((recompute_ci ? ndets : 0));
  boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>> H({dim, dim}, TG.Node());
  boost::multi::array<ComplexType, 1> energy(iextensions<1u>{2});
  using std::fill_n;
  fill_n(H.origin(), H.num_elements(), ComplexType(0.0));           // this call synchronizes
  fill_n(energy.origin(), energy.num_elements(), ComplexType(0.0)); // this call synchronizes
  ComplexType enuc = ham.getNuclearCoulombEnergy();
  for (int idet = 0; idet < ndets; idet++)
  {
    // These should already be sorted.
    boost::multi::array_ref<int, 1> deti(occs[idet].origin(), {nup + ndown});
    ComplexType cidet = coeff[idet];
    for (int jdet = idet; jdet < ndets; jdet++)
    {
      // Compute <Di|H|Dj>
      if ((idet * ndets + jdet) % TG.Global().size() == TG.Global().rank())
      {
        if (idet == jdet)
        {
          ComplexType Hii(0.0);
          Hii = slaterCondon0(ham, deti, NMO) + enuc;
          energy[0] += ma::conj(cidet) * cidet * Hii;
          energy[1] += ma::conj(cidet) * cidet;
          if (recompute_ci)
            H[idet][idet] = Hii;
        }
        else
        {
          ComplexType Hij(0.0);
          boost::multi::array_ref<int, 1> detj(occs[jdet].origin(), {nup + ndown});
          ComplexType cjdet = coeff[jdet];
          int perm          = 1;
          std::vector<int> excit;
          int nexcit = getExcitation(deti, detj, excit, perm);
          if (nexcit == 1)
          {
            Hij = ComplexType(perm) * slaterCondon1(ham, excit, detj, NMO);
          }
          else if (nexcit == 2)
          {
            Hij = ComplexType(perm) * slaterCondon2(ham, excit, NMO);
          }
          energy[0] += ma::conj(cidet) * cjdet * Hij + ma::conj(cjdet) * cidet * ma::conj(Hij);
          if (recompute_ci)
          {
            H[idet][jdet] = Hij;
            H[jdet][idet] = ma::conj(Hij);
          }
        }
      }
    }
  }
  TG.Node().barrier();
  if (TG.Node().root() && recompute_ci)
    TG.Cores().all_reduce_in_place_n(raw_pointer_cast(H.origin()), H.num_elements(), std::plus<>());
  TG.Global().all_reduce_in_place_n(energy.origin(), 2, std::plus<>());
  app_log(1," - Variational energy of trial wavefunction: {}", energy[0] / energy[1]);
  if (recompute_ci)
  {
    app_log(1," - Diagonalizing CI matrix.");
    using RVector = Vector<RealType>;
    using CMatrix = Matrix<ComplexType>;
    // Want a "unique" solution for all cores/nodes.
    if (TG.Global().rank() == 0)
    {
      std::pair<RVector, CMatrix> Sol = ma::symEigSelect<RVector, CMatrix>(H, 1);
      app_log(1," - Updating CI coefficients. ");
      app_log(1," - Recomputed coefficient of first determinant: {}", Sol.second[0][0]);
      for (int idet = 0; idet < ndets; idet++)
      {
        ComplexType ci = Sol.second[0][idet];
        // Do we want this much output?
        //app_log() << idet << " old: " << coeff[idet] << " new: " << ci << "";
        coeff[idet] = ci;
      }
      app_log(1," - Recomputed variational energy of trial wavefunction: {}",Sol.first[0]);
    }
    TG.Global().broadcast_n(raw_pointer_cast(coeff.data()), coeff.size(), 0);
  }
}

/ **
 * Compute the excitation level between two determinants.
 * /
int WavefunctionFactory::getExcitation(boost::multi::array_ref<int, 1>& deti,
                                       boost::multi::array_ref<int, 1>& detj,
                                       std::vector<int>& excit,
                                       int& perm)
{
  std::vector<int> from_orb, to_orb;
  // Work out which orbitals are excited from / to.
  std::set_difference(detj.begin(), detj.end(), deti.begin(), deti.end(), std::inserter(from_orb, from_orb.begin()));
  std::set_difference(deti.begin(), deti.end(), detj.begin(), detj.end(), std::inserter(to_orb, to_orb.begin()));
  int nexcit = from_orb.size();
  if (nexcit <= 2)
  {
    for (int i = 0; i < from_orb.size(); i++)
      excit.push_back(from_orb[i]);
    for (int i = 0; i < to_orb.size(); i++)
      excit.push_back(to_orb[i]);
    int nperm = 0;
    int nmove = 0;
    for (auto o : from_orb)
    {
      auto it = std::find(detj.begin(), detj.end(), o);
      int loc = std::distance(detj.begin(), it);
      nperm += loc - nmove;
      nmove += 1;
    }
    nmove = 0;
    for (auto o : to_orb)
    {
      auto it = std::find(deti.begin(), deti.end(), o);
      int loc = std::distance(deti.begin(), it);
      nperm += loc - nmove;
      nmove += 1;
    }
    perm = nperm % 2 == 1 ? -1 : 1;
  }
  return nexcit;
}

ComplexType WavefunctionFactory::slaterCondon0([[maybe_unused]] Hamiltonian& ham, 
                                               [[maybe_unused]] boost::multi::array_ref<int, 1>& det, 
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon0 Feature removed.");
/ *
  ComplexType one_body = ComplexType(0.0);
  ComplexType two_body = ComplexType(0.0);
  for (int i = 0; i < det.size(); i++)
  {
    int oi = det[i];
    one_body += ComplexType(ham.H(oi, oi));
    for (int j = i + 1; j < det.size(); j++)
    {
      int oj = det[j];
      two_body += ComplexType(ham.H(oi, oj, oi, oj)) - ComplexType(ham.H(oi, oj, oj, oi));
    }
  }
  return one_body + two_body;
* /
  return ComplexType(0.0);
}

ComplexType WavefunctionFactory::slaterCondon1([[maybe_unused]] Hamiltonian& ham,
                                               [[maybe_unused]] std::vector<int>& excit,
                                               [[maybe_unused]] boost::multi::array_ref<int, 1>& det,
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon1 Feature removed.");
/ *
  int i              = excit[0];
  int a              = excit[1];
  ComplexType one_body = ComplexType(ham.H(i, a));
  ComplexType two_body = ComplexType(0.0);
  for (auto j : det)
  {
    two_body += ComplexType(ham.H(i, j, a, j)) - ComplexType(ham.H(i, j, j, a));
  }
  return one_body + two_body;
* /
  return ComplexType(0.0);
}

ComplexType WavefunctionFactory::slaterCondon2([[maybe_unused]] Hamiltonian& ham, 
                                               [[maybe_unused]] std::vector<int>& excit, 
                                               [[maybe_unused]] int NMO)
{
  APP_ABORT("Error: slaterCondon2 Feature removed.");
/ *
  int i = excit[0];
  int j = excit[1];
  int a = excit[2];
  int b = excit[3];
  return ComplexType(ham.H(i, j, a, b) - ham.H(i, j, b, a));
* /
  return ComplexType(0.0);
}
*/
template<MEMORY_SPACE MEM>
void WavefunctionFactory<MEM>::build_PsiT_MO_phmsd(WALKER_TYPES walker_type, int npol, int NMO, int nup, 
      int ndown, int ndets, nda::array<ComplexType,1>& coeffs,
      nda::array<int,2>& occs, nda::array<PsiT_Matrix<HOST_MEMORY>,1>& PsiT_MO)
{
  using nda::range;
  auto all = range::all;
  ComplexType one(1.0, 0.0);
  bool trivial_ref = true;
  for(int i=0; i<nup; i++)
    if(occs(0,i) != i) {
      trivial_ref = false;
      break;
    }
  for(int i=0; i<ndown; i++)
    if(occs(0,nup+i) != NMO+i) {
      trivial_ref = false;
      break;
    }

  auto sort_with_sign = [](int NE, auto&& Iwork) {
    RealType ci=1.0;
    for (int i = 0; i < NE; i++)
      for (int j = i + 1; j < NE; j++)
      {
        if (Iwork[j] < Iwork[i])
        {
          ci *= RealType(-1.0);
          std::swap(Iwork[i], Iwork[j]);
        }
      }
    return ci;
  };

  if(trivial_ref) {

    PsiT_MO.resize(1);
    PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({npol*NMO,npol*NMO},1);

    // makes sense to move the reordering of non-compact excitations to here!
    for (int k = 0; k < npol*NMO; k++)
      PsiT_MO(0).emplace_back({k, k}, one);

  } else {

    // reference determinant is non-trivial (occupy bottom nalpha/nbeta states...)
    // build non-trivial reference and redefine excitations with respect to this new reference...
    app_log(1,"Found non-trivial reference determinant. Constructing appropriate reference state.");

    // if beta reference configuration has singly occupied states, 
    // you will need separate references
    bool separate_references = false;
    if(walker_type == COLLINEAR)
      for(int i=0; i<ndown; ++i) 
        if( *std::find(occs.begin(), occs.begin() + nup, occs(0,nup+i)-NMO) !=
          occs(0,nup+i)-NMO ) {
          separate_references = true;
          break;
        }

    if(separate_references) { // only if collinear and can't find a unique reference


      PsiT_MO.resize(2);
      PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({NMO,NMO},1);
      PsiT_MO(1) = PsiT_Matrix<HOST_MEMORY>({NMO,NMO},1);

      {

        // each spin has its own reference
        int dN_[2] = {0,NMO};
        int E0_[2] = {0,nup};
        int E1_[2] = {nup,nup+ndown};
        for(int is=0; is<2; is++) {
          int dN = dN_[is]; 
          int E0 = E0_[is]; 
          int E1 = E1_[is]; 
          std::vector<int> m(NMO,-1);  
          std::vector<int> im(NMO,-1);
          int norbs=0;  // number of states found so far
          // doubly occupied first, and since we checked all beta are doubly occp
          for(int i=E0; i<E1; i++) { 
            im[occs(0,i)-dN] = norbs;
            m[norbs++] = occs(0,i)-dN;
          }
          // now add all remaining states
          for(int n=1; n<occs.extent(0); ++n) { 
            for(int i=E0; i<E1; i++) 
              if(im[occs(n,i)-dN] < 0) {
                im[occs(n,i)-dN] = norbs;
                m[norbs++] = occs(n,i)-dN;
              } 
          }
          // now change occupation strings according to the generated map
          for(int n=0; n<occs.extent(0); ++n) { 
            for(int i=E0; i<E1; i++) occs(n,i) = im[occs(n,i)-dN]+dN;    
            coeffs[n] *= sort_with_sign(E1-E0,occs(n,range(E0,E1)));
          }
          for (int k = 0; k < norbs; k++) 
            PsiT_MO(is).emplace_back({k, m[k]}, one);
        } // is
    
      } 

    } else { // separate_references

      PsiT_MO.resize(1);
      PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({npol*NMO,npol*NMO},1); 

      {

        std::vector<int> m(npol*NMO,-1);  
        std::vector<int> im(npol*NMO,-1);
        int norbs=0;  // number of states found so far
        if(walker_type == NONCOLLINEAR) {
          for(int i=0; i<nup; i++) {
	    im[occs(0,i)] = norbs;
	    m[norbs++] = occs(0,i);
	  }
          // now add all remaining states
          for(int n=1; n<occs.extent(0); ++n) {
            for(int i=0; i<nup; i++) 
              if(im[occs(n,i)] < 0) {
                im[occs(n,i)] = norbs;
                m[norbs++] = occs(n,i);
              }
          }
        } else {
          // doubly occupied first, and since we checked all beta are doubly occp
          for(int i=0; i<ndown; i++) { 
            im[occs(0,nup+i)-NMO] = norbs;
            m[norbs++] = occs(0,nup+i)-NMO;
          }
          // singly occupied now
          for(int i=0; i<nup; ++i) 
            if( *std::find(occs.begin()+nup, occs.begin()+nup+ndown, 
		occs(0,i)+NMO) != occs(0,i)+NMO ) { 
              im[occs(0,i)] = norbs;
              m[norbs++] = occs(0,i);
	    }
          // now add all remaining states
          for(int n=1; n<occs.extent(0); ++n) { 
            for(int i=0; i<nup; i++) 
              if(im[occs(n,i)] < 0) {
                im[occs(n,i)] = norbs;
                m[norbs++] = occs(n,i);
              } 
            for(int i=nup; i<nup+ndown; i++)        
              if(im[occs(n,i)-NMO] < 0) {
                im[occs(n,i)-NMO] = norbs;
                m[norbs++] = occs(n,i)-NMO;
              }
          }
        }
        // now change occupation strings according to the generated map
        for(int n=0; n<occs.extent(0); ++n) { 
          for(int i=0; i<nup; i++) occs(n,i) = im[occs(n,i)];    
          if(walker_type == COLLINEAR)  
            for(int i=nup; i<nup+ndown; i++) occs(n,i) = im[occs(n,i)-NMO]+NMO;    
          coeffs[n] *= sort_with_sign(nup+ndown,occs(n,all));
        }
        for (int k = 0; k < norbs; k++) 
          PsiT_MO(0).emplace_back({k, m[k]}, one);

      } 

    } // separate_references

  } // trivial_ref
  // generate compact form
  for(int i=0; i<PsiT_MO.size(); ++i)
    PsiT_MO(i).remove_empty_spaces();
}

// Instantiate templates

template Wavefunction<HOST_MEMORY> WavefunctionFactory<HOST_MEMORY>::fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,const WavefunctionParameters&,WALKER_TYPES,bool,Hamiltonian&,int);

#if defined(ENABLE_DEVICE)

template Wavefunction<DEVICE_MEMORY> WavefunctionFactory<DEVICE_MEMORY>::fromHDF5(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>>,const WavefunctionParameters&,WALKER_TYPES,bool,Hamiltonian&,int);

#endif

} // namespace afqmc
} // namespace sfqmc
