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

#include <cstdlib>
#include <algorithm>
#include <complex>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <map>
#include <utility>
#include <vector>
#include <numeric>
#include <functional>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "hdf/hdf_multi.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/type_conversion.hpp"
#include "AFQMC/Utilities/kp_utilities.hpp"
#include "KPFactorizedHamiltonian.h"
#include "AFQMC/SlaterDeterminantOperations/rotate.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "upgradeOneBodyIntegrals.hpp"

namespace sfqmc
{
namespace afqmc
{

#if !defined(ENABLE_DEVICE)
template<bool MP> HamiltonianOperations<MP> 
KPFactorizedHamiltonian::getHamiltonianOperations_shared(WALKER_TYPES type,
							 std::vector<PsiT_Matrix>& PsiT,
							 TaskGroup_& TGprop,
							 TaskGroup_& TGwfn,
							 hdf_archive& hdf_restart)
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;
  using shmIMatrix    = boost::multi::array<int, 2, shared_allocator<int>>;
  using shmCMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using shmCTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using shmSpMatrix   = boost::multi::array<SPComplexType, 2, shared_allocator<SPComplexType>>;
  using IVector       = boost::multi::array<int, 1>;
  using SpMatrix      = boost::multi::array<SPComplexType, 2>;
  using SpTensor5     = boost::multi::array<SPComplexType, 5>;
  using SpMatrix_ref  = boost::multi::array_ref<SPComplexType, 2>;
  using Sp3Tensor_ref = boost::multi::array_ref<SPComplexType, 3>;

  std::string base_error(" Error in KPFactorizedHamiltonian::getHamiltonianOperations_shared: \n    ");

  if (TGprop.TG() != TGwfn.TG())
  {
    app_error(base_error + 
	" Requires nnodes to be the same in Wavefunction and Propagator xml blocks.");
    APP_ABORT("Error: Inconsistent nnodes in KPFactorizedHamiltonian ");
  }

  // hack until parallel hdf is in place
  bool write_hdf = false;
  if (TGwfn.Global().root())
    write_hdf = !hdf_restart.closed();
  //  if(TGwfn.Global().root()) write_hdf = (hdf_restart.file_id != hdf_archive::is_closed);
  TGwfn.Global().broadcast_value(write_hdf);
  bool unfold_ibz = false;

  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");
  int nspins = ((type != COLLINEAR) ? 1 : 2);
  int ndet   = PsiT.size() / nspins;
  int npol   = ((type == NONCOLLINEAR) ? 2 : 1);

  if (ndet > 1)
    APP_ABORT(base_error + "ndet > 1 not yet implemented");

  std::vector<int> hcore_shape(5);
  long nkpts;
  hdf_archive dump(TGwfn.Global());
  // right now only Node.root() reads
  if (TG.Node().root())
  {
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(base_error + "Error opening integral file");
  }
  std::string format = get_hamiltonian_format(dump,TG.Global());

  if (TG.Global().root())
  {
    if(format == "std") {
      std::vector<int> Idata(8);
      if (!dump.readEntry(Idata, "Hamiltonian/dims"))
        APP_ABORT(base_error + "Problems reading dims");
      nkpts = Idata[2];
    } else if(format == "coqui") {
      if (dump.push("System", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System not found. ");
      if (dump.push("BZ", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System/BZ not found. ");
      if (!dump.readAttributeEntry(nkpts, "number_of_kpoints"))
        APP_ABORT(base_error + "Problems reading attribute /System/BZ/number_of_kpoints");
      dump.pop();
      dump.pop();
    } else {
      APP_ABORT(base_error + "Unknown format:{}",format);
    } 
  }
  TG.Global().broadcast_n(&nkpts, 1, 0);
  app_log(1," nkpts: {}", nkpts);

  // partition Q over nodes if distributed Q
  IVector nmo_per_kp(iextensions<1u>{nkpts});
  IVector nchol_per_kp(iextensions<1u>{nkpts});
  IVector Qminus(iextensions<1u>{nkpts});
  IVector Qmap(iextensions<1u>{nkpts});
  shmIMatrix QKtok2({nkpts, nkpts}, shared_allocator<int>{TG.Node()});
  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;
  if (TG.Global().root())
  {
    if(format == "std") {
      if (!dump.readEntry(nmo_per_kp, "Hamiltonian/NMOPerKP"))
        APP_ABORT(base_error + " Problems reading NMOPerKP. ");
      if (!dump.readEntry(nchol_per_kp, "Hamiltonian/NCholPerKP"))
        APP_ABORT(base_error + "Problems reading NCholPerKP. ");
      if (!dump.readEntry(Qminus, "Hamiltonian/MinusK"))
        APP_ABORT(base_error + "Problems reading MinusK. ");
      if (!dump.readEntry(QKtok2, "Hamiltonian/QKTok2"))
        APP_ABORT(base_error + "Problems reading QKTok2. ");
      std::vector<RealType> E_(2);
    } else if(format == "coqui"){
      if (!dump.readEntry(Qminus, "/System/BZ/qminus"))
        APP_ABORT(base_error + "Problems reading /System/BZ/qminus. ");      
      if (!dump.readEntry(QKtok2, "/System/BZ/qk_to_k2"))
        APP_ABORT(base_error + "Problems reading /System/BZ/qk_to_k2. ");      
      long nbnd;
      if (dump.push("System", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System not found. ");
      if (!dump.readAttributeEntry(nbnd, "number_of_bands"))
        APP_ABORT(base_error + "Problems reading /System/number_of_bands. ");
      dump.pop();
      for(int k=0; k<nkpts; k++)
        nmo_per_kp[k] = nbnd; 
      std::vector<int> shape(6);
      for(int q=0; q<nkpts; q++) {
        if (!dump.getShape<RealType>("/Interaction/Vq"+std::to_string(q), shape))
          APP_ABORT(base_error + "Problems reading /Interaction/Vq" + std::to_string(q));      
        nchol_per_kp[q] = shape[0];
      }
    }
    for (int q = 0; q < nkpts; q++)
    {
      if (Qminus[q] < q)
        nchol_per_kp[q] = nchol_per_kp[Qminus[q]];
    }
    if (nmo_per_kp.size() != nkpts || nchol_per_kp.size() != nkpts || Qminus.size() != nkpts ||
        QKtok2.size(0) != nkpts || QKtok2.size(1) != nkpts) 
    {
      app_error(" Inconsistent dimension (NMOPerKP,NCholPerKP,QKtTok2)"); 
      app_error(" nkpts={}, nmo_per_kp.size={}, nchol_per_kp.size={}, QKtok2.shape:({},{}) ",
      		    nkpts, nmo_per_kp.size(), nchol_per_kp.size(), Qminus.size(), 
                    QKtok2.size(0), QKtok2.size(1));
      APP_ABORT(base_error);
    }
  }
  TG.Global().broadcast_n(&E0, 1, 0);
  TG.Global().broadcast_n(nmo_per_kp.origin(), nmo_per_kp.size(), 0);
  TG.Global().broadcast_n(nchol_per_kp.origin(), nchol_per_kp.size(), 0);
  TG.Global().broadcast_n(Qminus.origin(), Qminus.size(), 0);
  if (TG.Node().root())
    TG.Cores().broadcast_n(raw_pointer_cast(QKtok2.origin()), QKtok2.num_elements(), 0);
  TG.Node().barrier();

  int number_of_symmetric_Q = 0;
  int global_origin(0);
  // Defines behavior over Q vector:
  //   <0: Ignore (handled by another TG)
  //    0: Calculate, without rho^+ contribution
  //   >0: Calculate, with rho^+ contribution. LQKbln data located at Qmap[Q]-1
  std::fill_n(Qmap.origin(), Qmap.num_elements(), -1);
  {
    int ngrp(TGwfn.getNGroupsPerTG());
    int ig(TGwfn.getLocalGroupNumber());
    int work(0);
    // assign Q/Qm pairs of vectors to groups round-robin
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qminus[Q] == Q)
      {
        if (work % ngrp == ig)
          Qmap[Q] = 1 + (number_of_symmetric_Q++);
        if (work % ngrp < ig)
          global_origin += 2 * nchol_per_kp[Q];
        work++;
      }
      else if (Q < Qminus[Q])
      {
        if (work % ngrp == ig)
        {
          Qmap[Q]         = 0;
          Qmap[Qminus[Q]] = 0;
        }
        if (work % ngrp < ig)
          global_origin += 4 * nchol_per_kp[Q];
        work++;
      }
    }
    if (work < ngrp)
      APP_ABORT(" Error: Too many nodes in group (nnodes) for given number of kpoints. ");
  }
  // new communicator over nodes that share the same set of Q
  auto Qcomm=TG.Global().split(TGwfn.getLocalGroupNumber(), TG.Global().rank());
  auto Qcomm_roots=Qcomm.split(TGwfn.Node().rank(), Qcomm.rank());

  int nmo_max   = *std::max_element(nmo_per_kp.begin(), nmo_per_kp.end());
  int nchol_max = *std::max_element(nchol_per_kp.begin(), nchol_per_kp.end());
  int nspins_H1 = 1;
  int nkpts_H1_ints = 1;

  // check if H1 is spin dependent
  // right now, I only check if "H1_kp{2*nkpts-1}" exists. Other Qs are checked below
  if (type == COLLINEAR) {
    if (TG.Global().root()) {
      if(format == "std") {
        int K = 2*nkpts-1;
        int nmo_K = nmo_per_kp[nkpts-1];
        boost::multi::array<ComplexType, 2> h1({npol * nmo_K, npol * nmo_K});
        std::string h_id =  std::string("Hamiltonian/H1_kp") + std::to_string(K); 
        if (dump.readEntry(h1, h_id))  
          nspins_H1 = 2;
      } else if(format == "coqui") {
        // MAM: assuming all kpoints are consistent
        std::string h_id = std::string("System/H0");
        if (!dump.getShape<RealType>(h_id, hcore_shape))
          APP_ABORT(base_error + "Problems reading " + h_id);
        nspins_H1 = hcore_shape[0];
        nkpts_H1_ints = hcore_shape[1];  // this could be nkpts or nkpts_ibz!
      }
    }
    TG.Global().broadcast_value(nspins_H1);
    TG.Global().broadcast_value(nkpts_H1_ints);
    TG.Global().broadcast_n(hcore_shape.data(), 5, 0);
  } else if (type == NONCOLLINEAR) {
    if (TG.Global().root()) {
      if(format == "coqui") {
          // MAM: assuming all kpoints are consistent
          std::string h_id = std::string("System/H0");
          if (!dump.getShape<RealType>(h_id, hcore_shape))
            APP_ABORT(base_error + "Problems reading " + h_id);
          nspins_H1 = hcore_shape[0];
          nkpts_H1_ints = hcore_shape[1];  // this could be nkpts of nkpts_ibz!
      } else if (format == "std") {
        APP_ABORT(base_error + "Noncollinear not implemented for standard Hamiltonian format.");
      }
    }
    TG.Global().broadcast_value(nspins_H1);
    TG.Global().broadcast_value(nkpts_H1_ints);
    TG.Global().broadcast_n(hcore_shape.data(), 5, 0);
  }

  IVector kp_to_ibz(iextensions<1u>{nkpts});
  IVector kp_trev(iextensions<1u>{nkpts});
  if (format == "coqui" && nkpts_H1_ints != nkpts)
  {
    unfold_ibz = true;
    app_log(1, "Detected symmetry in Coquí HDF5 file. Unfolding kpoints from IBZ for H1. \n");
    if (TG.Global().root())
    {
      if (!dump.readEntry(kp_to_ibz, "System/BZ/kp_to_ibz"))
        APP_ABORT(base_error + "Problems reading System/BZ/kp_to_ibz");
      if (!dump.readEntry(kp_trev, "System/BZ/kp_trev_pair"))
        APP_ABORT(base_error + "Problems reading System/BZ/kp_trev_pair");
    }
    TG.Global().broadcast_n(kp_to_ibz.data(), nkpts, 0);
    TG.Global().broadcast_n(kp_trev.data(), nkpts, 0);
  }

  shmCTensor H1({nspins_H1*nkpts, npol * nmo_max, npol * nmo_max}, 
		 shared_allocator<ComplexType>{TG.Node()});
  std::vector<shmSpMatrix> LQKikn;
  LQKikn.reserve(nkpts);
  for (int Q = 0; Q < nkpts; Q++)
    if (Qmap[Q] >= 0 && Q <= Qminus[Q])
      LQKikn.emplace_back(
          shmSpMatrix({nkpts, nmo_max * nmo_max * nchol_per_kp[Q]}, shared_allocator<SPComplexType>{TG.Node()}));
    else
      LQKikn.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{TG.Node()}));

  if (TG.Node().root())
  {
    if(format == "std") {
      // now read H1_kpK
      for (int K = 0; K < nspins_H1*nkpts; K++)
      {
        int nmo_K = nmo_per_kp[K%nkpts];
        boost::multi::array<ComplexType, 2> h1({npol * nmo_K, npol * nmo_K});
        // until double_hyperslabs work!
        std::string h_id = std::string("Hamiltonian/H1_kp") + std::to_string(K); 
        if (!dump.readEntry(h1, h_id)) 
          APP_ABORT(base_error + " Problems reading " + h_id);
        // H1[K]({0,nmo_per_kp[K]},{0,nmo_per_kp[K]}) = h1;
        // using add to get raw pointer dispatch, otherwise matrix copy is going to sync
        ma::add(ComplexType(1.0), h1, ComplexType(0.0), h1, 
    	      H1[K]({0, npol * nmo_K}, {0, npol * nmo_K}));
      }
    } else if(format == "coqui") {
      if (unfold_ibz)
      {
      if (type == NONCOLLINEAR)
        if (hcore_shape[2] != npol * nmo_max || hcore_shape[3] != npol * nmo_max)
          APP_ABORT(base_error + "Upgrading H1 not yet implemented for noncollinear and unfolding from the irreducible Brillouin zone.");
      boost::multi::array<ComplexType, 4> h_ibz_({nspins_H1, nkpts_H1_ints, npol * nmo_max, npol * nmo_max});
      if (!dump.readEntry(h_ibz_, "System/H0"))
        APP_ABORT(base_error + " Problems reading System/H0");
      long Kibz;
      bool inversion_symm;
      boost::multi::array<ComplexType, 2> h_ibz({npol * nmo_max, npol * nmo_max});
      for (int K = 0; K < nkpts; K++)
      {
        for (int s = 0; s < nspins_H1; s++)
        {
        Kibz = kp_to_ibz[K];
        inversion_symm = (kp_trev[K] > 0);
        h_ibz = h_ibz_[s][Kibz];
        if (inversion_symm)
          for (int i = 0; i < npol * nmo_max; i++)
            for (int j = 0; j < npol * nmo_max; j++)
              h_ibz[i][j] = ma::conj(h_ibz[i][j]);
        ma::add(ComplexType(1.0), h_ibz, ComplexType(0.0), h_ibz, 
          H1[K]({0, npol * nmo_max}, {0, npol * nmo_max}));
        }
      }
      } else { // KE: need to be careful about spin symmetry in H1 : npol is determined by walker spin symmetry!!
          if (hcore_shape[2] != npol * nmo_max || hcore_shape[3] != npol * nmo_max )
          {
            app_log(1,"Detected mismatch between H1 shape in file and expected shape. Attempting to upgrade H1.");
            // upgrade H1 if necessary
            boost::multi::array<ComplexType, 4> h_({nspins_H1, nkpts, hcore_shape[2], hcore_shape[3]});
            if (!dump.readEntry(h_, "System/H0"))
            {  
              APP_ABORT(base_error + " Problems reading System/H0");
            }
            for(int k=0; k<nkpts; k++)
            {
              app_log(2,"Upgrading H1 for kpoint {} ", k);
              boost::multi::array_ref<ComplexType, 2> h_k(raw_pointer_cast(h_[0][k].origin()), {hcore_shape[2], hcore_shape[3]});
              boost::multi::array_ref<ComplexType, 2> H1_sk(raw_pointer_cast(H1[0][k].origin()), {npol * nmo_max, npol * nmo_max});
              std::vector<int> local_hcore_shape = {hcore_shape[2], hcore_shape[3]};
              upgradeOneBodyIntegrals<ComplexType>(h_k, H1_sk, local_hcore_shape, nmo_per_kp[k], npol, nspins_H1, type, "upgrading H1 in KPFactorizedHamiltonian");
            }
          } else {
            // fine to read directly, since nmo_per_kp == nbnd for all k
            boost::multi::array_ref<ComplexType, 4> h_(raw_pointer_cast(H1.origin()), 
                     {nspins_H1, nkpts, npol * nmo_max, npol * nmo_max});
            // no symmetry in H1, read it directly
            if (!dump.readEntry(h_, "System/H0")) 
            {
              APP_ABORT(base_error + " Problems reading System/H0");
            }
          }
      }
    }
    // read LQ
    for (int Q = 0; Q < nkpts; Q++)
    {
      using ma::conj;
      if (Qmap[Q] >= 0 && Q <= Qminus[Q])
      {
        if( format == "std" ) {
          if (!dump.readEntry(LQKikn[Q], std::string("Hamiltonian/KPFactorized/L") + std::to_string(Q)))
          {
            app_error(" Problems reading /Hamiltonian/KPFactorized/L{}", Q);
            APP_ABORT(base_error); 
          }
          if (LQKikn[Q].size(0) != nkpts || LQKikn[Q].size(1) != nmo_max * nmo_max * nchol_per_kp[Q])
          {
            app_error(" Problems reading /Hamiltonian/KPFactorized/L{}", Q);
            app_error(" Unexpected dimensions: ({}, {}) ",LQKikn[Q].size(0),LQKikn[Q].size(1));
            APP_ABORT(base_error); 
          }
        } else if( format == "coqui" ) {
          // VqQ(ichol, ispin, ik, ia, ib)
          SpTensor5 Vq;
          if (!dump.readEntry(Vq, std::string("Interaction/Vq") + std::to_string(Q)))
            APP_ABORT(base_error + "Error reading Interaction/Vq" + std::to_string(Q));
          if( (Vq.size(0) != nchol_per_kp[Q]) or
              (Vq.size(1) != 1) or
              (Vq.size(2) != nkpts) or
              (Vq.size(3) != nmo_max) or
              (Vq.size(4) != nmo_max) )
            APP_ABORT(base_error + "Invalid dimesnions in Interaction/Vq" + std::to_string(Q));
          //normalize
          ma::scal(SPComplexType(SPRealType(1.0/std::sqrt(SPRealType(nkpts)))), Vq);
          for (int K = 0; K < nkpts; K++)
          {
            int QK    = QKtok2[Q][K];
            int ni    = nmo_per_kp[K];
            int nk    = nmo_per_kp[QK];
            int nchol = nchol_per_kp[Q];
            Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {ni, nk, nchol});
            for(int n=0; n<nchol; ++n) 
              for(int i=0; i<ni; ++i) 
                for(int k=0; k<nk; ++k) 
                  Likn[i][k][n] = Vq[n][0][K][i][k];
          }
        }
      }
    }
  }
  TG.Node().barrier();

  // calculate vn0
  shmCTensor vn0({nkpts, nmo_max, nmo_max}, shared_allocator<ComplexType>{TG.Node()});

  // generate nocc_per_kp using PsiT and nmo_per_kp
  shmIMatrix nocc_per_kp({ndet, nspins * nkpts}, shared_allocator<int>{TG.Node()});
  if (TG.Node().root())
  {
    if (type == COLLINEAR)
    {
      for (int i = 0; i < ndet; i++)
      {
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[2 * i], nocc_per_kp[i]({0, nkpts})))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. "); 
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[2 * i + 1], nocc_per_kp[i]({nkpts, 2 * nkpts})))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. "); 
      }
    }
    else
    {
      for (int i = 0; i < ndet; i++)
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[i], nocc_per_kp[i], npol == 2))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. "); 
    }
  }
  TG.Node().barrier();
  int nocc_max = *std::max_element(raw_pointer_cast(nocc_per_kp.origin()),
                                   raw_pointer_cast(nocc_per_kp.origin()) + nocc_per_kp.num_elements());

  /* half-rotate LQ and H1:
   * Given that PsiT = H(SM),
   * h[K][a][k] = sum_i PsiT[K][a][i] * h[K][i][k]
   * L[Q][K][a][k][n] = sum_i PsiT[K][a][i] * L[Q][K][i][k][n]
   * L[Q][K][l][b][n] = sum_i PsiT[K][b][] * L[Q][K][l][k][n]*
   * LQKak has a special transposition to facilitate computations
   * of the energy, and they are stored with padding to max linear dimension
   * LQKank[Q][K][...] = LQKank[Q][K][a][n][k] = LQKakn[Q][K][a][k][n]
   */
  std::vector<shmSpMatrix> LQKank;
  LQKank.reserve(ndet * nspins * nkpts); // storing 2 components for Q=0, since it is not assumed symmetric
  shmCMatrix haj({ndet * nkpts, (type == COLLINEAR ? 2 : 1) * nocc_max * npol * nmo_max},
                 shared_allocator<ComplexType>{TG.Node()});
  if (TG.Node().root())
    std::fill_n(haj.origin(), haj.num_elements(), ComplexType(0.0));
  int ank_max = nocc_max * nchol_max * nmo_max;
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < nkpts; Q++)
      if (Qmap[Q] >= 0)
        LQKank.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{TG.Node()}));
      else
        LQKank.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{TG.Node()}));
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < nkpts; Q++)
        if (Qmap[Q] >= 0)
          LQKank.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{TG.Node()}));
        else
          LQKank.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{TG.Node()}));
    }
  }
  for (int nd = 0, nt = 0, nq0 = 0; nd < ndet; nd++, nq0 += nkpts * nspins)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] < 0)
        continue;
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % TG.Node().size() == TG.Node().rank())
        {
          std::fill_n(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), LQKank[nq0 + Q][K].num_elements(), SPComplexType(0.0));
          if (type == COLLINEAR)
          {
            std::fill_n(raw_pointer_cast(LQKank[nq0 + nkpts + Q][K].origin()), LQKank[nq0 + nkpts + Q][K].num_elements(),
                        SPComplexType(0.0));
          }
        }
      }
    }
  }

  // NOTE: LQKbnl is indexed by the K index of 'b', L[Q][Kb]
  std::vector<shmSpMatrix> LQKbnl;
  LQKbnl.reserve(ndet * nspins *
                 number_of_symmetric_Q); // storing 2 components for Q=0, since it is not assumed symmetric
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < number_of_symmetric_Q; Q++)
      LQKbnl.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{TG.Node()}));
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < number_of_symmetric_Q; Q++)
        LQKbnl.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{TG.Node()}));
    }
  }
  for (int nd = 0, nt = 0, nq0 = 0; nd < ndet; nd++, nq0 += number_of_symmetric_Q * nspins)
  {
    for (int Q = 0; Q < number_of_symmetric_Q; Q++)
    {
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % TG.Node().size() == TG.Node().rank())
        {
          std::fill_n(raw_pointer_cast(LQKbnl[nq0 + Q][K].origin()), LQKbnl[nq0 + Q][K].num_elements(), SPComplexType(0.0));
          if (type == COLLINEAR)
            std::fill_n(raw_pointer_cast(LQKbnl[nq0 + number_of_symmetric_Q + Q][K].origin()),
                        LQKbnl[nq0 + number_of_symmetric_Q + Q][K].num_elements(), SPComplexType(0.0));
        }
      }
    }
  }
  TG.Node().barrier();

  int Q0 = -1; // if K=(0,0,0) exists, store index here
  for (int Q = 0; Q < nkpts; Q++)
  {
    if (Qminus[Q] == Q)
    {
      bool found = true;
      for (int KI = 0; KI < nkpts; KI++)
        if (KI != QKtok2[Q][KI])
        {
          found = false;
          break;
        }
      if (found)
      {
        Q0 = Q;
        break;
      }
    }
  }
  if (Q0 < 0)
    APP_ABORT(" Error: Could not find Q=0. ");

  boost::multi::array<SPComplexType, 2> buff({npol * nmo_max, nchol_max});
  int nt = 0;
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int K = 0; K < nkpts; K++, nt++)
    {
      if (nt % TG.Global().size() == TG.Global().rank())
      {
        // haj and add half-transformed right-handed rotation for Q=0
        int na = nocc_per_kp[nd][K];
        int nb = (nspins == 2 ? nocc_per_kp[nd][nkpts + K] : na);
        int ni = nmo_per_kp[K];
        if (type == COLLINEAR)
        {
          { // Alpha
            auto Psi = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], K);
            RUNTIME_CHECK(Psi.size(0) == na, "");
            boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()), {na, ni});
            if (na > 0)
              ma::product(Psi, H1[K]({0, ni}, {0, ni}), haj_r);
          }
          { // Beta
            auto Psi = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], K);
            RUNTIME_CHECK(Psi.size(0) == nb, "");
            boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()) + na * ni, {nb, ni});
            if (nb > 0)
              ma::product(Psi, H1[K+(nspins_H1-1)*nkpts]({0, ni}, {0, ni}), haj_r);
          }
        }
        else
        {
          RealType scl = (type == CLOSED ? 2.0 : 1.0);
          auto Psi     = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[nd], K, npol == 2);
          RUNTIME_CHECK(Psi.size(0) == na, "");
          boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()), {na, npol * ni});
          if (na > 0)
            ma::product(ComplexType(scl), Psi, H1[K]({0, npol * ni}, {0, npol * ni}), ComplexType(0.0), haj_r);
        }
      }
    }
  }
  // Generate LQKank
  for (int nd = 0, nq0 = 0; nd < ndet; nd++, nq0 += nkpts * nspins)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] < 0)
        continue;
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % Qcomm.size() == Qcomm.rank())
        {
          // add half-transformed right-handed rotation for Q=0
          int Qm    = Qminus[Q];
          int QK    = QKtok2[Q][K];
          int na    = nocc_per_kp[nd][K];
          int nb    = (nspins == 2 ? nocc_per_kp[nd][nkpts + K] : na);
          int ni    = nmo_per_kp[K];
          int nk    = nmo_per_kp[QK];
          int nchol = nchol_per_kp[Q];
          if (type == COLLINEAR)
          {
            { // Alpha
              auto Psi = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], K);
              RUNTIME_CHECK(Psi.size(0) == nocc_per_kp[nd][K], "");
              if (Q <= Qm)
              {
                Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {ni, nk, nchol});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {na, nchol, nk});
                ma_rotate::getLank(Psi, Likn, Lank, buff);
              }
              else
              {
                Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nk, ni, nchol});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {na, nchol, nk});
                ma_rotate::getLank_from_Lkin(Psi, Lkin, Lank, buff);
              }
            }
            { // Beta
              auto Psi = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], K);
              RUNTIME_CHECK(Psi.size(0) == nb, "");
              if (Q <= Qm)
              {
                Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {ni, nk, nchol});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + nkpts + Q][K].origin()), {nb, nchol, nk});
                ma_rotate::getLank(Psi, Likn, Lank, buff);
              }
              else
              {
                Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nk, ni, nchol});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + nkpts + Q][K].origin()), {nb, nchol, nk});
                ma_rotate::getLank_from_Lkin(Psi, Lkin, Lank, buff);
              }
            }
          }
          else
          {
            auto Psi = get_PsiK<SpMatrix>(nmo_per_kp, PsiT[nd], K, npol == 2);
            RUNTIME_CHECK(Psi.size(0) == na, "");
            if (Q <= Qm)
            {
              Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {ni, nk, nchol});
              Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {na, nchol, npol * nk});
              ma_rotate::getLank(Psi, Likn, Lank, buff, npol == 2);
            }
            else
            {
              Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nk, ni, nchol});
              Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {na, nchol, npol * nk});
              ma_rotate::getLank_from_Lkin(Psi, Lkin, Lank, buff, npol == 2);
            }
          }
        }
      }
    }
  }

  // now generate LQKbnl if Q==(-Q)
  for (int nd = 0, nq0 = 0; nd < ndet; nd++, nq0 += number_of_symmetric_Q * nspins)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] <= 0)
        continue;
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % Qcomm.size() == Qcomm.rank())
        {
          // careful with subtle redefinition of na,nb,... here
          int QK    = QKtok2[Q][K];
          int na    = nocc_per_kp[nd][QK];
          int nb    = (nspins == 2 ? nocc_per_kp[nd][nkpts + QK] : na);
          int ni    = nmo_per_kp[K];
          int nk    = nmo_per_kp[QK];
          int nchol = nchol_per_kp[Q];
          Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {ni, nk, nchol});
          // NOTE: LQKbnl is indexed by the K index of 'b', L[Q][Kb]
          if (type == COLLINEAR)
          {
            { // Alpha
              auto PsiQK = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], QK);
              Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + Qmap[Q] - 1][QK].origin()), {na, nchol, ni});
              ma_rotate::getLank_from_Lkin(PsiQK, Likn, Lbnl, buff);
            }
            { // Beta
              auto PsiQK = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], QK);
              RUNTIME_CHECK(PsiQK.size(0) == nb, "");
              Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + number_of_symmetric_Q + Qmap[Q] - 1][QK].origin()),
                                 {nb, nchol, ni});
              ma_rotate::getLank_from_Lkin(PsiQK, Likn, Lbnl, buff);
            }
          }
          else
          {
            auto PsiQK = get_PsiK<SpMatrix>(nmo_per_kp, PsiT[nd], QK, npol == 2);
            RUNTIME_CHECK(PsiQK.size(0) == na, "");
            Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + Qmap[Q] - 1][QK].origin()), {na, nchol, npol * ni});
            ma_rotate::getLank_from_Lkin(PsiQK, Likn, Lbnl, buff, npol == 2);
          }
        }
      }
    }
  }
  Qcomm.barrier();
  if (TG.Node().root())
  {
    TG.Cores().all_reduce_in_place_n(raw_pointer_cast(haj.origin()), haj.num_elements(), std::plus<>());
    for (int Q = 0; Q < LQKank.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKank[Q].origin()), LQKank[Q].num_elements(), std::plus<>());
    for (int Q = 0; Q < LQKbnl.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKbnl[Q].origin()), LQKbnl[Q].num_elements(), std::plus<>());
    std::fill_n(raw_pointer_cast(vn0.origin()), vn0.num_elements(), ComplexType(0.0));
  }
  // need to broadcast haj from root of Qcomm with Qsym[0]>=0, to all other ones
  // NOTE NOTE NOTE
  TG.Node().barrier();

  // calculate vn0(I,L) = -0.5 sum_K sum_j sum_n L[0][K][i][j][n] ma::conj(L[0][K][l][j][n])
  for (int Q = 0; Q < nkpts; Q++)
  {
    if (Qmap[Q] < 0)
      continue;
    for (int K = 0; K < nkpts; K++)
    {
      if (K % TG.Node().size() == TG.Node().rank())
      {
        int QK = QKtok2[Q][K];
        int Qm = Qminus[Q];
        if (Q <= Qm)
        {
          boost::multi::array_ref<SPComplexType, 2> Likn(raw_pointer_cast(LQKikn[Q][K].origin()),
                                                         {nmo_per_kp[K], nmo_per_kp[QK] * nchol_per_kp[Q]});
          using ma::H;
	  if constexpr (MP) {
            Matrix<SPComplexType> v1_({nmo_per_kp[K], nmo_per_kp[K]});
            ma::product(SPComplexType(-0.5), Likn, H(Likn), SPComplexType(0.0), v1_);
            Matrix<ComplexType> v2_(v1_);
            ma::add(ComplexType(1.0), v2_, ComplexType(1.0), vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}),
                  vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}));
	  } else {
            ma::product(-0.5, Likn, H(Likn), 1.0, vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}));
	  }
        }
        else
        {
          boost::multi::array_ref<SPComplexType, 3> Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()),
                                                         {nmo_per_kp[QK], nmo_per_kp[K], nchol_per_kp[Qm]});
          boost::multi::array<SPComplexType, 3> buff3D({nmo_per_kp[K], nmo_per_kp[QK], nchol_per_kp[Qm]});
          using ma::conj;
          for (int i = 0; i < nmo_per_kp[K]; i++)
            for (int k = 0; k < nmo_per_kp[QK]; k++)
              for (int n = 0; n < nchol_per_kp[Qm]; n++)
                buff3D[i][k][n] = ma::conj(Lkin[k][i][n]);
          boost::multi::array_ref<SPComplexType, 2> L_(raw_pointer_cast(buff3D.origin()),
                                                       {nmo_per_kp[K], nmo_per_kp[QK] * nchol_per_kp[Qm]});
          using ma::H;
	  if constexpr (MP) {
            Matrix<SPComplexType> v1_({nmo_per_kp[K], nmo_per_kp[K]});
            ma::product(SPComplexType(-0.5), L_, H(L_), SPComplexType(0.0), v1_);
            Matrix<ComplexType> v2_(v1_);
            ma::add(ComplexType(1.0), v2_, ComplexType(1.0), vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}),
                  vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}));
	  } else {
            ma::product(-0.5, L_, H(L_), 1.0, vn0[K]({0, nmo_per_kp[K]}, {0, nmo_per_kp[K]}));
	  }
        }
      }
    }
    // need sync here to avoid having multiple Q's overwritting each other
    // either this or you need local storage
    TG.Node().barrier();
  }
  TG.Node().barrier();

  if (TG.Node().root())
    dump.close();

  int global_ncvecs = 2 * std::accumulate(nchol_per_kp.begin(), nchol_per_kp.end(), 0);

  std::vector<RealType> gQ(nkpts);
  if (nsampleQ > 0)
  {
    app_log(1," Sampling EXX energy using distribution over Q vector obtained from trial energy. ");

    if (npol == 2)
      APP_ABORT("Error: nsampleQ>0 not yet implemented for noncollinear.\n\n");

    RealType scl = (type == CLOSED ? 2.0 : 1.0);
    size_t nqk   = 0;
    for (int Q = 0; Q < nkpts; ++Q)
    { // momentum conservation index
      if (Qmap[Q] < 0)
        continue;
      int Qm = Qminus[Q];
      for (int Ka = 0; Ka < nkpts; ++Ka)
      {
        int Kk = QKtok2[Q][Ka];
        int Kb = Kk;
        int Kl = QKtok2[Qm][Kb];
        if ((Ka != Kl) || (Kb != Kk))
          APP_ABORT(" Error: Problems with EXX.");
        if ((nqk++) % Qcomm.size() == Qcomm.rank())
        {
          int nchol = nchol_per_kp[Q];
          int nl    = nmo_per_kp[Kl];
          int nb    = nocc_per_kp[0][Kb];
          int nk    = nmo_per_kp[Kk];
          int na    = nocc_per_kp[0][Ka];

          if (na == 0 || nb == 0)
            continue;

          SpMatrix_ref Lank(raw_pointer_cast(LQKank[Q][Ka].origin()), {na * nchol, nk});
          auto bnl_ptr=raw_pointer_cast(LQKank[Qm][Kb].origin());
          if (Qmap[Q] > 0)
            bnl_ptr = raw_pointer_cast(LQKbnl[Qmap[Q] - 1][Kb].origin());
          SpMatrix_ref Lbnl(bnl_ptr, {nb * nchol, nl});

          SpMatrix Tban({nb, na * nchol});
          Sp3Tensor_ref T3ban(Tban.origin(), {nb, na, nchol});
          SpMatrix Tabn({na, nb * nchol});
          Sp3Tensor_ref T3abn(Tabn.origin(), {na, nb, nchol});

          auto Gal = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[0], Ka, npol == 2);
          auto Gbk = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[0], Kb, npol == 2);
          for (int a = 0; a < na; ++a)
            for (int l = 0; l < nl; ++l)
              Gal[a][l] = ma::conj(Gal[a][l]);
          for (int b = 0; b < nb; ++b)
            for (int k = 0; k < nk; ++k)
              Gbk[b][k] = ma::conj(Gbk[b][k]);

          ma::product(Gal, ma::T(Lbnl), Tabn);
          ma::product(Gbk, ma::T(Lank), Tban);

          ComplexType E_(0.0);
          for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b)
              E_ += static_cast<SPComplexType>(ma::dot(T3abn[a][b], T3ban[b][a]));
          gQ[Q] -= scl * 0.5 * real(E_);
        }
        if (type == COLLINEAR)
        {
          APP_ABORT(" Finish UHF.\n ");
        }
      }
    }
    TG.Global().all_reduce_in_place_n(gQ.begin(), nkpts, std::plus<>());
    RealType E_ = std::accumulate(gQ.begin(), gQ.end(), RealType(0.0));
    for (auto& v : gQ)
      v /= E_;
    app_log(1," EXX: {}", E_);
    for (auto v : gQ)
    {
      if (v < 0.0)
        APP_ABORT(" Error: g(Q) < 0.0, implement shift to g(Q). ");
    }
  }

  return HamiltonianOperations<MP>(
      KP3IndexFactorization<MP>(TGwfn.TG_local(), type, std::move(nmo_per_kp), std::move(nchol_per_kp), std::move(Qminus),
                            std::move(nocc_per_kp), std::move(QKtok2), std::move(H1), std::move(haj), std::move(LQKikn),
                            std::move(LQKank), std::move(LQKbnl), std::move(Qmap), std::move(vn0), std::move(gQ),
                            nsampleQ, E0, global_origin, global_ncvecs));
}
#endif

template<bool MP> HamiltonianOperations<MP> 
KPFactorizedHamiltonian::getHamiltonianOperations_batched(WALKER_TYPES type,
                                                         std::vector<PsiT_Matrix>& PsiT,
                                                         TaskGroup_& TGprop,   
                                                         TaskGroup_& TGwfn,    
                                                         hdf_archive& hdf_restart)
{
  // For now doing setup in CPU and moving structures to GPU in HamOps constructor
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;
  using shmIMatrix    = boost::multi::array<int, 2, shared_allocator<int>>;
  using shmCMatrix    = boost::multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using shmCTensor    = boost::multi::array<ComplexType, 3, shared_allocator<ComplexType>>;
  using stdCTensor    = boost::multi::array<ComplexType, 3>;
  using shmSpMatrix   = boost::multi::array<SPComplexType, 2, shared_allocator<SPComplexType>>;
  using IVector       = boost::multi::array<int, 1>;
  using SpMatrix      = boost::multi::array<SPComplexType, 2>;
  using SpTensor3      = boost::multi::array<SPComplexType, 3>;
  using SpTensor5      = boost::multi::array<SPComplexType, 5>;
  using SpMatrix_ref  = boost::multi::array_ref<SPComplexType, 2>;
  using Sp3Tensor_ref = boost::multi::array_ref<SPComplexType, 3>;
  using Sp4Tensor_ref = boost::multi::array_ref<SPComplexType, 4>;

  std::string base_error(" Error in KPFactorizedHamiltonian::getHamiltonianOperations_shared: \n    ");

  if (TGprop.TG() != TGwfn.TG())
  {
    app_error(base_error + 
        " Requires nnodes to be the same in Wavefunction and Propagator xml blocks.");
    APP_ABORT("Error: Inconsistent nnodes in KPFactorizedHamiltonian ");
  }

  if (TG.TG_local().size() > 1)
    APP_ABORT(" Error: KPFactorizedHamiltonian::getHamiltonianOperations_batched expects ncores=1. ");

  // hack until parallel hdf is in place
  bool write_hdf = false;
  if (TGwfn.Global().root())
    write_hdf = (not hdf_restart.closed());
  TGwfn.Global().broadcast_value(write_hdf);

  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");
  int nspins = ((type != COLLINEAR) ? 1 : 2);
  int ndet   = PsiT.size() / nspins;
  int npol   = ((type == NONCOLLINEAR) ? 2 : 1);

  if (ndet > 1)
    APP_ABORT("Error: ndet > 1 not yet implemented in THCHamiltonian::getHamiltonianOperations.");

  auto Qcomm=TG.Global().split(TGwfn.getLocalGroupNumber(), TG.Global().rank());
  auto distNode=TG.Node().split(TGwfn.getLocalGroupNumber(), TG.Node().rank());
  auto Qcomm_roots=Qcomm.split(distNode.rank(), Qcomm.rank());

  long nkpts;
  hdf_archive dump(TGwfn.Global());
  // right now only Node.root() reads
  if (distNode.root())
  {
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(base_error + "Error opening integral file");
  }
  std::string format = get_hamiltonian_format(dump,TG.Global());

  if (TG.Global().root())
  {
    if(format == "std") {
      std::vector<int> Idata(8);
      if (!dump.readEntry(Idata, "Hamiltonian/dims"))
        APP_ABORT(base_error + "Problems reading dims");
      nkpts = Idata[2];
    } else if(format == "coqui") {
      if (dump.push("System", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System not found. ");
      if (dump.push("BZ", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System/BZ not found. ");
      if (!dump.readAttributeEntry(nkpts, "number_of_kpoints"))
        APP_ABORT(base_error + "Problems reading attribute /System/BZ/number_of_kpoints");
      dump.pop();
      dump.pop();
    }
  }
  TG.Global().broadcast_n(&nkpts, 1, 0);
  app_log(1," nkpts: {}", nkpts);

  // partition Q over nodes if distributed Q

  IVector nmo_per_kp(iextensions<1u>{nkpts});
  IVector nchol_per_kp(iextensions<1u>{nkpts});
  IVector Qminus(iextensions<1u>{nkpts});
  IVector Qmap(iextensions<1u>{nkpts});
  shmIMatrix QKtok2({nkpts, nkpts}, shared_allocator<int>{TG.Node()});
  ComplexType E0 = NuclearCoulombEnergy + FrozenCoreEnergy;
  if (TG.Global().root())
  {
    if(format == "std") {
      if (!dump.readEntry(nmo_per_kp, "Hamiltonian/NMOPerKP"))
        APP_ABORT(base_error + " Problems reading NMOPerKP. ");
      if (!dump.readEntry(nchol_per_kp, "Hamiltonian/NCholPerKP"))
        APP_ABORT(base_error + "Problems reading NCholPerKP. ");
      if (!dump.readEntry(Qminus, "Hamiltonian/MinusK"))
        APP_ABORT(base_error + "Problems reading MinusK. ");
      if (!dump.readEntry(QKtok2, "Hamiltonian/QKTok2"))
        APP_ABORT(base_error + "Problems reading QKTok2. ");
      std::vector<RealType> E_(2);
    } else if(format == "coqui"){
      if (!dump.readEntry(Qminus, "/System/BZ/qminus"))
        APP_ABORT(base_error + "Problems reading /System/BZ/qminus. ");
      if (!dump.readEntry(QKtok2, "/System/BZ/qk_to_k2"))
        APP_ABORT(base_error + "Problems reading /System/BZ/qk_to_k2. ");
      long nbnd;
      if (dump.push("System", false)<0)
        APP_ABORT(" Error in HamiltonianFactory::fromHDF5(): Group /System not found. ");
      if (!dump.readAttributeEntry(nbnd, "number_of_bands"))
        APP_ABORT(base_error + "Problems reading /System/number_of_bands. ");
      dump.pop();
      for(int k=0; k<nkpts; k++)
        nmo_per_kp[k] = nbnd;
      std::vector<int> shape(6);
      for(int q=0; q<nkpts; q++) {
        if (!dump.getShape<RealType>("/Interaction/Vq"+std::to_string(q), shape))
          APP_ABORT(base_error + "Problems reading /Interaction/Vq" + std::to_string(q));
        nchol_per_kp[q] = shape[0];
      }
    }
    for (int q = 0; q < nkpts; q++)
    { 
      if (Qminus[q] < q)
        nchol_per_kp[q] = nchol_per_kp[Qminus[q]];
    }
    if (nmo_per_kp.size() != nkpts || nchol_per_kp.size() != nkpts || Qminus.size() != nkpts ||
        QKtok2.size(0) != nkpts || QKtok2.size(1) != nkpts)
    {
      app_error(" Inconsistent dimension (NMOPerKP,NCholPerKP,QKtTok2)");
      app_error(" nkpts={}, nmo_per_kp.size={}, nchol_per_kp.size={}, QKtok2.shape:({},{}) ",
                    nkpts, nmo_per_kp.size(), nchol_per_kp.size(), Qminus.size(),
                    QKtok2.size(0), QKtok2.size(1));
      APP_ABORT(base_error);
    }
  }
  TG.Global().broadcast_n(&E0, 1, 0);
  TG.Global().broadcast_n(nmo_per_kp.origin(), nmo_per_kp.size(), 0);
  TG.Global().broadcast_n(nchol_per_kp.origin(), nchol_per_kp.size(), 0);
  TG.Global().broadcast_n(Qminus.origin(), Qminus.size(), 0);
  if (TG.Node().root())
    TG.Cores().broadcast_n(raw_pointer_cast(QKtok2.origin()), QKtok2.num_elements(), 0);
  TG.Node().barrier();

  // Defines behavior over Q vector:
  //   <0: Ignore (handled by another TG)
  //    0: Calculate, without rho^+ contribution
  //   >0: Calculate, with rho^+ contribution. LQKbln data located at Qmap[Q]-1
  int number_of_symmetric_Q = 0;
  int global_origin(0);
  std::fill_n(Qmap.origin(), Qmap.num_elements(), -1);
  {
    int ngrp(TGwfn.getNGroupsPerTG());
    int ig(TGwfn.getLocalGroupNumber());
    int work(0);
    // assign Q/Qm pairs of vectors to groups round-robin
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qminus[Q] == Q)
      {
        if (work % ngrp == ig)
          Qmap[Q] = 1 + (number_of_symmetric_Q++);
        if (work % ngrp < ig)
          global_origin += 2 * nchol_per_kp[Q];
        work++;
      }
      else if (Q < Qminus[Q])
      {
        if (work % ngrp == ig)
        {
          Qmap[Q]         = 0;
          Qmap[Qminus[Q]] = 0;
        }
        if (work % ngrp < ig)
          global_origin += 4 * nchol_per_kp[Q];
        work++;
      }
    }
    if (work < ngrp)
      APP_ABORT(" Error: Too many nodes in group (nnodes) for given number of kpoints. ");
  }

  int nmo_max   = *std::max_element(nmo_per_kp.begin(), nmo_per_kp.end());
  int nchol_max = *std::max_element(nchol_per_kp.begin(), nchol_per_kp.end());
  int nspins_H1 = 1;

  // check if H1 is spin dependent
  // right now, I only check if "H1_kp{2*nkpts-1}" exists. Other Qs are checked below
  if (type == COLLINEAR) {
    if (TG.Global().root()) {
      if(format == "std") {
        int K = 2*nkpts-1;
        int nmo_K = nmo_per_kp[nkpts-1];
        boost::multi::array<ComplexType, 2> h1({npol * nmo_K, npol * nmo_K});
        std::string h_id = std::string("Hamiltonian/H1_kp") + std::to_string(K); 
        if (dump.readEntry(h1,h_id)) 
          nspins_H1 = 2;
      } else if(format == "coqui") {
        std::string h_id = std::string("System/H0");
        std::vector<int> shape(5);
        if (!dump.getShape<RealType>(h_id, shape))
          APP_ABORT(base_error + "Problems reading " + h_id);
        nspins_H1 = shape[0];
      }
    }
    TG.Global().broadcast_value(nspins_H1);
  }

  shmCTensor H1({nspins_H1*nkpts, npol * nmo_max, npol * nmo_max}, 
		shared_allocator<ComplexType>{TG.Node()});
  std::vector<shmSpMatrix> LQKikn;
  LQKikn.reserve(nkpts);
  for (int Q = 0; Q < nkpts; Q++) {
    if (Qmap[Q] >= 0 && Q <= Qminus[Q])
      LQKikn.emplace_back(
          shmSpMatrix({nkpts, nmo_max * nmo_max * nchol_max}, shared_allocator<SPComplexType>{distNode}));
    else // Q > Qminus[Q]
      LQKikn.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{distNode}));
  }

  if (TG.Node().root())
  {
    if(format == "std") {
      // now read H1_kpK
      for (int K = 0; K < nspins_H1*nkpts; K++)
      {
        int nmo_K = nmo_per_kp[K%nkpts];
        // until double_hyperslabs work!
        boost::multi::array<ComplexType, 2> h1({npol * nmo_K, npol * nmo_K});
        std::string h_id = std::string("Hamiltonian/H1_kp") + std::to_string(K); 
        if (!dump.readEntry(h1, h_id)) 
          APP_ABORT(base_error + " Problems reading " + h_id);
        ma::add(ComplexType(1.0), h1, ComplexType(0.0), h1, 
          H1[K]({0, npol * nmo_K}, {0, npol * nmo_K}));
      }
    } else if(format == "coqui") {
       // fine to read directly, since nmo_per_kp == nbnd for all k
       boost::multi::array_ref<ComplexType, 4> h_(raw_pointer_cast(H1.origin()),
                     {nspins_H1, nkpts, npol * nmo_max, npol * nmo_max});
       if (!dump.readEntry(h_, "System/H0"))
         APP_ABORT(base_error + " Problems reading System/H0");
    }
  }
  if (distNode.root())
  {
    for (auto& v : LQKikn)
      std::fill_n(raw_pointer_cast(v.origin()), v.num_elements(), SPComplexType(0.0));
    // read LQ
    // read in compact form and transform to padded
    SpMatrix L_({1, 1});
    for (int Q = 0; Q < nkpts; Q++)
    {
      using ma::conj;
      int nchol = nchol_per_kp[Q];
      if (Qmap[Q] >= 0 && Q <= Qminus[Q])
      {
        if(format=="std") {
          if (!dump.readEntry(L_, std::string("Hamiltonian/KPFactorized/L") + std::to_string(Q)))
          {
            app_error(" Problems reading /Hamiltonian/KPFactorized/L{}", Q);
            APP_ABORT(base_error);
          }
          RUNTIME_CHECK(L_.size(0) == nkpts, "");
          Sp4Tensor_ref L2(raw_pointer_cast(LQKikn[Q].origin()), {nkpts, nmo_max, nmo_max, nchol_max});
          for (int K = 0; K < nkpts; ++K)
          {
            int QK = QKtok2[Q][K];
            int ni = nmo_per_kp[K];
            int nk = nmo_per_kp[QK];
            Sp3Tensor_ref L1(raw_pointer_cast(L_[K].origin()), {ni, nk, nchol});
            for (int i = 0; i < ni; i++)
              for (int k = 0; k < nk; k++)
                copy_n(L1[i][k].origin(), nchol, L2[K][i][k].origin());
          }
        } else if(format == "coqui") {
          // VqQ(ichol, ispin, ik, ia, ib)
          // transposing on host
          SpTensor5 Vq;
          if (!dump.readEntry(Vq, std::string("Interaction/Vq") + std::to_string(Q)))
            APP_ABORT(base_error + "Error reading Interaction/Vq" + std::to_string(Q));
          if( (Vq.size(0) != nchol_per_kp[Q]) or
              (Vq.size(1) != 1) or
              (Vq.size(2) != nkpts) or
              (Vq.size(3) != nmo_max) or
              (Vq.size(4) != nmo_max) )
            APP_ABORT(base_error + "Invalid dimesnions in Interaction/Vq" + std::to_string(Q));
          // normalize
          ma::scal(SPComplexType(SPRealType(1.0/std::sqrt(SPRealType(nkpts)))), Vq);
          SpTensor3 Vqt({nmo_max, nmo_max, nchol_max});
          Sp4Tensor_ref L_Kikn(raw_pointer_cast(LQKikn[Q].origin()), {nkpts, nmo_max, nmo_max, nchol_max});
          for (int K = 0; K < nkpts; K++)
          {
            int QK    = QKtok2[Q][K];
            int ni    = nmo_per_kp[K];
            int nk    = nmo_per_kp[QK];
            std::fill_n(raw_pointer_cast(Vqt.origin()), Vqt.num_elements(), SPComplexType(0.0));
            for(int n=0; n<nchol; ++n)
              for(int i=0; i<ni; ++i)
                for(int k=0; k<nk; ++k)
                  Vqt[i][k][n] = Vq[n][0][K][i][k];
            copy_n(Vqt.origin(), nmo_max*nmo_max*nchol_max, L_Kikn[K].origin());
          }
        }
      }
    }
  }
  TG.Node().barrier();

  // calculate vn0
  shmCTensor vn0({nkpts, nmo_max, nmo_max}, shared_allocator<ComplexType>{TG.Node()});

  // generate nocc_per_kp using PsiT and nmo_per_kp
  shmIMatrix nocc_per_kp({ndet, nspins * nkpts}, shared_allocator<int>{TG.Node()});
  TG.Node().barrier();
  if (TG.Node().root())
  {
    if (type == COLLINEAR)
    {
      for (int i = 0; i < ndet; i++)
      {
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[2 * i], nocc_per_kp[i]({0, nkpts})))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. ");
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[2 * i + 1], nocc_per_kp[i]({nkpts, 2 * nkpts})))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. ");
      }
    }
    else
    {
      for (int i = 0; i < ndet; i++)
        if (not get_nocc_per_kp(nmo_per_kp, PsiT[i], nocc_per_kp[i], npol == 2))
          APP_ABORT(base_error + " Only wavefunctions in block-diagonal form are accepted. ");
    }
  }
  TG.Node().barrier();
  int nocc_max = *std::max_element(raw_pointer_cast(nocc_per_kp.origin()),
                                   raw_pointer_cast(nocc_per_kp.origin()) + nocc_per_kp.num_elements());

  int nocc_tot = std::accumulate(raw_pointer_cast(nocc_per_kp.origin()),
                                 raw_pointer_cast(nocc_per_kp.origin()) + nocc_per_kp.num_elements(), 0);
  app_log(1," Total number of electrons: {}", nocc_tot);

  /* half-rotate LQ and H1:
   * Given that PsiT = H(SM),
   * h[K][a][k] = sum_i PsiT[K][a][i] * h[K][i][k]
   * L[Q][K][a][k][n] = sum_i PsiT[K][a][i] * L[Q][K][i][k][n]
   * Both permutations are stores, akn and ank, for performance reasons.
   */
  std::vector<shmSpMatrix> LQKank;
  LQKank.reserve(ndet * nspins * nkpts); // storing 2 components for Q=0, since it is not assumed symmetric
  shmCMatrix haj({ndet * nkpts, (type == COLLINEAR ? 2 : 1) * nocc_max * npol * nmo_max},
                 shared_allocator<ComplexType>{TG.Node()});
  if (TG.Node().root())
    std::fill_n(raw_pointer_cast(haj.origin()), haj.num_elements(), ComplexType(0.0));
  int ank_max = nocc_max * nchol_max * nmo_max;
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < nkpts; Q++)
      if (Qmap[Q] >= 0)
        LQKank.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{distNode}));
      else
        LQKank.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{distNode}));
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < nkpts; Q++)
        if (Qmap[Q] >= 0)
          LQKank.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{distNode}));
        else
          LQKank.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{distNode}));
    }
  }
  if (distNode.root())
    for (auto& v : LQKank)
      std::fill_n(raw_pointer_cast(v.origin()), v.num_elements(), SPComplexType(0.0));

  std::vector<shmSpMatrix> LQKakn;
  LQKakn.reserve(ndet * nspins * nkpts);
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] >= 0)
        LQKakn.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{distNode}));
      else
        LQKakn.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{distNode}));
    }
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < nkpts; Q++)
      {
        if (Qmap[Q] >= 0)
          LQKakn.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{distNode}));
        else
          LQKakn.emplace_back(shmSpMatrix({1, 1}, shared_allocator<SPComplexType>{distNode}));
      }
    }
  }
  if (distNode.root())
    for (auto& v : LQKakn)
      std::fill_n(raw_pointer_cast(v.origin()), v.num_elements(), SPComplexType(0.0));
  // NOTE: LQKbnl and LQKbln are indexed by the K index of 'b', L[Q][Kb]
  std::vector<shmSpMatrix> LQKbnl;
  LQKbnl.reserve(ndet * nspins *
                 number_of_symmetric_Q); // storing 2 components for Q=0, since it is not assumed symmetric
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < number_of_symmetric_Q; Q++)
      LQKbnl.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{distNode}));
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < number_of_symmetric_Q; Q++)
        LQKbnl.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{distNode}));
    }
  }
  if (distNode.root())
    for (auto& v : LQKbnl)
      std::fill_n(raw_pointer_cast(v.origin()), v.num_elements(), SPComplexType(0.0));

  std::vector<shmSpMatrix> LQKbln;
  LQKbln.reserve(ndet * nspins *
                 number_of_symmetric_Q); // storing 2 components for Q=0, since it is not assumed symmetric
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int Q = 0; Q < number_of_symmetric_Q; Q++)
    {
      LQKbln.emplace_back(shmSpMatrix({nkpts, npol * ank_max}, shared_allocator<SPComplexType>{distNode}));
    }
    if (type == COLLINEAR)
    {
      for (int Q = 0; Q < number_of_symmetric_Q; Q++)
      {
        LQKbln.emplace_back(shmSpMatrix({nkpts, ank_max}, shared_allocator<SPComplexType>{distNode}));
      }
    }
  }
  if (distNode.root())
    for (auto& v : LQKbln)
      std::fill_n(raw_pointer_cast(v.origin()), v.num_elements(), SPComplexType(0.0));

  int Q0 = -1; // if K=(0,0,0) exists, store index here
  for (int Q = 0; Q < nkpts; Q++)
  {
    if (Qminus[Q] == Q)
    {
      bool found = true;
      for (int KI = 0; KI < nkpts; KI++)
        if (KI != QKtok2[Q][KI])
        {
          found = false;
          break;
        }
      if (found)
      {
        Q0 = Q;
        break;
      }
    }
  }
  if (Q0 < 0)
    APP_ABORT(" Error: Could not find Q=0. ");

  TG.Node().barrier();
  boost::multi::array<SPComplexType, 2> buff({npol * nmo_max, nchol_max});
  int nt = 0;
  for (int nd = 0; nd < ndet; nd++)
  {
    for (int K = 0; K < nkpts; K++, nt++)
    {
      if (nt % TG.Global().size() == TG.Global().rank())
      {
        // haj and add half-transformed right-handed rotation for Q=0
        int na = nocc_per_kp[nd][K];
        int nb = (nspins == 2 ? nocc_per_kp[nd][nkpts + K] : na);
        int ni = nmo_per_kp[K];
        if (type == COLLINEAR)
        {
          { // Alpha
            auto Psi = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], K);
            RUNTIME_CHECK(Psi.size(0) == na, "");
            boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()),
                                                          {nocc_max, nmo_max});
            if (na > 0)
              ma::product(Psi, H1[K]({0, ni}, {0, ni}), haj_r({0, na}, {0, ni}));
          }
          { // Beta
            auto Psi = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], K);
            RUNTIME_CHECK(Psi.size(0) == nb, "");
            boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()) + nocc_max * nmo_max,
                                                          {nocc_max, nmo_max});
            if (nb > 0)
              ma::product(Psi, H1[K+(nspins_H1-1)*nkpts]({0, ni}, {0, ni}), haj_r({0, nb}, {0, ni}));
          }
        }
        else
        {
          RealType scl = (type == CLOSED ? 2.0 : 1.0);
          auto Psi     = get_PsiK<boost::multi::array<ComplexType, 2>>(nmo_per_kp, PsiT[nd], K, npol == 2);
          RUNTIME_CHECK(Psi.size(0) == na, "");
          boost::multi::array_ref<ComplexType, 2> haj_r(raw_pointer_cast(haj[nd * nkpts + K].origin()),
                                                        {nocc_max, npol * nmo_max});
          if (na > 0) {
            ma::product(ComplexType(scl), Psi, H1[K]({0, npol * ni}, {0, ni}), ComplexType(0.0),
                        haj_r({0, na}, {0, ni}));
	    if( npol == 2 )
              ma::product(ComplexType(scl), Psi, H1[K]({0, npol * ni}, {ni, 2*ni}), ComplexType(0.0),
                        haj_r({0, na}, {nmo_max, nmo_max+ni}));
	  }
        }
      }
    }
  }
  for (int nd = 0, nq0 = 0; nd < ndet; nd++, nq0 += nkpts * nspins)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] < 0)
        continue;
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % Qcomm.size() == Qcomm.rank())
        {
          // add half-transformed right-handed rotation for Q=0
          int Qm = Qminus[Q];
          int QK = QKtok2[Q][K];
          int na = nocc_per_kp[nd][K];
          int nb = (nspins == 2 ? nocc_per_kp[nd][nkpts + K] : na);
          if (type == COLLINEAR)
          {
            { // Alpha
              auto Psi = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], K);
              RUNTIME_CHECK(Psi.size(0) == na, "");
              if (Q <= Qm)
              {
                Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {nmo_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + Q][K].origin()), {nocc_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {nocc_max, nchol_max, nmo_max});
                ma_rotate_padded::getLakn_Lank(Psi, Likn, Lakn, Lank);
              }
              else
              {
                Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nmo_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {nocc_max, nchol_max, nmo_max});
                Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + Q][K].origin()), {nocc_max, nmo_max, nchol_max});
                ma_rotate_padded::getLakn_Lank_from_Lkin(Psi, Lkin, Lakn, Lank, buff);
              }
            }
            { // Beta
              auto Psi = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], K);
              RUNTIME_CHECK(Psi.size(0) == nb, "");
              if (Q <= Qm)
              {
                Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {nmo_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + nkpts + Q][K].origin()), {nocc_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + nkpts + Q][K].origin()), {nocc_max, nchol_max, nmo_max});
                ma_rotate_padded::getLakn_Lank(Psi, Likn, Lakn, Lank);
              }
              else
              {
                Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nmo_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + nkpts + Q][K].origin()), {nocc_max, nmo_max, nchol_max});
                Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + nkpts + Q][K].origin()), {nocc_max, nchol_max, nmo_max});
                ma_rotate_padded::getLakn_Lank_from_Lkin(Psi, Lkin, Lakn, Lank, buff);
              }
            }
          }
          else
          {
            auto Psi = get_PsiK<SpMatrix>(nmo_per_kp, PsiT[nd], K, npol == 2);
            RUNTIME_CHECK(Psi.size(0) == na, "");
            if (Q <= Qm)
            {
              Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {nmo_max, nmo_max, nchol_max});
              Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + Q][K].origin()), {nocc_max, npol * nmo_max, nchol_max});
              Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {nocc_max, nchol_max, npol * nmo_max});
              ma_rotate_padded::getLakn_Lank(Psi, Likn, Lakn, Lank, npol == 2);
            }
            else
            {
              Sp3Tensor_ref Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()), {nmo_max, nmo_max, nchol_max});
              Sp3Tensor_ref Lakn(raw_pointer_cast(LQKakn[nq0 + Q][K].origin()), {nocc_max, npol * nmo_max, nchol_max});
              Sp3Tensor_ref Lank(raw_pointer_cast(LQKank[nq0 + Q][K].origin()), {nocc_max, nchol_max, npol * nmo_max});
              ma_rotate_padded::getLakn_Lank_from_Lkin(Psi, Lkin, Lakn, Lank, buff, npol == 2);
            }
          }
        }
      }
    }
  }

  // now generate LQKbnl if Q==(-Q)
  for (int nd = 0, nq0 = 0; nd < ndet; nd++, nq0 += number_of_symmetric_Q * nspins)
  {
    for (int Q = 0; Q < nkpts; Q++)
    {
      if (Qmap[Q] <= 0)
        continue;
      for (int K = 0; K < nkpts; K++, nt++)
      {
        if (nt % Qcomm.size() == Qcomm.rank())
        {
          // careful with subtle redefinition of na,nb,... here
          int QK = QKtok2[Q][K];
          int na = nocc_per_kp[nd][QK];
          int nb = (nspins == 2 ? nocc_per_kp[nd][nkpts + QK] : na);
          Sp3Tensor_ref Likn(raw_pointer_cast(LQKikn[Q][K].origin()), {nmo_max, nmo_max, nchol_max});
          if (type == COLLINEAR)
          {
            { // Alpha
              auto PsiQK = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd], QK);
              RUNTIME_CHECK(PsiQK.size(0) == na, "");
              Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + Qmap[Q] - 1][QK].origin()), {nocc_max, nchol_max, nmo_max});
              Sp3Tensor_ref Lbln(raw_pointer_cast(LQKbln[nq0 + Qmap[Q] - 1][QK].origin()), {nocc_max, nmo_max, nchol_max});
              ma_rotate_padded::getLakn_Lank_from_Lkin(PsiQK, Likn, Lbln, Lbnl, buff);
            }
            { // Beta
              auto PsiQK = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[2 * nd + 1], QK);
              RUNTIME_CHECK(PsiQK.size(0) == nb, "");
              Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + number_of_symmetric_Q + Qmap[Q] - 1][QK].origin()),
                                 {nocc_max, nchol_max, nmo_max});
              Sp3Tensor_ref Lbln(raw_pointer_cast(LQKbln[nq0 + number_of_symmetric_Q + Qmap[Q] - 1][QK].origin()),
                                 {nocc_max, nmo_max, nchol_max});
              ma_rotate_padded::getLakn_Lank_from_Lkin(PsiQK, Likn, Lbln, Lbnl, buff);
            }
          }
          else
          {
            auto PsiQK = get_PsiK<SpMatrix>(nmo_per_kp, PsiT[nd], QK, npol == 2);
            RUNTIME_CHECK(PsiQK.size(0) == na, "");
            Sp3Tensor_ref Lbnl(raw_pointer_cast(LQKbnl[nq0 + Qmap[Q] - 1][QK].origin()),
                               {nocc_max, nchol_max, npol * nmo_max});
            Sp3Tensor_ref Lbln(raw_pointer_cast(LQKbln[nq0 + Qmap[Q] - 1][QK].origin()),
                               {nocc_max, npol * nmo_max, nchol_max});
            ma_rotate_padded::getLakn_Lank_from_Lkin(PsiQK, Likn, Lbln, Lbnl, buff, npol == 2);
          }
        }
      }
    }
  }
  Qcomm.barrier();

  if (TG.Node().root())
  {
    TG.Cores().all_reduce_in_place_n(raw_pointer_cast(haj.origin()), haj.num_elements(), std::plus<>());
    std::fill_n(raw_pointer_cast(vn0.origin()), vn0.num_elements(), ComplexType(0.0));
  }

  if (distNode.root())
  {
    for (int Q = 0; Q < LQKank.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKank[Q].origin()), LQKank[Q].num_elements(), std::plus<>());

    for (int Q = 0; Q < LQKakn.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKakn[Q].origin()), LQKakn[Q].num_elements(), std::plus<>());

    for (int Q = 0; Q < LQKbnl.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKbnl[Q].origin()), LQKbnl[Q].num_elements(), std::plus<>());

    for (int Q = 0; Q < LQKbln.size(); Q++)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(LQKbln[Q].origin()), LQKbln[Q].num_elements(), std::plus<>());
  }
  TG.Node().barrier();

  // local storage seems necessary
  stdCTensor vn0_({nkpts, nmo_max, nmo_max}, ComplexType(0.0));

  // calculate vn0(I,L) = -0.5 sum_K sum_j sum_n L[0][K][i][j][n] ma::conj(L[0][K][l][j][n])
  nt = 0;
  for (int Q = 0; Q < nkpts; Q++)
  {
    if (Qmap[Q] < 0)
      continue;
    for (int K = 0; K < nkpts; K++)
    {
      if (nt % Qcomm.size() == Qcomm.rank())
      {
        int QK = QKtok2[Q][K];
        int Qm = Qminus[Q];
        if (Q <= Qm)
        {
          Matrix_ref<SPComplexType> Likn(raw_pointer_cast(LQKikn[Q][K].origin()),
                                                         {nmo_max, nmo_max * nchol_max});
          using ma::H;
	  if constexpr (MP) {
            Matrix<SPComplexType> v1_({nmo_max, nmo_max});
            ma::product(SPComplexType(-0.5), Likn, H(Likn), SPComplexType(0.0), v1_);
            using std::copy_n;
            Matrix<ComplexType> v2_(v1_);
            ma::add(ComplexType(1.0), v2_, ComplexType(1.0), vn0_[K], vn0_[K]);
	  } else {
            ma::product(-0.5, Likn, H(Likn), 1.0, vn0_[K]);
	  }
        }
        else
        {
          Array_ref<SPComplexType, 3> Lkin(raw_pointer_cast(LQKikn[Qm][QK].origin()),
                                                         {nmo_max, nmo_max, nchol_max});
          Array<SPComplexType, 3> buff3D({nmo_max, nmo_max, nchol_max});
          using ma::conj;
          for (int i = 0; i < nmo_max; i++)
            for (int k = 0; k < nmo_max; k++)
              for (int n = 0; n < nchol_max; n++)
                buff3D[i][k][n] = ma::conj(Lkin[k][i][n]);
          Matrix_ref<SPComplexType> L_(raw_pointer_cast(buff3D.origin()), {nmo_max, nmo_max * nchol_max});
          using ma::H;
	  if constexpr (MP) {
          Matrix<SPComplexType> v1_({nmo_max, nmo_max});
          ma::product(SPComplexType(-0.5), L_, H(L_), SPComplexType(0.0), v1_);
          Matrix<ComplexType> v2_(v1_);
          ma::add(ComplexType(1.0), v2_, ComplexType(1.0), vn0_[K], vn0_[K]);
	  } else {
            ma::product(-0.5, L_, H(L_), 1.0, vn0_[K]);
	  }
        }
      }
    }
  }
  TG.Global().all_reduce_in_place_n(vn0_.origin(), vn0_.num_elements(), std::plus<>());
  copy_n(vn0_.origin(), vn0_.num_elements(), vn0.origin());

  if (TG.Node().root())
    dump.close();

  int global_ncvecs = 2 * std::accumulate(nchol_per_kp.begin(), nchol_per_kp.end(), 0);

  std::vector<RealType> gQ(nkpts);
  if (nsampleQ > 0)
  {
    app_log(1," Sampling EXX energy using distribution over Q vector obtained from trial energy,");

    if (npol == 2)
      APP_ABORT("Error: nsampleQ>0 not yet implemented for noncollinear.\n\n");

    RealType scl = (type == CLOSED ? 2.0 : 1.0);
    size_t nqk   = 0;
    for (int Q = 0; Q < nkpts; ++Q)
    { // momentum conservation index
      if (Qmap[Q] < 0)
        continue;
      int Qm = Qminus[Q];
      for (int Ka = 0; Ka < nkpts; ++Ka)
      {
        int Kk = QKtok2[Q][Ka];
        int Kb = Kk;
        int Kl = QKtok2[Qm][Kb];
        if ((Ka != Kl) || (Kb != Kk))
          APP_ABORT(" Error: Problems with EXX.");
        if ((nqk++) % Qcomm.size() == Qcomm.rank())
        {
          int nl    = nmo_per_kp[Kl];
          int nb    = nocc_per_kp[0][Kb];
          int nk    = nmo_per_kp[Kk];
          int na    = nocc_per_kp[0][Ka];

          if (na == 0 || nb == 0)
            continue;

          SpMatrix_ref Lank(raw_pointer_cast(LQKank[Q][Ka].origin()), {na * nchol_max, nmo_max});
          auto bnl_ptr=raw_pointer_cast(LQKank[Qm][Kb].origin());
          if (Qmap[Q] > 0)
            bnl_ptr = raw_pointer_cast(LQKbnl[Qmap[Q] - 1][Kb].origin());
          SpMatrix_ref Lbnl(bnl_ptr, {nb * nchol_max, nmo_max});

          SpMatrix Tban({nb, na * nchol_max});
          Sp3Tensor_ref T3ban(Tban.origin(), {nb, na, nchol_max});
          SpMatrix Tabn({na, nb * nchol_max});
          Sp3Tensor_ref T3abn(Tabn.origin(), {na, nb, nchol_max});

          auto Gal = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[0], Ka, npol == 2);
          auto Gbk = get_PsiK<boost::multi::array<SPComplexType, 2>>(nmo_per_kp, PsiT[0], Kb, npol == 2);
          for (int a = 0; a < na; ++a)
            for (int l = 0; l < nl; ++l)
              Gal[a][l] = ma::conj(Gal[a][l]);
          for (int b = 0; b < nb; ++b)
            for (int k = 0; k < nk; ++k)
              Gbk[b][k] = ma::conj(Gbk[b][k]);

          ma::product(Gal, ma::T(Lbnl), Tabn);
          ma::product(Gbk, ma::T(Lank), Tban);

          ComplexType E_(0.0);
          for (int a = 0; a < na; ++a)
            for (int b = 0; b < nb; ++b)
              E_ += ma::dot(T3abn[a][b], T3ban[b][a]);
          gQ[Q] -= scl * 0.5 * real(E_);
        }
        if (type == COLLINEAR)
        {
          APP_ABORT(" Finish UHF.\n ");
        }
      }
    }
    TG.Global().all_reduce_in_place_n(gQ.begin(), nkpts, std::plus<>());
    RealType E_ = std::accumulate(gQ.begin(), gQ.end(), RealType(0.0));
    for (auto& v : gQ)
      v /= E_;
    app_log(1," EXX: {}", E_);
    for (auto v : gQ)
    {
      if (v < 0.0)
        APP_ABORT(" Error: g(Q) < 0.0, implement shift to g(Q). ");
    }
  }

  if (out_of_core)
  {
    return HamiltonianOperations<MP>(
        KP3IndexFactorization_batched<MP,shmSpMatrix>(type, TG, std::move(nmo_per_kp), std::move(nchol_per_kp),
                                                   std::move(Qminus), std::move(nocc_per_kp), std::move(QKtok2),
                                                   std::move(H1), std::move(haj), std::move(LQKikn), std::move(LQKank),
                                                   std::move(LQKakn), std::move(LQKbnl), std::move(LQKbln),
                                                   std::move(Qmap), std::move(vn0), std::move(gQ), nsampleQ, E0,
                                                   device_allocator<ComplexType>{}, global_origin, global_ncvecs,
                                                   memory));
  }
  else
  {
    using devSpMatrix = boost::multi::array<SPComplexType, 2, device_allocator<SPComplexType>>;
    return HamiltonianOperations<MP>(
        KP3IndexFactorization_batched<MP,devSpMatrix>(type, TG, std::move(nmo_per_kp), std::move(nchol_per_kp),
                                                   std::move(Qminus), std::move(nocc_per_kp), std::move(QKtok2),
                                                   std::move(H1), std::move(haj), std::move(LQKikn), std::move(LQKank),
                                                   std::move(LQKakn), std::move(LQKbnl), std::move(LQKbln),
                                                   std::move(Qmap), std::move(vn0), std::move(gQ), nsampleQ, E0,
                                                   device_allocator<ComplexType>{}, global_origin, global_ncvecs,
                                                   memory));
  }
  return HamiltonianOperations<MP>{};
}

template<bool MP> HamiltonianOperations<MP> 
KPFactorizedHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
                                                  std::vector<PsiT_Matrix>& PsiT,
                                                  TaskGroup_& TGprop,   
                                                  TaskGroup_& TGwfn,    
                                                  hdf_archive& hdf_restart)
{
#if !defined(ENABLE_DEVICE)
  if (TG.TG_local().size() > 1 || (not batched))
  {
    return getHamiltonianOperations_shared<MP>(type, PsiT, TGprop, TGwfn, hdf_restart);
  }
  else
#endif
  {
    return getHamiltonianOperations_batched<MP>(type, PsiT, TGprop, TGwfn, hdf_restart);
  }
}; 

template 
HamiltonianOperations<true> KPFactorizedHamiltonian::getHamiltonianOperations<true>(
	WALKER_TYPES,std::vector<PsiT_Matrix>&, TaskGroup_&,TaskGroup_&,hdf_archive&);
template 
HamiltonianOperations<false> KPFactorizedHamiltonian::getHamiltonianOperations<false>(
	WALKER_TYPES,std::vector<PsiT_Matrix>&, TaskGroup_&,TaskGroup_&,hdf_archive&);

} // namespace afqmc
} // namespace sfqmc
