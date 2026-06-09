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
#include <mpi.h>

#include "config.h"
#include "Utilities/AppAbort.hpp"
#include "Utilities/type_traits/container_traits_multi.h"
#include "hdf/hdf_multi.h"
#include "hdf/hdf_archive.h"

#include "AFQMC/config.h"
#include "AFQMC/Utilities/Utils.hpp"
#include "AFQMC/Utilities/kp_utilities.hpp"
#include "RealDenseHamiltonian_v2.h"
#include "AFQMC/SlaterDeterminantOperations/rotate.hpp"
#include "upgradeOneBodyIntegrals.hpp"

namespace sfqmc
{
namespace afqmc
{

template<bool MP, bool REAL_ONEBODY> HamiltonianOperations<MP> 
RealDenseHamiltonian_v2::getHamiltonianOperations_impl(WALKER_TYPES type,
                                                       std::vector<PsiT_Matrix>& PsiT,
                                                       TaskGroup_& TGprop,   
                                                       [[maybe_unused]] TaskGroup_& TGwfn,    
                                                       [[maybe_unused]] hdf_archive& hdf_restart)
{
  using SPComplexType = typename to_working_precision<MP,ComplexType>::type;
  using SPRealType    = typename to_working_precision<MP,RealType   >::type;
  using ValueType     = typename std::conditional<REAL_ONEBODY, RealType, ComplexType>::type;

  using shmCMatrix    = multi::array<ComplexType, 2, shared_allocator<ComplexType>>;
  using shmSpRMatrix  = multi::array<SPRealType, 2, shared_allocator<SPRealType>>;
  using shmSp3Tensor  = multi::array<SPComplexType, 3, shared_allocator<SPComplexType>>;
  using CMatrix       = multi::array<ComplexType, 2>;
  using SpRMatrix     = multi::array<SPRealType, 2>;
  using SpRMatrix_ref = multi::array_ref<SPRealType, 2>;
  using CMatrix_ref   = multi::array_ref<ComplexType, 2>;

  using shmValueMatrix = multi::array<ValueType, 2 , shared_allocator<ValueType>>;
  using ValueMatrix   = multi::array<ValueType, 2>;
  using ValueMatrix_ref = multi::array_ref<ValueType, 2>;

  // for consistency checks
  int H1_SHAPE = 2;

  if (type == COLLINEAR)
    RUNTIME_CHECK(PsiT.size() % 2 == 0, "");
  int nspins = ((type != COLLINEAR) ? 1 : 2);
  int npol   = ((type == NONCOLLINEAR) ? 2 : 1);
  int ndet   = PsiT.size() / nspins;
  int nup    = PsiT[0].size(0);
  int ndown  = 0;
  if (nspins == 2)
    ndown = PsiT[1].size(0);
  int NEL = nup + ndown;

  // distribute work over equivalent nodes in TGprop.TG() across TG.Global()
  auto Qcomm=TG.Global().split(TGprop.getLocalGroupNumber(), TG.Global().rank());
#if defined(ENABLE_DEVICE)
  auto distNode=TG.Node().split(TGprop.getLocalGroupNumber(), TG.Node().rank());
#else
  auto distNode=TG.Node().split(0, TG.Node().rank());
#endif
  auto Qcomm_roots=Qcomm.split(distNode.rank(), Qcomm.rank());
  auto distNode_roots=TG.Global().split(distNode.rank(), TG.Global().rank());

  std::string base_error(" Error in RealDenseHamiltonian::getHamiltonianOperations():\n   ");

  hdf_archive dump(TG.Global());
  // right now only Node.root() reads
  if (distNode.root())
  {
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(base_error + "Problems opening integral file.");
    if (dump.push("Hamiltonian", false)<0)
      APP_ABORT(base_error + "Group Hamiltonian not found.");
  }

  std::vector<int> Idata(8);
  if (TG.Global().root())
  {
    if (!dump.readEntry(Idata, "dims"))
      APP_ABORT(base_error + "Problems reading dims. ");
  }
  TG.Global().broadcast_n(Idata.begin(), 8, 0);

  ComplexType E0;
  if (TG.Global().root())
  {
    std::vector<RealType> E_(2);
    if (!dump.readEntry(E_, "Energies"))
      APP_ABORT(base_error + "Problems reading Energies. ");
    E0 = E_[0] + E_[1];
  }
  TG.Global().broadcast_n(&E0, 1, 0);

  int global_ncvecs = Idata[7];
  int nc0, ncN;
  int node_number    = TGprop.getLocalGroupNumber();
  int nnodes_per_TG  = TGprop.getNGroupsPerTG();
  std::tie(nc0, ncN) = FairDivideBoundary(node_number, global_ncvecs, nnodes_per_TG);
  int local_ncv      = ncN - nc0;

  

  std::vector<int> hcore_shape;
  if (TG.Node().root())
  {
    if (!dump.getShape<ValueType>("hcore", hcore_shape))
      APP_ABORT(" Error in  RealDenseHamiltonian::getHamiltonianOperations_impl(): getShape(hcore) returned error. ");

    if (!(hcore_shape.size() == H1_SHAPE))
      APP_ABORT(base_error + "shape of H1 is not consistent with templated ValueType. ");
  }
  hcore_shape.resize(H1_SHAPE);
  TG.Node().broadcast_n(hcore_shape.begin(), H1_SHAPE, 0);

  // validate the combination of walker_type and hcore_shape
  if (type == CLOSED)
  {
    if (hcore_shape == std::vector<int>({2*NMO, NMO}))
      APP_ABORT(base_error + "COLLINEAR H1 Integrals for CLOSED walker type. Please generate CLOSED H1 integrals.");
    else if (hcore_shape == std::vector<int>({2*NMO, 2*NMO}))
      APP_ABORT(base_error + "NONCOLLINEAR H1 Integrals for CLOSED walker type. Please generate CLOSED H1 integrals.");
  }
  else if (type == COLLINEAR || type == FULLYPOLARIZED)
  {
    if (hcore_shape == std::vector<int>({2*NMO, 2*NMO}))
      APP_ABORT(base_error + "NONCOLLINEAR H1 Integrals for COLLINEAR / FULLYPOLARIZED walker type. "
                              "Please generate CLOSED or COLLINEAR H1 integrals.");
  }

  // this can be local, it's temporary
  ValueMatrix hcore({hcore_shape[0],hcore_shape[1]});
  if (TG.Node().root())
  {
    if (!dump.readEntry(hcore, "hcore"))
      APP_ABORT(base_error + "Problems reading /Hamiltonian/hcore.");
  }

  shmValueMatrix H1({nspins*npol*NMO,npol*NMO}, ValueType(0.0), shared_allocator<ValueType>{TG.Node()});

  // UPGRADE INTEGRALS CONSISTENT WITH WALKER_TYPE
  if (TG.Node().root())
  {
    ValueMatrix_ref _H1(raw_pointer_cast(H1.origin()), H1.extensions());
    std::fill_n(_H1.origin(), _H1.num_elements(), ValueType(0.0));
    upgradeOneBodyIntegrals(hcore,_H1,hcore_shape,NMO,npol,nspins,type,base_error);
  }

  shmSpRMatrix Likn({NMO * NMO, local_ncv}, SPRealType(0.0), shared_allocator<SPRealType>{distNode});
  if (distNode.root())
  {
    // read L
    if (dump.push("DenseFactorized", false)<0)
      APP_ABORT(base_error + " Group DenseFactorized not found. "); 
    SpRMatrix_ref L(raw_pointer_cast(Likn.origin()), Likn.extensions());
    hyperslab_proxy<SpRMatrix_ref, 2> hslab(L, std::array<int, 2>{NMO * NMO, global_ncvecs},
                                            std::array<int, 2>{NMO * NMO, local_ncv}, std::array<int, 2>{0, nc0});
    std::vector<int> shape;
    if (dump.getShape<RealType>("L", shape))
    {
      if (shape.size() == 3)
      {
        app_error(" Error: Found complex cholesky integrals in RealDenseHamiltonian_v2::getHamiltonianOperations().");
        APP_ABORT(base_error + " Please generate real integrals.");
      }
    }
    if (!dump.readEntry(hslab, std::string("L")))
      APP_ABORT(base_error + " Problems reading /Hamiltonian/DenseFactorized/L. ");
    if (Likn.size(0) != NMO * NMO || Likn.size(1) != local_ncv)
    {
      app_error(" Unexpected dimensins: ({}, {})", Likn.size(0), Likn.size(1));
      APP_ABORT(base_error + " Please generate real integrals.");
    }
    dump.pop();
  }
  TG.Node().barrier();

  shmCMatrix vn0({NMO, NMO}, ComplexType(0.0), shared_allocator<ComplexType>{distNode});
  shmCMatrix haj({ndet, NEL * npol * NMO}, ComplexType(0.0), shared_allocator<ComplexType>{TG.Node()});
  std::vector<shmSp3Tensor> Lnak;
  Lnak.reserve(PsiT.size());
  for (int nd = 0; nd < PsiT.size(); nd++)
    Lnak.emplace_back(shmSp3Tensor({local_ncv, PsiT[nd].size(0), npol * NMO}, shared_allocator<SPComplexType>{distNode}));
  TG.Node().barrier();

  // for simplicity
  CMatrix H1C({nspins * npol * NMO, npol* NMO}, ComplexType(0.0));
  copy_n_cast(raw_pointer_cast(H1.origin()), H1.size(0) * H1.size(1), H1C.origin());

  for (int nd = 0; nd < ndet; nd++)
  {
    // haj and add half-transformed right-handed rotation for Q=0
    if (nd % TG.Node().size() != TG.Node().rank())
      continue;
    if (type == COLLINEAR)
    {
      CMatrix_ref haj_r(raw_pointer_cast(haj[nd].origin()), {nup, NMO});
      ma::product(PsiT[2 * nd], H1C.sliced(0, NMO), haj_r);
      CMatrix_ref hbj_r(raw_pointer_cast(haj[nd].origin()) + (nup * NMO), {ndown, NMO});
      if (ndown > 0)
        ma::product(PsiT[2 * nd + 1], H1C.sliced(NMO, 2*NMO), hbj_r);
    }
    else
    {
      CMatrix_ref haj_r(raw_pointer_cast(haj[nd].origin()), {nup, npol*NMO});
      ma::product(ComplexType(1.0), PsiT[nd], H1C, ComplexType(0.0), haj_r);  // BUG: this fails due to a dimension mistmatch
    }
  }
  // Generate Lnak
  {
    CMatrix lik({npol*NMO, npol*NMO}, ComplexType(0.0));
    CMatrix lak({nup, npol*NMO}, ComplexType(0.0));
    for (int nd = 0; nd < ndet; nd++)
    {
      // all nodes across Qcomm share same segment {nc0,ncN}
      for (int nc = 0; nc < local_ncv; nc++)
      {
        if (nc % Qcomm.size() != Qcomm.rank())
          continue;
        for (int pol =0; pol < npol; pol++)
          for (int i = 0, ik = 0; i < NMO; i++)
            for (int k = 0; k < NMO; k++, ik++)
              lik[i+pol*NMO][k+pol*NMO] = ComplexType(static_cast<RealType>(Likn[ik][nc]), 0.0);
        ma::product(PsiT[nspins * nd], lik, lak);
        copy_n_cast(lak.origin(), nup * npol * NMO, raw_pointer_cast(Lnak[nspins * nd][nc].origin()));
        if (type == COLLINEAR)
        {
          ma::product(PsiT[2 * nd + 1], lik, lak.sliced(0, ndown));
          copy_n_cast(lak.origin(), ndown * NMO, raw_pointer_cast(Lnak[2 * nd + 1][nc].origin()));
        }
      }
    }
  }
  TG.Global().barrier();
  if (distNode.root())
  {
    for (auto& v : Lnak)
      Qcomm_roots.all_reduce_in_place_n(raw_pointer_cast(v.origin()), v.num_elements(), std::plus<>());
    std::fill_n(raw_pointer_cast(vn0.origin()), vn0.num_elements(), ComplexType(0.0));
  }
  TG.Node().barrier();

  // calculate vn0(i,l) = -0.5 sum_j sum_n L[i][j][n] L[j][l][n] = -0.5 sum_j sum_n L[i][j][n] L[l][j][n]
  {
    int i0, iN;
    std::tie(i0, iN) = FairDivideBoundary(Qcomm.rank(), NMO, Qcomm.size());
    if (iN > i0)
    {
      SpRMatrix v_({iN - i0, NMO}, SPRealType(0.0));
      SpRMatrix_ref Lijn(raw_pointer_cast(Likn.origin()), {NMO, NMO * local_ncv});
      ma::product(SPRealType(-0.5), Lijn.sliced(i0, iN), ma::T(Lijn), SPRealType(0.0), v_);
      copy_n_cast(v_.origin(), v_.num_elements(), raw_pointer_cast(vn0[i0].origin()));
    }
  }
  TG.Node().barrier();

  if (distNode.root())
  {
    distNode_roots.all_reduce_in_place_n(raw_pointer_cast(vn0.origin()), vn0.num_elements(), std::plus<>());
    dump.pop();
    dump.close();
  }

  return HamiltonianOperations<MP>(Real3IndexFactorization_batched_v2<MP,REAL_ONEBODY>(type, TGprop, std::move(H1), std::move(haj),
                                                                  std::move(Likn), std::move(Lnak), std::move(vn0), E0,
                                                                  device_allocator<ComplexType>{}, nc0, global_ncvecs,
                                                                  max_memory_MB));
}

template<bool MP> HamiltonianOperations<MP> 
RealDenseHamiltonian_v2::getHamiltonianOperations(WALKER_TYPES type,
                                                  std::vector<PsiT_Matrix>& PsiT,
                                                  TaskGroup_& TGprop,   
                                                  TaskGroup_& TGwfn,    
                                                  hdf_archive& hdf_restart)
{

  std::string base_error(" Error in RealDenseHamiltonian_v2::getHamiltonianOperations(): \n    ");

  int shape=-1;

  if (TG.Global().root())
  {
    // Open and inspect the dimensions of hcore
    hdf_archive dump(TG.Global(),false);
  
    if (!dump.open(fileName, H5F_ACC_RDONLY))
      APP_ABORT(base_error + "Problems opening integral file.");

    if (dump.push("Hamiltonian", false)<0)
      APP_ABORT(base_error + "Group Hamiltonian not found.");

    std::vector<int> shape_;
    if (!dump.getShape<RealType>("hcore", shape_))
      APP_ABORT(base_error + "getShape(hcore) returned error. ");
    shape = shape_.size();

    if (shape != 2 and shape != 3)
      APP_ABORT(base_error + "Inconsistent hcore shape. ");
    dump.close();
  }

  TG.Global().broadcast_n(&shape,1);

  if (shape == 3)
  {
    app_log(1, "reading complex-valued 1-body Hamiltonian");
    return getHamiltonianOperations_impl<MP,false>(type, PsiT, TGprop, TGwfn, hdf_restart);
  }
  else {
    app_log(1, "reading real-valued 1-body Hamiltonian");
    return getHamiltonianOperations_impl<MP,true>(type, PsiT, TGprop, TGwfn, hdf_restart);
  }
}

// instantiate templates
template HamiltonianOperations<true> RealDenseHamiltonian_v2::getHamiltonianOperations<true>(
	WALKER_TYPES, std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);
template HamiltonianOperations<false> RealDenseHamiltonian_v2::getHamiltonianOperations<false>(
	WALKER_TYPES, std::vector<PsiT_Matrix>&,TaskGroup_&,TaskGroup_&,hdf_archive&);

} // namespace afqmc
} // namespace sfqmc
