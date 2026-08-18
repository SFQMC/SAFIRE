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
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/h5_utils.hpp"
#include "numerics/nda_functions.hpp"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/HamiltonianOperations/detail/one_body.hpp"
#include "RealDenseHamiltonian.h"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
RealDenseHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  std::string base_error(" Error in RealDenseHamiltonian::getHamiltonianOperations():\n   ");
  auto all  = range::all;
  int nspin = ((type != COLLINEAR) ? 1 : 2);
  int npol  = ((type == NONCOLLINEAR) ? 2 : 1);
  int ndet = PsiT.extent(0);
  int nact_up = PsiT(0,0).extent(0);
  int NMO = PsiT(0,0).extent(1)/npol;
  int nspin_in_PsiT = PsiT.extent(1);
  int nspin_in_H1 = 1, npol_in_H1 = 1; // read/broadcast below
  int nspin_in_H2 = 1, npol_in_H2 = 1; // read/broadcast below
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.size(1)%npol != 0");
  utils::check(nspin_in_PsiT == nspin, base_error + "nspin mismatch in PsiT {} != {} expected", nspin_in_PsiT, nspin);
  utils::check(nspin==1 or npol==1, base_error + "Both nspin and npol can not be >1 simultaneously."); 

  int nact_dn = (type == COLLINEAR ? PsiT(0,1).extent(0) : 0l);

  std::vector<long> Idata(8);
  ComplexType E0;
  h5::file file;
  if (mpi->comm.root()) 
  {
    file = h5::file(fileName,'r');
    h5::group root = h5::group(file);
    h5::group g = root.open_group("Hamiltonian");

    h5::h5_read(g,"dims",Idata);

    E0 = read_energy_offset(root, "std", type, nact_up, nact_dn);

    {
      // Too many choices, consider forcing standard structure (e.g. [ns_f][np_f*NMO][np_f*NMO]
      auto l = h5::array_interface::get_dataset_info(g,"hcore");
      if(l.rank() == 2) {
        // [2*NMO][NMO] 
        if( l.lengths[0]==2*NMO and l.lengths[1]==NMO ) {
          nspin_in_H1=2;
          npol_in_H1=1; 
        } else {
          nspin_in_H1=1;
          utils::check(l.lengths[0] == l.lengths[1], base_error + "Size mismatch");
          utils::check(l.lengths[0]==NMO or l.lengths[0]==2*NMO, base_error +  "Size mismatch");
          npol_in_H1 = l.lengths[0]/NMO;
        }
      } else if(l.rank() == 3) {
        if (l.lengths[0]==2*NMO and l.lengths[1]==2*NMO)
        {
          nspin_in_H1=1;
          npol_in_H1=2;
          // checking l.lengths[0] == l.lengths[1] is unnecessary, see condition above
          utils::check(l.lengths[2]==2, base_error + "Incorrect tailing dimension of hcore in h5 file. Expected 2, got {}",l.lengths[2]);
        } else {
          nspin_in_H1=l.lengths[0];
          utils::check(l.lengths[1] == l.lengths[2], base_error + "Size mismatch");
          utils::check(l.lengths[1]==NMO or l.lengths[1]==2*NMO, base_error + "Size mismatch");
          npol_in_H1 = l.lengths[1]/NMO;
        }
      } else {
        utils::check(false, base_error + "Invalid hcore rank:{}",l.rank());
      }
      utils::check(walkerDimsAreConvertible(nspin_in_H1, npol_in_H1, nspin, npol), "Hamiltonian with nspin: {}, npol: {} cannot be broadcasted to {}", nspin_in_H1, npol_in_H1, walkerTypeToString(type));
    }
    {
      // cholesky tensor
      h5::group vgrp = g.open_group("DenseFactorized");
      auto l = h5::array_interface::get_dataset_info(vgrp,"L");
      if(l.rank()==2) {
        //[nspin_in_H2*npol_in_H2*NMO*NMO]][ncv]
        if( nspin_in_H1 > 1 ) nspin_in_H2 = l.lengths[0] / (NMO*NMO); 
        if( npol_in_H1 > 1 ) npol_in_H2 = l.lengths[0] / (NMO*NMO); 
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2*NMO*NMO, 
                      base_error + "Inconsistent size of DenseFactorized/L:({}, {}). Incompatible with nspin_in_H2:{}, npol_in_H2:{}, NMO:{} found in hcore",l.lengths[0],l.lengths[1],nspin_in_H2,npol_in_H2,NMO);
      } else if(l.rank()==3 or l.rank()==4) {
        //rank:3 [nspin_in_H2*npol_in_H2][NMO*NMO]][ncv]
        //rank:4 [nspin_in_H2*npol_in_H2][NMO][NMO]][ncv]
        if( nspin_in_H1 > 1 ) nspin_in_H2 = l.lengths[0]; 
        if( npol_in_H1 > 1 ) npol_in_H2 = l.lengths[0];  
        utils::check( l.lengths[0] == nspin_in_H2*npol_in_H2, 
                      base_error +  "Inconsistent size of DenseFactorized/L:({}, ...). Incompatible with nspin_in_H2:{}, npol_in_H2:{} found in hcore",l.lengths[0],nspin_in_H2,npol_in_H2);
      } else {
        utils::check(false, "Invalid Cholesky vector rank:{} ",l.rank());
      }
    }
  }
  mpi->comm.broadcast_n(Idata.begin(), 8, 0);
  mpi->comm.broadcast_n(&E0, 1, 0);
  mpi->comm.broadcast_n(&nspin_in_H1, 1, 0);
  mpi->comm.broadcast_n(&npol_in_H1, 1, 0);
  mpi->comm.broadcast_n(&nspin_in_H2, 1, 0);
  mpi->comm.broadcast_n(&npol_in_H2, 1, 0);

  // number of cholesky vectors
  int ncv = Idata[7];
  
  // allocate shared arrays

  auto H1 = memory::share_from_root(*mpi, [&]() {
    h5::group g = h5::group(file).open_group("Hamiltonian"); 
    memory::array<HOST_MEMORY, ComplexType, 3> H1(nspin_in_H1, npol_in_H1 * NMO, npol_in_H1 * NMO);
    sfqmc::utils::h5_read(g,"hcore",nda::reshape(H1(), nspin_in_H1 * npol_in_H1 * NMO, npol_in_H1 *NMO));
    return H1;
  });

  auto Likn = memory::share_from_root(*mpi, [&]() {
    h5::group g = h5::group(file).open_group("Hamiltonian"); 
    auto Likn = memory::array<MEM,RealType,4>(nspin_in_H2*npol_in_H2,NMO,NMO,ncv);

    h5::group vgrp = g.open_group("DenseFactorized"); 
    auto l = h5::array_interface::get_dataset_info(vgrp,"L");
    if(l.rank()==2) {
      utils::check_shape(l, "DenseFactorized/L", nspin_in_H2*npol_in_H2*NMO*NMO, ncv);
      auto L_ = nda::reshape(Likn(),std::array<long,2>{nspin_in_H2*npol_in_H2*NMO*NMO,ncv});
      utils::h5_read(vgrp,"L",L_);
    } else if(l.rank()==3) {
      utils::check_shape(l, "DenseFactorized/L", nspin_in_H2*npol_in_H2, NMO*NMO, ncv);
      auto L_ = nda::reshape(Likn(),std::array<long,3>{nspin_in_H2*npol_in_H2,NMO*NMO,ncv});
      utils::h5_read(vgrp,"L",L_);
    } else if(l.rank()==4) {
      utils::check_shape(l, "DenseFactorized/L", nspin_in_H2*npol_in_H2, NMO, NMO, ncv);
      auto L_ = nda::reshape(Likn(),std::array<long,4>{nspin_in_H2*npol_in_H2,NMO,NMO,ncv});
      utils::h5_read(vgrp,"L",L_);
    } else {
      utils::check(false, "Invalid Cholesky vector rank:{} ",l.rank());
    }
    return Likn;
  });
  
  mpi->comm.barrier();

  auto nel = std::to_array<long>({nact_up, (type == COLLINEAR ? nact_dn : 0l)});
  auto haj = half_rotate_hamiltonian<MEM>(*mpi, nel, nspin, npol, nspin_in_H1, npol_in_H1, NMO, PsiT(), H1());
  
  nda::array<memory::const_shared_array<MEM,ComplexType,5>,1> Lnak(nspin);
  for(int is = 0; is < nspin; is++) {
    Lnak(is) = memory::share_from_ranks<MEM,ComplexType,5,1>(*mpi,
        {ndet,npol,ncv,nel[is],NMO},
        [&,is](std::array<long,1> idx, auto&& block) {
      auto [id] = idx;
      if(nel[is] == 0) {
        return;  // empty sector (ndown==0): no rows to half-rotate
      }
      auto Aai = math::sparse::to_array<'N'>(PsiT(id,is));
      auto Aai_r = memory::to_real_view(Aai);
      auto L_r = memory::to_real_view(block);
      for(int p=0; p<npol; ++p) {
        int ip_f = (is*npol + p) % (nspin_in_H2*npol_in_H2);

        auto Aai_is = Aai_r(all,nda::range(p*NMO,(p+1)*NMO),all);
        nda::tensor::contract(RealType(1.0),Aai_is,"aic",Likn()(ip_f,all,all,all),"ijn",
                              RealType(0.0),L_r(p,all,all,all,all),"najc");
      }
    });
  }

  // exchange potential, parallelize over (isp,i) to avoid temporary memory
  // v0(i,l) = -0.5 sum_j sum_n L[i][j][n] L[j][l][n] = -0.5 sum_j sum_n L[i][j][n] L[l][j][n]
  auto v0 = memory::share_from_ranks<HOST_MEMORY,RealType,3,2>(*mpi,
      {nspin_in_H2*npol_in_H2, NMO, NMO},
      [&](std::array<long,2> idx, auto&& block) {
    auto [isp, i] = idx;
    if constexpr (MEM==HOST_MEMORY) {
      nda::tensor::contract(RealType(-0.5),Likn()(isp,i,all,all),"jn",
                                          Likn()(isp,all,all,all),"ljn",
                            RealType(0.0),block,"l");
    } else {
      memory::array<MEM,RealType,1> vt(NMO);
      nda::tensor::contract(RealType(-0.5),Likn()(isp,i,all,all),"jn",
                                          Likn()(isp,all,all,all),"ljn",
                            RealType(0.0),vt,"l");
      block = nda::to_host(vt);
    }
  });

  return HamiltonianOperations<MEM>(Real3IndexFactorization<MEM>(mpi,type,NMO,nact_up,nact_dn,
       std::move(H1), std::move(haj), std::move(Likn), std::move(Lnak), std::move(v0), E0,
       max_memory_MB));
}

// instantiate templates
template HamiltonianOperations<HOST_MEMORY> 
  RealDenseHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY>
  RealDenseHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc
