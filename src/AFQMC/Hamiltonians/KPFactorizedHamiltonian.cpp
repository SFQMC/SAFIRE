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
#include <iomanip>
#include <vector>
#include <utility>
#include <numeric>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/h5_utils.hpp"
#include "numerics/nda_functions.hpp"
#include "AFQMC/config.h"

#include "nda/h5.hpp"
#include "nda/tensor.hpp"

#include "KPFactorizedHamiltonian.h"
#include "AFQMC/Utilities/wfn_utils.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/HamiltonianOperations/detail/one_body.hpp"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{

// PsiT[idet][ispin][ik] -> csr_mat( nel_up/nel_dn, NMO )
template<MEMORY_SPACE MEM> HamiltonianOperations<MEM>
KPFactorizedHamiltonian::getHamiltonianOperations(WALKER_TYPES type,
               std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
               nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  std::string base_error(" Error in KPFactorizedHamiltonian::getHamiltonianOperations_shared: \n    ");
  auto all = range::all;
  std::string format;  // only meaningful at root
  double GBx = sizeof(ComplexType)/double(1024*1024*1024);
  ComplexType zero(0.0),one(1.0);
  long nspin = (type == COLLINEAR?2:1);
  long npol = (type == NONCOLLINEAR?2:1);
  int nspin_in_PsiT = PsiT.extent(1);
  long ndet = PsiT.extent(0);
  long nel_up = PsiT(0,0).extent(0);
  long NMO = PsiT(0,0).extent(1)/npol;
  utils::check(PsiT(0,0).extent(1)%npol==0, base_error + "Psi.extent(1)%npol != 0");
  utils::check(nspin_in_PsiT==nspin, "Size mismatch");
  utils::check(ndet==1, "Error: ndet > 1 not yet implemented in KPFactorizedHamiltonian::getHamiltonianOperations.");
  long nel_dn = (type == COLLINEAR ? PsiT(0,1).extent(0) : 0l);
  for(int i=0; i<ndet; ++i) {
    for(int ip=0; ip<npol; ++ip) {
      utils::check_shape(PsiT(i,0), "PsiT", nel_up, npol*NMO);
      if(type == COLLINEAR)
        utils::check_shape(PsiT(i,nspin_in_PsiT-1), "PsiT", nel_dn, NMO);
    }
  }
  utils::check(nel_up >= nel_dn, base_error + "nel_up:{} < nel_dn:{} not allowed.",nel_up,nel_dn);

  // Hamiltonian variables
  int nspin_in_H1 = nspin;
  int npol_in_H1 = npol;
  // BZ variables
  int nkpts = 1;
  int nqpts = 1;
  int nkpts_ibz = 1;
  int nqpts_ibz = 1;
  int nbnd = 1;
  int Q0_index = -1;
  nda::array<int,1> minusq;
  nda::array<int,2> qk_to_k2;
  nda::array<int,1> kp_to_ibz;
  nda::array<int,1> qp_to_ibz;
  nda::array<bool,1> kp_trev;
  nda::array<bool,1> qp_trev;
  nda::array<int,1> kp_trev_pair;
  nda::array<int,1> nchol;
  nda::array<double,2> qpoints;

  ComplexType E0(0);
  h5::file file;
  // Read nbnd, BZ info, etc from h5. Only root reads
  if (mpi->comm.root())
  {
    file = h5::file(fileName,'r');
    h5::group grp = h5::group(file);
    format = get_hamiltonian_format(grp);
    E0 = read_energy_offset(grp, format, type, nel_up, nel_dn);
    if(format == "coqui") {
      // open subgroup
      h5::group hgrp = grp.open_group("System");
      {
        h5::group bz = hgrp.open_group("BZ");
        // read nkpts
        h5::h5_read_attribute(bz,"number_of_kpoints",nkpts);
        h5::h5_read_attribute(bz,"number_of_kpoints_ibz",nkpts_ibz);
        h5::h5_read_attribute(bz,"number_of_qpoints",nqpts);
        utils::check(nqpts == nkpts, base_error + "nqpts != nkpts, nqpts:{}, nkpts:{}",nqpts,nkpts);
        h5::h5_read_attribute(bz,"number_of_qpoints_ibz",nqpts_ibz);
        // nbnd
        int nbnd_in_H1{};
        h5::h5_read_attribute(hgrp,"number_of_bands",nbnd_in_H1);  // per kpoint
        // check nbnd 
        nbnd = long(NMO/nkpts);
        utils::check(NMO%nkpts==0, base_error + "NMO%nkpts != 0, NMO: {}, nkpts: {}",NMO,nkpts);
        utils::check((NMO/nkpts)==nbnd_in_H1, base_error + "nbnd: {} differs from file: {}",nbnd,nbnd_in_H1);
        // read nspin_in_H1
        h5::h5_read_attribute(hgrp,"number_of_spins",nspin_in_H1);
        // read npol_in_H1
//        h5::h5_read_attribute(bz,"number_of_polarizations",n);
//        npol_in_H1=long(n);
        // TODO: does this format even support polarizations?
        npol_in_H1 = 1;

        utils::check(walkerDimsAreConvertible(nspin_in_H1, npol_in_H1, nspin, npol), "Hamiltonian with nspin: {}, npol: {} cannot be broadcasted to {}", nspin_in_H1, npol_in_H1, walkerTypeToString(type));
        minusq.resize(nkpts);
        nda::h5_read(bz,"qminus",minusq);
        qk_to_k2.resize(nkpts,nkpts);
        nda::h5_read(bz,"qk_to_k2",qk_to_k2);
        qp_to_ibz.resize(nkpts);
        nda::h5_read(bz,"qp_to_ibz",qp_to_ibz);
        kp_trev.resize(nkpts);
        nda::h5_read(bz,"kp_trev",kp_trev);
        kp_trev_pair.resize(nkpts);
        nda::h5_read(bz,"kp_trev_pair",kp_trev_pair);
        qp_trev.resize(nkpts);
        nda::h5_read(bz,"qp_trev",qp_trev);
        qpoints.resize(nkpts,3);
        nda::h5_read(bz,"qpoints",qpoints);
        Q0_index=-1;
        for(int i=0; i<nkpts; i++)
          if(nda::sum(qpoints(i,all)*qpoints(i,all)) < 1e-8) {
            utils::check(Q0_index<0, "Error: Multiple points with Q=0: {}, {}",Q0_index,i);
            Q0_index = i;
          }
        utils::check(Q0_index>=0, "Error: Problems finding Q=0");
      }
      {
        // interaction
        h5::group igrp = grp.open_group("Interaction");

        auto expected_shape = std::to_array<long>({nspin_in_H1 * npol_in_H1, nkpts, nbnd, nbnd});
        nchol.resize(nkpts); 
        for(int Q=0; Q<nkpts; ++Q) {
          auto l = h5::array_interface::get_dataset_info(igrp,"Vq"+std::to_string(Q));
          utils::check(l.rank() == 6, "Rank mismatch");
          utils::check(std::ranges::equal(std::span(l.lengths).subspan(1,4), expected_shape),
                       "Interaction/Vq{} size mismatch: {} != {} ", Q, std::span(l.lengths).subspan(1,4), expected_shape);
          if(Q <= minusq(Q))
            nchol(Q) = l.lengths[0];
          else
            nchol(Q) = nchol(minusq(Q)); 
        }
      }
    } else if(format == "std") {
      // MAM: The "std" format, written for pyscf and the old fortran QE converter,
      //      was/is limited to spin independent basis sets. Generalize this if needed...
      // Current implementation is limited to cases with a consistent number of bands 
      // per kpoint, unlikely we will go back to the more general case.
      nspin_in_H1 = 1;
      npol_in_H1  = 1;
      h5::group hgrp = grp.open_group("Hamiltonian");
      std::vector<int> Idata(8);
      h5::h5_read(hgrp,"dims",Idata);
      nkpts = Idata[2];
      nkpts_ibz = Idata[2];
      nqpts_ibz = Idata[2];
      utils::check(Idata[3] == NMO, " Error: NMO differs from value in integral file. ");

      Idata.resize(nkpts);
      std::vector<double> Ddata(nkpts);  
      h5::h5_read(hgrp,"NMOPerKP",Idata);
      utils::check(Idata.size() == nkpts, "size(NMOPerKP):{} != nkpts",Idata.size());
      nbnd = Idata[0];
      for(int i=1; i<nkpts; ++i)
        utils::check(Idata[i] == nbnd, "Inconsistent number of bands per kpoint. We now require all kpoints to have a consistent number of bands (NMOPerKP)."); 
      minusq.resize(nkpts);
      nda::h5_read(hgrp,"MinusK",minusq);
      qk_to_k2.resize(nkpts,nkpts);
      nda::h5_read(hgrp,"QKTok2",qk_to_k2);
      nchol.resize(nkpts);
      nda::h5_read(hgrp,"NCholPerKP",nchol);
      utils::check(NMO == nbnd*nkpts, " Error: NMO:{}, nkpts:{}, nbnd:{}",NMO,nkpts,nbnd); 
      Q0_index=-1;
      for (int Q = 0; Q < nkpts; Q++)
      {
        if (minusq(Q) == Q)
        {
          bool found = true;
          for (int KI = 0; KI < nkpts; KI++)
            if (KI != qk_to_k2(Q,KI))
            { 
              found = false;
              break;
            }
          if (found)
          {
            Q0_index = Q;
            break;
          }
        }
      }  
      utils::check(Q0_index>=0, "Error: Problems finding Q=0");
    } else {
      utils::check(false,"Unknown file format: {}",format);
    }
  } // root

  mpi->comm.broadcast_value(nspin_in_H1);
  mpi->comm.broadcast_value(npol_in_H1);
  mpi->comm.broadcast_value(nbnd);
  mpi->comm.broadcast_value(nkpts);
  mpi->comm.broadcast_value(nkpts_ibz);
  mpi->comm.broadcast_value(nqpts_ibz);
  mpi->comm.broadcast_value(Q0_index);
  mpi->comm.broadcast_value(E0);
  if(not mpi->comm.root()) {
    minusq.resize(nkpts);
    qk_to_k2.resize(nkpts,nkpts);
    nchol.resize(nkpts);
  }
  mpi->broadcast(nchol);
  mpi->broadcast(minusq);
  mpi->broadcast(qk_to_k2);

  double nchol_av = nda::sum(nchol)/double(nchol.extent(0));
  app_log(1," # k-points (IBZ): {} ({})", nkpts, nkpts_ibz);
  app_log(1," # Q-points (IBZ): {} ({})", nqpts, nqpts_ibz);
  app_log(1," Average # of cholesky vectors {:.2g}", nchol_av);

  // If q==minusq(q), qmap(q) has the position of q in Lbnk.
  nda::array<int,1> Qmap(nkpts,-1);  
  int number_of_symmetric_Q = 0;  // number of q==minusq(q)
  int number_of_allocated_Q = 0;
  for (int Q = 0; Q < nkpts; Q++) {
    if(minusq(Q) == Q) Qmap(Q) = (number_of_symmetric_Q++);
    if(Q <= minusq(Q)) number_of_allocated_Q++;
  }

  // Read hamiltonian components from h5. Only root reads
  // H0: /System/H0:   [nspin][nkpts][npol*nbnd][npol*nbnd]
  auto H1 = memory::share_from_root(*mpi, [&]() {
    nda::array<ComplexType,4> H1_h(nspin_in_H1,nkpts,npol_in_H1*nbnd,npol_in_H1*nbnd);
    h5::group grp = h5::group(file);

    if(format == "std") {

      utils::check(npol_in_H1==1 and nspin_in_H1==1, "KPFactorized: std format requires nspin_in_H1==1 and npol_in_H1==1.");

      h5::group hgrp = grp.open_group("Hamiltonian");

      // only spin independent hamiltonians right now!
      // now read H1_kpK
      for (int K = 0; K < nkpts; K++) {
        auto h_ = H1_h(0,K,all,all);
        utils::h5_read(hgrp, "H1_kp" + std::to_string(K), h_);
      }

    } else if(format == "coqui") {
      h5::group sgrp = grp.open_group("System");

      if(nkpts_ibz == nkpts) {
        utils::h5_read(sgrp,"H0",H1_h());
      } else {
        nda::array<ComplexType,4> H1_ibz(nspin_in_H1, nkpts_ibz, npol_in_H1*nbnd, npol_in_H1*nbnd);
        utils::h5_read(sgrp,"H0",H1_ibz());

        nda::vector<int> kp_to_ibz(nkpts);
        utils::h5_read(sgrp, "BZ/kp_to_ibz", kp_to_ibz());
        nda::vector<bool> kp_trev(nkpts);
        utils::h5_read(sgrp, "BZ/kp_trev", kp_trev());

        for(int k = 0; k < nkpts; k++) {
          bool inversion_symmetry = kp_trev(k);
          if(inversion_symmetry) {
            H1_h(all, k, all, all) = nda::conj(H1_ibz(all, kp_to_ibz[k], all, all));
          } else {
            H1_h(all, k, all, all) = H1_ibz(all, kp_to_ibz[k], all, all);
          }
        }
      }
    } // format
    return H1_h;
  });

  app_log(2, "KPFactorizedHamiltonian: Allocating Lijn: {} GB",
    number_of_allocated_Q*nspin_in_H1*npol_in_H1*nkpts*nbnd*nbnd*nchol_av*GBx);
  // L(Q)(ispin*ip,ik,i,j,n): Since each qpoint has its own nchol
  // Entries with Q > minusq(Q) are never read and stay default-constructed (empty)
  nda::array<memory::const_shared_array<MEM,ComplexType,6>,1> LQ(nkpts);
  for (int Q = 0; Q < nkpts; Q++) {
    if(Q > minusq(Q)) {
      continue;
    }
    // too much memory? replace by share_from_ranks?
    LQ(Q) = memory::share_from_root(*mpi, [&]() {
      nda::array<ComplexType,6> L_h(nspin_in_H1,npol_in_H1,nkpts,nbnd,nbnd,nchol(Q));
      h5::group grp = h5::group(file);

      if(format == "std") {
        h5::group lgrp = grp.open_group("Hamiltonian").open_group("KPFactorized");
        auto L2d = nda::reshape(L_h(0,0,nda::ellipsis{}),
                                std::array<long,2>{nkpts,nbnd*nbnd*nchol(Q)});
        utils::h5_read(lgrp, "L" + std::to_string(Q), L2d);
        // normalization (1/sqrt(nkpts)) assummed to be included
      } else if(format == "coqui") {
        h5::group igrp = grp.open_group("Interaction");
        nda::array<ComplexType,5> L(nchol(Q),nspin_in_H1*npol_in_H1,nkpts,nbnd,nbnd);
        utils::h5_read(igrp,"Vq"+std::to_string(Q),L);
        utils::check_shape(L, "Vq", nchol(Q), nspin_in_H1*npol_in_H1, nkpts, nbnd, nbnd);
        auto L2d = nda::reshape(L,std::array<long,2>{nchol(Q),nspin_in_H1*npol_in_H1*nkpts*nbnd*nbnd});
        auto LQ2d = nda::reshape(L_h(),std::array<long,2>{nspin_in_H1*npol_in_H1*nkpts*nbnd*nbnd,nchol(Q)});
        LQ2d() = nda::transpose(L2d());
        // normalize
        nda::tensor::scale(ComplexType(1.0/std::sqrt(RealType(nkpts))), L_h());
      } // format
      return memory::to_memory_space<MEM>(L_h);
    });
  } // Q

 // kpoint dependent occupations: for each (spin,kpoint), the list of PsiT row
 // indices occupying that kpoint. Computed on every rank (depends only on PsiT,
 // which is replicated), so no broadcast of the ragged list-of-lists is needed.
  auto nocc = nocc_per_kpoint(type,nkpts,PsiT);
  utils::check(nel_up == nelec_for_spin(nocc,0), "Error: Mismatch in number of electrons: nel_up:{} sum(nel_up(k)):{}",nel_up,nelec_for_spin(nocc,0));
  if(type == COLLINEAR)
    utils::check(nel_dn == nelec_for_spin(nocc,1), "Error: Mismatch in number of electrons: ndown:{} sum(ndown(k)):{}",nel_dn,nelec_for_spin(nocc,1));

  /* half-rotate LQ and H1:
   * Given that PsiT = H(SM),
   * Lank(Q)(idet,ispin,a,j,n) = sum_i PsiT(idet,ispin)(a,ip,i) * LQ(Q)(ispin*ip,ik,i,j,n)
   *  where a includes the kpoint index implicitly, e.g. a:{0,nup/ndown}
   */
  app_log(2, "KPFactorizedHamiltonian: Allocating Lank: {} GB",
        (number_of_symmetric_Q+nkpts)*ndet*nspin_in_H1*npol_in_H1*nel_up*nbnd*nchol_av*GBx);
  auto [Lank, Lbnk] = kpoint_half_rotate_cholesky<MEM>(*mpi, type, nspin_in_H1, npol_in_H1,
      NMO, nbnd, PsiT, nocc, minusq, qk_to_k2, Qmap, nchol, LQ);

  // haj(idet,a_is_ik,j_ip_ik) = sum_j PsiT(idet,is)(a_is,i) H1(is,ik,i,j_ip)
  auto nel = std::to_array<long>({nel_up, (type == COLLINEAR ? nel_dn : 0l)});

  auto hfull = memory::share_from_root(*mpi, [&](){
    memory::array<MEM,ComplexType,3> hfull(nspin_in_H1, npol_in_H1*nkpts*nbnd, npol_in_H1*nkpts*nbnd);
    hfull() = 0;
    auto hfull7 = nda::reshape(hfull, nspin_in_H1, npol_in_H1, nkpts, nbnd, npol_in_H1, nkpts, nbnd);
    for(int ik = 0; ik < nkpts; ik++) {
      hfull7(all, all, ik, all, all, ik, all) = nda::reshape(H1(), nspin_in_H1, nkpts, npol_in_H1, nbnd, npol_in_H1, nbnd)(all, ik, all, all, all, all);
    }

    return hfull;
  });
  auto haj = half_rotate_hamiltonian<MEM>(*mpi, nel, nspin, npol, nspin_in_H1, npol_in_H1, NMO, PsiT(), hfull());

  // calculate vn0
  // v0(s,k,i,l) = -0.5*sum_k,q sum_j <i_k,j_k-q|j_k-q,l_k>
  //         = -0.5 sum_kq sum_j,n L[Q](s,k,i,j,n) conj( L[Q](s,k,l,j) )
  auto v0 = memory::share_from_ranks<HOST_MEMORY,ComplexType,4,2>(*mpi,
      {nspin_in_H1*npol_in_H1, nkpts, nbnd, nbnd},
      [&](std::array<long,2> idx, auto&& block) {
    memory::buffered_array<MEM,ComplexType,2> vt(nbnd,nbnd);
    auto [isp,ik] = idx;
    long is_ = isp/npol_in_H1;
    long ip_ = isp%npol_in_H1;
    vt() = zero;
    for(long Q=0; Q<nkpts; Q++) {
      if(Q<=minusq(Q)) {
        auto Lijn = LQ(Q)()(is_,ip_,ik,nda::ellipsis{});
        nda::tensor::contract(ComplexType(-0.5),Lijn,"ijn",nda::conj(Lijn),"ljn",one,vt,"il");
      } else {
        // L[Q,k,k2](i,j,n) = conj( L[-Q,k2,k](j,i,n) )
        int Qm = minusq(Q);
        int k2 = qk_to_k2(Q,ik);
        auto Lijn = LQ(Qm)()(is_,ip_,k2,nda::ellipsis{});
        nda::tensor::contract(ComplexType(-0.5),nda::conj(Lijn),"jin",Lijn,"jln",one,vt,"il");
      }
    }
    block = vt();
  });

  return HamiltonianOperations<MEM>(
      KP3IndexFactorization<MEM>(mpi,type,nbnd,Q0_index,std::move(nocc),std::move(minusq),
        std::move(qk_to_k2),std::move(Qmap),std::move(H1),std::move(haj),std::move(LQ),
        std::move(Lank),std::move(Lbnk),std::move(v0),E0,buffer_size));
}

template HamiltonianOperations<HOST_MEMORY>
  KPFactorizedHamiltonian::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY> 
  KPFactorizedHamiltonian::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif 

} // namespace afqmc
} // namespace sfqmc
