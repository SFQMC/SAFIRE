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

#pragma once

#include <cassert>
#include <cstdlib>
#include <vector>
#include <type_traits>
#include "IO/app_loggers.h"

#include "config.h"
#include "IO/AppAbort.hpp"
#include "AFQMC/config.h"

#include "nda/h5.hpp"

namespace sfqmc
{
namespace afqmc
{
template<class WalkerSet>
bool dumpSamplesHDF5([[maybe_unused]] WalkerSet& wset,
                     [[maybe_unused]] h5::file& dump,
                     [[maybe_unused]] int nW_to_file)
{
  return true;
  APP_ABORT("Finish ");
/*
  if(nW_to_file==0) return true;
  if(head) { 

    int nW = numWalkers();

    std::vector<int> from(nproc_heads);
    MPI_Allgather(&nW,1,MPI_INT,from.data(),1,MPI_INT,TG.TG_heads().impl_);

    int nWtot = std::accumulate(from.begin(),from.end(),int(0));
    int w0 = std::accumulate(from.begin(),from.begin()+rank_heads,int(0));

    if(nW_to_file<0) nW_to_file=nWtot;
    nW_to_file = std::min(nW_to_file,nWtot);

    // careful here, avoid sending extra information (e.g. B mats for back propg)
    int wlk_nterms = (9 + nrow*ncol);
    int wlk_sz = wlk_nterms*sizeof(ComplexType);
    int nwlk_per_block = std::min(std::max(1,hdf_block_size/wlk_sz),nW_to_file);
    int nblks = nW_to_file/nwlk_per_block + ((nW_to_file%nwlk_per_block>0)?1:0);
    std::vector<int> wlk_per_blk;
  
    if(rank_heads==0) {

      // check that restart data doesnot exist 
      counts.resize(nproc_heads);
      displ.resize(nproc_heads);
      wlk_per_blk.reserve(nblks);

      commBuff.reserve(wlk_nterms*nwlk_per_block);

      std::vector<int> Idata(6);
      Idata[0]=nW_to_file;
      Idata[1]=nblks;
      Idata[2]=wlk_nterms;
      Idata[3]=wlk_sz;
      Idata[4]=nrow;
      Idata[5]=ncol;

      dump.push("Walkers");
      dump.write(Idata,"dims");
      
    }

    // to extract a single walker from walker storage 
    MPI_Datatype rtype, stype;
    {
      MPI_Datatype stype_;
      MPI_Type_contiguous(2*wlk_nterms, MPI_DOUBLE, &stype_);
      MPI_Type_commit(&stype_);
      MPI_Aint loc0, loc1;
      MPI_Get_address(walkers.values(), &loc0);
      MPI_Get_address(walkers.values()+walker_size, &loc1);
      MPI_Aint dis = loc1-loc0;
      MPI_Type_create_resized(stype_,0,dis,&stype); 
      MPI_Type_commit(&stype);
      MPI_Type_free(&stype_);

      // to deposit a single walker on contiguous memory
      MPI_Type_contiguous(2*wlk_nterms, MPI_DOUBLE, &rtype);
      MPI_Type_commit(&rtype);
    }

    int nsent=0;
    // ready to send walkers to head in blocks
    for(int i=0, ndone=0; i<nblks; i++, ndone+=nwlk_per_block) {

      int nwlk_tot = std::min(nwlk_per_block,nW_to_file-ndone);  
      int nw_to_send=0; 
      if( w0+nsent >= ndone && w0+nsent < ndone + nwlk_tot) 
        nw_to_send = std::min(nW-nsent,(ndone+nwlk_tot)-(w0+nsent)); 
 
      if(rank_heads==0) {

        for(int p=0, nt=0; p<nproc_heads; p++) {
   
          int n_ = 0;
          int nn = nt + nW;   
          if( ndone+nwlk_tot > nt && ndone < nt+nW ) {
            if(ndone <= nt)
              n_ = std::min(nW,(ndone+nwlk_tot)-nt);    
            else    
              n_ = std::min(nt+nW-ndone,nwlk_tot);    
          }    

          counts[p]=n_;
          nt+=from[p];

        }    
        displ[0]=0;
        for(int p=1, nt=0; p<nproc_heads; p++) {
          nt += counts[p-1];
          displ[p]=nt;
        }

        commBuff.resize(wlk_nterms*nwlk_tot);
      }  

      // assumes current storage structure
      MPI_Gatherv( getSM(nsent), nw_to_send, stype,
                    commBuff.data(), counts.data(), displ.data(), rtype, 
                    0, TG.TG_heads().impl_);
      nsent += nw_to_send;
      
      if(rank_heads==0) {  
        dump.write(commBuff,std::string("walkers_")+std::to_string(i));
        wlk_per_blk.push_back(nwlk_tot);
      }  
  
    }

    if(rank_heads==0) {
      dump.write(wlk_per_blk,"wlk_per_blk");
      dump.pop();
    }

    MPI_Type_free(&rtype);
    MPI_Type_free(&stype);

  }

  mpi->comm.barrier();  

  return true;
*/
}

// fh5 opened on all ranks with read-only
template<class WalkerSet>
bool restartFromHDF5(WalkerSet& wset,
                     int nW_per_rank,
                     h5::file& fh5,
                     bool set_to_target)
{
  auto all = nda::range::all;
  auto mpi = wset.get_mpi();

  std::vector<int> Idata(7);
  h5::group grp(fh5);
  utils::check(grp.has_subgroup("Walkers"), " restartFromHDF5: Missing Walkers dataset.");
  h5::group sgrp = grp.open_group("Walkers"); 
  utils::check(sgrp.has_subgroup("WalkerSet"), " restartFromHDF5: Missing WalkerSet dataset.");
  h5::group wgrp = sgrp.open_group("WalkerSet");
  h5::h5_read(wgrp, "dims", Idata);

  auto walker_type = wset.getWalkerType();

  int nWtot      = Idata[0];
  int wlk_nterms = Idata[2];
  int NMO        = Idata[4];
  int nup       = Idata[5];
  int ndn       = Idata[6];
  bool ft = wset.isFiniteTemperature();
  utils::check(wlk_nterms == wset.walkerSizeIO(), 
               " Inconsistent walker restart file: IO size, NMO, nup, ndown, WalkerType: {}, {}, {}, {}, {} ",
               wset.walkerSizeIO(), NMO, nup, ndn, walkerTypeToString(wset.getWalkerType()));

  // walker range belonging to this comm 
  int nW0, nWN;
  if (set_to_target)
  {
    utils::check(nWtot >= nW_per_rank * mpi->comm.size(),
                 " Error: Not enough walkers in restart file.");
    nW0 = nW_per_rank * mpi->comm.rank();
    nWN = nW0 + nW_per_rank;
  }
  else
  {
    utils::check(nWtot % mpi->comm.size() == 0, 
                 " Error: Number of walkers in restart file must be divisible by number of task groups.");
    nW0 = (nWtot / mpi->comm.size()) * mpi->comm.rank();
    nWN = nW0 + nWtot / mpi->comm.size();
  }

  int nw_local = nWN - nW0;
  { // to limit scope
    if(!ft){
      int nspin = ((walker_type == COLLINEAR) ? 2 : 1);
      int npol = ((walker_type == NONCOLLINEAR) ? 2 : 1);
      nda::array<ComplexType, 3> Psi(nspin, npol*NMO, nup);
      Psi() = ComplexType(0.0);
      wset.resize(nw_local, Psi);
    }
    else{
      int nspin = ((walker_type == COLLINEAR) ? 2 : 1);
      int npol = ((walker_type == NONCOLLINEAR) ? 2 : 1);
      nda::array<ComplexType, 3> U(nspin, npol*NMO, npol*NMO);
      nda::array<ComplexType, 2> D(nspin, npol*NMO);
      nda::array<ComplexType, 3> V(nspin, npol*NMO, npol*NMO);
      U() = ComplexType(0.0);
      D() = ComplexType(0.0);
      V() = ComplexType(0.0);
      wset.resize(nw_local, U, D, V);
    }
  }

  std::vector<int> wlk_per_blk;
  h5::h5_read(wgrp,"wlk_per_blk",wlk_per_blk);

  nda::array<ComplexType, 2> Data;

  // loop through blocks and read when necessary
  int ni = 0, nread = 0, bi = 0;
  while (nread < nw_local)
  {
    if (ni + wlk_per_blk[bi] > nW0)
    {
      // determine block of walkers to read
      int w0  = std::max(0, nW0 - ni);
      int nw_ = std::min(ni + wlk_per_blk[bi], nWN) - std::max(ni, nW0);
      Data.resize(nw_, wlk_nterms);

      nda::range r(w0,w0+nw_);
      nda::h5_read(wgrp,"walkers_"+std::to_string(bi),Data,std::tuple{r,all}); 
      for (int n = 0; n < nw_; n++, nread++)
        wset.copyFromIO(Data(n,all), nread);
    }
    ni += wlk_per_blk[bi++];
  }
  mpi->comm.barrier();
  return true;
}

template<class WalkerSet>
bool dumpToHDF5(WalkerSet& wset, h5::file& fh5)
{
  auto all = nda::range::all;
  auto mpi = wset.get_mpi();

  int nW = wset.size();
  auto nw_per_rank = mpi->comm.all_gather_value(nW);
  int nWtot = std::accumulate(nw_per_rank.begin(), nw_per_rank.end(), int(0));
  int w0    = std::accumulate(nw_per_rank.begin(), nw_per_rank.begin() + mpi->comm.rank(), int(0));

  auto walker_type = wset.getWalkerType();
  bool ft = wset.isFiniteTemperature();

  // careful here, avoid sending extra information (e.g. B mats for back propg)
  int wlk_nterms = wset.walkerSizeIO();
  int wlk_sz     = wlk_nterms * sizeof(ComplexType);

  // communicate to root
  int nwlk_per_block = std::min(std::max(1, WALKER_HDF_BLOCK_SIZE / wlk_sz), nWtot);
  int nblks          = (nWtot - 1) / nwlk_per_block + 1;
  std::vector<int> wlk_per_blk;

  nda::array<ComplexType, 2> RecvBuff;
  nda::array<int, 1> counts, displ;

  std::unique_ptr<h5::group> wgrp = nullptr;

  if (mpi->comm.root())
  {
    h5::group grp(fh5); 

    counts.resize(mpi->comm.size());
    displ.resize(mpi->comm.size());
    wlk_per_blk.reserve(nblks);

    [[maybe_unused]] long NMO = 0, nup = 0, ndn = 0;
    { // to limit the scope
      auto w = wset[0];
      if(!ft){
        auto SM = w.SlaterMatrix(Alpha);
        NMO = SM.extent(0); 
        nup = SM.extent(1); 
        if (walker_type == COLLINEAR)
          ndn = w.SlaterMatrix(Beta).extent(1);
        if (walker_type == NONCOLLINEAR)
          NMO /= 2;
      } else {
        auto UR = w.UMatrix(Alpha);
        NMO = UR.extent(0); 
        if (walker_type == NONCOLLINEAR)
          NMO /= 2;
      }
    }

    std::vector<int> Idata(7);
    Idata[0] = nWtot;
    Idata[1] = nblks;
    Idata[2] = wlk_nterms;
    Idata[3] = wlk_sz;
    Idata[4] = NMO;
    Idata[5] = nup;
    Idata[6] = ndn;

    h5::group sgrp = (grp.has_subgroup("Walkers") ?
            grp.open_group("Walkers")    :
            grp.create_group("Walkers", true));
    wgrp = std::make_unique<h5::group>(sgrp.create_group("WalkerSet", true)); 
    h5::h5_write(*wgrp, "dims", Idata);
  }

  int nsent = 0;
  // ready to send walkers to head in blocks
  for (int i = 0, ndone = 0; i < nblks; i++, ndone += nwlk_per_block)
  {
    nda::array<ComplexType, 2> SendBuff;
    int nwlk_tot   = std::min(nwlk_per_block, nWtot - ndone);
    int nw_to_send = 0;
    if (w0 + nsent >= ndone && w0 + nsent < ndone + nwlk_tot)
      nw_to_send = std::min(nW - nsent, (ndone + nwlk_tot) - (w0 + nsent));

    if (mpi->comm.root())
    {
      for (int p = 0, nt = 0; p < mpi->comm.size(); p++)
      {
        int n_ = 0;
        if (ndone + nwlk_tot > nt && ndone < nt + nW)
        {
          if (ndone <= nt)
            n_ = std::min(nW, (ndone + nwlk_tot) - nt);
          else
            n_ = std::min(nt + nW - ndone, nwlk_tot);
        }

        counts[p] = n_ * wlk_nterms;
        nt += nw_per_rank[p];
      }
      displ[0] = 0;
      for (int p = 1, nt = 0; p < mpi->comm.size(); p++)
      {
        nt += counts[p - 1];
        displ[p] = nt;
      }

      RecvBuff.resize(nwlk_tot, wlk_nterms);
    }

    if (nw_to_send > 0)
    {
      SendBuff.resize(nw_to_send, wlk_nterms);
      for (int p = 0; p < nw_to_send; p++)
      {
        wset.copyToIO(SendBuff(p,all), nsent + p);
      }
    }

    mpi->comm.gatherv_n(SendBuff.data(), SendBuff.size(), RecvBuff.data(), counts.data(),
                            displ.data(), 0);
    nsent += nw_to_send;

    if (mpi->comm.root())
    {
      nda::h5_write(*wgrp,std::string("walkers_") + std::to_string(i),RecvBuff,false);
      wlk_per_blk.push_back(nwlk_tot);
    }

    // not sure if necessary, but avoids avalanche of messages on head node
    mpi->comm.barrier();
  }

  if (mpi->comm.root())
    h5::h5_write(*wgrp, "wlk_per_blk", wlk_per_blk);

  mpi->comm.barrier();
  return true;
}

} // namespace afqmc

} // namespace sfqmc

