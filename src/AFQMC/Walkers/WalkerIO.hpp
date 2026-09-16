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
#include "AFQMC/parameters.hpp"

#include "nda/h5.hpp"

namespace sfqmc
{
namespace afqmc
{

// Reads a walker restart file and returns a fully constructed, populated walker
// set sized to this rank's share of the walkers in the file. fh5 opened on all
// ranks read-only. The set is built from the given parameters/rng with dimensions
// taken from the file, upholding the invariant that a walker set is always born
// fully populated (no empty intermediate state).
template<class WalkerSet, class MpiContext, class Rng>
WalkerSet readWalkersFromHDF5(std::shared_ptr<MpiContext> mpi,
                              std::shared_ptr<Rng> rng,
                              const WalkerSetParameters& params,
                              h5::file& fh5,
                              int nWalkers,
                              bool set_to_target)
{
  auto all = nda::range::all;
  // Finite-temperature restart is not supported: this routine always builds a
  // ground-state (finite_temperature = false) walker set from the file dims.

  h5::group grp(fh5);
  utils::check(grp.has_subgroup("Walkers"), " restartFromHDF5: Missing Walkers dataset.");
  h5::group sgrp = grp.open_group("Walkers");
  utils::check(sgrp.has_subgroup("WalkerSet"), " restartFromHDF5: Missing WalkerSet dataset.");
  h5::group wgrp = sgrp.open_group("WalkerSet");

  std::vector<int> Idata(7);
  h5::h5_read(wgrp, "dims", Idata);
  int nWtot      = Idata[0];
  int wlk_nterms = Idata[2];  // per-walker IO record length
  int NMO        = Idata[4];
  int nup        = Idata[5];
  int ndn        = Idata[6];

  std::array<int, 3> dims;  // {rows, naea, naeb} = wlk_desc[0..2]
  if (params.walker_type == NONCOLLINEAR)
    dims = {2 * NMO, nup + ndn, 0};
  else if (params.walker_type == COLLINEAR)
    dims = {NMO, nup, ndn};
  else
    dims = {NMO, nup, 0};

  // walker range belonging to this comm
  int nW0, nWN;  // [nW0, nWN) = global walker indices owned by this rank
  if (set_to_target)
  {
    utils::check(nWtot >= nWalkers * mpi->comm.size(),
                 " Error: Not enough walkers in restart file.");
    nW0 = nWalkers * mpi->comm.rank();
    nWN = nW0 + nWalkers;
  }
  else
  {
    utils::check(nWtot % mpi->comm.size() == 0,
                 " Error: Number of walkers in restart file must be divisible by number of task groups.");
    nW0 = (nWtot / mpi->comm.size()) * mpi->comm.rank();
    nWN = nW0 + nWtot / mpi->comm.size();
  }
  int nw_local = nWN - nW0;

  WalkerSet wset(mpi, rng, params, dims, nw_local, false);
  utils::check(wlk_nterms == wset.walkerSizeIO(),
               " Inconsistent walker restart file: IO size {} != walkerSizeIO {}.",
               wlk_nterms, wset.walkerSizeIO());

  std::vector<int> wlk_per_blk;
  h5::h5_read(wgrp, "wlk_per_blk", wlk_per_blk);

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

      nda::range r(w0, w0 + nw_);
      nda::h5_read(wgrp, "walkers_" + std::to_string(bi), Data, std::tuple{r, all});
      for (int n = 0; n < nw_; n++, nread++)
        wset.copyFromIO(Data(n, all), nread);
    }
    ni += wlk_per_blk[bi++];
  }
  mpi->comm.barrier();
  return wset;
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

