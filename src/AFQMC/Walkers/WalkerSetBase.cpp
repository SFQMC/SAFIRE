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

#include <cassert>
#include <cstdlib>

#include "AFQMC/Walkers/WalkerSetBase.h"
#include "IO/banner.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::parse(const WalkerSetParameters& params)
{
  app_log(1, section(std::format("Initializing Walker Set \"{}\"", params.name)));
  // walkerType is set from params in the constructor's member-init list, before setup()
  // needs it, so it is not assigned here.

  if constexpr (MEM == HOST_MEMORY)
    app_log(1, "Walker resides in CPU memory");
  else
    app_log(1, "Walker resides in GPU memory");

  app_log(1, "");
}

template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::setup(std::array<int, 3> dims)
{
  utils::check(walkerType != UNDEFINED_WALKER_TYPE,
               " Error: Undefined walker_type on WalkerSetBase::setup ");
  utils::check(!finite_temperature ||
               (walkerType == COLLINEAR || walkerType == NONCOLLINEAR),
               " Error: finite temperature requires a collinear or noncollinear walker type ");

  // wlk_descriptor: {nmo, naea, naeb, nback_prop, nCV, nRefs, nHist}
  // dims = {rows, naea, naeb}: rows already carries the 2*NMO factor for
  // noncollinear, naeb is 0 unless COLLINEAR. These are the resolved Slater
  // matrix dimensions (inferred from the initial guess or the HDF5 dims).
  wlk_desc = {dims[0], dims[1], dims[2], 0, 0, 0, 0};
  int nrow = dims[0];
  int ncol = dims[1] + dims[2];

  //   T = 0
  //   - SlaterMatrix:         NCOL*NROW
  //   T > 0
  //   - UMatrix:         NCOL*NROW
  //   - DMatrix:         NCOL
  //   - VMatrix:         NCOL*NROW
  
  //   - weight:               1
  //   - phase:                1
  //   - pseudo energy:        1
  //   - E1:                   1
  //   - EXX:                  1
  //   - EJ:                   1
  //   - overlap:              1
  //   - SlaterMatrixN:        Same size as Slater Matrix
  //   - SlaterMatrixAux:        Same size as Slater Matrix
  //   (T=0) Total: 10+2*NROW*NCOL+BP_SIZE+2*NBACK_PROP
  int cnt        = 0;
  if(!finite_temperature)
  {    
    data_displ[SM] = cnt;
    cnt += nrow * ncol;
  }
  else //finite_temperature 
  {
    data_displ[UR] = cnt;
    cnt += nrow * ncol;
    data_displ[DR] = cnt;
    cnt += ncol;
    data_displ[VR] = cnt;
    cnt += nrow * ncol;
  }
  data_displ[WEIGHT] = cnt;
  cnt += 1; // weight
  data_displ[PHASE] = cnt;
  cnt += 1; // phase
  data_displ[PSEUDO_ELOC_] = cnt;
  cnt += 1; // pseudo energy
  data_displ[E1_] = cnt;
  cnt += 1; // E1
  data_displ[EXX_] = cnt;
  cnt += 1; // EXX
  data_displ[EJ_] = cnt;
  cnt += 1; // EJ
  data_displ[OVLP] = cnt;
  cnt += 1; // overlap
  data_displ[LOGSCL_UP] = cnt;
  cnt += 1; // scale factor for D_up
  data_displ[LOGSCL_DN] = cnt;
  cnt += 1; // scale factor for D_dn
  data_displ[IS_UNITARY] = cnt;
  cnt += 1; // flag to track if UR is unitary
  data_displ[THETA] = cnt;
  cnt += 1; // theta
  walker_size                = cnt;
  walker_memory_usage        = walker_size * sizeof(ComplexType);
  data_displ[SMN]            = -1;
  data_displ[FIELDS]         = -1;
  data_displ[WEIGHT_FAC]     = -1;
  data_displ[WEIGHT_HISTORY] = -1;
  bp_walker_size             = 0;
  bp_walker_memory_usage     = bp_walker_size * sizeof(ComplexType);

  tot_num_walkers = 0;
}

/*
 * Increases the capacity of the containers to n.
 */
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::reserve(int n)
{
  if (walker_buffer.extent(0) < n  || walker_buffer.extent(1) != walker_size) 
  {
    // only preserve data if walker_buffer.size(1) == walker_size	
    if(walker_buffer.extent(1) == walker_size and walker_buffer.extent(0) > 0) {
      memory::array<MEM, ComplexType, 2> tmp(walker_buffer);
      walker_buffer.resize(n, walker_size);
      walker_buffer() = ComplexType(0.0); 
      walker_buffer(nda::range(tmp.extent(0)),nda::range::all) = tmp();
    } else {
      walker_buffer.resize(n, walker_size);
      walker_buffer() = ComplexType(0.0);
    }
  }
  if (bp_buffer.extent(0) < n || bp_buffer.extent(1) != bp_walker_size)
  {
    if(bp_walker_size > 0 and bp_buffer.extent(1) == bp_walker_size and bp_buffer.extent(0) > 0) {
      memory::array<MEM, ComplexType, 2> tmp(bp_buffer);
      bp_buffer.resize(n,bp_walker_size);
      bp_buffer() = ComplexType(0.0);
      bp_buffer(nda::range(tmp.extent(0)),nda::range::all) = tmp();
    } else {
      bp_buffer.resize(n,bp_walker_size);
      bp_buffer() = ComplexType(0.0);
    }
  }
}

/*
 * Adds/removes the number of walkers in the set to match the requested value.
 * Walkers are removed from the end of the set 
 *     and buffer capacity remains unchanged in this case.
 * New walkers are initialized from already existing walkers in a round-robin fashion. 
 * If the set is empty, calling this routine will abort. 
 * Capacity is increased if necessary.
 * Target Populations are set to n.
 */
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::resize(int n)
{
  auto all = nda::range::all;
  utils::check(tot_num_walkers>0, "WalkerSetBase::resize: empty set.");
  utils::check(walker_buffer.extent(1) == walker_size, "Shape mismatch: walker_size: {}",walker_size);

  reserve(n);
  if (n > tot_num_walkers)
  {
    auto pos = tot_num_walkers;
    auto i0  = 0;
    while (pos < n)
    {
      walker_buffer(pos++,all) = walker_buffer(i0,all);
      i0                       = (i0 + 1) % tot_num_walkers;
    }
  }
  tot_num_walkers  = n;
  targetN_per_rank = tot_num_walkers;
  targetN          = GlobalPopulation();
  utils::check(targetN == targetN_per_rank * mpi->comm.size(), 
           " Error in total walker population: targetN, targetN_per_rank, # of ranks: {}, {}, {}",
           targetN,targetN_per_rank,mpi->comm.size());
}

/*
 * Reserves capacity for n walkers and initializes all n of them to valid
 * default values (unit weight/overlap/phase, zero Slater matrices). This
 * always leaves the set fully populated with n walkers.
*/
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::allocate_walkers(int n)
{
  auto all = nda::range::all;
  reserve(n);
  tot_num_walkers = n;
  auto r = nda::range(0, n);
  walker_buffer(r, all) = ComplexType(0.0);
  walker_buffer(r, data_displ[WEIGHT]) = ComplexType(1.0);
  walker_buffer(r, data_displ[PHASE])  = ComplexType(1.0);
  walker_buffer(r, data_displ[THETA])  = ComplexType(0.0);
  if (finite_temperature)
  {
    // finite-T keeps log(ovlp) instead of ovlp
    walker_buffer(r, data_displ[LOGSCL_UP])  = ComplexType(0.0);
    walker_buffer(r, data_displ[LOGSCL_DN])  = ComplexType(0.0);
    walker_buffer(r, data_displ[IS_UNITARY]) = ComplexType(1.0);
  }
  else
  {
    walker_buffer(r, data_displ[OVLP]) = ComplexType(1.0);
  }
  targetN_per_rank = tot_num_walkers;
  targetN          = GlobalPopulation();
  utils::check(targetN == targetN_per_rank * mpi->comm.size(),
           " Error in total walker population: targetN, targetN_per_rank, # of ranks: {}, {}, {}",
           targetN,targetN_per_rank,mpi->comm.size());
}

/*
 * Copies the per-spin initial guess matrices into every walker's Slater matrix.
 * Each matrix is already exactly (rows x naea) / (NMO x naeb), so no truncation
 * is needed.
*/
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::populate_from_guess(WalkerSetInitialGuess::slater_guess guess)
{
  auto all = nda::range::all;
  utils::check((walkerType == COLLINEAR) == (guess.size() == 2),
               "WalkerSetBase::populate_from_guess: guess spin count does not match walker_type.");
  for (int i = 0; i < tot_num_walkers; i++)
  {
    reference w0(walker_buffer(i, all), data_displ, wlk_desc);
    w0.SlaterMatrix(Alpha) = guess[0];
    if (guess.size() > 1)
      w0.SlaterMatrix(Beta) = guess[1];
  }
}

/*
 * Copies the rank-4 finite-temperature guess {3, nspin, rows, naea} into every
 * walker's U/D/V matrices. The D slab is a full matrix; only its diagonal is
 * used. The set must already be sized and default-initialized (weights, phases,
 * LOGSCL_*, IS_UNITARY) by allocate_walkers, so this only fills U/D/V.
*/
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::populate_from_guess_ft(WalkerSetInitialGuess::udv_guess UDV)
{
  auto all = nda::range::all;
  int nspin = (walkerType == COLLINEAR ? 2 : 1);
  utils::check(UDV.shape() == std::array<long,4>{3,nspin,wlk_desc[0],wlk_desc[1]}, "Size mismatch.");
  for (int i = 0; i < tot_num_walkers; i++)
  {
    reference w0(walker_buffer(i,all), data_displ, wlk_desc);
    w0.UMatrix(Alpha) = UDV(0,0,nda::ellipsis{});
    w0.DMatrix(Alpha) = nda::diagonal(UDV(1,0,nda::ellipsis{}));
    w0.VMatrix(Alpha) = UDV(2,0,nda::ellipsis{});
    if (walkerType == COLLINEAR)
    {
      w0.UMatrix(Beta) = UDV(0,1,all,nda::range(wlk_desc[2]));
      w0.DMatrix(Beta) = nda::diagonal(UDV(1,1,all,nda::range(wlk_desc[2])));
      w0.VMatrix(Beta) = UDV(2,1,all,nda::range(wlk_desc[2]));
    }
  }
}

// for finite-T : resets walker set for start of each sweep
// UR, DR, VR --> Identity matrices, log scales --> 0
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::reset(int n)
{ 
  auto all = nda::range::all;

  reserve(n);

  // FT walker U/V blocks are square with leading dimension wlk_desc[0] (= npol*NMO),
  // and D is stored as its diagonal of the same length.
  int dim = wlk_desc[0];
  memory::array<MEM, ComplexType, 2> IMat(dim, dim);
  memory::array<MEM, ComplexType, 1> IVec(dim, ComplexType(1.0));

  math::set_identity(IMat);

  auto pos = 0;
  // careful here!!!
  while (pos < n)
  { 
    walker_buffer(pos,all) = ComplexType(0.0);
    reference w0(walker_buffer(pos,all), data_displ, wlk_desc);
    w0.UMatrix(Alpha) = IMat();
    w0.DMatrix(Alpha) = IVec();
    w0.VMatrix(Alpha) = IMat();
    if (walkerType == COLLINEAR)
    {
      w0.UMatrix(Beta) = IMat();
      w0.DMatrix(Beta) = IVec();
      w0.VMatrix(Beta) = IMat();
    }
    pos++;
  }
  auto r = nda::range(tot_num_walkers,n); 
  walker_buffer(r,data_displ[WEIGHT]) = ComplexType(1.0);
  walker_buffer(r,data_displ[OVLP]) = ComplexType(0.0); //finite-T keeps log(ovlp), instead of ovlp 
  walker_buffer(r,data_displ[PHASE]) = ComplexType(1.0);
  walker_buffer(r,data_displ[LOGSCL_UP]) = ComplexType(0.0);
  walker_buffer(r,data_displ[LOGSCL_DN]) = ComplexType(0.0);
  walker_buffer(r,data_displ[IS_UNITARY]) = ComplexType(1.0);
  walker_buffer(r,data_displ[THETA]) = ComplexType(0.0);

  tot_num_walkers = n;
  targetN_per_rank  = tot_num_walkers;
  targetN         = GlobalPopulation();
  utils::check(targetN == targetN_per_rank * mpi->comm.size(), 
           " Error in total walker population: targetN, targetN_per_rank, # of ranks: {}, {}, {}",
           targetN,targetN_per_rank,mpi->comm.size());
}

template<MEMORY_SPACE MEM>
bool WalkerSetBase<MEM>::clean()
{
  walker_buffer.resize(0, walker_size);
  bp_buffer.resize(0, bp_walker_size);
  tot_num_walkers = targetN = targetN_per_rank = 0;
  return true;
}

/*
* Resizes back propagation buffers
* Must be called before any call to bp-related routines.
*/
template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::resize_bp(int nbp, int nCV, int nref)
{
  utils::check(walker_buffer.extent(0) == bp_buffer.extent(0), "Size mismatch.");
  utils::check(bp_buffer.extent(1) == bp_walker_size, "Size mismatch.");
  utils::check(walker_buffer.extent(1) == walker_size, "Size mismatch.");
  // wlk_descriptor: {nmo, naea, naeb, nback_prop, nCV, nRefs, nHist}
  wlk_desc[3] = nbp;
  wlk_desc[4] = nCV;
  wlk_desc[5] = nref;
  wlk_desc[6] = 3 * nbp;
  // the FIELDS ring is indexed by history_pos % nbp, which only advances consistently
  // across the outer wrap if the field ring divides the history ring
  utils::check(nbp == 0 or wlk_desc[6] % nbp == 0,
               "resize_bp: field ring ({}) must divide history ring ({}).", nbp, wlk_desc[6]);
  history_pos = 0;
  // For all T=0 walker types (the only ones supported here), the Slater matrix
  // dimensions are recovered directly from wlk_desc: nrow = wlk_desc[0] (already
  // carries the 2*NMO factor for NONCOLLINEAR), ncol = wlk_desc[1] + wlk_desc[2]
  // (wlk_desc[2] is 0 for the non-COLLINEAR cases).
  int nrow, ncol;
  if (walkerType == CLOSED or walkerType == COLLINEAR or walkerType == NONCOLLINEAR)
  {
    nrow = wlk_desc[0];
    ncol = wlk_desc[1] + wlk_desc[2];
  }
  else
  {
    app_error(" Error: Incorrect walker_type on WalkerSetBase::resize_bp ");
    APP_ABORT("");
  }
  // store nbpx3 history of weights and factors in circular buffer
  int cnt            = 0;
  data_displ[FIELDS] = cnt;
  cnt += nbp * nCV;
  data_displ[WEIGHT_FAC] = cnt;
  cnt += wlk_desc[6];
  data_displ[WEIGHT_HISTORY] = cnt;
  cnt += wlk_desc[6];
  bp_walker_size = cnt;
  if (bp_buffer.extent(1) != bp_walker_size)
  {
    bp_buffer.resize(walker_buffer.extent(0), bp_walker_size);
    bp_buffer() = ComplexType(0.0);
    bp_buffer(nda::range::all, nda::range(data_displ[WEIGHT_FAC],data_displ[WEIGHT_FAC]+wlk_desc[6])) = ComplexType(1.0);
  }
  if (nbp > 0 && data_displ[SMN] < 0)
  {
    auto sz(walker_size);
    data_displ[SMN] = walker_size;
    walker_size += nrow * ncol;
    memory::array<MEM,ComplexType,2> wb(walker_buffer.extent(0),walker_size);
    wb(nda::range::all,nda::range(0,sz)) = walker_buffer();
    walker_buffer = std::move(wb);
  }
}  

template<MEMORY_SPACE MEM>
void WalkerSetBase<MEM>::benchmark(std::string& blist, int maxnW, int delnW, int repeat)
{
  if (blist.find("comm") != std::string::npos)
  {
    app_log(1," Testing communication times in WalkerHandler. ");
    app_log(1," This should be done using a single TG per node, ");
    app_log(1," to avoid timing communication between cores on the same node. ");
    std::ofstream out;
    if (mpi->comm.rank() == 0)
      out.open("benchmark.icomm.dat");

    std::vector<std::string> tags(3);
    tags[0] = "M1";
    tags[1] = "M2";
    tags[2] = "M3";


    int nw = 1;
    while (nw <= maxnW)
    {
      if (mpi->comm.rank() == 0 || mpi->comm.rank() == 1)
      {
        int sz = nw * walker_size;
        std::vector<ComplexType> Cbuff(sz);
        MPI_Request req;
        MPI_Status st;
        mpi->comm.barrier();
        for (int i = 0; i < repeat; i++)
        {
          if (mpi->comm.rank() == 0)
          {
            MPI_Isend(Cbuff.data(), 2 * Cbuff.size(), MPI_DOUBLE, 1, 999, mpi->comm.get(), &req);
            MPI_Wait(&req, &st);
          }
          else
          {
            MPI_Irecv(Cbuff.data(), 2 * Cbuff.size(), MPI_DOUBLE, 0, 999, mpi->comm.get(), &req);
            MPI_Wait(&req, &st);
          }
        }

        if (mpi->comm.rank() == 0)
        {
          out << nw << " ";
          out << std::endl;
        }
      }
      else 
      {
        mpi->comm.barrier();
      }

      if (delnW <= 0)
        nw *= 2;
      else
        nw += delnW;
    }
  }
  else if (blist.find("comm") != std::string::npos)
  {
    std::ofstream out;
    if (mpi->comm.rank() == 0)
      out.open("benchmark.comm.dat");
  }
}

template class WalkerSetBase<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
template class WalkerSetBase<DEVICE_MEMORY>;
#endif

} // namespace afqmc

} // namespace sfqmc

