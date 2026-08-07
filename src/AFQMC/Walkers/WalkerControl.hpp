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

#include <tuple>
#include <cassert>
#include <memory>
#include <span>
#include <stack>
#include <utility>
#include <mpi.h>
#include "AFQMC/config.h"
#include "utilities/FairDivide.hpp"
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/Walkers/WalkerUtilities.hpp"

#include "mpi3/communicator.hpp"
#include "mpi3/request.hpp"

namespace sfqmc
{
namespace afqmc
{
/** swap Walkers with Recv/Send
 *
 * The algorithm ensures that the load per node can differ only by one walker.
 * The communication is one-dimensional.
 */
template<class WlkBucket, class IVec = std::vector<int>>
inline int swapWalkersSimple(WlkBucket& wset,
                             nda::MemoryArrayOfRank<2> auto&& Wexcess,
                             IVec const& CurrNumPerNode,
                             IVec const& NewNumPerNode,
                             mpi3::communicator& comm)
{
  int wlk_size = wset.single_walker_size() + wset.single_walker_bp_size();
  int NumContexts, MyContext;
  NumContexts = comm.size();
  MyContext   = comm.rank();
  if (wlk_size != Wexcess.extent(1))
    throw std::runtime_error("Array dimension error in swapWalkersSimple().");
  if (1 != Wexcess.strides()[1])
    throw std::runtime_error("Array shape error in swapWalkersSimple().");
  if (CurrNumPerNode.size() < NumContexts || NewNumPerNode.size() < NumContexts)
    throw std::runtime_error("Array dimension error in swapWalkersSimple().");
  if (wset.capacity() < NewNumPerNode[MyContext])
    throw std::runtime_error("Insufficient capacity in swapWalkersSimple().");
  std::vector<int> minus, plus;
  int deltaN = 0;
  for (int ip = 0; ip < NumContexts; ip++)
  {
    int dn = CurrNumPerNode[ip] - NewNumPerNode[ip];
    if (ip == MyContext)
      deltaN = dn;
    if (dn > 0)
    {
      plus.insert(plus.end(), dn, ip);
    }
    else if (dn < 0)
    {
      minus.insert(minus.end(), -dn, ip);
    }
  }
  int nswap = std::min(plus.size(), minus.size());
  int nsend = 0;
  if (deltaN <= 0 && wset.size() != CurrNumPerNode[MyContext])
    throw std::runtime_error("error in swapWalkersSimple().");
  if (deltaN > 0 && (wset.size() != NewNumPerNode[MyContext] || int(Wexcess.extent(0)) != deltaN))
    throw std::runtime_error("error in swapWalkersSimple().");
  std::vector<ComplexType> buff;
  if (deltaN < 0)
    buff.resize(wlk_size);
  for (int ic = 0; ic < nswap; ic++)
  {
    if (plus[ic] == MyContext)
    {
      comm.send_n(std::addressof(Wexcess(nsend,0)), Wexcess.extent(1), minus[ic], plus[ic] + 999);
      ++nsend;
    }
    if (minus[ic] == MyContext)
    {
      comm.receive_n(buff.data(), buff.size(), plus[ic], plus[ic] + 999);
      auto v = nda::array_view<ComplexType, 2>({1,wlk_size},buff.data());
      wset.push_walkers(v);
    }
  }
  return nswap;
}

/** swap Walkers with Irecv/Send
 *
 * The algorithm ensures that the load per node can differ only by one walker.
 * The communication is one-dimensional.
 */
template<class WlkBucket, class IVec = std::vector<int>>
// eventually generalize MPI_Comm to a MPI wrapper
inline int swapWalkersAsync(WlkBucket& wset,
                            nda::MemoryArrayOfRank<2> auto&& Wexcess,
                            IVec const& CurrNumPerNode,
                            IVec const& NewNumPerNode,
                            mpi3::communicator& comm)
{
  int wlk_size = wset.single_walker_size() + wset.single_walker_bp_size();
  int NumContexts, MyContext;
  NumContexts = comm.size();
  MyContext   = comm.rank();
  if (wlk_size != Wexcess.extent(1))
    throw std::runtime_error("Array dimension error in swapWalkersAsync().");
  if (1 != Wexcess.strides()[1] || (Wexcess.extent(0) > 0 && 
      Wexcess.extent(1) != Wexcess.strides()[0]))
    throw std::runtime_error("Array shape error in swapWalkersAsync().");
  if (CurrNumPerNode.size() < NumContexts || NewNumPerNode.size() < NumContexts)
    throw std::runtime_error("Array dimension error in swapWalkersAsync().");
  if (wset.capacity() < NewNumPerNode[MyContext])
    throw std::runtime_error("Insufficient capacity in swapWalkersAsync().");
  std::vector<int> minus, plus;
  int deltaN = 0;
  for (int ip = 0; ip < NumContexts; ip++)
  {
    int dn = CurrNumPerNode[ip] - NewNumPerNode[ip];
    if (ip == MyContext)
      deltaN = dn;
    if (dn > 0)
    {
      plus.insert(plus.end(), dn, ip);
    }
    else if (dn < 0)
    {
      minus.insert(minus.end(), -dn, ip);
    }
  }
  int nswap     = std::min(plus.size(), minus.size());
  int nsend     = 0;
  int countSend = 1;
  if (deltaN <= 0 && wset.size() != CurrNumPerNode[MyContext])
    throw std::runtime_error("error(1) in swapWalkersAsync().");
  if (deltaN > 0 && (wset.size() != NewNumPerNode[MyContext] || int(Wexcess.extent(0)) != deltaN))
    throw std::runtime_error("error(2) in swapWalkersAsync().");
  std::vector<ComplexType*> buffers;
  std::vector<boost::mpi3::request> requests;
  std::vector<int> recvCounts;
  for (int ic = 0; ic < nswap; ic++)
  {
    if (plus[ic] == MyContext)
    {
      if ((ic < nswap - 1) && (plus[ic] == plus[ic + 1]) && (minus[ic] == minus[ic + 1]))
      {
        countSend++;
      }
      else
      {
        requests.emplace_back(comm.isend(std::addressof(Wexcess(nsend,0)), 
                                         std::addressof(Wexcess(nsend,0)) + countSend * Wexcess.extent(1),
                                         minus[ic], plus[ic] + 1999));
        nsend += countSend;
        countSend = 1;
      }
    }
    if (minus[ic] == MyContext)
    {
      if ((ic < nswap - 1) && (plus[ic] == plus[ic + 1]) && (minus[ic] == minus[ic + 1]))
      {
        countSend++;
      }
      else
      {
        ComplexType* bf = new ComplexType[countSend * wlk_size];
        buffers.push_back(bf);
        recvCounts.push_back(countSend);
        requests.emplace_back(comm.ireceive_n(bf, countSend * wlk_size, plus[ic], plus[ic] + 1999));
        countSend = 1;
      }
    }
  }
  if (deltaN < 0)
  {
    // receiving nodes
    for (int ip = 0; ip < requests.size(); ++ip)
    {
      requests[ip].wait();
      auto v = nda::array_view<ComplexType, 2>({recvCounts[ip],wlk_size},buffers[ip]);
      wset.push_walkers(v);
      delete[] buffers[ip];
    }
  }
  else
  {
    // sending nodes
    for (int ip = 0; ip < requests.size(); ++ip)
      requests[ip].wait();
  }
  return nswap;
}


/**
 * Implements Cafarrel's minimum branching algorithm.
 *   - buff: array of walker info (weight,num).
 */
inline void min_branch([[maybe_unused]] std::vector<std::pair<double, int>>& buff, 
                       [[maybe_unused]] utils::HostRandomGenerator& rng,
                       [[maybe_unused]] double max_c,
                       [[maybe_unused]] double min_c)
{
  APP_ABORT(" Error: min_branch not implemented yet. \n\n");
}

/**
 * Implements Cafarrel's minimum branching algorithm.
 *   - buff: array of walker info (weight,num).
 */
inline void serial_comb(std::vector<std::pair<double, int>>& buff, utils::HostRandomGenerator& rng)
{
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  int nW = buff.size();
  double norm = 0.0; 
  // since weights are reset to 1 at the end, using buff for sampling
  for( auto& v: buff ) norm += std::get<0>(v);    
  for( auto& v: buff ) v = std::pair<double, int>{std::get<0>(v)/norm,0};    

  // comb
  int idx=0;
  double s0(std::get<0>(buff[idx])),s1(0.0);
  for(int i=0; i<nW; ++i) {
    s1 = (i+distribution(rng.std_rng))/double(nW);
    while( s1 > s0 ) {
      idx++;
      s0 += std::get<0>(buff[idx]);
    }
    std::get<1>(buff[idx])++;  
  } 

  // now set all the weights to 1
  for( auto& v: buff ) std::get<0>(v) = 1.0; 
}

/**
 * Implements the paired branching algorithm on a population of walkers,
 * given a list of walker weights. For each walker in the list, returns the weight
 * and number of times the walker should appear in the new list.
 *   - buff: array of walker info (weight,num).
 */
inline void pair_branch(std::vector<std::pair<double, int>>& buff, utils::HostRandomGenerator& rng, double max_c, double min_c)
{
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  typedef std::tuple<double, int, int> tp;
  typedef std::vector<tp>::iterator tp_it;
  // slow for now, not efficient!!!
  int nw = buff.size();
  std::vector<tp> wlks(nw);
  for (int i = 0; i < nw; i++)
    wlks[i] = tp{buff[i].first, 1, i};

  std::sort(wlks.begin(), wlks.end(), [](const tp& a, const tp& b) { return std::get<0>(a) < std::get<0>(b); });

  tp_it it_s = wlks.begin();
  tp_it it_l = wlks.end() - 1;

  while (it_s < it_l)
  {
    if (std::abs(std::get<0>(*it_s)) < min_c || std::abs(std::get<0>(*it_l)) > max_c)
    {
      double w12 = std::get<0>(*it_s) + std::get<0>(*it_l);
      if (distribution(rng.std_rng) < std::get<0>(*it_l) / w12)
      {
        std::get<0>(*it_l) = 0.5 * w12;
        std::get<0>(*it_s) = 0.0;
        std::get<1>(*it_l) = 2;
        std::get<1>(*it_s) = 0;
      }
      else
      {
        std::get<0>(*it_s) = 0.5 * w12;
        std::get<0>(*it_l) = 0.0;
        std::get<1>(*it_s) = 2;
        std::get<1>(*it_l) = 0;
      }
      it_s++;
      it_l--;
    }
    else
      break;
  }

  int nnew  = 0;
  int nzero = 0;
  for (auto& w : wlks)
  {
    buff[std::get<2>(w)] = {std::get<0>(w), std::get<1>(w)};
    nnew += std::get<1>(w);
    if (std::get<1>(w) > 0 && std::abs(std::get<0>(w)) < 1e-7)
      nzero++;
  }
  if (nzero > 0)
  {
    utils::check(false, "Found {} walkers with zero weight after branch. Try reducing subSteps or reducing the time step.", nzero);
  }
  if (nw != nnew)
    APP_ABORT("Error: Problems with pair_branching.");
}

/**
 * Implements the paired branching algorithm on a population of walkers,
 * given a list of walker weights. For each walker in the list, returns the branching
 * count {0, 1, 2} and the index of the walker paired to the current walker, if branching count is 2. 
 *   - buff: array of walker weights. 
 * Note: For replicated walkers, the weight of the new walkers will be:
 *    - For a walker in position I with bdata[I][:] = {2, Is}, wnew = 0.5*(weight[I] + weight[Is])  
 */
inline void pair_branch_for_correlated(std::vector<double> const& buff, nda::array<int,2>& bdata, utils::HostRandomGenerator& rng, double max_c, double min_c)
{ 
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  typedef std::tuple<double, int> tp;
  typedef std::vector<tp>::iterator tp_it;
  // slow for now, not efficient!!!
  int nw = buff.size();
  std::vector<tp> wlks(nw);
  for (int i = 0; i < nw; i++)
    wlks[i] = tp{buff[i], i};
  
  std::sort(wlks.begin(), wlks.end(), [](const tp& a, const tp& b) { return std::get<0>(a) < std::get<0>(b); });
  
  tp_it it_s = wlks.begin();
  tp_it it_l = wlks.end() - 1;
  
  for( int i=0; i<bdata.extent(0); ++i ) {
    bdata(i,0) = 1; // by default do nothing
    bdata(i,1) = -1; // coupled to nothing
  }
  while (it_s < it_l)
  { 
    if (std::abs(std::get<0>(*it_s)) < min_c || std::abs(std::get<0>(*it_l)) > max_c)
    { 
      int i_l = std::get<1>(*it_l);
      int i_s = std::get<1>(*it_s);
      double w12 = std::get<0>(*it_s) + std::get<0>(*it_l);
      if (distribution(rng.std_rng) < std::get<0>(*it_l) / w12)
      { 
        bdata(i_l,0) = 2;   // replicate large
        bdata(i_l,1) = i_s; // coupled to small
        bdata(i_s,0) = 0;   // kill small
      }
      else
      { 
        bdata(i_s,0) = 2;   // replicate small
        bdata(i_s,1) = i_l; // coupled to large
        bdata(i_l,0) = 0;   // kill large
      }
      it_s++;
      it_l--;
    }
    else
      break;
  }
  
  // some checks
  int nnew  = 0;
  for (int i=0; i<nw; i++)
    nnew += bdata(i,0);
  if (nw != nnew)
    APP_ABORT("Error: Problems with pair_branching_for_correlated.");
}

/**
 * Implements the serial branching algorithm on the set of walkers.
 * Serial branch involves gathering the list of weights on the root node
 * and making the decisions locally. The new list of walker weights is then bcasted.
 * This implementation requires contiguous walkers and fixed population walker sets.
 */
template<class WalkerSet,
         typename = typename std::enable_if<(WalkerSet::contiguous_walker)>::type,
         typename = typename std::enable_if<(WalkerSet::fixed_population)>::type>
inline void SerialBranching(WalkerSet& wset,
                            BranchingAlgorithm type,
                            double min_,
                            double max_,
                            std::vector<int>& wlk_counts,
                            nda::MemoryArrayOfRank<2> auto& Wexcess,
                            utils::HostRandomGenerator& rng,
                            mpi3::communicator& comm)
{
  using nda::range;
  std::vector<std::pair<double, int>> buffer(wset.get_global_target_population());

  // assemble list of weights
  getGlobalListOfWalkerWeights(wset, buffer, comm);

  // using global weight list, use pair branching algorithm
  if (comm.root())
  {
    if (type == BranchingAlgorithm::pair)
      pair_branch(buffer, rng, max_, min_);
    else if (type == BranchingAlgorithm::min_branch)
      min_branch(buffer, rng, max_, min_);
    else if (type == BranchingAlgorithm::serial_comb)
      serial_comb(buffer, rng);
    else
      APP_ABORT("Error: Unknown branching type in SerialBranching. ");
  }

  // bcast walker information and calculate new walker counts locally
  comm.broadcast_n(buffer.data(),buffer.size());

  int target = wset.get_target_population();
  wlk_counts.resize(comm.size());
  for (int i = 0, p = 0; i < comm.size(); i++)
  {
    int cnt = 0;
    for (int k = 0; k < target; k++, p++)
      cnt += buffer[p].second;
    wlk_counts[i] = cnt;
  }
  if (wset.get_global_target_population() != std::accumulate(wlk_counts.begin(), wlk_counts.end(), 0))
  {
    app_error(" Error: targetN != nwold: {}, {} ",target,
                  std::accumulate(wlk_counts.begin(), wlk_counts.end(), 0));
    APP_ABORT(" Error: targetN != nwold.");
  }

  // reserve space for extra walkers
  if (wlk_counts[comm.rank()] > target)
    Wexcess.resize(std::max(0, wlk_counts[comm.rank()] - target), 
		   wset.single_walker_size() + wset.single_walker_bp_size());

  // perform local branching
  // walkers beyond target go in Wexcess
  wset.branch(std::span(buffer).subspan(target * comm.rank(), target), Wexcess);
}

/**
 * Implements the serial branching algorithm for a group of correlated walker sets. 
 * The list of weights is provided as an argument.
 * The "collective" weight is generated and branching decisions are made based on it.
 * New weights and branching counts, for each system, are calculated and updated in the buffer.
 * comm is a communicator with the roots of all Global communicators. 
 */
inline void correlatedSerialBranching(BranchingAlgorithm branch_type,
			    std::string combine_type,		
                            double min_,
                            double max_,
			    nda::array<std::pair<double, int>,2>& buffer,
                            utils::HostRandomGenerator& rng,
                            mpi3::communicator& comm)
{
  // generate collective weights
  int n_sys = buffer.extent(0);
  int nW = buffer.extent(1);
  std::vector<double> collW(nW,0.0); 
  for(int s=0; s<n_sys; s++) {
    if(combine_type == "max" or combine_type == "mod-max") {
      for (int i = 0; i < nW; ++i)
        collW[i] = std::max(std::abs(buffer(s,i).first), collW[i]);
    } else if(combine_type == "mean" or combine_type == "mod-mean") {
      for (int i = 0; i < nW; ++i)
        collW[i] += std::abs(buffer(s,i).first)/double(n_sys);
    } else {
      APP_ABORT("Error: Unknown combine_type in correlatedSerialBranch.");
    }
  }
  if(combine_type == "max" or combine_type == "mod-max") 
    comm.reduce_in_place_n(collW.data(), nW, boost::mpi3::max<>());
  else if(combine_type == "mean" or combine_type == "mod-mean") 
    comm.reduce_in_place_n(collW.data(), nW, std::plus<>());

  // make branching decision based on collective weights
  nda::array<int,2> branch_data(nW,2);
  if (comm.root())
  {
    if(combine_type == "mean" or combine_type == "mod-mean")
      for(int i=0; i<nW; i++)
        collW[i] = collW[i]/double(comm.size());
    if (branch_type == BranchingAlgorithm::pair) {
      pair_branch_for_correlated(collW, branch_data, rng, max_, min_);
    } else {
      APP_ABORT("Error: Unknown branching type in correlatedSerialBranching. ");
    }
  }   
  comm.broadcast_n(branch_data.data(),branch_data.size());

  // modify original buffer, with new weights and branching counts for each walker
  // depends on branch_type
  if(branch_type == BranchingAlgorithm::pair) {     
    for(int i=0; i<nW; i++) {
      if(branch_data(i,0)==2) { // branch 
	int I_coupled = branch_data(i,1);   	
        for(int s=0; s<n_sys; s++) {
	  // new weight 
          if(combine_type == "mod-mean" or combine_type == "mod-max") {
	    double wnew = 0.5*(buffer(s,i).first + buffer(s,I_coupled).first);
	    buffer(s,i) = std::make_pair(wnew,2); 
	  } else if(combine_type == "mean" or combine_type == "max") {
            double wx = 0.5 * (collW[i] + collW[I_coupled]) / collW[i];
            buffer(s,i).first *= wx; 
            buffer(s,i).second = 2;
          }
	}
      } else if(branch_data(i,0)!=0 and branch_data(i,0)!=1) {
	APP_ABORT("Error in correlatedSerialBranching: Unknown branching count.");
      }
    }
    // now reset kill weights/counts
    for(int i=0; i<nW; i++) {
      if(branch_data(i,0)==0) { // kill 
        for(int s=0; s<n_sys; s++)
          buffer(s,i) = std::make_pair(0.0,0);
      } else if(branch_data(i,0)==1) { // set cnt to 1, leave weight unchanged 
        for(int s=0; s<n_sys; s++)
          buffer(s,i).second = 1;
      }
    }       
  } else {
    APP_ABORT("Error: Unknown branching type in correlatedSerialBranching. ");
  } 
}

/**
 * Implements the distributed comb branching algorithm.
 */
template<class WalkerSet,
         typename = typename std::enable_if<(WalkerSet::contiguous_walker)>::type,
         typename = typename std::enable_if<(WalkerSet::fixed_population)>::type>
inline void CombBranching([[maybe_unused]] WalkerSet& wset,
                          [[maybe_unused]] BranchingAlgorithm type,
                          [[maybe_unused]] std::vector<int>& wlk_counts,
                          [[maybe_unused]] nda::MemoryArrayOfRank<2> auto& Wexcess,
                          [[maybe_unused]] utils::HostRandomGenerator& rng,
                          [[maybe_unused]] mpi3::communicator& comm)
{
  APP_ABORT("Error: comb not implemented yet. ");
}

/* 
 * Population Control algorithm
 * curData:
 *  0: factor used to rescale the weights
 *  1: sum_i w_i * Eloc_i   (where w_i is the unnormalized weight)
 *  2: sum_i w_i            (where w_i is the unnormalized weight)
 *  3: sum_i abs(w_i)       (where w_i is the unnormalized weight)
 *  4: sum_i abs(<psi_T|phi_i>)
 *  5: total number of walkers
 *  6: total number of "healthy" walkers (those with weight > 1e-6, ovlp>1e-8, etc)
 */ 
template<class WalkerSet,
         typename = typename std::enable_if<(WalkerSet::contiguous_walker)>::type,
         typename = typename std::enable_if<(WalkerSet::fixed_population)>::type
	>
void correlatedPopulationControl(std::vector<std::reference_wrapper<WalkerSet>>& wlks,
				nda::array<ComplexType,2>& curData,
				std::string combine_type, 
				bool skip = false)
{
  app_log(0," correlatedPopulationControl probably broken! Fix Fix Fix.");
  if(wlks.size() == 0)
    APP_ABORT("Error: Empty walker vector in correlatedPopulationControl.");
  auto& mpi=wlks[0].get().get_mpi(); 
  auto rng=wlks[0].get().getRNG();

  int LoadBalance_timer = AFQMCTimer.add("WalkerSetBase::loadBalance");
  int Branching_timer = AFQMCTimer.add("WalkerSetBase::branching");
  AFQMCTimer.start(Branching_timer);

  int n_sys = wlks.size();
  if(curData.extent(0) != n_sys or curData.extent(1) != 7)
    curData = nda::array<ComplexType,2>(n_sys,7);
  curData() = ComplexType(0);

  auto [pop_control, min_weight, max_weight] = wlks[0].get().population_control_parameters();
  int tot_num_walkers = wlks[0].get().size();
  int target = wlks[0].get().get_target_population();
  int global_target = wlks[0].get().get_global_target_population();
  if (target != tot_num_walkers)
    APP_ABORT("Error: tot_num_walkers!=target");
  if (target*mpi.comm.size() != global_target) 
    APP_ABORT("Error: Mismatched global populations"); 

  // safety check
  for( int s=0; s<n_sys; s++ ) { 
    if (wlks[s].get().size() != tot_num_walkers)
      APP_ABORT("Error: Inconsistent number of walkers in cs_systems"); 
    if (wlks[s].get().get_target_population() != target) 
      APP_ABORT("Error: Inconsistent target populations in cs_systems"); 
  }

  // gather data and walker information
  {
    for(int s=0; s<n_sys; s++) 
    {
      afqmc::BasicWalkerData(wlks[s].get(), curData(s,nda::range::all), mpi.comm);
      RealType scl = 1.0 / curData(s,0).real();
      wlks[s].get().scaleWeight(scl, true);
    }
  }
  if (mpi.comm.size() > 1)
    mpi.broadcast(curData);
  for(int s=0; s<n_sys; s++) 
    wlks[s].get().adjustLogOverlapFactor(std::log(std::abs(curData(s,4))));

  if(skip) return;

  // gather weights 
  nda::array<std::pair<double, int>,2> buffer(n_sys,global_target);
  for(int s=0; s<n_sys; s++)
    getGlobalListOfWalkerWeights(wlks[s].get(), buffer(s,nda::range::all), mpi.comm);

  // make correlated branching decisions
// MAM: this needs to be done over the "global" communicator, which does not exist yet 
//      in mpi_context. FIX FIX FIX
  if( mpi.comm.root() ) { 
    // population control on master node
    if (pop_control == BranchingAlgorithm::pair || pop_control == BranchingAlgorithm::serial_comb ||
        pop_control == BranchingAlgorithm::min_branch) {
      correlatedSerialBranching(pop_control, combine_type, min_weight, max_weight, buffer, *rng, mpi.comm);
    } else {
      APP_ABORT("Error: Unknown population control algorithm.");
    }
  }
  
  // bcast branching decisions
  int n_excess=0;     
  std::vector<int> nwalk_counts_old, nwalk_counts_new;
  {
    nwalk_counts_old.resize(mpi.comm.size());
    nwalk_counts_new.resize(mpi.comm.size());
    std::fill(nwalk_counts_new.begin(), nwalk_counts_new.end(), target);
    mpi.broadcast(buffer);
    // all systems should have identical branching instructions!
    for (int i = 0, p = 0; i < mpi.comm.size(); i++)
    {
      int cnt = 0;
      for (int k = 0; k < target; k++, p++)
        cnt += buffer(0,p).second;
      nwalk_counts_old[i] = cnt;
    }   
    n_excess = std::max(0,nwalk_counts_old[mpi.comm.rank()]-target);
  }

  int walker_size = wlks[0].get().single_walker_size() + wlks[0].get().single_walker_bp_size();
  nda::array<ComplexType,2> Wexcess(n_excess, walker_size);
  AFQMCTimer.stop(Branching_timer);
  for(int s=0; s<n_sys; s++) {

    if (wlks[s].get().single_walker_size() + wlks[s].get().single_walker_bp_size() != walker_size) 
      APP_ABORT("Error in correlated sampling: Walkers with different size found. FIX"); 

    AFQMCTimer.start(Branching_timer);
    // perform local branching
    // walkers beyond target go in Wexcess
    auto buff_s = buffer(s,nda::range::all);
    wlks[s].get().branch(std::span(buff_s.data() + target * mpi.comm.rank(), target), Wexcess);
    AFQMCTimer.stop(Branching_timer);

    // load balance
    AFQMCTimer.start(LoadBalance_timer);
//     load balance after population control events
    wlks[s].get().loadBalance(Wexcess,nwalk_counts_old,nwalk_counts_new);
    AFQMCTimer.stop(LoadBalance_timer);  
  }

/*
*/ 
}

} // namespace afqmc

} // namespace sfqmc

