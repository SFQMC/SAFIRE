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
#include "utilities/Random.hpp"
#include "IO/app_loggers.h"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"

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
 * Gathers the magnitudes of every walker weight in the population, paired with a
 * multiplicity of 1, into buffer on all ranks.
 */
template<class WlkBucket, class Vec,
         typename = typename std::enable_if<(WlkBucket::fixed_population)>::type>
inline void getGlobalListOfWalkerWeights(WlkBucket& wlk,
                                         Vec&& buffer,
                                         mpi3::communicator& comm)
{
  using Type = std::pair<double, int>;
  static_assert( std::is_same_v<Type,
                                typename std::decay_t<Vec>::value_type>,
                 "Type mismatch.");
  int target = wlk.get_target_population();
  int nW     = wlk.size();
  if (buffer.size() < target * comm.size())
    APP_ABORT(" Error in getGlobalListOfWalkerWeights(): Array dimensions.");
  if (nW > target)
    APP_ABORT(" Error in getGlobalListOfWalkerWeights(): size > target.");
  std::vector<Type> blocal(target);
  std::vector<Type>::iterator itv = blocal.begin();
  nda::array<ComplexType, 1> w_data(nW);
  wlk.getProperty(WEIGHT, w_data);
  for (int i = 0; i < nW; ++i, ++itv)
    *itv = {std::abs(w_data(i)), 1};
  MPI_Allgather(blocal.data(), blocal.size() * sizeof(Type), MPI_CHAR, buffer.data(), blocal.size() * sizeof(Type),
                MPI_CHAR, comm.get());
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

} // namespace afqmc

} // namespace sfqmc

