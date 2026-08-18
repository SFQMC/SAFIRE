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
#include <memory>
#include <mpi.h>
#include "configuration.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "AFQMC/config.h"

namespace sfqmc
{
namespace afqmc
{
// This should use meaningful names from an enum instead of an implicit known order
template<class WlkBucket, class DVec>
inline void BasicWalkerData(WlkBucket& wlk, DVec&& curData, mpi3::communicator& comm)
{
  using nda::range;
  utils::check(curData.size() >= 7, "Size mismatch.");
  std::fill(curData.begin(), curData.begin() + 7, 0);
  int nW            = wlk.size();
  std::vector<double> data(8, 0.);
  nda::array<ComplexType, 2> w_data(nW, 3);
  wlk.getProperty(WEIGHT, w_data(range(0, nW), 0));
  wlk.getProperty(OVLP, w_data(range(0, nW), 1));
  wlk.getProperty(PSEUDO_ELOC_, w_data(range(0, nW), 2));
  bool modified = false;
  for (int iw = 0; iw < nW; iw++)
  {
    ComplexType weight = w_data(iw,0);
    ComplexType ovlp   = w_data(iw,1);
    ComplexType eloc   = w_data(iw,2);
    data[6]++; // all walkers
    if (std::abs(weight) <= 1e-6 || (!std::isfinite(std::abs(ovlp))) || (!std::isfinite((weight * eloc).real())) ||
        (!std::isfinite((weight * eloc).imag())))
    {
      w_data(iw,0) = ComplexType(0.0, 0.0);
      w_data(iw,1) = ComplexType(0.0, 0.0); 
      w_data(iw,2) = ComplexType(0.0, 0.0);
      modified      = true;
      continue;
    }
    data[0] += (weight * eloc).real();
    data[1] += (weight * eloc).imag();
    data[2] += weight.real();
    data[3] += weight.imag();
    data[4] += std::abs(weight);
    data[5] += ovlp.real();
    data[7]++; // healthy walkers
  }
  comm.all_reduce_in_place_n(data.begin(), data.size(), std::plus<>());
  curData[0] = ComplexType(data[4] / static_cast<RealType>(wlk.get_global_target_population()), 0.0);
  curData[1] = ComplexType(data[0] / data[6], data[1] / data[6]);
  curData[2] = ComplexType(data[2] / data[6], data[3] / data[6]);
  curData[3] = data[4];
  curData[4] = data[5] / data[6];
  curData[5] = data[6];
  curData[6] = data[7];
  if (modified)
  {
    wlk.setProperty(WEIGHT, w_data(range(0, nW), 0));
    wlk.setProperty(OVLP, w_data(range(0, nW), 1));
    wlk.setProperty(PSEUDO_ELOC_, w_data(range(0, nW), 2));
  }
}

template<class WlkBucket, class IVec = std::vector<int>>
inline void CountWalkers(WlkBucket& wlk, IVec& WCnt, mpi3::communicator& comm)
{
  WCnt.resize(comm.size(), 0);
  int nw = wlk.size();
  comm.all_gather_value(nw, WCnt.begin());
}

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

} // namespace afqmc

} // namespace sfqmc

