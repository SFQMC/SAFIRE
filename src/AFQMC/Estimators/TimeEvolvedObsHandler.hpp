/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#pragma once

#include <vector>
#include <string>
#include <iostream>

#include "AFQMC/config.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Estimators/FullObsHandler.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

#include "AFQMC/Estimators/Observables/Observable.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Handler class for time-evolved observables. 
 */
template<MEMORY_SPACE MEM>
class TimeEvolvedObsHandler
{

public:
  TimeEvolvedObsHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                 std::string name_,
                 ptree pt,
                 WALKER_TYPES wlk,
                 int nave,
                 int NMO_,
                 Wavefunction<MEM>& wfn_)
      : mpi(_mpi),
        walker_type(wlk),
        wfn(std::addressof(wfn_)),
        ncalls(0),
        number_of_averages(nave),
        name(name_)
  {
    using std::fill_n;
    if(number_of_averages <= 0)
      APP_ABORT("Error:  Empty measure_at.");

    for(const ptree::value_type &it : pt)
    {
      std::string cname = it.first;
      io::tolower(cname);
      if (cname == "onerdm")
      {
        properties_1body.emplace_back(full1rdm(mpi, it.second, walker_type, NMO_, number_of_averages));
      }

//       else if (cname == "gfock" || cname == "genfock" || cname == "ekt")
//       {
//         properties.emplace_back(
//             generalizedFockMatrix(TG, info, it.second, walker_type, wfn0, number_of_averages, block_size));
//       }
      else if (cname == "diag2rdm")
      {
        properties.emplace_back(diagonal2rdm<MEM>(mpi, it.second, walker_type, NMO_, number_of_averages));
      }
      else if (cname == "twordm")
      {
        properties.emplace_back(full2rdm<MEM>(mpi, it.second, walker_type, NMO_, number_of_averages));
      }
//       else if (cname == "n2r" || cname == "ontop2rdm")
//       {
// #if defined(ENABLE_DEVICE)
//         ptree pt1 = it.second;
//         bool use_host_memory = pt1.get<bool>("use_host_memory", false);
//         if (use_host_memory)
//         {
//           properties.emplace_back(
//               n2r<device_allocator<ComplexType>>(TG, info, it.second, walker_type, false, device_allocator<ComplexType>{},
//                                                  device_allocator<ComplexType>{}, number_of_averages, block_size));
//         }
//         else
// #endif
//         {
//           properties.emplace_back(
//               n2r<shared_allocator<ComplexType>>(TG, info, it.second, walker_type, true,
//                                                  shared_allocator<ComplexType>{TG.TG_local()},
//                                                  shared_allocator<ComplexType>{TG.Node()}, number_of_averages, block_size));
//         }
//       }
//       else if (cname == "realspace_correlators")
//       {
//         properties.emplace_back(realspace_correlators(TG, info, it.second, walker_type, number_of_averages, block_size));
//       }
//       else if (cname == "correlators")
//       {
//         properties.emplace_back(atomcentered_correlators(TG, info, it.second, walker_type, number_of_averages, block_size));
//       }
      else if (cname == "pair_correlators")
      {
        properties.emplace_back(pair_correlator(mpi, it.second, walker_type, NMO_, number_of_averages));
      }
      else if (cname == "spinspin")
      {
        properties.emplace_back(spinspinobs(mpi, it.second, walker_type, NMO_, number_of_averages));
      }

    }

    utils::check(properties.size()+properties_1body.size() > 0, "empty observables list is not allowed.");

    denominator.resize(number_of_averages);
    denominator() = ComplexType(0.0, 0.0);
  }

  void print(int iblock, h5::group* g) 
  {
    denominator() *= ComplexType(1.0 / double(ncalls));
    mpi->all_reduce(denominator,std::plus<>());

    for (auto& v : properties_1body)
      v.print(iblock, g, denominator);
    for (auto& v : properties)
      v.print(iblock, g, denominator);

    ncalls=0;
    denominator() = ComplexType(0.0);
  }

  template<class WlkSet>
  void accumulate(int iav, WlkSet& wset, nda::MemoryVector auto&& wgt, 
                  nda::MemoryArrayOfRank<4> auto const& X, 
                  nda::MemoryArrayOfRank<4> auto const& Y, 
                  nda::MemoryArrayOfRank<4> auto const& M, bool importanceSampling) 
  {
    // accumulate denominator
    if (importanceSampling)
      denominator(iav) += nda::sum(wgt);
    else
    {
      APP_ABORT(" Finish implementation of free projection. \n\n");
    } 
    ncalls++;

    wfn->accumulate_estimators(iav, wset, wgt, properties_1body, properties, 
        std::addressof(X), std::addressof(Y), std::addressof(M), true, importanceSampling);
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  Wavefunction<MEM>* wfn;

  int ncalls = 1;

  int number_of_averages = 1;

  std::string name;

  std::vector<Observable<MEM>> properties_1body;
  std::vector<Observable<MEM>> properties;

  nda::array<ComplexType,1> denominator;

};

} // namespace afqmc

} // namespace sfqmc

