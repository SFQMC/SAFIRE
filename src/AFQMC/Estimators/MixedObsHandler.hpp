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
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * This class manages a list of observables evaluated at the mixed distribution.
 * Mixed distribution observables are those that have the trial wave function as the left
 * hand side. 
 * Given a walker set and an array of slater matrices (references), 
 * this routine will calculate and accumulate all requested observables.
 * This class also handles all the hdf5 I/O (given a hdf archive).
 * Since n-body observables (n>1) require an explicit sum over references, 
 * they are treated separately if the number of references is > 1.
 * A customized implementation for PHMSD will be built in the future if needed. 
 */
template<MEMORY_SPACE MEM>
class MixedObsHandler
{
public:
  MixedObsHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                 std::string name_,
                 const EstimatorParameters& params,
                 WALKER_TYPES wlk,
                 int NMO_,
                 Wavefunction<MEM>& wfn_)
      : mpi(_mpi),
        walker_type(wlk),
        wfn(std::addressof(wfn_)),
        name(name_),
        denominator(1)
  {
    if(params.onerdm) {
      properties_1body.emplace_back(full1rdm(mpi, *params.onerdm, walker_type, NMO_, 1));
    }
    if(params.diag2rdm) {
      properties.emplace_back(diagonal2rdm<MEM>(mpi, *params.diag2rdm, walker_type, NMO_, 1));
    }
    if(params.twordm) {
      properties.emplace_back(full2rdm<MEM>(mpi, *params.twordm, walker_type, NMO_, 1));
    }
    if(params.pair_correlators) {
      properties.emplace_back(pair_correlator(mpi, *params.pair_correlators, walker_type, NMO_, 1));
    }
    if(params.spinspin) {
      properties.emplace_back(spinspinobs(mpi, *params.spinspin, walker_type, NMO_, 1));
    }

    ncalls = 0;
    denominator() = ComplexType(0.0, 0.0);
    utils::check((properties.size()+properties_1body.size()) != 0, "empty observables list is not allowed.");
  }

  void print(int iblock, h5::group *g)
  {
    denominator(0) *= ComplexType(1.0 / double(ncalls));
    mpi->comm.all_reduce_in_place_n(&denominator(0),1,std::plus<>());

    for (auto& v : properties_1body)
      v.print(iblock, g, denominator);
    for (auto& v : properties)
      v.print(iblock, g, denominator);

    ncalls=0;
    denominator(0) = ComplexType(0.0, 0.0);
  }

  template<class WlkSet>
  void accumulate(WlkSet& wset)
  {
    int nwalk = wset.size();

    nda::array<ComplexType, 1> wgt(nwalk);
    wset.getProperty(WEIGHT, wgt);
    ncalls++;
    denominator(0) += std::accumulate(wgt.begin(), wgt.end(), ComplexType(0.0));

    wfn->accumulate_estimators(0,wset,wgt,properties_1body,properties);
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  Wavefunction<MEM>* wfn;

  int ncalls = 0;

  std::string name;

  std::vector<Observable<MEM>> properties_1body;
  std::vector<Observable<MEM>> properties;

  nda::array<ComplexType,1> denominator;

};

} // namespace afqmc

} // namespace sfqmc

