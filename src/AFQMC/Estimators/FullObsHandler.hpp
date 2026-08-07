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

#include <vector>
#include <string>
#include <iostream>

#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Estimators/Observables/Observable.hpp"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/SlaterDeterminantOperations/density_matrix.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * This class manages a list of "full" observables.
 * Full observables are those that have a walker dependent left-hand side, 
 * which result from back propagation. 
 * This implementation of the class assumes a multi-determinant trial wavefunction,  
 * resulting in the loop over "references" (each determinant in the trial wavefunciton
 * being back-propagated). 
 * Given a walker set and an array of (back propagated) slater matrices, 
 * this routine will calculate and accumulate all requested observables.
 * To make the implementation of the BackPropagated class cleaner, 
 * this class also handles all the hdf5 I/O (given a hdf archive).
 */
template<MEMORY_SPACE MEM>
class FullObsHandler
{

public:
  FullObsHandler(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                 std::string name_,
                 const EstimatorParameters& params,
                 int nave_,
                 WALKER_TYPES wlk,
                 int NMO_,
                 Wavefunction<MEM>& wfn)
      : mpi(_mpi),
        walker_type(wlk),
        wfn0(std::addressof(wfn)),
        ncalls(0),
        nave(nave_),
        name(name_)
  {
    utils::check(nave>0, "naverages <= 0 is not allowed.");

    if(params.onerdm) {
      properties.emplace_back(full1rdm(mpi, *params.onerdm, walker_type, NMO_, nave));
    }
    if(params.diag2rdm) {
      properties.emplace_back(diagonal2rdm<MEM>(mpi, *params.diag2rdm, walker_type, NMO_, nave));
    }
    if(params.twordm) {
      properties.emplace_back(full2rdm<MEM>(mpi, *params.twordm, walker_type, NMO_, nave));
    }
    if(params.pair_correlators) {
      properties.emplace_back(pair_correlator(mpi, *params.pair_correlators, walker_type, NMO_, nave));
    }
    if(params.spinspin) {
      properties.emplace_back(spinspinobs(mpi, *params.spinspin, walker_type, NMO_, nave));
    }

    utils::check(properties.size() > 0, "empty observables list is not allowed.");

    denominator.resize(nave);
    denominator() = ComplexType(0.0);
  }

  void print(int iblock, h5::group *g) 
  {
    denominator() *= ComplexType(1.0 / double(ncalls));
    mpi->all_reduce(denominator,std::plus<>());

    for (auto& v : properties)
      v.print(iblock, g, denominator);

    ncalls=0;
    denominator() = ComplexType(0.0);
  }

  /*
   * Write basic equations here!!!
   */
  template<class WlkSet>
  void accumulate(int iav, WlkSet& wset, nda::MemoryArrayOfRank<4> auto&& Refs, 
                  nda::MemoryVector auto&& wgt, nda::MemoryArrayOfRank<2> auto && logdetR, 
                  bool impsamp)
  // Refs, logdetR should be on MEM
  // wgt should be on host
  // requires(  ) 
  {
    using nda::range;
    auto all = range::all;
    int npol   = ( walker_type == NONCOLLINEAR ? 2 : 1 );
    int nspins = ( walker_type == COLLINEAR ? 2 : 1 );
    int nwalk = wset.size();
    int nrefs = Refs.extent(1);
    utils::check(iav >= 0 and iav < nave, "full1rdm::accumulate: iav out of range.");
    utils::check(logdetR.extent(0) == nwalk and logdetR.extent(1) == nrefs, "Size mismatch");

    auto SMA = wset.SlaterMatricesN(Alpha);
    auto SMB = wset.SlaterMatricesN( (walker_type == COLLINEAR) ? Beta : Alpha );

    int NMO = SMA.extent(1)/npol;

    int nup = SMA.extent(2); 
    int ndown = SMB.extent(2); 
    utils::check(SMB.extent(1) == npol*NMO, "Size mismatch");

    memory::buffered_array<MEM,ComplexType,4> G4D(nwalk,nspins,npol*NMO,npol*NMO);
    memory::buffered_array<MEM,ComplexType,1> Ov(nwalk, ComplexType(0.0)); 

    // host copy/view
    auto G4D_h = nda::to_host(G4D());

    memory::buffered_array<HOST_MEMORY,ComplexType,1> Xw(nwalk);
    memory::buffered_array<HOST_MEMORY,ComplexType,1> scl_wgt(wgt);
    memory::buffered_array<HOST_MEMORY,ComplexType,2> logdetR_h(logdetR); 

    // add contribution from down electrons if CLOSED 
    if (walker_type == CLOSED) logdetR_h() *= 2.0;

    if (impsamp)
      denominator[iav] += nda::sum(wgt);
    else
    {
      utils::check(false, " Finish implementation of free projection. \n\n");
    }
    // calculate all overlaps and accumulate denominator 
    if( nrefs > 1 ) {

      // logdetR_shift[w] = (1/Nd) * sum_d logdetR[w][d]
      // apply shift: logdetR[w][d] = logdetR[w][d] - logdetR_shift[w]
      for(int iw=0; iw<nwalk; ++iw) { 
        auto shift = nda::sum(logdetR_h(iw,all))/double(nrefs);
        logdetR_h(iw,all) -= shift; 
      }
        
// MAM: no reference overlap is being substracted yet, 
// find a suitable common reference at this stage
      Xw() = ComplexType(0.0);
      for (int iref = 0; iref < nrefs; iref++)
      {
        Ov() = ComplexType(0.0);

        //1. Calculate Green functions
        det_ops::Log_Overlap(Refs(all,iref,all,range(nup)), SMA, Ov);

        if (walker_type == COLLINEAR)
          det_ops::Log_Overlap(Refs(all,iref,all,range(nup,nup+ndown)), SMB, Ov);

        //2.accumulate CI[n] * Ov[n] * R[n]
        ComplexType CIcoeff(std::conj(wfn0->getReferenceWeight(iref)));
        auto Ov_h = nda::to_host(Ov());
        if (walker_type == CLOSED) Ov_h() *= 2.0;
        Xw() += (CIcoeff * nda::exp(Ov_h + nda::conj(logdetR_h(all,iref))));
      }
      
      // scale walker weights
      scl_wgt() /= Xw();

    }

    // calculate GF and accumulate
    Xw() = ComplexType(1.0);
    for (int iref = 0; iref < nrefs; iref++)
    {
      Ov() = ComplexType(0.0);

      //1. Calculate Green functions
      // Full GF
      det_ops::MixedDensityMatrix(Refs(all,iref,all,range(nup)), SMA, G4D(all,0,all,all), Ov, false);

      if (walker_type == COLLINEAR)
        det_ops::MixedDensityMatrix(Refs(all,iref,all,range(nup,nup+ndown)), SMB, G4D(all,1,all,all), Ov, false);

      //2. calculate and accumulate appropriate weights
      Xw() = scl_wgt();
      if (nrefs > 1)
      {
        // conjugated here!
        // MAM: Observables divide by the number of calls to accumulate. 
        //      We multiply here by nrefs to cancel that term. 
        //      This way a single interface can be used by all handlers 
        ComplexType CIcoeff(std::conj(wfn0->getReferenceWeight(iref)) * ComplexType(nrefs));
        auto Ov_h = nda::to_host(Ov());
        if (walker_type == CLOSED) Ov_h() *= 2.0;
        Xw() *= (CIcoeff * nda::exp(Ov_h + nda::conj(logdetR_h(all,iref))));
      }

      if constexpr (MEM == DEVICE_MEMORY) 
        G4D_h() = G4D();

      //3. accumulate references
      for (auto& v : properties)
        v.accumulate(iav, G4D, G4D_h, Xw, impsamp);
    }
    ncalls ++;
  }

private:
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  Wavefunction<MEM>* wfn0;

  int ncalls = 0; 

  int nave = 1;

  std::string name;

  std::vector<Observable<MEM>> properties;

  // denominator (nave, ...)
  nda::array<ComplexType,1> denominator;

};

} // namespace afqmc

} // namespace sfqmc

