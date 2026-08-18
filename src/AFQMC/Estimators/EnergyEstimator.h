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

#include "AFQMC/config.h"
#include <vector>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>

#include "AFQMC/Utilities/AFQMCTimer.h"

#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Walkers/WalkerConfig.hpp"


namespace sfqmc
{
namespace afqmc
{

// use atan2 to enforce [-pi,pi] range of values
inline ComplexType mod2pi(ComplexType x){
  RealType x_r = std::real(x);
  RealType x_i = std::imag(x);
  return ComplexType(atan2(sin(x_r),cos(x_r)),
                     atan2(sin(x_i),cos(x_i))
                     );
}

template<MEMORY_SPACE MEM>
class EnergyEstimator : public EstimatorBase<MEM>
{
public:
  EnergyEstimator(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> _mpi,
                  const EstimatorParameters& params,
                  int measure_interval_,
                  Wavefunction<MEM>& wfn,
                  bool impsamp_ = true)
      : mpi(_mpi),
        wfn0(std::addressof(wfn)),
        importanceSampling(impsamp_),
        energy_components(params.print_components)
  {
    print_sign    = params.print_sign;
    nblocks_equil = params.equil;
    nblocks_skip  = params.skip;
    utils::check(nblocks_equil >= 0 and nblocks_skip >= 0, "EnergyEstimator: equil, skip must both be > 0");
    measure_interval = measure_interval_;

    data.resize(11); // hardcoded to the current number of working fields
  }

  void accumulate_step([[maybe_unused]] double total_time, [[maybe_unused]] WalkerSet<MEM>& wlks,
                       [[maybe_unused]] std::vector<ComplexType>& curData) {}

  void accumulate_block([[maybe_unused]] double total_time, WalkerSet<MEM>& wset)
  {
    auto all = nda::range::all;
    // the global timer feeds the summary table, the member one the per-estimator output column
    auto energy_time = timers.energy.start();
    auto estimator_time = energy_timer.start();
    long nwalk = wset.size();

    ComplexType dum, et;
    // MAM: if nblocks_skip > 0, this will produce data filled with zeros.
    //      Can output to hdf5 instead for better format
    if( iblock < nblocks_equil or (iblock-nblocks_equil)%(nblocks_skip+1) != 0) {
      data() = ComplexType(0.0);
    } else {
      nda::array<ComplexType,2> eloc(nwalk,3);
      nda::array<ComplexType,1> ovlp(nwalk);
      int nt = wset.getTauStep();
      if constexpr (MEM == HOST_MEMORY) {  
        wfn0->Energy(wset, eloc, ovlp, nt);
      } else {
        memory::buffered_array<MEM,ComplexType,2> eloc_d(nwalk,3);
        memory::buffered_array<MEM,ComplexType,1> ovlp_d(nwalk);
        wfn0->Energy(wset, eloc_d, ovlp_d, nt);
        eloc() = eloc_d(); 
        ovlp() = ovlp_d(); 
      }

      nda::array<ComplexType,2> wprop(7,nwalk);
      wset.getProperty(WEIGHT, wprop(0,all));
      wset.getProperty(OVLP, wprop(1,all));
      wset.getProperty(PHASE, wprop(2,all));
      wset.getProperty(PHASE1, wprop(3,all));
      wset.getProperty(PHASE2, wprop(4,all));
      wset.getProperty(PHASE3, wprop(5,all));
      wset.getProperty(THETA, wprop(6,all));
      data() = ComplexType(0.0);
      for (int i = 0; i < nwalk; i++)
      {
        if (std::isnan(real(wprop(0,i)))) continue;
        if (importanceSampling)
        {
          dum = wprop(0,i) * std::exp( ovlp(i) - wprop(1,i) );
        }
        else
        {
          dum = wprop(0,i) * std::exp(ovlp(i)) * wprop(2,i);
        }
        et = eloc(i,0) + eloc(i,1) + eloc(i,2);
        if ((!std::isfinite(real(dum))) || (!std::isfinite(real(et * dum))))
          continue;
        data(1) += dum;
        data(0) += et * dum;
        data(2) += eloc(i,0) * dum;
        data(3) += eloc(i,1) * dum;
        data(4) += eloc(i,2) * dum;
        data(5) += ( dum / std::abs(dum) );  
        data(6) += ( wprop(2,i) );  
        data(7) += ( wprop(3,i) );  
        data(8) += ( wprop(4,i) );  
        data(9) += ( wprop(5,i) );
        data(10) += ( mod2pi(wprop(6,i)) );
      }
      mpi->all_reduce(data, std::plus<>());
    }
    // increase counter
    iblock ++;
    estimator_time.stop();
    energy_time.stop();
  }

  void tags(std::ofstream& out)
  {
    if (mpi->comm.root())
    {
      out << "EnergyEstim_" << name << "_nume_real  EnergyEstim_" << name << "_nume_imag "
          << "EnergyEstim_" << name << "_deno_real  EnergyEstim_" << name << "_deno_imag "
          << "EnergyEstim_" << name << "_timer ";
      if(print_sign) 
      {
        /*
        printing the real and imaginary parts of:
        - data[5] = dum /std::abs(dum)  with dum = w_i * O_i^(n+1) / O_i^(n)   (i is a walker index)
        - data[6] = wset property PHASE  (KE: unclear exactly what phase this is)
        - data[7] = PHASE1 = prod_k exp(-(i dt)/2 * (E_k + E_k')) / scale  : where scale is the constraint factor
        - data[8] = PHASE2 = O_i^(n+1) : i.e. the "new" overlap
        - data[9] = PHASE3 = w_i * scale : i.e. the "new" weight (this is cumulative product!)

        Printing just the real part of the following
        - THETA = arg( O_i^(n+1) / O_i^(n) ) - Im[ x <V> ]  : real by construction (note: <V> is the mean-field subtraction)
        */
        out << " Energy_sign_real   Energy_sign_imag "
            << "  Phase_real   Phase_imag   "
            << "  Phase1_real   Phase1_imag   "
            << "  NewOverlap_real   NewOverlap_imag   " 
            << "  ConstraintWeight_real   ConstraintWeight_imag   "
            << "  ConstraintTheta_real    ConstraintTheta_imag  ";
      }  
      if (energy_components)
      {
        out << "OneBodyEnergyEstim__nume_real "
            << "EXXEnergyEstim__nume_real "
            << "ECoulEnergyEstim__nume_real ";
      }
    }
  }

  int get_measurement_interval()
  {
    return measure_interval;
  }

  void print(std::ofstream& out, [[maybe_unused]] h5::file& file, WalkerSet<MEM>& wset)
  {
    if (mpi->comm.root())
    {
      int n = wset.get_global_target_population();
      out << data(0).real() / n << " " << data(0).imag() / n << " " << data(1).real() / n << " " << data(1).imag() / n
          << " " << energy_timer.total_time << " ";
      if(print_sign) 
      {
        out <<data(5).real() / n <<" " <<data(5).imag() / n <<" " 
            <<data(6).real() / n <<" " <<data(6).imag() / n <<" "
            <<data(7).real() / n <<" " <<data(7).imag() / n <<" " 
            <<data(8).real() / n <<" " <<data(8).imag() / n <<" " 
            <<data(9).real() / n <<" " <<data(9).imag() / n <<" "
            <<data(10).real() / n <<" " <<data(10).imag() / n <<" ";
      }
      if (energy_components)
      {
        out << data(2).real() / n << " " << data(3).real() / n << " " << data(4).real() / n << " ";
      }
      energy_timer.reset();
    }
  }

private:
  std::string name;

  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi;

  Wavefunction<MEM>* wfn0 = nullptr;

  int nblocks_skip = 0;
  int nblocks_equil = 0;
  int iblock = 0;
  int measure_interval = 1;

  nda::array<std::complex<double>,1> data;

  bool importanceSampling = true;
  bool energy_components = false;
  bool print_sign = false;

  utils::Timer energy_timer{"energy"};

};
} // namespace afqmc
} // namespace sfqmc

