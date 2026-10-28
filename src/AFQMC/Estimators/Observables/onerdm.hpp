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

#include "configuration.hpp"
#include "AFQMC/config.h"
#include "AFQMC/parameters.hpp"

#include "utilities/check_shape.hpp"
#include "AFQMC/Walkers/WalkerSet.hpp"
#include "AFQMC/Estimators/Measurements.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc::afqmc {

template<MEMORY_SPACE MEM>
class OneRDM {
public:
  
  OneRDM(utils::mpi_context_t<boost::mpi3::communicator>& mpi, OneRDMParameters const& params, WALKER_TYPES walker_type, int NMO) {
    auto [nspin, npol] = walkerTypeToDims(walker_type);
    if(params.rotation) {
      rotation_ = params.rotation->load_shared<MEM, ComplexType, 3>(mpi, "RotationMatrix");
      utils::check_shape(rotation_, "RotationMatrix", nspin, rotation_.extent(1), npol*NMO);
    }
  }

  template<typename RefLoop>
  void measure(utils::mpi_context_t<boost::mpi3::communicator>& mpi, MeasurementOutput& output, MeasurementInputs<MEM, RefLoop>& inputs) {
    int nwalk = inputs.weights.size();
    auto [nspin, npol] = walkerTypeToDims(inputs.walkerType);
    int npolNMO = npol * inputs.NMO;
    
    memory::buffered_array<MEM,ComplexType,3> avgG(nspin, npolNMO, npolNMO);
    memory::buffered_array<MEM,ComplexType,1> weightedCoeff(nwalk);
    avgG() = 0.0;

    // avgG[s,i,j] = sum_d sum_w weights[w] * refCoeff_d[w] * singleRefG_d[w,s,i,j]
    inputs.referenceLoop([&](memory::array_view<MEM, ComplexType, 1> refCoeff, memory::array_view<MEM, ComplexType, 4> singleRefG) {
      auto G2D = nda::reshape(singleRefG, nwalk, nspin*npolNMO*npolNMO);
      weightedCoeff() = inputs.weights();
      nda::tensor::elementwise(1.0, refCoeff, 1.0, weightedCoeff, nda::tensor::binary_op::PROD);
      nda::blas::gemv(1.0, nda::transpose(G2D), weightedCoeff(), 1.0, nda::flatten(avgG));
    });
        
    if(rotation_.size() > 0) {
      int num_out   = rotation_.extent(0);
      memory::buffered_array<MEM,ComplexType,3> rotG(nspin, num_out, npolNMO); 
      memory::buffered_array<MEM,ComplexType,3> rotGrot(nspin, num_out, num_out); 

      nda::tensor::contract(nda::conj(rotation_()),"sai",avgG,"sij",rotG,"saj");
      nda::tensor::contract(rotG,"saj",rotation_(),"sbj",rotGrot,"sab");

      output.measure(mpi, "OneRDM", rotGrot);
    } else {
      output.measure(mpi, "OneRDM", avgG);
    }
  }

private:
  memory::const_shared_array<MEM, ComplexType, 3> rotation_;
};

} // namespace sfqmc::afqmc

