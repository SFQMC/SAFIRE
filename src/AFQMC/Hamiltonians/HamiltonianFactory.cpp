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

#include <optional>
#include <complex>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>

#include "nda/h5.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "AFQMC/config.h"
#include "HamiltonianFactory.h"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

#include "AFQMC/Hamiltonians/RealDenseHamiltonian.h"
#include "AFQMC/Hamiltonians/THCHamiltonian.h"
#include "AFQMC/Hamiltonians/KPFactorizedHamiltonian.h"
#include "AFQMC/Hamiltonians/ModelHamOpsGenerator.h"

#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{
Hamiltonian HamiltonianFactory::fromHDF5(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi, ptree pt)
{
  std::string filename = pt.get<std::string>("filename");
  std::string format;  // only meaningful at root
  HamiltonianTypes htype = UNKNOWN;

  app_log(1," Initializing Hamiltonian from file: {}", filename);

  h5::file file;
  std::optional<h5::group> grp, hgrp;
  if (mpi->comm.root())
  {
    file = h5::file(filename,'r');
    grp = std::make_optional(h5::group(file));
    format = get_hamiltonian_format(*grp);
    app_log(1, " Found hamiltonian with format: {}", format);
    htype = peekHamType(*grp,format);
    // open subgroup
    if(format == "coqui") {
      hgrp = std::make_optional(grp->open_group("System"));
    } else {
      hgrp = std::make_optional(grp->open_group("Hamiltonian"));
    } 
  }
  {
    int htype_ = int(htype);
    mpi->comm.broadcast_n(&htype_, 1, 0);
    htype = HamiltonianTypes(htype_);
  }

  std::vector<int> Idata(8);
  if (mpi->comm.root()) {
    if(format == "coqui") { // coqui always complex for now!
      h5::h5_read_attribute(*hgrp,"number_of_bands",Idata[3]);  // per kpoint
      h5::group bz = hgrp->open_group("BZ");
      h5::h5_read_attribute(bz,"number_of_kpoints",Idata[2]);
      Idata[3] *= Idata[2];
    } else { // assuming only coqui or std
      h5::h5_read(*hgrp,"dims",Idata); 
    }
  }
  mpi->comm.broadcast(Idata.begin(), Idata.end());

  mpi->comm.barrier();
  if (htype == KPTHC)
  {
    if(mpi->comm.root())
      utils::check(format == "coqui", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(KPTHCHamiltonian(pt));
  }
  else if (htype == KPFactorized)
  {
    if(mpi->comm.root())
      utils::check(format == "coqui" or format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(KPFactorizedHamiltonian(pt));
  }
  else if (htype == RealDenseFactorized)
  {
    // CoQui does not generate real cholesky yet, it is hardwired to be complex
    if(mpi->comm.root())
      utils::check(format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(RealDenseHamiltonian(pt));
  }
  else if ( htype == ModelHamiltonian ) 
  {
    if(mpi->comm.root())
      utils::check(format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(ModelHamOpsGenerator(pt));
  }
  else if ( htype == THC )
  {
    if(mpi->comm.root())
      utils::check(format == "coqui", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(THCHamiltonian(pt));
  }

  utils::check(false, " Error in HamiltonianFactory::fromHDF5(): Unknown Hamiltonian Type. ");
  return Hamiltonian{};
}
} // namespace afqmc
} // namespace sfqmc
