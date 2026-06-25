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
  std::string info;
  info = pt.get<std::string>("system", "");
  utils::check(InfoMap.find(info) != InfoMap.end(), "ERROR: Undefined system in execute block. ");

  AFQMCInfo& AFinfo = InfoMap[info];

  int NMO  = AFinfo.NMO;
  int nup = AFinfo.nup;
  int ndown = AFinfo.ndown;

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

  // safety check!!!
  utils::check(Idata[3] == NMO, " Error: NMO differs from value in integral file. ");

  ComplexType NuclearCoulombEnergy(0);
  ComplexType FrozenCoreEnergy(0);
  ComplexType ElecSelfIntEnergy(0);

  if (mpi->comm.root())
  {
    if(format == "std") {
      std::vector<RealType> Rdata(2,RealType(0.0));
   
      h5::h5_read(*hgrp,"Energies",Rdata);  
      if (Rdata.size() > 0)
        NuclearCoulombEnergy = Rdata[0];
      if (Rdata.size() > 1)
        FrozenCoreEnergy = Rdata[1];
    } else if(format == "coqui") {
      NuclearCoulombEnergy = 0.0;
      if( H5Aexists(h5::hid_t(*hgrp),"nuclear_energy") )
        h5::h5_read_attribute(*hgrp,"nuclear_energy",NuclearCoulombEnergy);  
      if( H5Aexists(h5::hid_t(*hgrp),"frozen_core_energy") )
        h5::h5_read_attribute(*hgrp,"frozen_core_energy",FrozenCoreEnergy);
      if( H5Aexists(h5::hid_t(*hgrp),"madelung_constant") ) {
        h5::h5_read_attribute(*hgrp,"madelung_constant",ElecSelfIntEnergy);
        ElecSelfIntEnergy *= -1.0*(nup+ndown);
      }
    }
  }
  mpi->comm.broadcast_n(&NuclearCoulombEnergy, 1, 0);
  mpi->comm.broadcast_n(&FrozenCoreEnergy, 1, 0);
  mpi->comm.broadcast_n(&ElecSelfIntEnergy, 1, 0);

  app_log(2, "");
  app_log(2, " - Nuclear coulomb energy: {}",NuclearCoulombEnergy);
  app_log(2, " - Frozen Core energy: {}",FrozenCoreEnergy);
  app_log(2, " - Electron self-interaction energy: {}",ElecSelfIntEnergy);

  // MAM: FrozenCoreEnergy is not handled correctly! FIX! Add all terms into a single E0
  NuclearCoulombEnergy += ElecSelfIntEnergy;

  mpi->comm.barrier();
  if (htype == KPTHC)
  {
    if(mpi->comm.root())
      utils::check(format == "coqui", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(KPTHCHamiltonian(pt, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if (htype == KPFactorized)
  {
    if(mpi->comm.root())
      utils::check(format == "coqui" or format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(KPFactorizedHamiltonian(pt, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if (htype == RealDenseFactorized)
  {
    // CoQui does not generate real cholesky yet, it is hardwired to be complex
    if(mpi->comm.root())
      utils::check(format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(RealDenseHamiltonian(pt, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if ( htype == ModelHamiltonian ) 
  {
    if(mpi->comm.root())
      utils::check(format == "std", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(ModelHamOpsGenerator(pt, NuclearCoulombEnergy, FrozenCoreEnergy));
  }
  else if ( htype == THC )
  {
    if(mpi->comm.root())
      utils::check(format == "coqui", "Error: format: {} not yet implemented with this hamiltonian type.", format);
    return Hamiltonian(THCHamiltonian(pt, NuclearCoulombEnergy, FrozenCoreEnergy));
  }

  utils::check(false, " Error in HamiltonianFactory::fromHDF5(): Unknown Hamiltonian Type. ");
  return Hamiltonian{};
}
} // namespace afqmc
} // namespace sfqmc
