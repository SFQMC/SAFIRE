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

#include <algorithm>
#include <format>
#include <initializer_list>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "nda/h5.hpp"

#include "AFQMC/config.h"
#include "AFQMC/Hamiltonians/Hamiltonian.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/parameter_defaults.hpp"
#include "IO/app_loggers.h"
#include "IO/banner.hpp"
#include "utilities/check.hpp"

namespace sfqmc::afqmc {

Hamiltonian Hamiltonian::from_params(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                                     const HamiltonianParameters& params) {
  app_log(1, section(std::format("Initializing Hamiltonian \"{}\"", params.name)));

  const std::string& filename = params.filename;
  utils::check(!filename.empty(), "Error: hamiltonian must contain a filename.");
  std::string format; // only meaningful at root

  const HamiltonianType htype = peek_hamiltonian_type(params, *mpi);

  h5::file file;
  std::optional<h5::group> grp, hgrp;
  if(mpi->comm.root()) {
    file = h5::file(filename, 'r');
    grp  = std::make_optional(h5::group(file));
    format = get_hamiltonian_format(*grp);
    app_log(1, "Found hamiltonian with format: {}", format);
    // open subgroup
    if(format == "coqui") {
      hgrp = std::make_optional(grp->open_group("System"));
    } else {
      hgrp = std::make_optional(grp->open_group("Hamiltonian"));
    }
  }

  std::vector<int> Idata(8);
  if(mpi->comm.root()) {
    if(format == "coqui") { // coqui always complex for now!
      h5::h5_read_attribute(*hgrp, "number_of_bands", Idata[3]); // per kpoint
      h5::group bz = hgrp->open_group("BZ");
      h5::h5_read_attribute(bz, "number_of_kpoints", Idata[2]);
      Idata[3] *= Idata[2];
    } else { // assuming only coqui or std
      h5::h5_read(*hgrp, "dims", Idata);
    }
  }
  mpi->comm.broadcast(Idata.begin(), Idata.end());

  mpi->comm.barrier();

  // the formats a hamiltonian type can be read from. Only the root knows the format, so only it
  // can check.
  auto check_format = [&](std::initializer_list<std::string_view> supported) {
    if(mpi->comm.root()) {
      utils::check(std::ranges::find(supported, format) != supported.end(),
                   "Error: format: {} not yet implemented with this hamiltonian type.", format);
    }
  };

  switch(htype) {
  case HamiltonianType::kpthc:
    check_format({"coqui"});
    return Hamiltonian(KPTHCHamiltonian(params));
  case HamiltonianType::kp_factorized:
    check_format({"coqui", "std"});
    return Hamiltonian(KPFactorizedHamiltonian(params));
  case HamiltonianType::real_dense_factorized:
    // CoQui does not generate real cholesky yet, it is hardwired to be complex
    check_format({"std"});
    return Hamiltonian(RealDenseHamiltonian(params));
  case HamiltonianType::model_hamiltonian:
    check_format({"std"});
    return Hamiltonian(ModelHamOpsGenerator(params));
  case HamiltonianType::thc:
    check_format({"coqui"});
    return Hamiltonian(THCHamiltonian(params));
  }

  APP_ABORT("Error in Hamiltonian::from_params(): Unknown Hamiltonian Type.");
}

} // namespace sfqmc::afqmc
