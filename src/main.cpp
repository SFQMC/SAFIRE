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

#include <iostream>
#include <vector>
#include <cxxopts.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "IO/app_loggers.h"
#include "IO/AppAbort.hpp"

#include "config.h"
#include "configuration.hpp"

#include "utilities/check.hpp"
#include "arch/arch.h"
#include "utilities/mpi_context.h"
#include "utilities/app_version.h"

#include "AFQMC/AFQMCFactory.h"

ptree parse_input(std::string filename) {
  std::filesystem::path p{filename};

  ptree pt;
  if(p.extension() == ".json") {
      boost::property_tree::read_json(filename, pt);
  } else if(p.extension() == ".xml") {
      boost::property_tree::read_xml(filename, pt);
  } else {
    throw std::runtime_error{std::format("Unknown file extension '{}' for input file '{}'", p.extension().string(), filename)};
  }

  return pt;
}

/*
 * *** execution blocks are processed sequentially, so order is important.
 *     Communication between blocks occurs though appropriate hdf5 I/O. *** 
 */ 

/** @file safire.cpp
 */
int main_impl(int argc, char** argv)
{
  using namespace sfqmc;
  auto world = mpi3::environment::get_world_instance();
  bool root(world.root());

  
  #if defined(ENABLE_DEVICE)
  std::vector<std::string> allowed_devices = {"gpu", "cpu"};
  std::string allowed_devices_str = std::format("{}, {}", allowed_devices[0], allowed_devices[1]);
  std::string default_compute = "gpu";
  #else
  std::vector<std::string> allowed_devices = {"cpu"};
  std::string allowed_devices_str = "cpu";
  std::string default_compute = "cpu";
  #endif
  std::string compute;

  constexpr const char *welcome{
    "███████╗ █████╗ ███████╗██╗██████╗ ███████╗\n"
    "██╔════╝██╔══██╗██╔════╝██║██╔══██╗██╔════╝\n"
    "███████╗███████║█████╗  ██║██████╔╝█████╗  \n"
    "╚════██║██╔══██║██╔══╝  ██║██╔══██╗██╔══╝  \n"
    "███████║██║  ██║██║     ██║██║  ██║███████╗\n"
    "╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝\n"
    "\n"};

  // parse command line inputs
  std::vector<std::string> inputs;
  int output_level, debug_level; 

  // parse command line inputs
  cxxopts::Options options(argv[0],"SAFIRE");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
    ("h,help", "print help message")
    ("v,version", "print version message")
    ("verbosity", "higher means more (allowed: 0, 1, 2, 3)", cxxopts::value<int>()->default_value("2"))
    ("debug", "higher means more (allowed: 0, 1, 2, 3)", cxxopts::value<int>()->default_value("0"))
    ("compute", std::format("where to execute (allowed: {})", allowed_devices_str), cxxopts::value<std::string>()->default_value(default_compute))
    ("filenames", "input filenames", cxxopts::value<std::vector<std::string>>())
  ;
  options.parse_positional({"filenames"});

  cxxopts::ParseResult args{};
  try {
    args = options.parse(argc, argv);
  } catch(const cxxopts::exceptions::exception &e) {
    throw AppAbortException{e.what()};
  }

  // record program options
  if (args.count("help"))
  {
    if(root)
      std::cout << options.help({"", "Group"}) << std::endl;
    return 0;
  }
  output_level = args["verbosity"].as<int>();
  // check program options
  if (output_level < 0) 
  {
    throw AppAbortException{std::format("verbosity ({}) has to be >= 0!", output_level)};
  }
  debug_level = args["debug"].as<int>();
  if (debug_level < 0) 
  {
    throw AppAbortException{std::format("debug ({}) has to be >= 0!", output_level)};
  }
  if (args.count("version"))
  {
    if(root) {
      std::cout << welcome;
      print_version();  // print everything
    }
    return 0;
  }
  compute = args["compute"].as<std::string>();
  if (std::ranges::find(allowed_devices, compute) == allowed_devices.end())
  {
    if(compute == "gpu") {
      throw AppAbortException{std::format("Attempted to run with --compute gpu, but this is not a gpu build!")};
    }
      
    throw AppAbortException{std::format("Invalid compute: {} (allowed values: {})", compute, allowed_devices_str)};
  }

  // input files are positional arguments
  int nfile = args.count("filenames");
  if (nfile < 1)
  {
    throw AppAbortException{"no input file given; exiting ..."};
  } else {
    inputs = args["filenames"].as<std::vector<std::string>>();
  }

  // setup output loggers
  setup_loggers(root, output_level, debug_level);

  app_log(1, welcome);      

  if(root)
    print_version();


  sfqmc::arch::init(compute == "gpu");

  // !!!! assume a single input for now
  std::string myinput = inputs[0];
  ptree pt;
  try {
    pt = parse_input(myinput);
  } catch (std::exception const& e) {
    throw AppAbortException(fmt::format("Could not parse input file: {}", e.what()));
  }

  auto mpi = std::make_shared<utils::mpi_context_t<boost::mpi3::communicator>>(utils::make_mpi_context(world));

  for(auto const& it : pt)
  { // go through all simulation requests
    std::string cname = it.first;
    if (cname == "afqmc") {
      ptree sim = it.second;
#if defined(ENABLE_DEVICE)
      if(compute=="gpu") { 
        sfqmc::arch::check_device_configuration();
        auto afqmc_fac = afqmc::AFQMCFactory<DEVICE_MEMORY>("afqmc", mpi, sim);
      } else 
#endif
        auto afqmc_fac = afqmc::AFQMCFactory<HOST_MEMORY>("afqmc", mpi, sim);
    } else if (cname == "ftafqmc") {
      ptree sim = it.second;
#if defined(ENABLE_DEVICE)
      if(compute=="gpu" or compute=="default") { 
        sfqmc::arch::check_device_configuration();
        auto afqmc_fac = afqmc::AFQMCFactory<DEVICE_MEMORY>("ftafqmc", mpi, sim);
      } else 
#endif
        auto afqmc_fac = afqmc::AFQMCFactory<HOST_MEMORY>("ftafqmc", mpi, sim);
    }
     else if(cname == "cs_afqmc" || cname == "csafqmc") {
      ptree sim = it.second;
      int n_groups = sim.get<int>("project.n_groups", 1);
// need new strategy for n_group>1, need to add a new "global" communicator to the context.
#if defined(ENABLE_DEVICE)
      if(compute=="gpu") { 
        sfqmc::arch::check_device_configuration();
        auto afqmc_fac = afqmc::AFQMCFactory<DEVICE_MEMORY>("csafqmc",mpi,sim,n_groups);
      } else 
#endif
        auto afqmc_fac = afqmc::AFQMCFactory<HOST_MEMORY>("csafqmc",mpi,sim,n_groups);
    } else {
      app_error("unknown calculation type: {} \n",cname.c_str());
      throw sfqmc::AppAbortException("APP_ABORT triggered");
    }
  } // simulations end

  mpi->shared_windows.collective_free_unused();
  if(!mpi->shared_windows.isempty()) {
    app_warning("MPI shared windows were still in use when the simulation is already over. This is probably a bug.");
  }
  return 0;
}

int main(int argc, char** argv) {
  mpi3::environment env(argc, argv);
  try {
    return main_impl(argc, argv);
  } catch (const std::exception& e) {
    // avoid collective calls here!
    if(env.get_world_instance().root()) {
      std::cerr << fmt::format("\nError: {}\n", e.what());
    }
    env.get_world_instance().abort(1);
    return 1;
  } 
  // destructor of env calls MPI finalize
}
