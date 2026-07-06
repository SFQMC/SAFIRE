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

#include<fstream>
#include<random>
#include<vector>
#include<string>
#include<tuple>
#include<memory>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <catch2/interfaces/catch_interfaces_capture.hpp>

#include <nda/nda.hpp>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "utilities/mpi_context.h"
#include "AFQMC/config.h"
#include "IO/app_loggers.h"

namespace sfqmc {
namespace utils {

// mpi context for unit tests
namespace detail {
extern std::shared_ptr<mpi_context_t<boost::mpi3::communicator>> __unit_test_mpi_context__;
}

/* Path to unit test files included in the code base */
inline constexpr std::string unit_test_base() 
{
  //std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/tests/unit_test_files/";
  std::string pre = std::string(PROJECT_SOURCE_DIR_STR) + "/utils/tests/functional/";
  return pre;
}

namespace TestFiles {
  using Flags = std::uint32_t;
  constexpr Flags RHF = 1<<0;
  constexpr Flags UHF = 1<<1;
  constexpr Flags GHF = 1<<2;
  constexpr Flags NOMSD = 1<<3;
  constexpr Flags PHMSD = 1<<4;
  constexpr Flags FINITE_T = 1<<5;
  constexpr Flags MOLECULES = 1<<6;
  constexpr Flags LATTICES = 1<<7;
  constexpr Flags SOLIDS = 1<<8;
  constexpr Flags ALL_SYSTEMS = MOLECULES | LATTICES | SOLIDS;
};


inline constexpr auto molecule_unit_tests_files(TestFiles::Flags flags) 
{
  std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> > files;
  auto pre = unit_test_base() + "molecules/";
  if(flags & TestFiles::NOMSD) {
    if(flags & TestFiles::RHF) {
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_closed.h5", 
                                          pre + "BH/afqmc_inputs/afqmc_rhf_nomsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
    } 
    if(flags & TestFiles::UHF) {
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_uhf_nomsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_uhf_nomsd_init_rhf.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "Li/afqmc_inputs/hamil_closed.h5",
                                          pre + "Li/afqmc_inputs/rohf_nomsd_fullypolarized.h5",
                                          afqmc::FULLYPOLARIZED);
    }
    if(flags & TestFiles::GHF) {
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_noncollinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_ghf_nomsd.h5",
                                          afqmc::NONCOLLINEAR);
      files.emplace_back(pre + "Pb/afqmc_inputs/afqmc_H_rhf_basis_noncollinear_sf.h5",
                                          pre + "Pb/afqmc_inputs/afqmc_ghf_sf_nomsd.h5",
                                          afqmc::NONCOLLINEAR);
      files.emplace_back(pre + "Pb/afqmc_inputs/afqmc_H_rhf_basis_noncollinear_soc.h5",
                                          pre + "Pb/afqmc_inputs/afqmc_ghf_soc_nomsd.h5",
                                          afqmc::NONCOLLINEAR);

    }
  }
  if (flags & TestFiles::PHMSD) {
    if (flags & TestFiles::UHF) {
      // edge case: leading det only
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_casci_uhf_1phmsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "BH/afqmc_inputs/afqmc_H_rhf_collinear.h5",
                                          pre + "BH/afqmc_inputs/afqmc_casci_uhf_phmsd.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      // may be redundant with above test: good for diversity of inputs
      files.emplace_back(pre + "N2/afqmc_inputs/cas_basis_hamil.h5",
                                        pre + "N2/afqmc_inputs/cas_wfn.h5",
                                        afqmc::UNDEFINED_WALKER_TYPE);
      }
  }
  return files;
}


inline constexpr auto lattice_unit_test_files(TestFiles::Flags flags) 
{
  std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> > files;
  auto pre = unit_test_base() + "models/";
  if(flags & TestFiles::NOMSD) {
    if(flags & TestFiles::RHF) {
      // Closed spin symmetry is not implemented - no tests expected to pass
    } 
    if(flags & TestFiles::UHF) {
      // HST is discrete spin for the following case
      files.emplace_back(pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_collinear.h5",
                                          pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/uhf_U0.1_wfn_nup5_ndn5.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_collinear_cont_spin.h5",
                                          pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/uhf_U0.1_wfn_nup5_ndn5.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_collinear_Um4_cont_charge.h5",
                                          pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/uhf_U0.1_wfn_nup5_ndn5.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_collinear_Um4_disc_charge.h5",
                                          pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/uhf_U0.1_wfn_nup5_ndn5.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);

      files.emplace_back(pre + "square_6x1_hubbard_kanamori_nup6_ndn6/afqmc_inputs/ham_collinear.h5",
                                          pre + "square_6x1_hubbard_kanamori_nup6_ndn6/afqmc_inputs/wfn_fe_collinear.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
    }    
    if(flags & TestFiles::GHF) {
      files.emplace_back(pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/ham_noncollinear.h5",
                                          pre + "square_4x4_hubbard_nup5_ndn5/afqmc_inputs/wfn_fe_noncollinear.h5",
                                          afqmc::NONCOLLINEAR);
    }
    if(flags & TestFiles::FINITE_T) {
      files.emplace_back(pre + "finiteT/square_2x2_hubbard_Beta3_nt100/afqmc_inputs/ham_collinear.h5",
                                          pre + "finiteT/square_2x2_hubbard_Beta3_nt100/afqmc_inputs/wfn_collinear_ft.h5",
                                          afqmc::COLLINEAR_FT);
    }
  }
  if (flags & TestFiles::PHMSD) {
    // no phmsd tests for lattice models yet
  }
  return files;
}


inline constexpr auto solid_unit_test_files(TestFiles::Flags flags) {
  std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> > files;
  auto pre = unit_test_base() + "solids/";
  if(flags & TestFiles::NOMSD) {
    if(flags & TestFiles::RHF) {
      // Closed spin symmetry is not implemented - no tests expected to pass
    } 
    if(flags & TestFiles::UHF) {
      // Cholesky cases
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_chol_1e-5.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe_closed.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_chol_1e-5.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);                          
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_2x2x2_chol_1e-5.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_2x2x2_pbe.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      // THC cases
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_thc_1e-6.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe_closed.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_thc_1e-6.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe.h5",
                                          afqmc::UNDEFINED_WALKER_TYPE);

    }    
    if(flags & TestFiles::GHF) {
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_chol_1e-5.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe_noncollinear.h5",
                                          afqmc::NONCOLLINEAR);
      files.emplace_back(pre + "C_diamond_coqui/afqmc_inputs/ham_thc_1e-6.h5",
                                          pre + "C_diamond_coqui/afqmc_inputs/wfn_mf_pbe_noncollinear.h5",
                                          afqmc::NONCOLLINEAR);
    }
  }
  if (flags & TestFiles::PHMSD) {
    // no phmsd tests for lattice models yet
  }
  return files;
}


inline constexpr auto get_unit_tests_files(TestFiles::Flags flags) {
  auto unit_test_files = std::vector< std::tuple<std::string, std::string, afqmc::WALKER_TYPES> >{};

  if (flags & TestFiles::MOLECULES) {
    auto mol_files = molecule_unit_tests_files(flags);
    unit_test_files.insert(unit_test_files.end(), mol_files.begin(), mol_files.end());
  }
  if (flags & TestFiles::LATTICES) {
    auto lat_files = lattice_unit_test_files(flags);
    unit_test_files.insert(unit_test_files.end(), lat_files.begin(), lat_files.end());
  }
  if (flags & TestFiles::SOLIDS) {
    // add solid state unit test files here
    auto solid_files = solid_unit_test_files(flags);
    unit_test_files.insert(unit_test_files.end(), solid_files.begin(), solid_files.end());
  }
  return unit_test_files;
}

inline void catch_test_exceptions(std::string_view name, auto func) {
  try {
    func();
  } catch(const sfqmc::AppAbortException &e) {
    FAIL_CHECK(std::format("APP_ABORT in test {}:\n{}", name, e.what()));
  } catch(const std::exception &e) {
    FAIL_CHECK(std::format("Exception in test {}:\n{}", name, e.what()));
  }
}

template<typename F>
inline void run_test_with_files(F func, std::string hamil_file, std::string wfn_file, TestFiles::Flags flags) {
  std::string name = Catch::getResultCapture().getCurrentTestName();

  auto run = [&](std::string hamil_file, std::string wfn_file, afqmc::WALKER_TYPES walker_type) {
    std::string subsection = std::format("{}({},{},{})", name, hamil_file, wfn_file, afqmc::walkerTypeToString(walker_type));
    
    INFO(std::format("Subsection: {} CPU", subsection));
    catch_test_exceptions(std::format("{} CPU", subsection), [&]() {
      func.template operator()<HOST_MEMORY>(hamil_file, wfn_file, walker_type);
    });
    #if defined(ENABLE_DEVICE)
      INFO(std::format("Subsection: {} GPU", subsection));
      catch_test_exceptions(std::format("{} GPU", subsection), [&]() {
        func.template operator()<DEVICE_MEMORY>(hamil_file, wfn_file, walker_type);
      });
    #endif    
  };

  if(hamil_file != "" && wfn_file != "") {
    app_log(0, "Running test with user-provided files: {}\n --hamil {} \\\n --wfn {}", name, hamil_file, wfn_file);
    run(hamil_file, wfn_file, afqmc::UNDEFINED_WALKER_TYPE);
  } else {
    auto [std_hamil_file, std_wfn_file, walker_type] = GENERATE_COPY(from_range(get_unit_tests_files(flags)));

    app_log(0, "Running test with files: \"{}\"\n --hamil {} \\\n --wfn {}", 
      name, std_hamil_file, std_wfn_file
    );
  
    run(std_hamil_file, std_wfn_file, walker_type);
  }  
}

/* Checks if a file exists in the file system */
inline bool file_exists(const std::string& name)
{
  std::ifstream f(name.c_str());
  return f.good();
}

inline std::shared_ptr<mpi_context_t<mpi3::communicator>>& make_unit_test_mpi_context()
{
  if(not detail::__unit_test_mpi_context__) {
    detail::__unit_test_mpi_context__ =
         std::make_shared<mpi_context_t<boost::mpi3::communicator>>(make_mpi_context());
  }
  return detail::__unit_test_mpi_context__;
}

template<nda::Array Array>
struct ApproxArrayMatcher : Catch::Matchers::MatcherGenericBase {
  ApproxArrayMatcher(Array const& array, RealType abstol, RealType reltol)
    : array{array},
      abstol{abstol},
      reltol{reltol}
  {}

  bool match(nda::Array auto const& other) const {
    if(other.shape() != array.shape()) {
      return false;
    }
    // make_regular required for flatten in case array is not contiguous
    auto other_host = nda::flatten(nda::make_regular(nda::to_host(other)));
    auto array_host = nda::flatten(nda::make_regular(nda::to_host(array)));
    auto diffnorm = nda::norm(other_host - array_host, INFINITY);
    auto valnorm = std::max(nda::norm(array_host, INFINITY), nda::norm(other_host, INFINITY));

    return diffnorm < std::max(abstol, reltol * valnorm);
  }

  std::string describe() const override {
      return std::format("Approx(atol={}, rtol={}): {}", abstol, reltol, array);
  }

private:
  Array array;
  RealType abstol{};
  RealType reltol{};
};

template<typename T>
struct ApproxScalarMatcher : Catch::Matchers::MatcherGenericBase {
  ApproxScalarMatcher(T val, RealType abstol, RealType reltol)
    : val{val},
      abstol{abstol},
      reltol{reltol}
  {}

  bool match(T other) const {
    return std::abs(val-other) < std::max(abstol, reltol * std::abs(val));
  }

  std::string describe() const override {
      return std::format("Approx(atol={}, rtol={}): {}", abstol, reltol, val);
  }

private:
  T val;
  RealType abstol{};
  RealType reltol{};
};


inline auto Approx(nda::Array auto&& array, RealType abstol=1e-8, RealType reltol = 1e-8) {
  return ApproxArrayMatcher{array, abstol, reltol};
}
template<typename T>
  requires (not nda::Array<T>)
inline auto Approx(T val, RealType abstol=1e-8, RealType reltol = 1e-8) {
  // + 0.0 makes sure we get a floating point matcher even if val is integral
  return ApproxScalarMatcher{val + 0.0, abstol, reltol};
}


template<nda::Array Arr>
void fillRandomArray(Arr&& A, double a = 0.0, double b = 1.0)
{
  using T = typename std::decay_t<Arr>::value_type;
  std::mt19937 generator(0);
  if constexpr (nda::is_complex_v<T>) {
    // for float, extract base type from T is complex
    std::uniform_real_distribution<double> distribution(a,b);
    for( auto& v: A )  { v  = T{distribution(generator),distribution(generator)}; }
  } else {
    std::uniform_real_distribution<T> distribution(T{a},T{b});
    for( auto& v: A )  { v  = distribution(generator); }
  }
}

template<typename T>
auto make_random(long N)
{ 
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(2*N);
    nda::array<T,1> res(N);
    for(int i=0; i<N; i++) res(i) = T{a(2*i),a(2*i+1)};     
    return res;
  } else { 
    return nda::rand<T>(N);
  }
}

template<typename T>
auto make_random(long N1, long N2)
{
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(N1,2*N2);
    nda::array<T,2> res(N1,N2);
    for(int i=0; i<N1; i++) 
      for(int j=0; j<N2; j++) 
        res(i,j) = T{a(i,2*j),a(i,2*j+1)};
    return res;
  } else {
    return nda::rand<T>(N1,N2);
  }
}

template<typename T>
auto make_random(long N1, long N2, long N3)
{
  if constexpr (nda::is_complex_v<T>) {
    auto a = nda::rand<remove_complex_t<T>>(N1,N2,2*N3);
    nda::array<T,3> res(N1,N2,N3);
    for(int i=0; i<N1; i++)
      for(int j=0; j<N2; j++)
        for(int k=0; k<N3; k++)
          res(i,j,k) = T{a(i,j,2*k),a(i,j,2*k+1)};
    return res;
  } else {
    return nda::rand<T>(N1,N2,N3);
  }
}

} // utils
} // sfqmc
