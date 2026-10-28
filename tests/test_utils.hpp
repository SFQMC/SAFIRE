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

#include <complex>
#include <random>
#include <algorithm>

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"

namespace sfqmc
{
namespace afqmc
{

template<typename T>
struct TEST_DATA
{
  nda::array<T, 1> E1, EJ, EXX;
  nda::array<T, 2> vbias;
  nda::array<T, 4> VHS;
  bool available;
};

template<typename T>
TEST_DATA<T> read_test_results_from_hdf(std::string fileName, std::string wfn_type = "")
{
  h5::file file(fileName,'r');
  h5::group grp(file);
  nda::array<T, 1> E1, EJ, EXX;
  nda::array<T, 2> vbias;
  nda::array<T, 4> VHS;
  bool available{};
// MAM: put test results inside Wavefunction dataset, to avoid needing to provide wfn_type.
//      This also allows a single Hamiltonian to be used in multiple tests with different wfn_types.
  if (grp.has_key("TEST_RESULTS"))
  {
    h5::group tgrp = grp.open_group("TEST_RESULTS");
    if (tgrp.has_key(wfn_type)) {
      available = true;
      h5::group wgrp = tgrp.open_group(wfn_type);
      h5::read(wgrp, "E1", E1);
      h5::read(wgrp, "EJ", EJ);
      h5::read(wgrp, "EXX", EXX);
      h5::read(wgrp, "vbias", vbias);
      h5::read(wgrp, "VHS", VHS);
    }
  }

  return TEST_DATA<T>{std::move(E1), std::move(EJ), std::move(EXX), vbias, VHS, available};
}

template<typename T>
void write_test_results_to_hdf(std::string filename, std::string wfn_type, const TEST_DATA<T>& data) {
  h5::file file(filename,'a');
  h5::group grp = h5::group{file};
  h5::group tgrp = grp.create_group("TEST_RESULTS");
  h5::group wgrp = tgrp.create_group(wfn_type);

  h5::write(wgrp, "E1", data.E1);
  h5::write(wgrp, "EJ", data.EJ);
  h5::write(wgrp, "EXX", data.EXX);
  h5::write(wgrp, "vbias", data.vbias);
  h5::write(wgrp, "VHS", data.VHS);

}

/*
// Create a fake output hdf5 filename for unit tests.
inline std::string create_test_hdf(std::string& wfn_file, std::string& hamil_file)
{
  std::size_t startw   = wfn_file.find_last_of("\\/");
  std::size_t endw     = wfn_file.find_last_of(".");
  std::string wfn_base = wfn_file.substr(startw + 1, endw - startw - 1);

  std::size_t starth   = hamil_file.find_last_of("\\/");
  std::size_t endh     = hamil_file.find_last_of(".");
  std::string ham_base = hamil_file.substr(starth + 1, endh - starth - 1);

  return wfn_base + "_" + ham_base + ".h5";
}

// generate matrix of random integers between [a0, a0+range)
inline void fillRandomMatrix(std::vector<int>& vec, int range, int a=0)
{
  std::mt19937 generator(0);
  std::uniform_int_distribution<int> distribution(a, a+range-1);
  // avoid uninitialized warning
  [[maybe_unused]] int tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    int val  = distribution(generator);
    vec[i] = val;
  }
}

template<typename T>
void fillRandomMatrix(std::vector<T>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  // avoid uninitialized warning
  T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T val  = distribution(generator);
    vec[i] = val;
  }
}

template<typename T>
void fillRandomMatrix(std::vector<std::complex<T>>& vec)
{
  std::mt19937 generator(0);
  std::normal_distribution<T> distribution(0.0, 1.0);
  [[maybe_unused]] T tmp = distribution(generator);
  for (int i = 0; i < vec.size(); i++)
  {
    T re   = distribution(generator);
    T im   = distribution(generator);
    vec[i] = std::complex<T>(re, im);
  }
}
*/

} // namespace afqmc
} // namespace sfqmc
