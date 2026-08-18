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

#include <iostream>
#include <fstream>
#include <string>

#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace sfqmc {
namespace utils
{

// This is needed when value_t doesn't have a default constructor (in the case of nda::range it is deprecated)
// Thanks to Nils for the implementation!
template<long N, typename value_t>
auto array_of_objects(value_t const& val) {
  return []<auto ... Is>(auto v, std::index_sequence<Is...>) {
    return std::array{((void)Is, v)...};
  }(val,std::make_index_sequence<N>());
};

template<long N>
auto default_array_of_ranges() {
  return array_of_objects<N>(::nda::range(0));
};

inline auto read_file_to_string(const std::string& filename) {
  std::ifstream file(filename);
  utils::check(file.is_open(), "Error: Could not open file: {}", filename);
  std::stringstream buffer;
  buffer << file.rdbuf(); // Read the entire file's content into the stringstream buffer
  file.close();
  return buffer.str(); // Convert the stringstream buffer to a std::string
};

// compares 2 strings, ignoring white spaces
inline bool string_equal(std::string const& s1, std::string const& s2)
{
  int N = std::min(s1.length(),s2.length());
  for(int i=0; i<N; i++) 
    if(s1[i] != s2[i]) return false; 
  for(int i=N; i<s1.length(); ++i)
    if(s1[i] != ' ') return false; 
  for(int i=N; i<s2.length(); ++i)
    if(s2[i] != ' ') return false; 
  return true;
}

}
}

