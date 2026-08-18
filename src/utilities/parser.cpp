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
////////////////////////////////////////////////////////////////////////////////

#include "parser.h"

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace sfqmc {
namespace utils
{

std::vector<std::string> split(std::string const& str, std::string const& delim)
{
  std::vector<std::string> w;
  auto beg = str.find_first_not_of(delim);
  while(beg != std::string::npos) {
    auto end=str.find_first_of(delim, beg+1);
    if(end == std::string::npos) {
      w.emplace_back(str.substr(beg,str.size()-beg)); 
      break;
    }
    w.emplace_back(str.substr(beg,end-beg)); 
    beg = str.find_first_not_of(delim,end+1);
  }  
  return w;
}

}
}
