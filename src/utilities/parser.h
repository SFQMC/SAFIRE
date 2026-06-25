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

#pragma once

#include <algorithm>
#include <string>
#include <vector>
#include <format>
#include "utilities/check.hpp"

namespace sfqmc {
namespace utils
{

std::vector<std::string> split(std::string const&, std::string const&);


template<typename T>
struct EnumInputSpec {
  T type{};
  std::string_view explanation{};
  std::vector<std::string> names{};
};
  
// consider unifying this together with the other walker functions into a separate file
template<typename T>
T parse_enum(std::string_view input, std::initializer_list<EnumInputSpec<T>> specs, std::string_view enum_name) {
  std::string lower_input{input};
  std::ranges::transform(lower_input, lower_input.begin(),
                         [](unsigned char c) { return std::tolower(c); });

  for(const auto& spec : specs) {
    for(const auto &name: spec.names) {
      if(lower_input == name) {
        std::string explanation{};
        if(!spec.explanation.empty()) {
          explanation = std::format(" ({})", spec.explanation);
        }
        app_log(1,"Using {}{} {}.", spec.names.front(), explanation, enum_name); 
        return spec.type;
      }
    }
  }
  std::string allowed;
  for(const auto& spec : specs) {
    if(!allowed.empty()) {
      allowed += ", ";
    }
    allowed += std::format("\"{}\"", spec.names.front());
  }
    
  check(false, "Unknown {} type \"{}\" has to be one of {{{}}}.", enum_name, lower_input, allowed);
  return T{};
}

}
}
