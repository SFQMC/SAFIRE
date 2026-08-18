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

#pragma once

#include <iostream>
#include<string>
#include <array>
#include <string_view>
#include <source_location>
#include <type_traits>
#include "IO/AppAbort.hpp"
#include "nda/nda.hpp"

namespace sfqmc {
namespace utils
{

/**
 * Checks whether the cond is true, and abort otherwise with provided message (args)
 * @tparam Args
 * @param cond - condition to be verified
 * @param args - messages
 */
template<class... Args>
struct check
{
  check(bool cond, const std::string_view format_string, Args&&... args, const std::source_location& loc = std::source_location::current())
  {
    if(not cond) {
      if constexpr (sizeof...(Args) > 0)
        APP_ABORT_with_source(loc, format_string, std::forward<Args>(args)...);
      else
        APP_ABORT_with_source(loc, format_string);
    }
  }
};

template <typename... Args>
check(bool, const std::string_view, Args&&...) -> check<Args...>;

} // namespace utils
} // namespace sfqmc

