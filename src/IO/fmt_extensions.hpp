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

#include <complex>
#include <format>
#include <nda/nda.hpp>

#if defined(ENABLE_SPDLOG)

#include <spdlog/fmt/bundled/format.h>
#include <spdlog/fmt/bundled/ranges.h>

template <> struct fmt::formatter<std::complex<double>> {
  fmt::formatter<double> inner;
  
  constexpr auto parse(format_parse_context& ctx) {
    return inner.parse(ctx);
  }

  template <typename FormatContext>
  auto format(const std::complex<double>& p, FormatContext& ctx) {
    auto out = ctx.out();
    *out++ = '(';
    ctx.advance_to(out);
    out = inner.format(std::real(p), ctx);
    *out++ = ',';
    ctx.advance_to(out);
    out = inner.format(std::imag(p), ctx);
    *out++ = ')';
    return out;
  }
};

#endif

template <typename T> struct std::formatter<std::complex<T>> {
  std::formatter<T> inner;
  
  constexpr auto parse(format_parse_context& ctx) {
    return inner.parse(ctx);
  }

  template <typename FormatContext>
  auto format(const std::complex<T>& p, FormatContext& ctx) const {
    auto out = ctx.out();
    *out++ = '(';
    ctx.advance_to(out);
    out = inner.format(std::real(p), ctx);
    *out++ = ',';
    ctx.advance_to(out);
    out = inner.format(std::imag(p), ctx);
    *out++ = ')';
    return out;
  }
};

// C++23 provides a std::formatter for every input_range. nda arrays are input
// ranges, so that generic specialization would be ambiguous with the one below.
// Opt nda arrays out of range formatting to give ours precedence. The
// input_range and same_as constraints mirror those of the standard library's
// format_kind partial specialization so that ours is more specialized.
#ifdef __cpp_lib_format_ranges
template <nda::Array A>
  requires std::ranges::input_range<A> && std::same_as<A, std::remove_cvref_t<A>>
constexpr std::range_format std::format_kind<A> = std::range_format::disabled;
#endif

template <nda::Array A, typename CharT>
struct std::formatter<A, CharT> : std::formatter<std::string_view, CharT> {
  auto format(A const& a, auto& ctx) const {
    std::ostringstream os;
    os << a;
    return std::formatter<std::string_view, CharT>::format(os.str(), ctx);
  }
};

/*
template <nda::Array Arr>
struct formatter<Arr> 
{

  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template<typename FormatContext>
  auto format(Arr const& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    *ctx.out()++ = '[';
    bool first = true;
    for(auto const& v : p) {
      if (!first) {
        *ctx.out()++ = ',';
      }
      std::format_to(ctx.out(), "{}", v);
      first = false;
    }
    *ctx.out()++ = ']';
    return ctx.out(); 
  }

};
*/

