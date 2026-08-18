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

#include <algorithm>
#include <format>
#include <string>
#include <string_view>

namespace sfqmc {

/// Width in columns of every rule drawn by the helpers below.
constexpr int default_banner_width = 75;

/// U+2550 BOX DRAWINGS DOUBLE HORIZONTAL, drawn by banner().
constexpr std::string_view double_rule_glyph{"═"};
/// U+2500 BOX DRAWINGS LIGHT HORIZONTAL, drawn by section() and hrule().
constexpr std::string_view light_rule_glyph{"─"};

namespace detail {

/// The rule glyphs are multi-byte in UTF-8, so std::string(n, c) cannot build a rule.
inline std::string repeat(std::string_view glyph, int n) {
  n = std::max(n, 0);
  std::string buf;
  buf.reserve(glyph.size() * n);
  for(int i = 0; i < n; i++) {
    buf.append(glyph);
  }
  return buf;
}

/// Columns occupied by a UTF-8 string: continuation bytes do not advance the cursor.
/// Assumes no wide or combining characters, which holds for every title we print.
constexpr int display_width(std::string_view s) {
  int width{};
  for(char c : s) {
    if((static_cast<unsigned char>(c) & 0xc0) != 0x80) {
      width++;
    }
  }
  return width;
}

} // namespace detail

/// A three-line band with the title centered between two full-width double rules:
///
///     ═══════════════════════════════════
///              Input parameters
///     ═══════════════════════════════════
///
/// The result carries no leading or trailing newline, since app_log() terminates the
/// last line itself; prefix "\n{}" where a blank line above the band is wanted. Log it
/// as app_log(level, banner(title)): with no format arguments the text is passed
/// through unformatted, so a title containing '{' is safe.
inline std::string banner(std::string_view title, int width = default_banner_width) {
  const std::string rule = detail::repeat(double_rule_glyph, width);
  const int indent = std::max((width - detail::display_width(title)) / 2, 0);
  return std::format("\n{}\n{}{}\n{}", rule, std::string(indent, ' '), title, rule);
}

/// A one-line section header, the title anchored left in a light rule:
///
///     ── Project ────────────────────────
inline std::string section(std::string_view title, int width = default_banner_width) {
  constexpr int lead = 6;
  const int used = lead + 2 + detail::display_width(title); // 2 for the spaces around the title
  std::string buf = std::format("\n{} {}", detail::repeat(light_rule_glyph, lead), title);
  if(used < width) {
    buf += " ";
    buf += detail::repeat(light_rule_glyph, width - used);
  }
  return buf;
}

/// A plain full-width light rule, for delimiting tables.
inline std::string hrule(int width = default_banner_width) {
  return detail::repeat(light_rule_glyph, width);
}

} // namespace sfqmc
