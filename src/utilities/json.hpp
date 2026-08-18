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
#include <cctype>
#include <charconv>
#include <format>
#include <system_error>
#include <optional>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include <nlohmann/json.hpp>

#include "utilities/check.hpp"

namespace nlohmann {
// nlohmann/json does not serialize std::optional: 3.11 has no support at all and the 3.12
// implementation is behind an `#ifndef JSON_USE_IMPLICIT_CONVERSIONS` that macro_scope.hpp
// always defines. A null or absent value maps to nullopt.
template<typename T>
struct adl_serializer<std::optional<T>> {
  template<typename BasicJson>
  static void to_json(BasicJson& j, const std::optional<T>& opt) {
    if(opt) {
      j = *opt;
    } else {
      j = nullptr;
    }
  }
  static void from_json(const json& j, std::optional<T>& opt) {
    if(j.is_null()) {
      opt = std::nullopt;
    } else {
      opt = j.get<T>();
    }
  }
};

// The serializer of a BlockRef has to be a specialization rather than a free function found by
// argument dependent lookup: BlockRef is an alias, so the associated namespaces of the variant
// are std and the one of T, never the one the alias is declared in.
template<typename T>
struct adl_serializer<std::variant<std::string, T>> {
  template<typename BasicJson>
  static void to_json(BasicJson& j, const std::variant<std::string, T>& ref) {
    std::visit([&j](const auto& block) { j = block; }, ref);
  }
  static void from_json(const json& j, std::variant<std::string, T>& ref) {
    if(j.is_string()) {
      ref = j.get<std::string>();
    } else {
      ref = j.get<T>();
    }
  }
};
} // namespace nlohmann

// Strict replacements for the nlohmann serialization macros, which quietly accept malformed input:
// NLOHMANN_JSON_SERIALIZE_ENUM maps anything it does not recognize -- an unknown string, a number, an
// object -- to the first entry of the table, and NLOHMANN_DEFINE_TYPE_* ignores keys that do not
// correspond to a member, so a typo in the input silently does nothing.
//
// Like the nlohmann macros, these have to be invoked in the namespace the type is declared in, so
// that the generated to_json/from_json are found by argument dependent lookup.

namespace sfqmc::utils {

/// A block inside an execute block is either the name of a block declared outside of it, or a
/// full declaration. When it is absent altogether, a default constructed block is used.
template<typename T>
using BlockRef = std::variant<std::string, T>;

namespace detail {

/// nlohmann::json stores the members of an object in a std::map, which sorts them by key, while
/// nlohmann::ordered_json keeps them in insertion order. The two are distinct types, so the
/// generated to_json is written against either of them.
template<typename T>
concept basic_json = nlohmann::detail::is_basic_json<T>::value;

template<std::ranges::input_range Names>
std::string json_quoted_list(Names&& names) {
  std::string list;
  for(const std::string_view name : names) {
    if(!list.empty()) {
      list += ", ";
    }
    list += std::format("\"{}\"", name);
  }
  return list;
}

template<typename BasicJson, typename Enum, std::size_t N>
void json_enum_to(BasicJson& j, Enum value, const std::pair<Enum, std::string_view> (&table)[N],
                  std::string_view enum_name) {
  for(const auto& [candidate, name] : table) {
    if(candidate == value) {
      j = name;
      return;
    }
  }
  check(false, "Cannot serialize {} value {}, which is none of {{{}}}.", enum_name,
        static_cast<std::underlying_type_t<Enum>>(value), json_quoted_list(table | std::views::values));
}

/// The comparison is case insensitive, so that inputs written before the names were fixed
/// (e.g. "COLLINEAR") keep working.
template<typename Enum, std::size_t N>
void json_enum_from(const nlohmann::json& j, Enum& value, const std::pair<Enum, std::string_view> (&table)[N],
                    std::string_view enum_name) {
  check(j.is_string(), "Expected a string for {}, but found {} ({}). It has to be one of {{{}}}.", enum_name,
        j.type_name(), j.dump(), json_quoted_list(table | std::views::values));

  const auto& input = j.get_ref<const nlohmann::json::string_t&>();
  std::string lower_input{input};
  std::ranges::transform(lower_input, lower_input.begin(), [](unsigned char c) { return std::tolower(c); });

  for(const auto& [candidate, name] : table) {
    if(lower_input == name) {
      value = candidate;
      return;
    }
  }
  check(false, "Unknown {} \"{}\", it has to be one of {{{}}}.", enum_name, input,
        json_quoted_list(table | std::views::values));
}

template<typename T>
void json_read_value(const nlohmann::json& j, T& out);
inline void json_read_value(const nlohmann::json& j, bool& out);
template<typename T>
  requires(std::is_arithmetic_v<T>)
void json_read_value(const nlohmann::json& j, T& out);
template<typename T>
void json_read_value(const nlohmann::json& j, std::vector<T>& out);
template<typename T>
void json_read_value(const nlohmann::json& j, std::optional<T>& out);

template<typename T>
void json_read_value(const nlohmann::json& j, T& out) {
  out = j.get<T>();
}

/// Input written for the property tree reader, which stored every value as a string, may
/// quote its numbers and booleans. Accept that rather than break those files.
inline void json_read_value(const nlohmann::json& j, bool& out) {
  if(!j.is_string()) {
    out = j.get<bool>();
    return;
  }
  const auto& text = j.get_ref<const nlohmann::json::string_t&>();
  if(text == "true" || text == "1") {
    out = true;
  } else if(text == "false" || text == "0") {
    out = false;
  } else {
    check(false, "Cannot read a boolean from \"{}\", it has to be one of {{true, false}}.", text);
  }
}

template<typename T>
  requires(std::is_arithmetic_v<T>)
void json_read_value(const nlohmann::json& j, T& out) {
  if(!j.is_string()) {
    out = j.get<T>();
    return;
  }
  const auto& text = j.get_ref<const nlohmann::json::string_t&>();
  const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), out);
  check(error == std::errc{} && end == text.data() + text.size(), "Cannot read a number from \"{}\".", text);
}

/// A lone value stands for a one element list, so that a block that may be repeated does not
/// have to be wrapped in brackets in the common case where there is only one of it.
template<typename T>
void json_read_value(const nlohmann::json& j, std::vector<T>& out) {
  out.clear();
  if(j.is_array()) {
    out.reserve(j.size());
    for(const auto& element : j) {
      json_read_value(element, out.emplace_back());
    }
  } else {
    json_read_value(j, out.emplace_back());
  }
}

template<typename T>
void json_read_value(const nlohmann::json& j, std::optional<T>& out) {
  if(j.is_null()) {
    out.reset();
  } else {
    json_read_value(j, out.emplace());
  }
}

/// Reads the member at key, or leaves it at the value it has in a default constructed object.
template<typename T>
void json_read_member(const nlohmann::json& j, const char* key, const T& fallback, T& out) {
  const auto it = j.find(key);
  if(it == j.end()) {
    out = fallback;
  } else {
    json_read_value(*it, out);
  }
}

/// Rejects anything that is not an object with a subset of the given keys.
inline void json_check_keys(const nlohmann::json& j, std::span<const std::string_view> keys,
                            std::string_view type_name) {
  check(j.is_object(), "Expected an object for {}, but found {} ({}).", type_name, j.type_name(), j.dump());

  for(const auto& item : j.items()) {
    bool known = false;
    for(const auto& key : keys) {
      if(key == item.key()) {
        known = true;
        break;
      }
    }
    if(!known) {
      if(keys.empty()) {
        check(false, "Unknown key \"{}\" in {}, which takes no parameters.", item.key(), type_name);
      }
      check(false, "Unknown key \"{}\" in {}, it has to be one of {{{}}}.", item.key(), type_name,
            json_quoted_list(keys));
    }
  }
}

} // namespace sfqmc::utils::detail
} // namespace sfqmc::utils

/// Serialize ENUM_TYPE through an explicit {{value, "name"}, ...} table. Values or names that are
/// not in the table are an error in both directions.
#define SAFIRE_DEFINE_ENUM(ENUM_TYPE, ...)                                                                   \
  template<::sfqmc::utils::detail::basic_json SafireJson>                                                    \
  void to_json(SafireJson& safire_json_j, const ENUM_TYPE& safire_json_value) {                              \
    using safire_json_enum_type = ENUM_TYPE;                                                                 \
    static constexpr std::pair<safire_json_enum_type, std::string_view> safire_json_table[] = __VA_ARGS__;   \
    ::sfqmc::utils::detail::json_enum_to(safire_json_j, safire_json_value, safire_json_table, #ENUM_TYPE);   \
  }                                                                                                          \
  inline void from_json(const nlohmann::json& safire_json_j, ENUM_TYPE& safire_json_value) {                 \
    using safire_json_enum_type = ENUM_TYPE;                                                                 \
    static constexpr std::pair<safire_json_enum_type, std::string_view> safire_json_table[] = __VA_ARGS__;   \
    ::sfqmc::utils::detail::json_enum_from(safire_json_j, safire_json_value, safire_json_table, #ENUM_TYPE); \
  }

#define SAFIRE_JSON_ENUM_NAME_ENTRY(NAME) {safire_json_enum_type::NAME, #NAME},

/// Serialize ENUM_TYPE using the enumerator names themselves as the json strings, e.g.
/// SAFIRE_DEFINE_ENUM_NAMES(DriverType, afqmc, ftafqmc).
#define SAFIRE_DEFINE_ENUM_NAMES(ENUM_TYPE, ...)       \
  SAFIRE_DEFINE_ENUM(ENUM_TYPE, {NLOHMANN_JSON_EXPAND( \
      NLOHMANN_JSON_PASTE(SAFIRE_JSON_ENUM_NAME_ENTRY, __VA_ARGS__))})

#define SAFIRE_JSON_KEY(MEMBER) #MEMBER,
#define SAFIRE_JSON_TO_MEMBER(MEMBER) safire_json_j[#MEMBER] = safire_json_t.MEMBER;
#define SAFIRE_JSON_FROM_MEMBER(MEMBER)                                    \
  ::sfqmc::utils::detail::json_read_member(safire_json_j, #MEMBER,         \
                                           safire_json_defaults.MEMBER,    \
                                           safire_json_t.MEMBER);

/// Serialize the listed members of TYPE. Members that are absent from the input keep the value they
/// have in a default constructed TYPE; keys that do not name a listed member are an error.
#define SAFIRE_DEFINE_PARAMETERS(TYPE, ...)                                                          \
  template<::sfqmc::utils::detail::basic_json SafireJson>                                            \
  void to_json(SafireJson& safire_json_j, [[maybe_unused]] const TYPE& safire_json_t) {              \
    safire_json_j = SafireJson::object();                                                            \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(SAFIRE_JSON_TO_MEMBER, __VA_ARGS__))                    \
  }                                                                                                  \
  inline void from_json(const nlohmann::json& safire_json_j, [[maybe_unused]] TYPE& safire_json_t) { \
    static constexpr std::string_view safire_json_keys[] = {                                         \
        NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(SAFIRE_JSON_KEY, __VA_ARGS__))};                    \
    ::sfqmc::utils::detail::json_check_keys(safire_json_j, safire_json_keys, #TYPE);                 \
    [[maybe_unused]] const TYPE safire_json_defaults{};                                              \
    NLOHMANN_JSON_EXPAND(NLOHMANN_JSON_PASTE(SAFIRE_JSON_FROM_MEMBER, __VA_ARGS__))                  \
  }

/// SAFIRE_DEFINE_PARAMETERS for a TYPE without any members: the only valid input is an empty object.
#define SAFIRE_DEFINE_EMPTY_PARAMETERS(TYPE)                           \
  template<::sfqmc::utils::detail::basic_json SafireJson>              \
  void to_json(SafireJson& safire_json_j, const TYPE&) {               \
    safire_json_j = SafireJson::object();                              \
  }                                                                    \
  inline void from_json(const nlohmann::json& safire_json_j, TYPE&) {  \
    ::sfqmc::utils::detail::json_check_keys(safire_json_j, {}, #TYPE); \
  }
