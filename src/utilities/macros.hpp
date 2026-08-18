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

// Macros to simplify use of visitors in interface classes

#define VISITOR(NAME,VAR,CONST) \
  auto NAME() CONST \
  { return std::visit( [&](auto&& v) { return v.NAME(); }, VAR ); } 

#define VOID_VISITOR(NAME,VAR,CONST) \
  void NAME() CONST \
  { std::visit( [&](auto&& v) { v.NAME(); }, VAR ); } 

#define VISITOR_ARGS(NAME,VAR,CONST) \
  template<class...  Args> auto NAME(Args&&... args) CONST \
  { return std::visit( [&](auto&& v) { return v.NAME(std::forward<Args>(args)...); }, VAR ); } 

#define VOID_VISITOR_ARGS(NAME,VAR,CONST) \
  template<class...  Args> void NAME(Args&&... args) CONST \
  { std::visit( [&](auto&& v) { v.NAME(std::forward<Args>(args)...); }, VAR ); } 

