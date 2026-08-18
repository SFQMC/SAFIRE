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

#include <iostream>
#include <format>
#include "config.h"

#include "app_version.h"
#include "git-rev.h"

void print_version() {
  constexpr const char* mkl_feature =
  #ifdef NDA_USE_MKL
      " MKL";
  #else
      "";
  #endif

  constexpr const char* cuda_feature =
  #ifdef ENABLE_CUDA
      " CUDA";
  #else
      "";
  #endif

  constexpr const char* cpptrace_feature =
  #ifdef ENABLE_CPPTRACE
      " cpptrace";
  #else
      "";
  #endif

  constexpr const char* device_rng_from_host_feature =
  #ifdef DEVICE_RNG_FROM_HOST
      " DeviceRNGFromHost";
  #else
      "";
  #endif

  constexpr const char* git_info =
  #ifdef AF_APP_GIT_BRANCH
      "Git branch:  " AF_APP_GIT_BRANCH "\n"
      "Git commit:  " AF_APP_GIT_HASH " (" AF_APP_GIT_COMMIT_LAST_CHANGED ")\n";
  #else
      "";
  #endif


  std::cout << std::format(
      "Version:     " AF_APP_VERSION "\n"
      "{}"
      "Build type:  " AF_APP_BUILD_TYPE "\n"
      "Features:   {}{}{}{}\n",
      git_info,
      mkl_feature,
      cuda_feature,
      cpptrace_feature,
      device_rng_from_host_feature
  );
}
