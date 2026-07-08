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

#ifndef AFQMC_KP_UTILITIES_HPP
#define AFQMC_KP_UTILITIES_HPP

#include <algorithm>
#include "Numerics/ma_operations.hpp"
#include "multi/array.hpp"
#include "multi/array_ref.hpp"

/*
 * Return true if PsiT is in block diagonal form, otherwise return false;
 * If it is block diagonal form, it also returns the number of states in each k-point block. 
 */
template<class Vector, class CSR, class Array>
bool get_nocc_per_kp(Vector const& nmo_per_kp, CSR const& PsiT, Array&& nocc_per_kp, bool noncolin = false)
{
  int nkpts = nmo_per_kp.size();
  int N     = PsiT.size(0);
  int M     = PsiT.size(1);
  int npol  = noncolin ? 2 : 1;
  RUNTIME_CHECK(M % npol == 0, "");
  RUNTIME_CHECK(nocc_per_kp.size() == nkpts, "");

  std::fill_n(raw_pointer_cast(nocc_per_kp.origin()), nkpts, 0);
  std::vector<int> bounds(npol * nkpts + 1);
  bounds[0] = 0;
  for (int k = 0; k < npol * nkpts; k++)
    bounds[k + 1] = bounds[k] + nmo_per_kp[k % nkpts];
  int Q = 0;
  for (int i = 0; i < N; i++)
  {
    auto nt = PsiT.num_non_zero_elements(i);
    if (nt == 0)
    {
      std::fill_n(raw_pointer_cast(nocc_per_kp.origin()), nkpts, 0);
      return false;
    }
    auto col = PsiT.non_zero_indices2_data(i);
    // check the kp index of the first non-zero column. Must be either >= Q
    auto it = std::lower_bound(bounds.begin(), bounds.end(), *col + 1) - 1;
    RUNTIME_CHECK(it != bounds.end(), "");
    int Q_   = std::distance(bounds.begin(), it) % nkpts;
    int pol_ = std::distance(bounds.begin(), it) / nkpts;
    RUNTIME_CHECK(Q_ >= 0 && Q_ < nkpts, "");
    RUNTIME_CHECK(pol_ == 0 || pol_ == 1, "");
    if (Q_ < Q)
    {
      std::fill_n(raw_pointer_cast(nocc_per_kp.origin()), nkpts, 0);
      return false;
    }
    Q = Q_;
    for (int ni = 0; ni < nt; ++ni, ++col)
    {
      if ((*col < bounds[Q] || *col >= bounds[Q + 1]) &&
          (*col < bounds[(npol - 1) * nkpts + Q] || *col >= bounds[(npol - 1) * nkpts + Q + 1]))
      {
        std::fill_n(raw_pointer_cast(nocc_per_kp.origin()), nkpts, 0);
        return false;
      }
    }
    ++nocc_per_kp[Q];
  }
  return true;
}

template<class Array, class Vector, class CSR>
Array get_PsiK(Vector const& nmo_per_kp, CSR const& PsiT, int K, bool noncolin = false)
{
  int nkpts = nmo_per_kp.size();
  int N     = PsiT.size(0);
  int M     = PsiT.size(1);
  int npol  = noncolin ? 2 : 1;
  RUNTIME_CHECK(M % npol == 0, "");

  int nel = 0;
  std::vector<int> bounds(npol * nkpts + 1);
  bounds[0] = 0;
  for (int k = 0; k < npol * nkpts; k++)
    bounds[k + 1] = bounds[k] + nmo_per_kp[k % nkpts];
  int Q = 0;
  for (int i = 0; i < N; i++)
  {
    auto nt = PsiT.num_non_zero_elements(i);
    if (nt == 0)
      APP_ABORT("Error: PsiT not in block-diagonal form in get_PsiK.");
    auto col = PsiT.non_zero_indices2_data(i);
    // check the kp index of the first non-zero column. Must be either >= Q
    auto it = std::lower_bound(bounds.begin(), bounds.end(), *col + 1) - 1;
    RUNTIME_CHECK(it != bounds.end(), "");
    int Q_   = std::distance(bounds.begin(), it) % nkpts;
    int pol_ = std::distance(bounds.begin(), it) / nkpts;
    RUNTIME_CHECK(Q_ >= 0 && Q_ < nkpts, "");
    RUNTIME_CHECK(pol_ == 0 || pol_ == 1, "");
    if (Q_ < Q)
      APP_ABORT("Error: PsiT not in block-diagonal form in get_PsiK.");
    Q = Q_;
    for (int ni = 0; ni < nt; ++ni, ++col)
      if ((*col < bounds[Q] || *col >= bounds[Q + 1]) &&
          (*col < bounds[(npol - 1) * nkpts + Q] || *col >= bounds[(npol - 1) * nkpts + Q + 1]))
        APP_ABORT("Error: PsiT not in block-diagonal form in get_PsiK.");
    if (Q == K)
      nel++;
  }
  using element = typename std::decay<Array>::type::element;
  Array A({nel, npol * nmo_per_kp[K]});
  using std::fill_n;
  fill_n(A.origin(), A.num_elements(), element(0));
  nel = 0;
  for (int i = 0; i < N; i++)
  {
    auto nt  = PsiT.num_non_zero_elements(i);
    auto col = PsiT.non_zero_indices2_data(i);
    auto val = PsiT.non_zero_values_data(i);
    // check the kp index of the first non-zero column. Must be either >= Q
    auto it = std::lower_bound(bounds.begin(), bounds.end(), *col + 1) - 1;
    Q   = std::distance(bounds.begin(), it) % nkpts;
    if (Q == K)
    {
      for (int ni = 0; ni < nt; ++ni, ++col, ++val)
      {
        if (*col < bounds[K + 1]) // alpha
          A[nel][*col - bounds[K]] = static_cast<element>(*val);
        else // beta
          A[nel][*col - bounds[nkpts + K] + nmo_per_kp[K]] = static_cast<element>(*val);
      }
      nel++;
    }
    if (Q > K)
      break;
  }
  return A;
}

#endif
