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

#include <cmath>
#include <concepts>

#include "utilities/check.hpp"

namespace math
{

/// The bracket a bisection search ended on.
template<std::floating_point T>
struct bisect_result
{
  T lower;         ///< lower end of the final bracket, and the root to use
  T upper;         ///< upper end of the final bracket
  int iterations;  ///< number of halvings performed
  bool converged;  ///< whether the bracket was narrowed as far as it can be, see bisect()
};

/// Finds a root of f in [lower, upper] by repeated halving of the bracket. The interval given has
/// to bracket the root, that is f has to change sign over it. Halving stops once the bracket is
/// narrower than tol, once it is down to two neighbouring representable values, or after max_iter
/// halvings, whichever comes first.
///
/// converged reports all but the last of those, so a caller that needs the root to a known accuracy
/// has to check it: a search that runs out of iterations returns a bracket of arbitrary width.
template<std::floating_point T, typename F>
bisect_result<T> bisect(F&& f, T lower, T upper, T tol, int max_iter)
{
  sfqmc::utils::check(lower <= upper, "bisect: the bracket [{}, {}] is in the wrong order.", lower, upper);

  T flower = f(lower);
  if(flower == T{0}) {
    return {lower, lower, 0, true};
  }
  const T fupper = f(upper);
  if(fupper == T{0}) {
    return {upper, upper, 0, true};
  }
  sfqmc::utils::check(flower * fupper < T{0}, "bisect: f does not change sign over the bracket [{}, {}], where it "
                      "takes the values {} and {}, so the bracket holds no root, or more than one.",
                      lower, upper, flower, fupper);

  int iterations = 0;
  bool converged = upper - lower < tol;
  while(!converged && iterations < max_iter) {
    const T mid = (lower + upper) / 2;
    if(mid == lower || mid == upper) {
      // the ends are neighbouring floating point values, the bracket cannot be narrowed further
      converged = true;
      break;
    }
    ++iterations;

    const T fmid = f(mid);
    if(fmid == T{0}) {
      return {mid, mid, iterations, true};
    }
    // keep the end whose sign the middle does not share, so that the bracket still holds the root
    if(std::signbit(fmid) == std::signbit(flower)) {
      lower  = mid;
      flower = fmid;
    } else {
      upper = mid;
    }
    converged = upper - lower < tol;
  }

  return {lower, upper, iterations, converged};
}

} // namespace math
