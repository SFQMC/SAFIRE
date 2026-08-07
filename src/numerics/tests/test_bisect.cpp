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

#undef NDEBUG

#include <cmath>
#include <numbers>

#include "catch2/catch_test_macros.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"

#include "numerics/operations/bisect.hpp"
#include "test_common.hpp"

namespace sfqmc
{

TEST_CASE("bisect: root of a monotonic function", "[numerics][bisect]")
{
  constexpr double tol = 1e-12;
  const auto f         = [](double x) { return x * x - 2.0; };

  const auto root = math::bisect(f, 0.0, 2.0, tol, 300);
  CHECK(root.converged);
  CHECK(root.upper - root.lower < tol);
  CHECK_THAT(root.lower, utils::Approx(std::numbers::sqrt2, tol, 0.0));
  // halving a bracket of width 2 down to tol
  CHECK(root.iterations == 41);
}

TEST_CASE("bisect: root of a decreasing function", "[numerics][bisect]")
{
  constexpr double tol = 1e-10;
  // the same shape the discrete propagator solves for: decreasing, with the root inside
  const auto f = [](double x) { return std::cos(x) - 0.5; };

  const auto root = math::bisect(f, 0.0, 2.0, tol, 300);
  CHECK(root.converged);
  CHECK_THAT(root.lower, utils::Approx(std::numbers::pi / 3.0, tol, 0.0));
}

TEST_CASE("bisect: a root on an end of the bracket", "[numerics][bisect]")
{
  const auto f = [](double x) { return x * x - 4.0; };

  const auto lower = math::bisect(f, -2.0, 3.0, 1e-12, 300);
  CHECK(lower.converged);
  CHECK(lower.lower == -2.0);
  CHECK(lower.upper == -2.0);
  CHECK(lower.iterations == 0);

  const auto upper = math::bisect(f, -3.0, 2.0, 1e-12, 300);
  CHECK(upper.converged);
  CHECK(upper.lower == 2.0);
  CHECK(upper.upper == 2.0);
  CHECK(upper.iterations == 0);
}

TEST_CASE("bisect: a bracket the root is hit exactly in", "[numerics][bisect]")
{
  // the first midpoint is the root, so the search ends there rather than at the tolerance
  const auto root = math::bisect([](double x) { return x - 1.0; }, 0.0, 2.0, 1e-12, 300);
  CHECK(root.converged);
  CHECK(root.lower == 1.0);
  CHECK(root.upper == 1.0);
  CHECK(root.iterations == 1);
}

TEST_CASE("bisect: a bracket narrower than the tolerance", "[numerics][bisect]")
{
  const auto root = math::bisect([](double x) { return x - 1.5; }, 1.0, 2.0, 4.0, 300);
  CHECK(root.converged);
  CHECK(root.iterations == 0);
}

TEST_CASE("bisect: running out of iterations does not converge", "[numerics][bisect]")
{
  const auto root = math::bisect([](double x) { return x * x - 2.0; }, 0.0, 2.0, 1e-12, 3);
  CHECK(!root.converged);
  CHECK(root.iterations == 3);
  // the bracket still holds the root, it is just no narrower than three halvings make it
  CHECK(root.lower < std::numbers::sqrt2);
  CHECK(root.upper > std::numbers::sqrt2);
  CHECK(root.upper - root.lower == 0.25);
}

TEST_CASE("bisect: a bracket without a sign change aborts", "[numerics][bisect]")
{
  const auto f = [](double x) { return x * x - 2.0; };
  CHECK_THROWS_AS(math::bisect(f, 2.0, 3.0, 1e-12, 300), AppAbortException);
  // two roots, so the ends share a sign
  CHECK_THROWS_AS(math::bisect(f, -2.0, 2.0, 1e-12, 300), AppAbortException);
  // ends in the wrong order
  CHECK_THROWS_AS(math::bisect(f, 2.0, 0.0, 1e-12, 300), AppAbortException);
}

} // namespace sfqmc
