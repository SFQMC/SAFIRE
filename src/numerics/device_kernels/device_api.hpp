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

// Every entry point nvcc defines and the host compiler calls, in one place.
//
// Each .cu includes this so its definitions are checked against these declarations, and each
// host-side wrapper includes it to call them. Because view<> is layout- and address-space-erased,
// one declaration here covers every nda layout and both device and unified memory.
//
// Nothing here may name an nda type: these declarations are compiled by nvcc. A wrapper may carry
// the same name as its kernel; it is the one taking nda arrays, and it lives in the namespace of
// whatever calls it, never in this one.

#include <complex>
#include <tuple>

#include "numerics/device_kernels/device_view.hpp"

namespace kernels::device
{

/// The elementwise operations apply() runs itself, i.e. the ones cutensor's permute rejects.
/// nda::tensor::unary_op is mapped onto this by the host-side wrapper, so nda stays out of nvcc.
enum class unary_op
{
  SQRT,
  ABS,
  RCP,
  EXP,
  LOG
};

// argmax_min.cu
//
// Index and value of the extremum by real part; ties resolve to the smaller index. These take a
// flat buffer rather than a view, so there is no layout to erase.
template<typename T>
std::tuple<long, T> argmax(T const* x, long N);
template<typename T>
std::tuple<long, T> argmin(T const* x, long N);

// copy.cu, accumulate.cu
//
// b(...) = B(a(...)) and b(...) += B(alpha*a(...)). One template covers the same-type and the
// casting case: with A == B the conversion is a copy. The pairs instantiated are in
// cuda/cast_pairs.cuh.
//
// None of A, B, R is deducible from view<> (native_t is an alias template, dextents expands a pack),
// so callers spell all three: accumulate<TA,TB,R>(alpha, to_view(A), to_view(B)).
template<typename A, typename B, int R>
void copy(view<A const, R> a, view<B, R> b);
template<typename A, typename B, int R>
void accumulate(A alpha, view<A const, R> a, view<B, R> b);

// Highest rank each is instantiated for. Each (pair, rank) is a separate device kernel, so these
// bounds are a code-size choice: the caller handles anything above them by peeling off outer
// dimensions until what is left fits.
inline constexpr int max_copy_rank       = 6;
inline constexpr int max_accumulate_rank = 4;

// apply.cu
template<typename T, int R>
void apply(T alpha, view<T, R> a, unary_op op);

// phmsd_determinants.cu
//
// The closed-form small-matrix cases live on the device; above them the host wrapper allocates
// scratch and drives getrf/getri, so only the extract kernel is needed here.
inline constexpr int phmsd_det_max_closed_form     = 5;
inline constexpr int phmsd_inverse_max_closed_form = 4;

void phmsd_det_small(int nex, int const* iex, view<std::complex<double> const, 3> T,
                     view<std::complex<double>, 2> ov);
void phmsd_det_extract(int nex, int const* iex, view<std::complex<double> const, 3> T,
                       view<std::complex<double>, 4> M);

void phmsd_compact_R_inverse_small(int nex, int const* iex, view<std::complex<double> const, 3> T,
                                   view<std::complex<double>, 2> ov,
                                   view<std::complex<double>, 4> M);
void phmsd_compact_R_extract(int nex, int const* iex, view<std::complex<double> const, 3> T,
                             view<std::complex<double>, 4> M);
void phmsd_compact_R_assemble(int nex, int const* refc, int const* iex,
                              view<std::complex<double> const, 3> T,
                              view<std::complex<double> const, 2> ov,
                              view<std::complex<double> const, 4> M,
                              view<std::complex<double>, 4> R);

void phmsd_reduce_R(int nex, int const* refc, int const* iex, view<std::complex<double> const, 2> wgt,
                    view<std::complex<double> const, 4> Rbuff, view<std::complex<double>, 3> R);

// phmsd_energy.cu -- block-cooperative (cub::BlockReduce + shared memory), so these stay
// hand-written __global__s rather than a flat for-each. The walker loop of the 2-body term runs in
// the host wrapper, which slices T/R/w/KE per walker.
void ph_excited_1body_energy(int const* refc, int const* iexcit, view<std::complex<double> const, 3> S,
                             view<std::complex<double> const, 4> R,
                             view<std::complex<double> const, 2> w, view<std::complex<double>, 1> E);

void ph_excited_2body_energy_dense_cholesky_Tpna_walker(
    int const* refc, int const* iexcit, view<std::complex<double> const, 3> T,
    view<std::complex<double> const, 3> R, view<std::complex<double> const, 1> w,
    view<std::complex<double>, 1> EX, view<std::complex<double>, 1> EJ, long iw,
    view<std::complex<double>, 2> KE);

// phmsd_vmf.cu
void vmf_offdiag(long total_pairs, int nrows, int rank, int size, int nelec, long const* pair_off,
                 int const* coup_rbegin, int const* coup_rend, int const* coup_jdet,
                 std::complex<double> const* coup_val, int const* configs,
                 view<std::complex<double> const, 2> O, view<std::complex<double>, 2> G);

// add_scalar.cu
//
// a(...) = alpha + beta*a(...), with alpha crossing as a value or as an address in device memory.
// beta = 0 makes it a fill, and a is then not read -- which is the case nda's own device fill
// cannot cover, since that needs a block layout, i.e. at most one strided dimension.
template<typename T, int R>
void add_scalar(scalar_arg<T> alpha, T beta, view<T, R> a);

inline constexpr int max_add_scalar_rank = 4;

// copy_select.cu
template<typename T, typename I>
void copy_select(bool expand, view<I const, 1> m, T alpha, view<T const, 1> A, T scl, view<T, 1> B);
template<typename T, typename I>
void copy_select(bool expand, view<I const, 1> m, view<T const, 1> s, T alpha, view<T const, 1> A,
                 T scl, view<T, 1> B);
template<typename T, typename I>
void copy_select(bool expand, int dim, view<I const, 1> m, T alpha, view<T const, 2> A, T scl,
                 view<T, 2> B);
template<typename T, typename I>
void copy_select(bool expand, int dim, view<I const, 1> m, view<T const, 1> s, T alpha,
                 view<T const, 2> A, T scl, view<T, 2> B);

// determinants.cu
void log_determinant_from_getrf(view<std::complex<double> const, 3> a, view<int const, 2> pivot,
                                view<std::complex<double>, 1> res);
void log_determinant_from_geqrf(view<std::complex<double> const, 3> a,
                                view<std::complex<double>, 2> scl,
                                view<std::complex<double>, 1> res);

// complex_tools.cu
template<int R>
void zero_imag(view<std::complex<double>, R> a);

// split_singular_vals.cu
void splitDmatrix(view<std::complex<double> const, 2> A, view<std::complex<double>, 2> B,
                  view<std::complex<double>, 2> C, view<std::complex<double>, 1> res,
                  view<std::complex<double> const, 1> scl);

// construct_fields.cu
void construct_X(bool zero, bool fp, double sqrtdt, double vbias_bound, view<int const, 1> FieldTypes,
                 view<std::complex<double> const, 1> vMF, view<std::complex<double>, 1> mf_factor,
                 view<std::complex<double>, 1> hybrid_weight, view<double const, 2> RN,
                 view<std::complex<double>, 2> X);

} // namespace kernels::device
