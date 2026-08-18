#pragma once

#include <type_traits>
#include "configuration.hpp"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"

#if defined(ENABLE_DEVICE)
#include "nda/blas/interface/cublas_interface.hpp"
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/device_kernels/to_view.hpp"
#endif

namespace math
{

#if defined(ENABLE_DEVICE)

using kernels::device::to_view;

namespace detail
{

/**
 * @brief Stride of the arithmetic sequence A's elements form in memory.
 *
 * Only meaningful together with is_strided_1d(). min_stride() cannot be used: it reads the stride
 * of the fastest dimension even when that dimension has extent 1, where the stride is arbitrary.
 */
template<nda::MemoryArray A_t>
long blas_stride(A_t const& A)
{
  constexpr int R = nda::get_rank<A_t>;
  for(int d = R - 1; d >= 0; --d) {
    int dd = std::decay_t<decltype(A.indexmap())>::stride_order[d];
    if(A.extent(dd) > 1) {
      return A.strides()[dd];
    }
  }
  return 1;
}

/// cublas takes int for the length and the increments, and a negative increment would mean
/// counting from the far end of the buffer rather than the base pointer.
inline bool fits_blas(long n, long inc)
{
  return inc > 0 and n <= std::numeric_limits<int>::max() and inc <= std::numeric_limits<int>::max();
}

// Slice A at a fixed index along axis D, keeping every other axis whole. D is a constant, so
// the scalar/range argument mix is built by the pack expansion rather than dispatched on.
template<int D, std::size_t I>
auto slice_index(long i)
{
  if constexpr(int(I) == D) {
    return i;
  } else {
    (void)i;
    return nda::range::all;
  }
}

template<int D, typename A_t, std::size_t... I>
auto slice_at(A_t&& A, long i, std::index_sequence<I...>)
{
  return A(slice_index<D, I>(i)...);
}

} // namespace detail

#endif

// hermitization of A along the last two dimensions
void hermitize(nda::Array auto&& A) {
  constexpr int rank = nda::get_rank<decltype(A)>;
  static_assert(rank > 2, "A needs at least rank 2");

  int n = A.size() / (A.extent(rank-2) * A.extent(rank-1));
  auto A3 = reshape(A, n, A.extent(rank-2), A.extent(rank-1));

  for(int k = 0; k < A3.extent(0); k++) {
    for(int i = 0; i < A3.extent(1); i++) {
      for(int j = i; j < A3.extent(2); j++) {
        if(i == j) {
          A3(k, i, i) = nda::real(A3(k, i, i));
        } else {
          A3(k, i, j) = 0.5 * A3(k, i, j) + 0.5* nda::conj(A3(k, j, i));
          A3(k, j, i) = nda::conj(A3(k, i, j));
        }
      }
    }
  }
}

// B(...) = T(A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> )
void copy(A_t const& A, B_t && B) {
  using TA [[maybe_unused]] = std::remove_const_t<nda::get_value_t<A_t>>;
  using TB [[maybe_unused]] = std::remove_const_t<nda::get_value_t<B_t>>;
  [[maybe_unused]] constexpr int R = nda::get_rank<A_t>;
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
#if defined(ENABLE_DEVICE)
  if constexpr(nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
    if constexpr(nda::get_layout_info<A_t>.stride_order == nda::get_layout_info<B_t>.stride_order) {
      if(A.is_contiguous() && B.is_contiguous()) {
        if constexpr(std::is_same_v<TA,TB>) {
          // a flat memcpy, rather than cub, whose mdspan Copy asserts on overlapping spans with a
          // check that also rejects two allocations that merely touch
          nda::mem::memcpy<nda::mem::get_addr_space<std::decay_t<B_t>>,
                           nda::mem::get_addr_space<std::decay_t<A_t>>>(B.data(), A.data(),
                                                                        A.size() * sizeof(TA));
          return;
        } else if constexpr(R > 1) {
          // a cast still needs a kernel, but a rank-1 one
          math::copy(nda::flatten(A), nda::flatten(B));
          return;
        }
      }
    }
    if constexpr(R > kernels::device::max_copy_rank) {
      // peel off the slowest-varying dimension and recurse, so a rank beyond the instantiated
      // range still works.
      constexpr int  d   = std::decay_t<decltype(A.indexmap())>::stride_order[0];
      constexpr auto seq = std::make_index_sequence<R>{};
      for(long i = 0; i < A.extent(d); ++i) {
        math::copy(detail::slice_at<d>(A, i, seq), detail::slice_at<d>(B, i, seq));
      }
    } else {
      kernels::device::copy<TA, TB, R>(to_view(A), to_view(B));
    }
  } else
#endif
  {
    B() = A();
  }
}

// B(...) += T(alpha*A(...)), where T = nda::get_value_t<decltype(B)>
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
requires( nda::get_rank<A_t> == nda::get_rank<B_t> )
void accumulate(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B) {
  using TA = std::remove_const_t<nda::get_value_t<A_t>>;
  using TB = std::remove_const_t<nda::get_value_t<B_t>>;
  [[maybe_unused]] constexpr int R = nda::get_rank<A_t>;
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
#if defined(ENABLE_DEVICE)
  if constexpr(nda::mem::have_device_compatible_addr_space<A_t,B_t>) {
    long n = A.size();
    if(n == 0) {
      return;
    }
    // Whenever both sides are a single arithmetic sequence in memory this is exactly cublas
    // axpy, whose incx/incy express the stride.
    if constexpr(std::is_same_v<TA,TB> && nda::is_blas_lapack_v<TA> &&
                 nda::get_layout_info<A_t>.stride_order == nda::get_layout_info<B_t>.stride_order) {
      long ia = detail::blas_stride(A);
      long ib = detail::blas_stride(B);
      if(A.indexmap().is_strided_1d() && B.indexmap().is_strided_1d() &&
         detail::fits_blas(n, ia) && detail::fits_blas(n, ib)) {
        nda::blas::device::axpy(int(n), alpha, A.data(), int(ia), B.data(), int(ib));
        return;
      }
    }
    // one rank-1 kernel rather than a rank-R walk
    if constexpr(R > 1 &&
                 nda::get_layout_info<A_t>.stride_order == nda::get_layout_info<B_t>.stride_order) {
      if(A.is_contiguous() && B.is_contiguous()) {
        math::accumulate(alpha, nda::flatten(A), nda::flatten(B));
        return;
      }
    }
    if constexpr(R <= kernels::device::max_accumulate_rank) {
      kernels::device::accumulate<TA, TB, R>(alpha, to_view(A), to_view(B));
    } else if constexpr(std::is_same_v<TA,TB>) {
      // above the instantiated ranks, and not flat in memory: cutensor
      nda::tensor::add(alpha,A,1.0,B);
    } else {
      // cutensor takes a single value type, so peel off the slowest-varying dimension and recurse
      // until the rank fits a kernel.
      constexpr int  d   = std::decay_t<decltype(A.indexmap())>::stride_order[0];
      constexpr auto seq = std::make_index_sequence<R>{};
      for(long i = 0; i < A.extent(d); ++i) {
        math::accumulate(alpha, detail::slice_at<d>(A, i, seq), detail::slice_at<d>(B, i, seq));
      }
    }
  } else
#endif
  {
    if constexpr(std::is_same_v<TA,TB>) {
      nda::tensor::add(alpha,A,1.0,B);
    } else {
      B() += alpha*A();
    }
  }
}

/*
 *   a(...) = alpha + beta*a(...)
 *
 * beta = 0 is a fill, and a is then not read, so it may be uninitialized.
 */
template<nda::MemoryArray A_t>
void add_scalar(nda::get_value_t<A_t> alpha, nda::get_value_t<A_t> beta, A_t && a)
{
  using T = std::remove_const_t<nda::get_value_t<A_t>>;
#if defined(ENABLE_DEVICE)
  if constexpr (nda::mem::have_device_compatible_addr_space<A_t>) {
    static_assert(nda::get_rank<A_t> <= kernels::device::max_add_scalar_rank, "Rank mismatch");
    kernels::device::add_scalar<T, nda::get_rank<A_t>>(kernels::device::scalar_value(alpha), beta,
                                                       to_view(a));
  } else
#endif
  {
    nda::for_each(a.shape(),
                  [&](auto... i) { a(i...) = (beta == T{}) ? alpha : alpha + beta * a(i...); });
  }
}

/*
 *   a(...) = alpha(0) + beta*a(...)
 *
 * alpha is read where it lives, so on the device this costs no device-to-host copy and no
 * synchronization.
 */
template<nda::MemoryVector V_t, nda::MemoryArray A_t>
void add_scalar(V_t const& alpha, nda::get_value_t<A_t> beta, A_t && a)
{
  using T = std::remove_const_t<nda::get_value_t<A_t>>;
#if defined(ENABLE_DEVICE)
  if constexpr (nda::mem::have_device_compatible_addr_space<V_t,A_t>) {
    static_assert(nda::get_rank<A_t> <= kernels::device::max_add_scalar_rank, "Rank mismatch");
    kernels::device::add_scalar<T, nda::get_rank<A_t>>(kernels::device::scalar_at(alpha.data()),
                                                       beta, to_view(a));
  } else
#endif
  {
    math::add_scalar(T(alpha(0)), beta, a);
  }
}

template<nda::MemoryArray Arr>
void zero_imag(Arr && A)
{
  using value_type = nda::get_value_t<Arr>;
  if constexpr (nda::is_complex_v<value_type>) {
    if constexpr(nda::mem::on_host<Arr>) {
      for( auto& v : A ) v = value_type(v.real(),0.0);
    } else {
#if defined(ENABLE_DEVICE)
      if(A.is_contiguous()) {
        auto Ac = nda::reshape(A, std::array<long, 1>{A.size()});
        kernels::device::zero_imag<1>(to_view(Ac));
      } else {
        sfqmc::utils::check(nda::get_rank<Arr> <= 3, "Rank mismatch");
        kernels::device::zero_imag<nda::get_rank<Arr>>(to_view(A));
      }
#else
      sfqmc::utils::check(false,"Error: Found device array without ENABLE_DEVICE.");
#endif
    }
  }
}

template<nda::MemoryArray Arr>
void set_identity(Arr && A)
requires( nda::get_rank<Arr> > 1 and nda::get_rank<Arr> < 6 and std::decay_t<Arr>::is_stride_order_C())
{
  auto A_diag = memory::diagonal_view(A);
  A() = 0.0;
  // not A_diag() = 1: from rank 3 up the diagonal is strided twice, once between blocks and once
  // within one, and nda's own device fill covers a block layout only
  math::add_scalar(1.0, 0.0, A_diag);
}

}
