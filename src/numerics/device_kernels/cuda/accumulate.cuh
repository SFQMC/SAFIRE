
#pragma once

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename A, typename B> 
void accumulate_impl(nda::get_value_t<A> alpha, A const& a, B & b);

template<typename A> 
void scale_impl(nda::get_value_t<A> alpha, A& a);

// B(...) = A(...) for arbitrary rank and arbitrary strides. Extents and strides are runtime
// arguments, so a layout and any permutation of it share one instantiation.
//
// Instantiated for every rank in [1,max_copy_rank] by accumulate.cu. Each rank is a separate
// device kernel, so the bound is a code-size choice: copy() handles anything above it by
// peeling off outer dimensions until what is left fits.
inline constexpr int max_copy_rank = 6;

template<typename T, int R>
void copy_strided(T const* src, std::array<long,R> const& src_str,
                  T* dst, std::array<long,R> const& dst_str,
                  std::array<long,R> const& ext);

// Slice A at a fixed index along axis D, keeping every other axis whole. D is a constant, so
// the scalar/range argument mix is built by the pack expansion rather than dispatched on.
template<int D, std::size_t I>
auto slice_index(long i)
{
  if constexpr (int(I) == D) {
    return i;
  } else {
    (void) i;
    return nda::range::all;
  }
}

template<int D, typename A_t, std::size_t... I>
auto slice_at(A_t && A, long i, std::index_sequence<I...>)
{
  return A(slice_index<D,I>(i)...);
}

}

/*
 *   B(...) += alpha * A(...)
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
void accumulate(nda::get_value_t<A_t> alpha, A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
{
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  if(A.is_contiguous() and B.is_contiguous() and nda::get_rank<A_t> > 1) {
    kernels::device::accumulate(alpha,nda::flatten(A),nda::flatten(B));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  auto B_b = to_basic_layout(B());
  if constexpr (nda::get_rank<A_t> < 5)
    detail::accumulate_impl(alpha,A_b,B_b);
  else
    sfqmc::utils::check(false, "Calling accumulate_cast with rank > 4 array. Finish");
} 

/*
 *   A(...) *= alpha
 */
template<nda::MemoryArray A_t>
void scale(nda::get_value_t<A_t> alpha, A_t && A)
{
  if(A.is_contiguous() and nda::get_rank<A_t> > 1) { 
    kernels::device::scale(alpha,nda::flatten(A));
    return;
  }
  // careful with ranks here 
  auto A_b = to_basic_layout(A());
  if constexpr (nda::get_rank<A_t> < 5)
    detail::scale_impl(alpha,A_b);
  else
    sfqmc::utils::check(false, "Calling accumulate_cast with rank > 2 array. Finish");
}

/*
 *   B(...) = A(...) 
 */
template<nda::MemoryArray A_t, nda::MemoryArray B_t>
void copy(A_t const& A, B_t && B)
requires( nda::get_rank<A_t> == nda::get_rank<B_t> ) 
{
  using T = nda::get_value_t<A_t>;
  constexpr int R = nda::get_rank<A_t>;
  sfqmc::utils::check(A.shape() == B.shape(), "Shape mismatch");
  // a flat memcpy, rather than cub, whose mdspan Copy asserts on overlapping spans with a
  // check that also rejects two allocations that merely touch
  constexpr bool same_stride_order = std::decay_t<A_t>::layout_t::stride_order_encoded ==
                                     std::decay_t<B_t>::layout_t::stride_order_encoded;
  if constexpr (same_stride_order) {
    if(A.is_contiguous() and B.is_contiguous()) {
      nda::mem::memcpy<nda::mem::get_addr_space<std::decay_t<B_t>>,
                       nda::mem::get_addr_space<std::decay_t<A_t>>>(
          B.data(), A.data(), A.size()*sizeof(T));
      return;
    }
  }
  if constexpr (R > detail::max_copy_rank) {
    // peel off the slowest-varying dimension and recurse, so a rank beyond the instantiated
    // range still works.
    constexpr int d = std::decay_t<decltype(A.indexmap())>::stride_order[0];
    constexpr auto seq = std::make_index_sequence<R>{};
    for(long i = 0; i < A.extent(d); ++i) {
      kernels::device::copy(detail::slice_at<d>(A,i,seq),detail::slice_at<d>(B,i,seq));
    }
    return;
  }
  else {
    detail::copy_strided<T,R>(A.data(),A.strides(),B.data(),B.strides(),A.shape());
  }
}


} // namespace kernels::device

