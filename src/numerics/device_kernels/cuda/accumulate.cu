

//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/accumulate.cuh"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include "cub/device/device_copy.cuh"

namespace kernels::device::detail
{

template<typename A, typename B>
void accumulate_impl(nda::get_value_t<A> alpha, A const& a, B & b)
{
  static_assert(nda::get_rank<A> == nda::get_rank<B> and nda::get_rank<A> < 5, "Rank mismatch");
  constexpr int rank = nda::get_rank<A>;
  sfqmc::utils::check(a.shape() == b.shape(), "Shape mismatch");
  auto a_d = to_cuda_std_mdspan(a);
  auto b_d = to_cuda_std_mdspan(b);
  auto alpha_d = cuda_std_value_cast(alpha);
  long sz = a.size();
  if constexpr (rank==1) {
    auto f = [=] __device__(long n) {
      b_d(n) += alpha_d*a_d(n);
    };
    cub::DeviceFor::Bulk(sz,f);
  } else if constexpr (rank==2) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long N = a.extent(1);
      auto f = [=] __device__(long n) {
        long i = n/N;
        n = n - i*N;
        b_d(i,n) += alpha_d*a_d(i,n);
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long N = a.extent(0);
      auto f = [=] __device__(long n) {
        long i = n/N;
        n = n - i*N;
        b_d(n,i) += alpha_d*a_d(n,i);
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  } else if constexpr (rank==3) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long Nz = a.extent(2);
      long Nyz = a.extent(1)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nyz;
        n = n - i*Nyz;
        long j = n/Nz;
        n = n - j*Nz;
        b_d(i,j,n) += alpha_d*a_d(i,j,n);
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long Nz = a.extent(2);
      long Nyz = a.extent(1)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nyz;
        n = n - i*Nyz;
        long j = n/Nz;
        n = n - j*Nz;
        b_d(n,j,i) += alpha_d*a_d(n,j,i);
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  } else if constexpr (rank==4) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long Nz = a.extent(3);
      long Nxyz = a.extent(1)*a.extent(2)*Nz;
      long Nyz = a.extent(2)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nxyz;
        n = n - i*Nxyz;
        long j = n/Nyz;
        n = n - j*Nyz;
        long k = n/Nz;
        n = n - k*Nz; 
        b_d(i,j,k,n) += alpha_d*a_d(i,j,k,n);
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long Nz = a.extent(3);
      long Nxyz = a.extent(1)*a.extent(2)*Nz;
      long Nyz = a.extent(2)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nxyz;
        n = n - i*Nxyz;
        long j = n/Nyz;
        n = n - j*Nyz;
        long k = n/Nz;
        n = n - k*Nz;     
        b_d(n,k,j,i) += alpha_d*a_d(n,k,j,i);
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  }
}

template<typename A>
void scale_impl(nda::get_value_t<A> alpha, A & a)
{
  static_assert(nda::get_rank<A> < 5, "Rank mismatch");
  constexpr int rank = nda::get_rank<A>;
  auto a_d = to_cuda_std_mdspan(a);
  auto alpha_d = cuda_std_value_cast(alpha);
  long sz = a.size();
  if constexpr (rank==1) {
    auto f = [=] __device__(long n) {
      a_d(n) *= alpha_d;
    };
    cub::DeviceFor::Bulk(sz,f);
  } else if constexpr (rank==2) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long N = a.extent(1);
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        a_d(i,j) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long N = a.extent(0);
      auto f = [=] __device__(long n) {
        long i = n/N;
        long j = n - i*N;
        a_d(j,i) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  } else if constexpr (rank==3) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long Nz = a.extent(2);
      long Nyz = a.extent(1)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nyz;
        n = n - i*Nyz;
        long j = n/Nz;
        n = n - j*Nz;
        a_d(i,j,n) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long Nz = a.extent(2);
      long Nyz = a.extent(1)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nyz;
        n = n - i*Nyz;
        long j = n/Nz;
        n = n - j*Nz;
        a_d(n,j,i) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  } else if constexpr (rank==4) {
    if constexpr (std::decay_t<A>::is_stride_order_C()) {
      long Nz = a.extent(3);
      long Nxyz = a.extent(1)*a.extent(2)*Nz;
      long Nyz = a.extent(2)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nxyz;
        n = n - i*Nxyz;
        long j = n/Nyz;
        n = n - j*Nyz;
        long k = n/Nz;
        n = n - k*Nz;
        a_d(i,j,k,n) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    } else {
      long Nz = a.extent(3);
      long Nxyz = a.extent(1)*a.extent(2)*Nz;
      long Nyz = a.extent(2)*Nz;
      auto f = [=] __device__(long n) {
        long i = n/Nxyz;
        n = n - i*Nxyz;
        long j = n/Nyz;
        n = n - j*Nyz;
        long k = n/Nz;
        n = n - k*Nz;     
        a_d(n,k,j,i) *= alpha_d;
      };
      cub::DeviceFor::Bulk(sz,f);
    }
  }
}

template<typename MdIn, typename MdOut>
struct copy_functor
{
  MdIn  a;
  MdOut b;
  // ForEachInExtents passes the linear iteration index ahead of the coordinates
  template<typename Idx, typename... I>
  __device__ void operator()(Idx, I... idx) { b(idx...) = a(idx...); }
};

template<typename T, int R>
void copy_strided(T const* src, std::array<long,R> const& src_str,
                  T* dst, std::array<long,R> const& dst_str,
                  std::array<long,R> const& ext)
{
  using dext = cuda::std::dextents<long,R>;
  using cuda::std::layout_stride;

  cuda::std::array<long,R> e, ss, ds;
  std::copy_n(ext.begin(),R,e.begin());
  std::copy_n(src_str.begin(),R,ss.begin());
  std::copy_n(dst_str.begin(),R,ds.begin());

  auto sp = cuda_std_ptr_cast(src);
  auto dp = cuda_std_ptr_cast(dst);
  using src_t = typename std::pointer_traits<decltype(sp)>::element_type;
  using dst_t = typename std::pointer_traits<decltype(dp)>::element_type;
  auto a_d = cuda::std::mdspan<src_t,dext,layout_stride>(sp,layout_stride::mapping<dext>(e,ss));
  auto b_d = cuda::std::mdspan<dst_t,dext,layout_stride>(dp,layout_stride::mapping<dext>(e,ds));

  auto status = cub::DeviceFor::ForEachInExtents(a_d.extents(),
                    copy_functor<decltype(a_d),decltype(b_d)>{a_d,b_d});
  // cuda_check(status)
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

// permuted layouts
// 021 -> 120 where each integer is written in 4 bit binary, e.g. 120 -> 0001 0010 0000 -> 288 
using perm021_layout_t = typename nda::basic_layout<0, 288ul, nda::layout_prop_e::none>;

#define _inst_(T,V) \
template void accumulate_impl(T,V<const T,1,basic_layout_t<1>> const&, V<T,1,basic_layout_t<1>> &);  \
template void accumulate_impl(T,V<const T,2,basic_layout_t<2>> const&, V<T,2,basic_layout_t<2>> &);  \
template void accumulate_impl(T,V<const T,3,basic_layout_t<3>> const&, V<T,3,basic_layout_t<3>> &);  \
template void accumulate_impl(T,V<const T,4,basic_layout_t<4>> const&, V<T,4,basic_layout_t<4>> &);  \
template void accumulate_impl(T,V<const T,3,basic_layout_t<3>> const&, V<T,3,perm021_layout_t> &);  \
template void scale_impl(T,V<T,1,basic_layout_t<1>> &);  \
template void scale_impl(T,V<T,2,basic_layout_t<2>> &);  \
template void scale_impl(T,V<T,3,basic_layout_t<3>> &);  \
template void scale_impl(T,V<T,4,basic_layout_t<4>> &);

_inst_(double,device_array_view)
_inst_(std::complex<double>,device_array_view)

// copy_strided takes extents and strides as arguments, so layout is not part of its type
static_assert(max_copy_rank == 6, "_inst_copy_ranks_ must cover every rank up to max_copy_rank");

#define _inst_copy_(T,R) \
template void copy_strided<T,R>(T const*, std::array<long,R> const&, \
                                T*, std::array<long,R> const&, std::array<long,R> const&);

#define _inst_copy_ranks_(T) \
_inst_copy_(T,1) _inst_copy_(T,2) _inst_copy_(T,3) _inst_copy_(T,4) _inst_copy_(T,5) _inst_copy_(T,6)

_inst_copy_ranks_(double)
_inst_copy_ranks_(std::complex<double>)


} // namespace kernels::device::detail

