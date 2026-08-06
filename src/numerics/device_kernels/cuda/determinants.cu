#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>
#include "numerics/operations/determinants_impl.hpp"

namespace kernels::device::detail
{

template<typename V1, typename V2, typename V3> 
void log_determinant_from_getrf_impl(V1 const& a, V2 const& ipiv, V3& res)
{
  long N = a.extent(0);
  if(N==0) return;
  auto a_d = to_cuda_std_mdspan(a);
  auto ipiv_d = to_cuda_std_mdspan(ipiv);
  auto res_d = to_cuda_std_mdspan(res);
  auto F = math::detail::log_determinant_from_getrf_impl<decltype(a_d),decltype(ipiv_d),decltype(res_d)>{a_d,ipiv_d,res_d};

  cub::DeviceFor::Bulk(N,F);
}

template<typename V1, typename V2, typename V3>
void log_determinant_from_geqrf_impl(V1 const& a, V2& scl, V3& res)
{
  long N = a.extent(0);
  if(N==0) return;
  auto a_d = to_cuda_std_mdspan(a);
  auto scl_d = to_cuda_std_mdspan(scl);
  auto res_d = to_cuda_std_mdspan(res);
  auto F = math::detail::log_determinant_from_geqrf_impl<decltype(a_d),decltype(scl_d),decltype(res_d)>{a_d,scl_d,res_d};

  cub::DeviceFor::Bulk(N,F);
}
  
using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

template void log_determinant_from_getrf_impl(
    device_array_view<const std::complex<double>,3,basic_layout_t<3>> const&,
    device_array_view<const int,2,basic_layout_t<2>> const&,
    device_array_view<std::complex<double>,1,basic_layout_t<1>>&);

template void log_determinant_from_geqrf_impl(
    device_array_view<const std::complex<double>,3,basic_layout_t<3>> const&,
    device_array_view<std::complex<double>,2,basic_layout_t<2>>&,
    device_array_view<std::complex<double>,1,basic_layout_t<1>>&);

} //kernels::device::detail
