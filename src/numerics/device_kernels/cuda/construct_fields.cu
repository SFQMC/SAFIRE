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
#include "AFQMC/Propagators/construct_X.hpp"

namespace kernels::device::detail
{

template<typename V1, typename V2, typename V3, typename V4, typename V5>
void construct_X_impl(bool zero, bool fp, double sqrtdt, double vbias_bound, V1 const& FieldTypes, 
   V2 const& vMF, V3& mf_factor, V3& hybrid_weight, V4 const& RN, V5& X)
{ 
  long N = X.size();
  if(N==0) return;
  auto FT_d = to_cuda_std_mdspan(FieldTypes);
  auto vMF_d = to_cuda_std_mdspan(vMF);
  auto MF_d = to_cuda_std_mdspan(mf_factor);
  auto HW_d = to_cuda_std_mdspan(hybrid_weight);
  auto RN_d = to_cuda_std_mdspan(RN);
  auto X_d = to_cuda_std_mdspan(X);
  sfqmc::afqmc::detail::construct_X_impl construct_X{zero,fp,sqrtdt,vbias_bound,FT_d,vMF_d,MF_d,HW_d,RN_d,X_d};

  cub::DeviceFor::Bulk(N,construct_X);
}

using memory::device_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

template void construct_X_impl(bool,bool,double,double,
    device_array_view<const int,1,basic_layout_t<1>> const&, 
    device_array_view<const std::complex<double>,1,basic_layout_t<1>> const&,
    device_array_view<std::complex<double>,1,basic_layout_t<1>>&, 
    device_array_view<std::complex<double>,1,basic_layout_t<1>>&, 
    device_array_view<const double,2,basic_layout_t<2>> const&,
    device_array_view<std::complex<double>,2,basic_layout_t<2>>&); 

}
