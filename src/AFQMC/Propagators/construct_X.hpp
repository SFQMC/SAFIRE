
#pragma once

#include "nda/nda.hpp"

#include "AFQMC/Propagators/construct_X_kernel.hpp"

#if defined(ENABLE_DEVICE)
#include "numerics/device_kernels/device_api.hpp"
#include "numerics/device_kernels/to_view.hpp"
#endif

namespace sfqmc::afqmc
{

#if defined(ENABLE_DEVICE)

/**
 * @brief Fill X and accumulate the mean-field and hybrid weights on the device.
 *
 * The host path drives construct_X_impl over (iw,m) itself; here the loop is the kernel, so all
 * this does is hand the arrays across the nvcc boundary.
 */
void construct_X(bool zero, bool fp, double sqrtdt, double vbias_bound,
                 nda::MemoryVector auto const& FT, nda::MemoryVector auto const& vMF,
                 nda::MemoryVector auto&& MF, nda::MemoryVector auto&& HW,
                 nda::MemoryMatrix auto const& RN, nda::MemoryMatrix auto&& X)
{
  using kernels::device::to_view;
  kernels::device::construct_X(zero, fp, sqrtdt, vbias_bound, to_view(FT), to_view(vMF),
                               to_view(MF), to_view(HW), to_view(RN), to_view(X));
}

#endif

} // namespace sfqmc::afqmc
