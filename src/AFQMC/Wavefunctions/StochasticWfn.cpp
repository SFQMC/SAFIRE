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

#include "AFQMC/Wavefunctions/StochasticWfn.hpp"
#include "AFQMC/Propagators/Propagator.hpp"
#include "AFQMC/Wavefunctions/StochasticWfn.icc"

namespace sfqmc
{
namespace afqmc
{

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
