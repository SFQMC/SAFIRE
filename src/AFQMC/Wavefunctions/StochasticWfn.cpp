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

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, class devPsiT>
Propagator<MEM>& StochasticWfn<MEM, devPsiT>::inner_propagator()
{
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::inner_propagator: inner propagator not built.");
  return inner_stack_->propagator();
}

template<MEMORY_SPACE MEM, class devPsiT>
Propagator<MEM> const& StochasticWfn<MEM, devPsiT>::inner_propagator() const
{
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::inner_propagator: inner propagator not built.");
  return inner_stack_->propagator();
}

template<MEMORY_SPACE MEM, class devPsiT>
void StochasticWfn<MEM, devPsiT>::maybe_advance_inner_ensemble()
{
  if (inner_nsteps_ <= 0)
  {
    inner_step_pending_ = false;
    return;
  }
  if (not inner_step_pending_)
    return;
  inner_step_pending_ = false;

  if (not inner_ensemble_.initialized || inner_ensemble_.wset == nullptr)
    APP_ABORT("Error in StochasticWfn::maybe_advance_inner_ensemble: inner walkers not initialized.");
  if (not inner_stack_->has_propagator())
    APP_ABORT("Error in StochasticWfn::maybe_advance_inner_ensemble: inner propagator not built.");

  WalkerSet<MEM>& inner = *inner_ensemble_.wset;
  const bool collinear  = (inner_stack_->nomsd().getWalkerType() == COLLINEAR);
  auto all              = nda::range::all;
  const int P           = inner.size();
  nda::array<ComplexType, 3> anchor_on_mem = inner_anchor_;
#if defined(ENABLE_DEVICE)
  if constexpr (MEM == DEVICE_MEMORY)
    anchor_on_mem = nda::to_device(inner_anchor_);
#endif

  for (int ip = 0; ip < P; ++ip)
  {
    inner.SlaterMatrices(Alpha)(ip, all, all) = anchor_on_mem(0, all, all);
    if (collinear)
    {
      int naeb = int(inner.SlaterMatrices(Beta).extent(2));
      inner.SlaterMatrices(Beta)(ip, all, all) = anchor_on_mem(1, all, nda::range(naeb));
    }
  }

  RealType eshift(0);
  RealType dt(inner_timestep_);
  for (int step = 0; step < inner_nsteps_; ++step)
    inner_propagator().Propagate(inner, eshift, dt, 0);
  mpi_->comm.barrier();
}

template class StochasticWfn<HOST_MEMORY, PsiT_Matrix<HOST_MEMORY>>;
template class StochasticWfn<HOST_MEMORY, memory::const_shared_array<HOST_MEMORY, ComplexType, 2>>;

#if defined(ENABLE_DEVICE)

template class StochasticWfn<DEVICE_MEMORY, PsiT_Matrix<DEVICE_MEMORY>>;
template class StochasticWfn<DEVICE_MEMORY, memory::const_shared_array<DEVICE_MEMORY, ComplexType, 2>>;

#endif

} // namespace afqmc
} // namespace sfqmc
