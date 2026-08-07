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

#include <variant>

#include "AFQMC/config.h"

#include "AFQMC/HamiltonianOperations/HamiltonianOperations.h"
#include "AFQMC/HamiltonianOperations/THCOps.hpp"
#include "AFQMC/HamiltonianOperations/KPTHCOps.hpp"
#include "AFQMC/HamiltonianOperations/KP3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/Real3IndexFactorization.hpp"
#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"


// MAM: Once all hamiltonians are implemented, measure the compilation time 
//      with and without instantiations. Remove all this and go back to the
//      header only version if the compilation times are not reduced significantly

namespace sfqmc
{
namespace afqmc
{

  // move constructor
  template<MEMORY_SPACE M>
  template<typename HOps>
  HamiltonianOperations<M>::HamiltonianOperations(HOps&& other) : var(std::move(other)) {}

  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,true>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,false>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(KPTHCOps<HOST_MEMORY>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(ModelHamOps<HOST_MEMORY,true>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(ModelHamOps<HOST_MEMORY,false>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(Real3IndexFactorization<HOST_MEMORY>&&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(KP3IndexFactorization<HOST_MEMORY>&&);

#if defined(ENABLE_DEVICE)
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,true>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,false>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(KPTHCOps<DEVICE_MEMORY>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(ModelHamOps<DEVICE_MEMORY,true>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(ModelHamOps<DEVICE_MEMORY,false>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(Real3IndexFactorization<DEVICE_MEMORY>&&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(KP3IndexFactorization<DEVICE_MEMORY>&&);
#endif

  // copy constructor
  template<MEMORY_SPACE M>
  template<typename HOps>
  HamiltonianOperations<M>::HamiltonianOperations(HOps const& other) : var(other) {}

  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,true>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(THCOps<HOST_MEMORY,false>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(KPTHCOps<HOST_MEMORY>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(ModelHamOps<HOST_MEMORY,true>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(ModelHamOps<HOST_MEMORY,false>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(Real3IndexFactorization<HOST_MEMORY>const&);
  template HamiltonianOperations<HOST_MEMORY>::HamiltonianOperations(KP3IndexFactorization<HOST_MEMORY>const&);

#if defined(ENABLE_DEVICE)
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,true>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(THCOps<DEVICE_MEMORY,false>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(KPTHCOps<DEVICE_MEMORY>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(ModelHamOps<DEVICE_MEMORY,true>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(ModelHamOps<DEVICE_MEMORY,false>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(Real3IndexFactorization<DEVICE_MEMORY>const&);
  template HamiltonianOperations<DEVICE_MEMORY>::HamiltonianOperations(KP3IndexFactorization<DEVICE_MEMORY>const&);
#endif


  // getOneBodyPropagatorMatrix_impl
  template<MEMORY_SPACE M>
  nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(dt,vMF); },
                                var);
  }

#define __getOneBodyPropagatorMatrix__(M)    \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<HOST_MEMORY,const ComplexType,1> const&);   \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<HOST_MEMORY,const ComplexType,1,nda::C_layout> const&);   \
  template nda::array<ComplexType,3> HamiltonianOperations<M>::getOneBodyPropagatorMatrix_impl(double, memory::array_view<HOST_MEMORY,const ComplexType,1,nda::basic_layout<0,nda::C_stride_order<1>,nda::layout_prop_e::strided_1d>> const&);
__getOneBodyPropagatorMatrix__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__getOneBodyPropagatorMatrix__(DEVICE_MEMORY)
#endif

  //energy_impl
  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::energy_impl(nda::MemoryArrayOfRank<2> auto& E, 
                                        nda::MemoryArrayOfRank<2> auto const& G,
              int idet, bool addH1, bool addEJ, bool addEXX)
  {
    std::visit([&](auto&& a) { a.energy(E,G,idet,addH1,addEJ,addEXX); }, var);
  }

#define __energy__(M) \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2>const&,int,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,int,bool,bool,bool);  
__energy__(HOST_MEMORY) 
#if defined(ENABLE_DEVICE)
__energy__(DEVICE_MEMORY) 
#endif

/*
  // vHS_impl
  template<MEMORY_SPACE M> 
  auto HamiltonianOperations<M>::vHS_impl(nda::MemoryMatrix auto& X, double dt)
  {
    return std::visit([&](auto&& s) { return s.vHS(X,dt); }, var);
  }
#define __vHS__(M) \
  template memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS_impl(memory::array_view<M, ComplexType,2>&,double); \
  template memory::buffered_array<M,ComplexType,4> HamiltonianOperations<M>::vHS_impl(memory::array_view<M, ComplexType,2,nda::C_layout>&,double); 
__vHS__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__vHS__(DEVICE_MEMORY)
#endif
*/

  // vbias_impl
  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::vbias_impl(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    std::visit([&](auto&& s) { s.vbias(G,v,dt); }, var);
  }
#define __vbias__(M) \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2>const&,memory::array_view<M,ComplexType,2>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2,nda::C_layout>const&,memory::array_view<M,ComplexType,2>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2>const&,memory::array_view<M,ComplexType,2,nda::C_layout>&,double); \
  template void HamiltonianOperations<M>::vbias_impl(memory::array_view<M,ComplexType const,2,nda::C_layout>const&,memory::array_view<M,ComplexType,2,nda::C_layout>&,double); 
__vbias__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__vbias__(DEVICE_MEMORY)
#endif

  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::energy_fullG_impl(nda::MemoryArrayOfRank<2> auto& E,
                                                   nda::MemoryArrayOfRank<2> auto const& G,
                                                   bool addH1,
                                                   bool addEJ,
                                                   bool addEXX)
  {
    std::visit([&](auto&& a) { a.energy_fullG(E, G, addH1, addEJ, addEXX); }, var);
  }

#define __energy_fullG__(M) \
  template void HamiltonianOperations<M>::energy_fullG_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2>const&,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_fullG_impl(memory::array_view<M,ComplexType,2>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_fullG_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2>const&,bool,bool,bool); \
  template void HamiltonianOperations<M>::energy_fullG_impl(memory::array_view<M,ComplexType,2,nda::C_layout>&, memory::array_view<M,const ComplexType,2,nda::C_layout>const&,bool,bool,bool);
__energy_fullG__(HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__energy_fullG__(DEVICE_MEMORY)
#endif

  // update_potential
  template<MEMORY_SPACE M>
  void HamiltonianOperations<M>::update_potentials_impl(double dt, nda::MemoryVector auto const& nMF, nda::MemoryVector auto& vMF, bool natural_shift)
  {
    std::visit([&](auto&& s) { s.update_potentials(dt,nMF,vMF,natural_shift); }, var);
  }

#define __update_potentials__(M1,M2) \
  template void HamiltonianOperations<M1>::update_potentials_impl(double,memory::array_view<M2,ComplexType const,1>const&,memory::array_view<M1,ComplexType,1>&,bool); \
  template void HamiltonianOperations<M1>::update_potentials_impl(double,memory::array_view<M2,ComplexType const,1>const&,memory::array_view<M1,ComplexType,1,nda::C_layout>&,bool); \
  template void HamiltonianOperations<M1>::update_potentials_impl(double,memory::array_view<M2,ComplexType const,1,nda::C_layout>const&,memory::array_view<M1,ComplexType,1>&,bool); \
  template void HamiltonianOperations<M1>::update_potentials_impl(double,memory::array_view<M2,ComplexType const,1,nda::C_layout>const&,memory::array_view<M1,ComplexType,1,nda::C_layout>&,bool); 
__update_potentials__(HOST_MEMORY,HOST_MEMORY)
#if defined(ENABLE_DEVICE)
__update_potentials__(DEVICE_MEMORY,DEVICE_MEMORY)
__update_potentials__(DEVICE_MEMORY,HOST_MEMORY)
#endif

  // accessors
  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_cholesky_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  int HamiltonianOperations<M>::number_of_ke_vectors() const
  {
    return std::visit([&](auto&& a) { return a.number_of_ke_vectors(); }, var);
  }

  template<MEMORY_SPACE M>
  std::tuple<int,int> HamiltonianOperations<M>::vHS_dims() const
  {
    return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
  }

  template<MEMORY_SPACE M>
  HamiltonianTypes HamiltonianOperations<M>::getHamType() const
  {
    return std::visit([&](auto&& a) { return a.getHamType(); }, var);
  }

  template<MEMORY_SPACE M>
  nda::array<int,1> HamiltonianOperations<M>::getFieldTypes() const
  {
    return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
  }

  template<MEMORY_SPACE M>
  bool HamiltonianOperations<M>::has_fullG_vbias() const
  {
    return std::visit([&](auto&& a) { return a.has_fullG_vbias(); }, var);
  }

  template int HamiltonianOperations<HOST_MEMORY>::number_of_cholesky_vectors() const;
  template int HamiltonianOperations<HOST_MEMORY>::number_of_ke_vectors() const;
  template std::tuple<int,int> HamiltonianOperations<HOST_MEMORY>::vHS_dims() const;
  template HamiltonianTypes HamiltonianOperations<HOST_MEMORY>::getHamType() const;
  template nda::array<int,1> HamiltonianOperations<HOST_MEMORY>::getFieldTypes() const;
  template bool HamiltonianOperations<HOST_MEMORY>::has_fullG_vbias() const;

#if defined(ENABLE_DEVICE)
  template int HamiltonianOperations<DEVICE_MEMORY>::number_of_cholesky_vectors() const;
  template int HamiltonianOperations<DEVICE_MEMORY>::number_of_ke_vectors() const;
  template std::tuple<int,int> HamiltonianOperations<DEVICE_MEMORY>::vHS_dims() const;
  template HamiltonianTypes HamiltonianOperations<DEVICE_MEMORY>::getHamType() const;
  template nda::array<int,1> HamiltonianOperations<DEVICE_MEMORY>::getFieldTypes() const;
  template bool HamiltonianOperations<DEVICE_MEMORY>::has_fullG_vbias() const;
#endif

} // namespace afqmc

} // namespace sfqmc

