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
#include "AFQMC/Wavefunctions/Wavefunction.hpp"

// The dispatch lives here so that the NOMSD/PHMSD/NOMSD_FT bodies are instantiated
// once, in this TU, instead of in every TU that calls into Wavefunction.

namespace sfqmc {
namespace afqmc {

template<MEMORY_SPACE MEM>
MEMORY_SPACE Wavefunction<MEM>::get_memory_space() const {
  return std::visit([&](auto&& a) { return a.get_memory_space(); }, var);
}

template<MEMORY_SPACE MEM>
int Wavefunction<MEM>::number_of_cholesky_vectors() const {
  return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::runtime_optimization(WalkerSet<MEM>& wset) {
  std::visit([&](auto&& a) { a.runtime_optimization(wset); }, var);
}

template<MEMORY_SPACE MEM>
WALKER_TYPES Wavefunction<MEM>::getWalkerType() const {
  return std::visit([&](auto&& a) { return a.getWalkerType(); }, var);
}

template<MEMORY_SPACE MEM>
bool Wavefunction<MEM>::isFiniteTemperature() const {
  return std::visit([&](auto&& a) { return a.isFiniteTemperature(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::vMF(memory::array_view<MEM,ComplexType,1> v, double dt) {
  std::visit([&](auto&& a) { a.vMF(v, dt); }, var);
}

template<MEMORY_SPACE MEM>
memory::const_shared_array<HOST_MEMORY,ComplexType,3> Wavefunction<MEM>::G_MF() {
  return std::visit([&](auto&& a) { return a.G_MF(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v,
                              double dt) {
  std::visit([&](auto&& a) { a.vbias(wset, v, dt); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::vbias(WalkerSet<MEM>& wset, memory::array_view<MEM,ComplexType,2> v,
                              double dt, int nt) {
  std::visit([&](auto&& a) { a.vbias(wset, v, dt, nt); }, var);
}

template<MEMORY_SPACE MEM>
memory::buffered_array<MEM,ComplexType,4> Wavefunction<MEM>::vHS(
    memory::array_view<MEM,ComplexType,2> X, double dt) {
  return std::visit([&](auto&& a) { return a.vHS(X, dt); }, var);
}

template<MEMORY_SPACE MEM>
nda::array_view<math::sparse::csr_matrix<ComplexType,MEM,int,int>,1>
    Wavefunction<MEM>::vHS_sparse(memory::array_view<MEM,const ComplexType,2> X, double dt) {
  return std::visit([&](auto&& a) { return a.vHS_sparse(X, dt); }, var);
}

template<MEMORY_SPACE MEM>
std::tuple<int,int> Wavefunction<MEM>::vHS_dims() const {
  return std::visit([&](auto&& a) { return a.vHS_dims(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Energy(WalkerSet<MEM>& wset) {
  std::visit([&](auto&& a) { a.Energy(wset); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Energy(WalkerSet<MEM>& wset, int nt) {
  std::visit([&](auto&& a) { a.Energy(wset, nt); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Energy(WalkerSet<MEM> const& wset,
                               memory::array_view<MEM,ComplexType,2> E,
                               memory::array_view<MEM,ComplexType,1> Ov) {
  std::visit([&](auto&& a) { a.Energy(wset, E, Ov); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Energy(WalkerSet<MEM> const& wset,
                               memory::array_view<MEM,ComplexType,2> E,
                               memory::array_view<MEM,ComplexType,1> Ov, int nt) {
  std::visit([&](auto&& a) { a.Energy(wset, E, Ov, nt); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::MixedDensityMatrix(WalkerSet<MEM> const& wset,
                                           memory::array_view<MEM,ComplexType,2> G, bool compact) {
  std::visit([&](auto&& a) { a.MixedDensityMatrix(wset, G, compact); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Log_Overlap(WalkerSet<MEM>& wset) {
  std::visit([&](auto&& a) { a.Log_Overlap(wset); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::Log_Overlap(WalkerSet<MEM> const& wset,
                                    memory::array_view<MEM,ComplexType,1> Ov) {
  std::visit([&](auto&& a) { a.Log_Overlap(wset, Ov); }, var);
}

template<MEMORY_SPACE MEM>
int Wavefunction<MEM>::total_number_of_references() const {
  return std::visit([&](auto&& a) { return a.total_number_of_references(); }, var);
}

template<MEMORY_SPACE MEM>
int Wavefunction<MEM>::getNMO() const {
  return std::visit([&](auto&& a) { return a.getNMO(); }, var);
}

template<MEMORY_SPACE MEM>
ComplexType Wavefunction<MEM>::getReferenceWeight(int i) {
  return std::visit([&](auto&& a) { return a.getReferenceWeight(i); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs) {
  std::visit([&](auto&& a) { a.getReferences(Refs); }, var);
}

template<MEMORY_SPACE MEM>
HamiltonianTypes Wavefunction<MEM>::getHamType() const {
  return std::visit([&](auto&& a) { return a.getHamType(); }, var);
}

template<MEMORY_SPACE MEM>
nda::array<int,1> Wavefunction<MEM>::getFieldTypes() {
  return std::visit([&](auto&& a) { return a.getFieldTypes(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::update_potentials(double dt,
                       memory::array_view<HOST_MEMORY,const ComplexType,1> nMF,
                       memory::array_view<MEM,ComplexType,1> vMF, bool natural_shift) {
  std::visit([&](auto&& a) { a.update_potentials(dt, nMF, vMF, natural_shift); }, var);
}

template<MEMORY_SPACE MEM>
nda::array<ComplexType,3> Wavefunction<MEM>::getOneBodyPropagatorMatrix(double dt,
                       memory::array_view<HOST_MEMORY,const ComplexType,1> vMF) {
  return std::visit([&](auto&& a) { return a.getOneBodyPropagatorMatrix(dt, vMF); }, var);
}

template<MEMORY_SPACE MEM>
ComplexType Wavefunction<MEM>::getLogScale(SpinTypes s) {
  return std::visit([&](auto&& a) { return a.getLogScale(s); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::resetLogScale() {
  std::visit([&](auto&& a) { a.resetLogScale(); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::setLogPT0(memory::array_view<MEM,ComplexType,1> v) {
  std::visit([&](auto&& a) { a.setLogPT0(v); }, var);
}

template<MEMORY_SPACE MEM>
memory::array<MEM,ComplexType,1> Wavefunction<MEM>::getLogPT0() {
  using R = memory::array<MEM,ComplexType,1>;
  return std::visit([&](auto&& a) -> R { return a.getLogPT0(); }, var);
}

template class Wavefunction<HOST_MEMORY>;
#if defined(ENABLE_DEVICE)
template class Wavefunction<DEVICE_MEMORY>;
#endif

} // namespace afqmc

} // namespace sfqmc
