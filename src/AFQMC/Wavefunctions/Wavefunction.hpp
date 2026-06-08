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

#ifndef SFQMC_AFQMC_WAVEFUNCTION_HPP
#define SFQMC_AFQMC_WAVEFUNCTION_HPP


#include "AFQMC/config.h"

#include "AFQMC/SlaterDeterminantOperations/SlaterDetOperations.hpp"
#include "AFQMC/Wavefunctions/NOMSD.hpp"
#include "AFQMC/Wavefunctions/PHMSD.hpp"
#include "AFQMC/Wavefunctions/StochasticWfn.hpp"

namespace sfqmc
{
namespace afqmc
{
namespace dummy
{
/*
 * Empty class to avoid need for default constructed Wavefunctions.
 * Throws is any visitor is called. 
 */
class dummy_wavefunction
{
private:
  std::vector<ComplexType> ci;
  std::vector<PsiT_Matrix> orbs;

public:
  dummy_wavefunction(){};

  int size_of_G_for_vbias() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return 0;
  }
  int local_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return 0;
  }
  int global_number_of_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return 0;
  }
  int global_origin_cholesky_vector() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return 0;
  }
  int number_of_references_for_back_propagation() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return 0;
  }
  bool distribution_over_cholesky_vectors() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return false;
  }

  WALKER_TYPES getWalkerType() const { return UNDEFINED_WALKER_TYPE; }

  bool transposed_G_for_vbias() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return false;
  }

  bool transposed_G_for_E() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return false;
  }

  bool transposed_vHS() const
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return false;
  }

  template<class Vec>
  void vMF([[maybe_unused]] Vec&& v, [[maybe_unused]] double dt)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class MatG, class MatA>
  void vbias([[maybe_unused]] const MatG& G, [[maybe_unused]] MatA&& v, [[maybe_unused]] double dt, [[maybe_unused]] double a = 1.0)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class MatX, class MatA>
  void vHS([[maybe_unused]] MatX&& X, [[maybe_unused]] MatA&& v, [[maybe_unused]] double dt, [[maybe_unused]] double a = 1.0)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  void G_MF([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WSet>
  void Energy([[maybe_unused]] WSet& wset)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet, class Mat, class TVec>
  void Energy([[maybe_unused]] const WlkSet& wset, [[maybe_unused]] Mat&& E, [[maybe_unused]] TVec&& Ov)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet, class MatG>
  void MixedDensityMatrix([[maybe_unused]] const WlkSet& wset, [[maybe_unused]] MatG&& G, [[maybe_unused]] bool compact = true, [[maybe_unused]] bool transpose = false)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix([[maybe_unused]] const WlkSet& wset, [[maybe_unused]] MatG&& G, [[maybe_unused]] TVec&& Ov, [[maybe_unused]] bool compact = true, [[maybe_unused]] bool transpose = false)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias([[maybe_unused]] const WlkSet& wset, [[maybe_unused]] MatG&& G)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  void DensityMatrix ([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet, class TVec>
  void Overlap([[maybe_unused]] const WlkSet& wset, [[maybe_unused]] TVec&& Ov)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class WlkSet>
  void Overlap([[maybe_unused]] WlkSet& wset)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  void accumulate_estimators([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  ComplexType getReferenceWeight([[maybe_unused]] int i)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return ComplexType(0.0, 0.0);
  }

  template<class Mat>
  void getReferencesForBackPropagation([[maybe_unused]] Mat&& A)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  HamiltonianTypes getHamType()
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return UNKNOWN; 
  }

  template<class... Args>
  void getFieldTypes([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  void update_potentials([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
  }

  template<class... Args>
  multi::array<ComplexType, 2> getOneBodyPropagatorMatrix([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return multi::array<ComplexType, 2>{};
  }

  template<class... Args>
  std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*> vHS_sparse([[maybe_unused]] Args&&... args)
  {
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    dev_csr_Matrix<ComplexType> const* t(nullptr);
    return std::make_tuple(t,t);
  }

  SlaterDetOperations SDet;
  SlaterDetOperations* getSlaterDetOperations() { return std::addressof(SDet); }

  bool spin_dependent_vHS() const { 
    throw std::runtime_error("calling visitor on dummy_wavefunction object");
    return false;
  } 

};
} // namespace dummy

class Wavefunction : public boost::variant<dummy::dummy_wavefunction,
                                           NOMSD<true,local_csr_Matrix<ComplexType>>,
                                           NOMSD<false,local_csr_Matrix<ComplexType>>,
                                           NOMSD<true,ComplexMatrix<node_allocator<ComplexType>>>,
                                           NOMSD<false,ComplexMatrix<node_allocator<ComplexType>>>,
                                           StochasticWfn<true,local_csr_Matrix<ComplexType>>,
                                           StochasticWfn<false,local_csr_Matrix<ComplexType>>,
                                           StochasticWfn<true,ComplexMatrix<node_allocator<ComplexType>>>,
                                           StochasticWfn<false,ComplexMatrix<node_allocator<ComplexType>>>,
                                           PHMSD<true>,
                                           PHMSD<false>>
{
public:
  Wavefunction() { APP_ABORT(" Error: Reached default constructor of Wavefunction. "); }
  explicit Wavefunction(NOMSD<true,local_csr_Matrix<ComplexType>>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(NOMSD<true,local_csr_Matrix<ComplexType>> const& other) = delete;

  explicit Wavefunction(NOMSD<false,local_csr_Matrix<ComplexType>>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(NOMSD<false,local_csr_Matrix<ComplexType>> const& other) = delete;

  explicit Wavefunction(NOMSD<true,ComplexMatrix<node_allocator<ComplexType>>>&& other) : 
			variant(std::move(other)) {}
  explicit Wavefunction(NOMSD<true,ComplexMatrix<node_allocator<ComplexType>>> const& other) = delete;

  explicit Wavefunction(NOMSD<false,ComplexMatrix<node_allocator<ComplexType>>>&& other) : 
			variant(std::move(other)) {}
  explicit Wavefunction(NOMSD<false,ComplexMatrix<node_allocator<ComplexType>>> const& other) = delete;

  explicit Wavefunction(StochasticWfn<true,local_csr_Matrix<ComplexType>>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(StochasticWfn<true,local_csr_Matrix<ComplexType>> const& other) = delete;

  explicit Wavefunction(StochasticWfn<false,local_csr_Matrix<ComplexType>>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(StochasticWfn<false,local_csr_Matrix<ComplexType>> const& other) = delete;

  explicit Wavefunction(StochasticWfn<true,ComplexMatrix<node_allocator<ComplexType>>>&& other) :
                        variant(std::move(other)) {}
  explicit Wavefunction(StochasticWfn<true,ComplexMatrix<node_allocator<ComplexType>>> const& other) = delete;

  explicit Wavefunction(StochasticWfn<false,ComplexMatrix<node_allocator<ComplexType>>>&& other) :
                        variant(std::move(other)) {}
  explicit Wavefunction(StochasticWfn<false,ComplexMatrix<node_allocator<ComplexType>>> const& other) = delete;

  explicit Wavefunction(PHMSD<true>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(PHMSD<true> const& other) = delete;

  explicit Wavefunction(PHMSD<false>&& other) : variant(std::move(other)) {}
  explicit Wavefunction(PHMSD<false> const& other) = delete;

  Wavefunction(Wavefunction const& other) = delete;
  Wavefunction(Wavefunction&& other)      = default;

  Wavefunction& operator=(Wavefunction const& other) = delete;
  Wavefunction& operator=(Wavefunction&& other) = default;

  int size_of_G_for_vbias() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.size_of_G_for_vbias(); }, *this);
  }

  int local_number_of_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.local_number_of_cholesky_vectors(); }, *this);
  }

  int global_number_of_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.global_number_of_cholesky_vectors(); }, *this);
  }

  int global_origin_cholesky_vector() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.global_origin_cholesky_vector(); }, *this);
  }

  int number_of_references_for_back_propagation() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.number_of_references_for_back_propagation(); }, *this);
  }

  bool distribution_over_cholesky_vectors() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.distribution_over_cholesky_vectors(); }, *this);
  }

  bool transposed_G_for_vbias() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_G_for_vbias(); }, *this);
  }

  bool transposed_G_for_E() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_G_for_E(); }, *this);
  }

  bool transposed_vHS() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.transposed_vHS(); }, *this);
  }

  WALKER_TYPES getWalkerType() const
  {
    return boost::apply_visitor([&](auto&& a) { return a.getWalkerType(); }, *this);
  }

  template<class... Args>
  void vMF(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.vMF(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void G_MF(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.G_MF(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void vbias(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.vbias(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void vHS(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.vHS(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void Energy(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.Energy(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void DensityMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.DensityMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void MixedDensityMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.MixedDensityMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void MixedDensityMatrix_for_vbias(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.MixedDensityMatrix_for_vbias(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void Overlap(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.Overlap(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  ComplexType getReferenceWeight(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.getReferenceWeight(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void getReferencesForBackPropagation(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.getReferencesForBackPropagation(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.accumulate_estimators(std::forward<Args>(args)...); }, *this);
  }

  SlaterDetOperations* getSlaterDetOperations()
  {
    return boost::apply_visitor([&](auto&& a) { return a.getSlaterDetOperations(); }, *this);
  }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.generalizedFockMatrix(std::forward<Args>(args)...); }, *this);
  } 

  HamiltonianTypes getHamType() 
  {
    return boost::apply_visitor([&](auto&& a) { return a.getHamType(); }, *this);
  }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.getFieldTypes(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    boost::apply_visitor([&](auto&& a) { a.update_potentials(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  multi::array<ComplexType, 2> getOneBodyPropagatorMatrix(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.getOneBodyPropagatorMatrix(std::forward<Args>(args)...); }, *this);
  }

  template<class... Args>
  std::tuple<dev_csr_Matrix<ComplexType> const*, dev_csr_Matrix<ComplexType> const*> vHS_sparse(Args&&... args)
  {
    return boost::apply_visitor([&](auto&& a) { return a.vHS_sparse(std::forward<Args>(args)...); }, *this);
  }

  bool spin_dependent_vHS() const { 
    return boost::apply_visitor([&](auto&& a) { return a.spin_dependent_vHS(); }, *this);
  }

};

} // namespace afqmc

} // namespace sfqmc

#endif
