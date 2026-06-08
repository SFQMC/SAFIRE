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

#ifndef SFQMC_AFQMC_STOCHASTICWFN_H
#define SFQMC_AFQMC_STOCHASTICWFN_H

#include "io/ptree/ptree_utilities.hpp"
#include "AFQMC/config.h"
#include "AFQMC/Wavefunctions/NOMSD.hpp"

namespace sfqmc
{
namespace afqmc
{
/*
 * Stochastic trial wavefunction wrapper.
 * Currently delegates all operations to an internal NOMSD object.
 */
template<bool MP, class devPsiT>
class StochasticWfn : public AFQMCInfo
{
  NOMSD<MP, devPsiT> nomsd_;

public:
  template<class MType>
  StochasticWfn(AFQMCInfo& info,
                ptree pt_in,
                afqmc::TaskGroup_& tg_,
                SlaterDetOperations&& sdet_,
                HamiltonianOperations<MP>&& hop_,
                std::vector<ComplexType>&& ci_,
                std::vector<MType>&& orbs_,
                WALKER_TYPES wlk,
                ComplexType nce,
                [[maybe_unused]] int targetNW = 1)
      : AFQMCInfo(info),
        nomsd_(info, pt_in, tg_, std::move(sdet_), std::move(hop_), std::move(ci_), std::move(orbs_), wlk, nce,
               targetNW)
  {
    ptree pt = interpret_inputs(pt_in);
    app_log(2, "\nStochasticWfn input:\n{}\n", io::to_string(pt));
  }

  static ptree interpret_inputs(const ptree pt0)
  {
    ptree pt1 = NOMSD<MP, devPsiT>::interpret_inputs(pt0);
    std::unordered_set<std::string> pass_through_keys = {
        "system",
        "name",
        "ndets_to_read",
        "restart_file",
        "filename",
        "stochastic",
    };
    io::compare_known_keys("Stochastic trial wavefunction (StochasticWfn)", pt1, pt0, pass_through_keys);
    return pt1;
  }

  ~StochasticWfn() = default;

  StochasticWfn(StochasticWfn const& other)            = delete;
  StochasticWfn& operator=(StochasticWfn const& other) = delete;
  StochasticWfn(StochasticWfn&& other)                   = default;
  StochasticWfn& operator=(StochasticWfn&& other)        = delete;

  NOMSD<MP, devPsiT>& inner_wfn() { return nomsd_; }
  NOMSD<MP, devPsiT> const& inner_wfn() const { return nomsd_; }

  int local_number_of_cholesky_vectors() const { return nomsd_.local_number_of_cholesky_vectors(); }
  int global_number_of_cholesky_vectors() const { return nomsd_.global_number_of_cholesky_vectors(); }
  int global_origin_cholesky_vector() const { return nomsd_.global_origin_cholesky_vector(); }
  bool distribution_over_cholesky_vectors() const { return nomsd_.distribution_over_cholesky_vectors(); }
  bool spin_dependent_vHS() const { return nomsd_.spin_dependent_vHS(); }

  int size_of_G_for_vbias() const { return nomsd_.size_of_G_for_vbias(); }

  bool transposed_G_for_vbias() const { return nomsd_.transposed_G_for_vbias(); }
  bool transposed_G_for_E() const { return nomsd_.transposed_G_for_E(); }
  bool transposed_vHS() const { return nomsd_.transposed_vHS(); }
  WALKER_TYPES getWalkerType() const { return nomsd_.getWalkerType(); }

  template<class Vec>
  void vMF(Vec&& v, double dt);

  template<class Mat>
  void G_MF(Mat&& G);

  SlaterDetOperations* getSlaterDetOperations() { return nomsd_.getSlaterDetOperations(); }

  template<class... Args>
  void generalizedFockMatrix(Args&&... args)
  {
    nomsd_.generalizedFockMatrix(std::forward<Args>(args)...);
  }

  HamiltonianTypes getHamType() { return nomsd_.getHamType(); }

  template<class... Args>
  void getFieldTypes(Args&&... args)
  {
    nomsd_.getFieldTypes(std::forward<Args>(args)...);
  }

  template<class... Args>
  void update_potentials(Args&&... args)
  {
    nomsd_.update_potentials(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto getOneBodyPropagatorMatrix(Args&&... args)
  {
    return nomsd_.getOneBodyPropagatorMatrix(std::forward<Args>(args)...);
  }

  template<class... Args>
  auto vHS_sparse(Args&&... args)
  {
    return nomsd_.vHS_sparse(std::forward<Args>(args)...);
  }

  template<class MatG, class MatA>
  void vbias(const MatG& G, MatA&& v, double dt, double a = 1.0)
  {
    nomsd_.vbias(G, std::forward<MatA>(v), dt, a);
  }

  template<class MatX, class MatA>
  void vHS(MatX&& X, MatA&& v, double dt, double a = 1.0)
  {
    nomsd_.vHS(std::forward<MatX>(X), std::forward<MatA>(v), dt, a);
  }

  template<class WlkSet>
  void Energy(WlkSet& wset)
  {
    nomsd_.Energy(wset);
  }

  template<class WlkSet, class Mat, class TVec>
  void Energy(const WlkSet& wset, Mat&& E, TVec&& Ov)
  {
    nomsd_.Energy(wset, std::forward<Mat>(E), std::forward<TVec>(Ov));
  }

  template<class WlkSet, class MatG>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, bool compact = true, bool transpose = false)
  {
    nomsd_.MixedDensityMatrix(wset, std::forward<MatG>(G), compact, transpose);
  }

  template<class WlkSet, class MatG, class TVec>
  void MixedDensityMatrix(const WlkSet& wset, MatG&& G, TVec&& Ov, bool compact = true, bool transpose = false)
  {
    nomsd_.MixedDensityMatrix(wset, std::forward<MatG>(G), std::forward<TVec>(Ov), compact, transpose);
  }

  template<class WlkSet, class MatA, class MatB, class MatG, class TVec>
  void DensityMatrix(const WlkSet& wset,
                     MatA&& RefA,
                     MatB&& RefB,
                     MatG&& G,
                     TVec&& Ov,
                     bool herm,
                     bool compact,
                     bool transposed)
  {
    nomsd_.DensityMatrix(wset, std::forward<MatA>(RefA), std::forward<MatB>(RefB), std::forward<MatG>(G),
                        std::forward<TVec>(Ov), herm, compact, transposed);
  }

  template<class WlkSet, class MatG>
  void MixedDensityMatrix_for_vbias(const WlkSet& wset, MatG&& G)
  {
    nomsd_.MixedDensityMatrix_for_vbias(wset, std::forward<MatG>(G));
  }

  template<class WlkSet, class TVec>
  void Overlap(const WlkSet& wset, TVec&& Ov)
  {
    nomsd_.Overlap(wset, std::forward<TVec>(Ov));
  }

  template<class WlkSet>
  void Overlap(WlkSet& wset)
  {
    nomsd_.Overlap(wset);
  }

  template<class... Args>
  void accumulate_estimators(Args&&... args)
  {
    nomsd_.accumulate_estimators(std::forward<Args>(args)...);
  }

  int number_of_references_for_back_propagation() const
  {
    return nomsd_.number_of_references_for_back_propagation();
  }

  ComplexType getReferenceWeight(int i) const { return nomsd_.getReferenceWeight(i); }

  template<class Mat, class Ptr = ComplexType*>
  void getReferencesForBackPropagation(Mat&& A)
  {
    nomsd_.getReferencesForBackPropagation(std::forward<Mat>(A));
  }
};

} // namespace afqmc

} // namespace sfqmc

#include "AFQMC/Wavefunctions/StochasticWfn.icc"

#endif
