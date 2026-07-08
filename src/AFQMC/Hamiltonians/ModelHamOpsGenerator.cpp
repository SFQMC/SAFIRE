/*
 * This file is distributed under the Apache License, Version 2.0 License.
 * See LICENSE file in top directory for details.
 *
 * Copyright (c) 2021-2025 The Simons Foundation, Inc.
 *
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 */

#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#include <utility>
#include <vector>

#include "config.h"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"

#include "AFQMC/config.h"
#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

#include "ModelHamOpsGenerator.h"
//#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"
//#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
//#include "AFQMC/HamiltonianOperations/ModelHamOps.hpp"

namespace sfqmc
{
namespace afqmc
{

namespace detail
{
template<class csrM>
csrM spin_to_walker_type(int NMO, WALKER_TYPES type, std::string stype, csrM& hij)
{
  std::string base_error("Error in ModelHamOpsGenerator::spin_to_walker_type(...): ");
  if(stype == "closed") {
    utils::check_shape(hij, "one_body::tij", NMO, NMO);
    if(type == CLOSED) {
      utils::check(false," Error: Model Hamiltonians not allowed with CLOSED walkers. ");
    } else if(type == COLLINEAR) {
      return math::sparse::closed_to_collinear(hij);
    } else if(type == NONCOLLINEAR) {
      return math::sparse::closed_to_noncollinear(hij);
    } else {
      utils::check(false,base_error + " Bad Walker Type!");
    }
  } else if(stype == "collinear") {
    utils::check_shape(hij, "one_body::tij", 2*NMO, NMO);
    if(type == CLOSED) {
      utils::check(false," Error: Model Hamiltonians not allowed with CLOSED walkers. ");
    } else if(type == COLLINEAR) {
      return hij;
    } else if(type == NONCOLLINEAR) {
      return math::sparse::collinear_to_noncollinear(hij);
    } else {
      utils::check(false,base_error + " Bad Walker Type!");
    }
  } else if(stype == "noncollinear") {
    utils::check_shape(hij, "one_body::tij", 2*NMO, 2*NMO);
    if(type == NONCOLLINEAR) {
      return hij;
    } else { 
      utils::check(false,base_error + " Bad Walker Type!");
    }
  } else {
    utils::check(false,base_error + " Unknown spin_type: " + stype);
  }
  return hij;
}
}

template<MEMORY_SPACE MEM, bool REAL> HamiltonianOperations<MEM> 
ModelHamOpsGenerator::getHamiltonianOperations_impl(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  using nda::range;
  auto all = range::all;
  using ValueType = typename std::conditional_t<REAL, RealType, ComplexType>;
  using csrMat    = math::sparse::csr_matrix<ValueType, HOST_MEMORY, int, int>;   

  if(type == CLOSED)
  utils::check(type != CLOSED and type != FULLYPOLARIZED, " Error in ModelHamOpsGenerator::getHamiltonianOperations: CLOSED or FULLYPOLARIZED walker types not allowed with Model Hamiltonians. "); 

  // make sure there is at least 1 one-body hamiltonian in the components
  bool one_body_term(false);

  std::string base_error("Error in ModelHamOpsGenerator::getHamiltonianOperations(): ");

  int ndet   = PsiT.extent(0); 
  int nspin  = ((type == COLLINEAR) ? 2 : 1);
  int npol   = ((type == NONCOLLINEAR) ? 2 : 1);
  int NMO = PsiT(0, 0).extent(1)/npol;
  int nel_up = PsiT(0,0).extent(0); 
  int nel_dn = ( type == COLLINEAR ? PsiT(0,1).extent(0) : 0 ); 
  int nspin_in_PsiT = PsiT.extent(1);
  utils::check(nspin_in_PsiT==1 or nspin_in_PsiT==nspin, base_error + "Invalid PsiT shape");

  // generate trial wavefunctions in appropriate form
  // ModelHamOps expects a vector of PsiC(i,a) = (psiT(a,i)) (complex of trial wfn Slater Matrix)
  // PsiT is replicated on every rank, so the determinants can be filled rank-by-rank.
  auto PsiC = memory::share_from_ranks<MEM,ComplexType,4,1>(*mpi,
      {ndet,nspin,npol*NMO,nel_up},
      [&](std::array<long,1> idx, auto&& block) {
        auto [id] = idx;
        utils::check_shape(PsiT(id,0), "PsiT", nel_up, npol*NMO);
        block(0,all,all) = math::sparse::to_array<'T'>(PsiT(id,0));
        if(type == COLLINEAR) {
          utils::check_shape(PsiT(id,nspin_in_PsiT-1), "PsiT", nel_dn, npol*NMO);
          block(1,all,range(nel_dn)) = math::sparse::to_array<'T'>(PsiT(id,nspin_in_PsiT-1));
        }
      });

  // everyone reads for simplicity, change to single reader if it becomes a problem
  h5::file file = h5::file(fileName,'r'); 
  h5::group grp = h5::group(file).open_group("Hamiltonian");

  std::vector<long> Idata(8);
  h5::read(grp,"dims",Idata);
  ValueType E0;
  {
    std::vector<RealType> E_(2);
// MAM: dataset is currently long, fix!!! 
    //h5::h5_read(grp,"Energies",E_);
    E0 = 0.0; //E_[0] + E_[1];
  }

  h5::group mgrp = grp.open_group("ModelHamiltonian"); 
  
  long num_components(0);
  h5::read(mgrp,"number_of_components",num_components);
  // can check file and determine maximum_connections directly!
  long maximum_connections(12);
  h5::read(mgrp,"maximum_connectivity",maximum_connections);
  
  std::vector<ModelComponent<MEM,REAL>> Hams;
  Hams.reserve(num_components);
  csrMat hij({0,0});

  // accumulating terms 
  /*
   * Map:
   *   collect_U  ( opposite spin from 0-M, same spin from M-2M )  
   *   0: continuous charge 
   *   1: continuous spin 
   *   2: discrete charge 
   *   3: discrete spin 
   *   collect_J
   *   0: continuous charge 
   *   1: continuous spin 
   *   2: empty-container, for call to addComponent with discrete 
   */
  std::vector<csrMat> collect_U;  
  std::vector<csrMat> collect_J;  
  collect_U.reserve(4);
  collect_J.reserve(3);
  for(int i=0; i<4; i++) 
    collect_U.emplace_back(csrMat({2*NMO, NMO},maximum_connections));
  // [2] is a place-holder for the empty case
  for(int i=0; i<3; i++)
    collect_J.emplace_back(csrMat({NMO, NMO},maximum_connections));

  for(int n=0; n<num_components; n++)  {
    h5::group gn = mgrp.open_group("ModelComponent_"+std::to_string(n));
    
    std::string model_type("dummy");
    h5::h5_read(gn,"model_type",model_type); 
    std::transform(model_type.begin(), model_type.end(), model_type.begin(), (int (*)(int))tolower);

    if( model_type == "one_body" ) 
    {
      utils::check(not one_body_term, base_error + " Multiple one_body components defined.");  
      one_body_term=true;

      h5::group dn = gn.open_group("tij");
      auto tij = math::sparse::HDF2CSR<ValueType,HOST_MEMORY,int,int>(dn);

      std::string stype;
      h5::h5_read(gn,"spin_type",stype);
      std::transform(stype.begin(), stype.end(), stype.begin(), (int (*)(int))tolower);

      // returns a sparse matrix with the 1-body hamiltonian consistent with type
      hij = detail::spin_to_walker_type(NMO, type, stype, tij);
    }
    else if( model_type == "hubbard_u" ) 
    {
      h5::group dn = gn.open_group("Uij");
      csrMat Uij =  math::sparse::HDF2CSR<ValueType,HOST_MEMORY,int,int>(dn); 

      utils::check((Uij.extent(0) == NMO or Uij.extent(0) == 2*NMO) and 
                   Uij.extent(1) == NMO, 
                   base_error + " Found Hubbard_U model with inconsistent dimensions. ");
      // for safety
      utils::check(Uij.nnz() != 0, base_error + " Found empty Hubbard_U model. "); 

      std::string hst_type; 
      h5::h5_read(gn,"hst_type",hst_type);
      std::transform(hst_type.begin(), hst_type.end(), hst_type.begin(), (int (*)(int))tolower);

      int where_(-1);
      if(hst_type == "continuous_charge") where_ = 0; 
      else if(hst_type == "continuous_spin") where_ = 1;
      else if(hst_type == "discrete_charge") where_ = 2; 
      else if(hst_type == "discrete_spin") where_ = 3;
      else
        utils::check(false,base_error + " Unknown hst_type: " + hst_type);

      app_log(1, "Hamiltonian component {} is using Hubbard-Stratonovich transformation type {} or Hubbard U", n, hst_type);

      {
        // doing this "by hand" to impose condition i>j
        // opposite spin (i<=j)
        // MAM: ignoring lower diagonal terms seems to confuse users, consider reading everything
        //      and just transposing the terms here when you write them in collect_U/J
        auto vals = Uij.values();
        auto cols = Uij.columns();
        for( int r=0; r<NMO; ++r) {
          for(long i=Uij.row_begin(r); i<Uij.row_end(r); ++i) {
            auto u_ = vals(i);
            int j = cols(i);
            if( std::abs(u_) < 1e-6 or r>j ) {
              app_warning("Ignoring opposite spin Uij, i>j: i:{}, j:{}, U:{}",r,j,u_);
              continue;
	    }
            collect_U[where_].add( {r, j}, u_ );
          }
        }
        for( int r=NMO; r<Uij.extent(0); ++r) {
          for(long i=Uij.row_begin(r); i<Uij.row_end(r); ++i) {
            auto u_ = vals(i);
            int j = cols(i);
            if( std::abs(u_) < 1e-6 or (r-NMO)>=j ) { 
              app_warning("Ignoring same spin Uij, i=>j: i:{}, j:{}, U:{}",r,j,u_);
              continue;
            }
            collect_U[where_].add( {r, j}, u_ );
          }
        }
      }
    }
    else if( model_type == "hubbard_j" )  
    { 
      h5::group dn = gn.open_group("Jij");
      csrMat Jij =  math::sparse::HDF2CSR<ValueType,HOST_MEMORY,int,int>(dn);
      
      utils::check_shape(Jij, "Jij", NMO, NMO);
      // for safety
      utils::check(Jij.nnz() != 0, base_error + " Found empty Hubbard_U model. ");

      std::string hst_type;
      h5::h5_read(gn,"hst_type",hst_type);
      std::transform(hst_type.begin(), hst_type.end(), hst_type.begin(), (int (*)(int))tolower);
      
      int where_(-1);
      if(hst_type == "continuous_charge") where_ = 0; 
      else if(hst_type == "continuous_spin") where_ = 1; 
      else if(hst_type == "discrete_charge") { 
        utils::check(false,base_error + " Discrete HS transformations not allowed with Hubbard_J");
      } else if(hst_type == "discrete_spin") {
        utils::check(false,base_error + " Discrete HS transformations not allowed with Hubbard_J");
      } else 
        utils::check(false,base_error + " Unknown hst_type: " + hst_type);
    
      app_log(1, "Hamiltonian component {} is using Hubbard-Stratonovich transformation type {} for Hubbard J", n, hst_type);

      { 
        // doing this "by hand" to impose condition i>j
        // opposite spin (i<=j)
        auto vals = Jij.values();
        auto cols = Jij.columns();
        for( int r=0; r<NMO; ++r) {
          for(long i=Jij.row_begin(r); i<Jij.row_end(r); ++i) {
            auto v_ = vals(i);
            int j = cols(i);
            if( std::abs(v_) < 1e-6 or r>=j ) continue;
            collect_J[where_].add( {r, j}, v_ );
          }
        }
      }
    }
    else  
      utils::check(false,base_error + " Unknown model type: " + model_type);

  }
  utils::check(one_body_term, base_error + " Missing one_body component in ModelHamiltonian.");  
  utils::check(hij.nnz() != 0, 
               base_error + " Something went wrong, empty one_body component.");  

  // combine all U/J matrices for energy evaluation. 
  csrMat combined_U({2*NMO,NMO}, maximum_connections);
  csrMat combined_J({NMO,NMO}, maximum_connections);

  {
    for( auto& v: collect_U ) 
      math::sparse::accumulate(ValueType(1.0), v, combined_U);
    for( auto& v: collect_J ) 
      math::sparse::accumulate(ValueType(1.0), v, combined_J);
  }  
  // make compact, since some libraries require it!
  combined_U.remove_empty_spaces();  
  combined_J.remove_empty_spaces();  
  hij.remove_empty_spaces();

  // create energy evaluation model class
  SparseEnergy<MEM,REAL> ET( make_SparseEnergy<MEM,REAL>(mpi, type, hij, combined_U, combined_J, E0) );

  // Add Jij terms to Uij. 
  {
    auto vals = collect_J[0].values();
    auto cols = collect_J[0].columns();
    for( int r=0; r<NMO; ++r) 
      // terms go in same spin component, so shift row by NMO
      for(long i=collect_J[0].row_begin(r); i<collect_J[0].row_end(r); ++i) 
        collect_U[0].add( {r+NMO, cols(i)}, vals(i) );
  }
  {
    auto vals = collect_J[1].values();
    auto cols = collect_J[1].columns();
    for( int r=0; r<NMO; ++r) 
      // terms go in same spin component, so shift row by NMO
      for(long i=collect_J[1].row_begin(r); i<collect_J[1].row_end(r); ++i)
        collect_U[0].add( {r+NMO, cols(i)}, ValueType(-1.0)*vals(i) );
  }

  // generate "present" one-body terms
  using map_t = std::unordered_map<size_t, int>;
  nda::array<long,1> n2IJ( find_occupied_pairs(type, collect_U, collect_J) );
  map_t IJ2n;
  IJ2n.reserve(n2IJ.size()); 
  for(int n=0; n<n2IJ.size(); n++) 
    IJ2n.insert(std::make_pair(n2IJ[n], n));

  addComponent<MEM,REAL>( type, ContinuousChargePropagator, mpi, collect_U[0], 
		collect_J[0], Hams, n2IJ, IJ2n);
  addComponent<MEM,REAL>( type, ContinuousSpinPropagator, mpi, collect_U[1], 
		collect_J[1], Hams, n2IJ, IJ2n);
  // note: collect_J[2] should be empty
  utils::check(collect_J[2].nnz() == 0, "");
  addComponent<MEM,REAL>( type, DiscreteChargePropagator, mpi, collect_U[2], 
		collect_J[2], Hams, n2IJ, IJ2n);
  addComponent<MEM,REAL>( type, DiscreteSpinPropagator, mpi, collect_U[3], 
		collect_J[2], Hams, n2IJ, IJ2n);

  return HamiltonianOperations<MEM>(ModelHamOps<MEM,REAL>(mpi, type, nel_up, nel_dn,
       std::move(PsiC), std::move(ET), std::move(Hams), n2IJ)); 
}

template<MEMORY_SPACE MEM> HamiltonianOperations<MEM> 
ModelHamOpsGenerator::getHamiltonianOperations(WALKER_TYPES type,
                 std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                 nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  bool Real = true;
  if(mpi->comm.root()) { 
    // check file structure, check types
    std::string base_error("Error in ModelHamOpsGenerator::getHamiltonianOperations(): ");
    h5::file file(fileName,'r');
    h5::group grp(file);
    h5::group mgrp = grp.open_group("Hamiltonian").open_group("ModelHamiltonian");
    long num_components;
    h5::h5_read(mgrp,"number_of_components",num_components);
    for(int n=0; n<num_components; n++)  {
      h5::group gn = mgrp.open_group("ModelComponent_"+ std::to_string(n));

      std::string dset;
      std::string model_type("dummy");
      h5::h5_read(gn, "model_type", model_type);
      std::transform(model_type.begin(), model_type.end(), model_type.begin(), (int (*)(int))tolower);

      if( model_type == "one_body" ) { 
        utils::check(gn.has_key("tij"), "Missing dataset tij with model_type=one_body.");
        dset = "tij";
      } else if( model_type == "hubbard_u" ) {
        utils::check(gn.has_key("Uij"), "Missing dataset Uij with model_type=hubbard_u.");
        dset = "Uij";
      } else if( model_type == "hubbard_j" ) {
        utils::check(gn.has_key("Jij"), "Missing dataset Jij with model_type=hubbard_j.");
        dset = "Jij";
      } else
        utils::check(false, base_error + " Unknown model type: " + model_type);

      // Allowing mixed types
      h5::group dn = gn.open_group(dset);
      auto l = h5::array_interface::get_dataset_info(dn,"data_");
      utils::check((l.rank() == 1) or (l.rank() == 2), "Rank mismatch");
      if(l.has_complex_attribute or (l.rank() == 2)) Real = false;
    } // for(n)
  }
  mpi->comm.broadcast_n(&Real, 1, 0);

  if(Real)
    return getHamiltonianOperations_impl<MEM,true>(type, mpi, PsiT);
  else
    return getHamiltonianOperations_impl<MEM,false>(type, mpi, PsiT);
}   

template HamiltonianOperations<HOST_MEMORY> 
ModelHamOpsGenerator::getHamiltonianOperations<HOST_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<HOST_MEMORY>,2>const&);
#if defined(ENABLE_DEVICE)
template HamiltonianOperations<DEVICE_MEMORY>
ModelHamOpsGenerator::getHamiltonianOperations<DEVICE_MEMORY>(WALKER_TYPES,
     std::shared_ptr<utils::mpi_context_t<mpi3::communicator>>,
     nda::array<PsiT_Matrix<DEVICE_MEMORY>,2>const&);
#endif

} // namespace afqmc
} // namespace sfqmc

