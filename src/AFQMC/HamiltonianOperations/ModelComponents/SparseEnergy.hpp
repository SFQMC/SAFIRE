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

#pragma once

#include <vector>
#include <type_traits>

#include "config.h" // NOLINT(misc-include-cleaner)
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
  
#include "numerics/sparse/sparse.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, bool REAL>
class SparseEnergy
{
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:

  SparseEnergy(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
                          WALKER_TYPES type,
                          std::vector<csrMat<ValueType>> && hij_,
                          std::vector<csrMat<ValueType>> && vj_,
                          std::vector<csrMat<ValueType>> && vx_,
                          csrMat<ValueType> && u_,
                          csrMat<ValueType> && j_,
                          nda::MemoryVector auto && n2ij_,
                          ComplexType e0 = 0.0
                )
      : mpi(_mpi), 
        walker_type(type),
        n2IJ_host(std::move(n2ij_)),
        n2IJ_dev(n2IJ_host),
        E0(e0),
        hij(std::move(hij_)),
        SpVJ(std::move(vj_)),
        SpVX(std::move(vx_)),
        U(std::move(u_)),
        J(std::move(j_))
  {
    // expect hij as a sparse Matrix with a single row (to reuse csr_matrix class) 
    int nIJ = n2IJ_host.extent(0);
    int nspin = type == COLLINEAR ? 2 : 1;
    utils::check(SpVJ.size() == 1 + (nspin-1)*2, "Size mismatch");
    utils::check(SpVX.size() == 1 + (nspin-1), "Size mismatch");
    utils::check(hij[0].shape() == std::array<long,2>{1,nIJ}, "Size mismatch");
    utils::check(SpVJ[0].shape() == std::array<long,2>{nIJ,nIJ}, "Size mismatch");
    utils::check(SpVX[0].shape() == std::array<long,2>{nIJ,nIJ}, "Size mismatch");
    if(type == COLLINEAR) {
      utils::check(hij[1].shape() == std::array<long,2>{1,nIJ}, "Size mismatch");
      utils::check(SpVJ[1].shape() == std::array<long,2>{nIJ,nIJ}, "Size mismatch");
      utils::check(SpVJ[2].shape() == std::array<long,2>{nIJ,nIJ}, "Size mismatch");
      utils::check(SpVX[1].shape() == std::array<long,2>{nIJ,nIJ}, "Size mismatch");
    }
    for(int n=0; n<hij.size(); n++) 
      utils::check(hij[n].compact(), "Expect compact csr_matrix.");
    for(int n=0; n<SpVJ.size(); n++) { 
      utils::check(SpVJ[n].compact(), "Expect compact csr_matrix.");
      nnz_VJ[n]  = SpVJ[n].nnz();
    }
    for(int n=0; n<SpVX.size(); n++) {
      utils::check(SpVX[n].compact(), "Expect compact csr_matrix.");
      nnz_VXX[n] = SpVX[n].nnz();
    }
  }

  auto get_n2IJ() const { return n2IJ_host(); };
  auto get_n2IJ_dev() const { return n2IJ_dev(); }  

  void addOneBodyPropagatorMatrix(nda::array<ComplexType, 3> & H1, double dt)
  {
    using nda::range;
    auto all = range::all;
    int nspin  = walker_type == COLLINEAR ? 2 : 1;
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int NMO = H1.extent(1) / npol;
    int nIJ = n2IJ_host.extent(0);
    utils::check(H1.shape() == std::array<long,3>{nspin, npol*NMO, npol*NMO}, "Shape mismatch");

    auto nnz0 = hij[0].nnz(0);
    auto nnz1 = hij[nspin-1].nnz(0);
    if(nnz0+nnz1 == 0) return;

    auto H2d = nda::reshape(H1, std::array<long,2>{nspin,npol*NMO*npol*NMO});
    // alpha component
    {
      auto vals = nda::to_host(hij[0].values()(range(nIJ)));
      // hij should be compact
      nda::copy_select(true, n2IJ_host, ComplexType(dt), vals, ComplexType(1.0), H2d(0,all));
    }
    // beta component
    if(walker_type == COLLINEAR) {
      auto vals = nda::to_host(hij[1].values()(range(nIJ)));
      nda::copy_select(true, n2IJ_host, ComplexType(dt), vals, ComplexType(1.0), H2d(1,all));
    }
  }

  void accumulate_energy(int ispin,
              nda::MemoryMatrix auto&& E,
              nda::MemoryMatrix auto const& G,
              nda::MemoryMatrix auto && EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using nda::range;
    auto all = range::all;
    int nwalk  = G.extent(1);
    int nIJ = n2IJ_dev.extent(0);
    utils::check(E.shape() == std::array<long,2>{nwalk,3}, "Shape mismatch");
    utils::check(G.extent(0) == nIJ, "Size mismatch");

    if( addE1 ) {
      if(ispin == 0) 
        E(all,0) = E0;
      auto vals = hij[ispin].values()(range(nIJ));
      if constexpr (REAL) {
        auto G3d = memory::to_real_view(G);
        auto E3d = memory::to_real_view(E);
        nda::tensor::contract(ValueType(1.0),G3d,"nwc",vals,"n",ValueType(1.0),E3d(all,0,all),"wc");
      } else {
        nda::tensor::contract(ComplexType(1.0),G,"nw",vals,"n",ComplexType(1.0),E(all,0),"w");
      }
    }

    if( (not addEJ) and
        (not addEXX or nnz_VXX[ispin] == 0) )
      return;

    // working array
    memory::buffered_array<MEM,ComplexType,2> VG(nIJ,nwalk);

    if( addEJ ) { 
      if( nnz_VJ[ispin] > 0 ) {
        // VJ[nIJ,nIJ] * G[nw,nij] = VG[nw,nIJ] 
        math::sparse::csrmm<'N'>(SpVJ[ispin], G, VG); 
        // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
        nda::tensor::contract(ComplexType(1.0), G, "nw", VG, "nw", 
    		              ComplexType(1.0), E(all,2), "w");
      }	
      if(walker_type == COLLINEAR) {
        utils::check(EJn.shape() == G.shape(), "Size mismatch");
        if(ispin==0) {
          if( nnz_VJ[2] > 0 ) {
            // SpVJ[2] = VJ(beta,alpha) + T( VJ(alpha,beta) )
            math::sparse::csrmm<'N'>(SpVJ[2], G, EJn);
	  } else {
            EJn() = ComplexType(0.0);
	  }
        } else {
          EJn() = G();
	}
      }
    }

    if( addEXX and nnz_VXX[ispin] > 0 ) {
      math::sparse::csrmm<'N'>(SpVX[ispin], G, VG);
      // y[incy*n] = beta * y[incy*n] + alpha sum_m^{0,M} opA(A)[n,m] * opB(B)[n,m]  
      nda::tensor::contract(ComplexType(1.0), G, "nw", VG, "nw", 
  		            ComplexType(1.0), E(all,1), "w");
    }
  }

  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    utils::check(false," Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

private:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  /* defines the compact ordering */
  memory::array<HOST_MEMORY, int,1> n2IJ_host;  
  memory::array<MEM, int,1> n2IJ_dev;  

  ComplexType E0;

  int nnz_VJ[3];
  int nnz_VXX[2];

  // bare one body hamiltonian
  // [0]:(alpha,alpha), [1]:(beta,beta)
  std::vector<csrMat<ValueType>> hij;

  // coulomb 2-e matrix elements 
  // [0]:(alpha,alpha), [1]:(beta,beta), [2]:[ (beta,alpha) + T( (alpha,beta) ) ]
  std::vector<csrMat<ValueType>> SpVJ; 

  // exchange 2-e matrix elements 
  // [0]:(alpha,alpha), [1]:(beta,beta)
  std::vector<csrMat<ValueType>> SpVX; 

  // combined U matrices 
  csrMat<ValueType> U;

  // combined J matrices 
  csrMat<ValueType> J;

};

} // namespace afqmc

} // namespace sfqmc

