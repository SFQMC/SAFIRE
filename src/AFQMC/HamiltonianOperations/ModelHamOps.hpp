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

#include "config.h"
#include "AFQMC/config.h"
#include "arch/arch.h"
#include "nda/nda.hpp"                                 
#include "nda/tensor.hpp"
#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "utilities/check_strides.hpp"
#include "utilities/memory_utils.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/nda_functions.hpp"

#include "AFQMC/HamiltonianOperations/ModelComponents/SparseEnergy.hpp"
#include "AFQMC/HamiltonianOperations/ModelComponents/ModelComponent.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, bool REAL>
class ModelHamOps
{
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;
  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:
  static const HamiltonianTypes HamOpType = ModelHamiltonian; 
  HamiltonianTypes getHamType() const { return ModelHamiltonian; }

  ModelHamOps() {};

  ModelHamOps(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
              WALKER_TYPES type,
              int nel_up, int nel_dn,
              memory::const_shared_array<MEM,ComplexType,4>&& psi_,
              SparseEnergy<MEM,REAL>&& et_,
              std::vector<ModelComponent<MEM,REAL>>&& h_,
              nda::MemoryVector auto&& n2ij_
             )
      : mpi(_mpi),
        walker_type(type),
        nel{nel_up,nel_dn},
        PsiC(std::move(psi_)),
        ET(std::move(et_)),
        Hams(std::move(h_)),
        spvHS(0),
        n2IJ(std::move(n2ij_)),
        n2IJ_dev(n2IJ),
        nIJ_first_beta(n2IJ.extent(0)),  // default value for noncollinear
        n2IJ_vHS_dev(n2IJ_dev)           // default value for collinear
  {
    num_ke_vectors=0;
    nCV=0;
    for(auto const& v: Hams) {
      field_ranges.emplace_back(nda::range(nCV,nCV+v.number_of_cholesky_vectors())); 
      ke_pos.emplace_back(nda::range(num_ke_vectors,num_ke_vectors+v.number_of_ke_vectors())); 
      num_ke_vectors += v.number_of_ke_vectors();
      nCV += v.number_of_cholesky_vectors();
    }

    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int M     = PsiC.extent(2);
    int NMO   = M/npol;
    if( walker_type == COLLINEAR) {
      for(int n=0; n<n2IJ.size(); ++n) {
        if( n2IJ[n]/M >= NMO ) {
          nIJ_first_beta=n;
          break;
        }
      }
    }
 
    // setup n2IJ_vHS_dev, only needed for NONCOLLINEAR 
    // for COLLINEAR, n2IJ_vHS == n2IJ, so no need to update 
    if(walker_type == NONCOLLINEAR) {
      nda::array<long,1> n2IJ_vHS_h(n2IJ);
      for(auto& v : n2IJ_vHS_h) {
        long In = long(v/M);
        long Jn = long(v%M);
        v = In*NMO + (Jn%NMO);  
      }
      n2IJ_vHS_dev() = n2IJ_vHS_h();
    }
    mpi->comm.barrier();
  
  }

  nda::array<ComplexType,3> getOneBodyPropagatorMatrix(double dt,
                                                       nda::MemoryVector auto const& vMF)
  {
    utils::check(vMF.size() == number_of_cholesky_vectors(), "Size mismatch"); 
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;

    nda::array<ComplexType, 3> H1(nspin, npol*NMO, npol*NMO);
    H1() = ComplexType(0.0);

    // accumulate terms from 1body
    ET.addOneBodyPropagatorMatrix(H1,dt);

    for(int i=0; i<Hams.size(); i++) 
      Hams[i].addOneBodyPropagatorMatrix(H1,dt, vMF(field_ranges[i]), n2IJ);

    return H1;
  }

  void runtime_optimization(nda::MemoryArrayOfRank<2> auto const& G)
  {  
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = G.extent(0);

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO},G.data());

    // determines sparse_G_eval 
    find_alg(G3d);
  } 

  nda::array<int,1> getFieldTypes() const {
    int nvc = number_of_cholesky_vectors();
    nda::array<int,1> v(nvc);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].getFieldTypes(v(field_ranges[i])); 
    return v;
  }
 
  void update_potentials(double dt, nda::MemoryVector auto const& nMF_, nda::MemoryVector auto&& vMF_, bool natural_shift)
  {
    // work with host copies
    auto nMF = nda::to_host(nMF_); 
    nda::array<ComplexType,1> vMF(vMF_.extent(0),ComplexType(0.0));
    std::unordered_map<long, int> IJ2n;
    if(IJ2n.size()==0) {
      IJ2n.reserve(n2IJ.size());
      for(int n=0; n<n2IJ.size(); n++)
        IJ2n.insert(std::make_pair(n2IJ[n], n));
    } 
    for(int i=0; i<Hams.size(); i++) 
      Hams[i].update(dt,nMF,n2IJ,IJ2n,vMF(field_ranges[i]),natural_shift);
    // copy result 
    vMF_() = vMF();
  }

  void energy(nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    using nda::range;
    auto all = range::all;
    int npol  = (walker_type == NONCOLLINEAR) ? 2 : 1;
    int nspin = (walker_type == COLLINEAR) ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = G.extent(0);
    
    E() = ComplexType(0.0);
    utils::check(idet >= 0 and idet < PsiC.extent(0), "idet out of bounds:{} ndet:{}",idet, PsiC.extent(0));

    // ET.get_n2IJ() runs over [0,M^2) in COLLINEAR case 
    int nIJ(ET.get_n2IJ().extent(0));
    bool allocate_EJn (addEJ and walker_type==COLLINEAR);
    memory::buffered_array<MEM,ComplexType,3> EJn(nspin, (allocate_EJn?nIJ:1), (allocate_EJn?nwalk:1)); 

    utils::check(G.is_contiguous(), "Layout mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO},G.data());

    EJn() = ComplexType(0.0);
    
    // generate GIJ with custom mapping
    if( sparse_G_eval ) {
      for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) {
        auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
        auto GIJ = getGIJ_for_energy(idet,is,Gwaj);
        ET.accumulate_energy(is, E, GIJ, EJn(is,all,all), addE1, addEJ, addEXX); 
      }
    } else {
      // generate full G
      auto ET_n2IJ = ET.get_n2IJ_dev();
      memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);
      for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) {
        auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
        auto Gfull = getGFull_for_energy(idet,is,Gwaj);
        // B[n][:] = A[ I[n] ][:] 
        nda::copy_select(false, 0, ET_n2IJ, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
        ET.accumulate_energy(is, E, GIJ, EJn(is,all,all), addE1, addEJ, addEXX); 
      }
    }

    // opposite spin EJ contribution
    if(addEJ and walker_type == COLLINEAR)
      nda::tensor::contract(ComplexType(1.0), EJn(0,all,all), "iw", EJn(1,all,all), "iw",
                            ComplexType(1.0), E(all,2), "w");
  }

  void energy(SpinTypes spin_component,
              nda::MemoryArrayOfRank<2> auto && E,
              nda::MemoryArrayOfRank<2> auto const& G,
              int idet,
              nda::MemoryArrayOfRank<2> auto && EJn,
              bool addE1  = true,
              bool addEJ  = true,
              bool addEXX = true)
  {
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int ispin = ( spin_component==Alpha ? 0 : 1 );
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = G.extent(0);
    int nIJ(ET.get_n2IJ().extent(0));
    
    utils::check(idet >= 0 and idet < PsiC.extent(0), "idet out of bounds:{} ndet:{}",idet, PsiC.extent(0));

    utils::check(G.is_contiguous(), "Layout mismatch");
    utils::check(G.shape() == std::array<long,2>{nwalk,nel[ispin]*npol*NMO}, "Size mismatch");
    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nel[ispin],npol*NMO},G.data());

    E() = ComplexType(0.0);
    bool allocate_EJn (addEJ and walker_type==COLLINEAR);
    if(allocate_EJn) {
      utils::check(EJn.shape() == std::array<long,2>{nwalk,nIJ}, "Size mismatch");
      EJn() = ComplexType(0.0);
    }

    memory::buffered_array<MEM,ComplexType,2> J((allocate_EJn ? nIJ : 0),nwalk); 
    if( sparse_G_eval ) {
      // generate GIJ with custom mapping
      auto GIJ = getGIJ_for_energy(idet,ispin,G3d);
      ET.accumulate_energy(ispin, E, GIJ, J, addE1, addEJ, addEXX); 
    } else {
      auto ET_n2IJ = ET.get_n2IJ_dev();
      auto Gfull = getGFull_for_energy(idet,ispin,G3d);
      memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);
      // B[n][:] = A[ I[n] ][:] 
      nda::copy_select(false, 0, ET_n2IJ, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
      ET.accumulate_energy(ispin, E, GIJ, J, addE1, addEJ, addEXX);
    }
    if(allocate_EJn) {
      if constexpr (MEM==HOST_MEMORY) {
        EJn() = nda::transpose(J);
      } else {
        nda::tensor::add(ComplexType(1.0),J,"nw",ComplexType(1.0),EJn,"wn");
      }
    }
  }

  template<class... Args>
  void ph_reference_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_reference_energy not implemented yet. ");
  }

  template<class... Args>
  void ph_excited_energy([[maybe_unused]] Args&&... args)
  {
    APP_ABORT(" Error: ph_excited_energy not implemented yet. ");
  }

  auto vHS(nda::MemoryMatrix auto&& X, double dt)
  {
    constexpr MEMORY_SPACE MEM_X = memory::get_memory_space<decltype(X)>();
    static_assert(MEM == MEM_X, "Memory space mismatch");
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = X.extent(0);
    int nIJ = n2IJ.size();
    // sanity checks!
    utils::check(X.extent(1) == nCV, "Size mismatch");

    memory::buffered_array<MEM_X,ComplexType,2> vIJ(nwalk, nIJ);
    vIJ() = ComplexType(0.0);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(X(nda::range::all,field_ranges[i]), vIJ, dt);

    memory::buffered_array<MEM,ComplexType,4> v(nspin,nwalk,npol*NMO,NMO);
    v() = ComplexType(0.0);

    memory::buffered_array<MEM,ComplexType,4> v_(nwalk,nspin,npol*NMO,NMO);
    auto v2d = nda::reshape(v_,std::array<long,2>{nwalk,nspin*npol*NMO*NMO});
    v_() = ComplexType(0.0);
    // B[:][I[n]] += A[:][n] 
    nda::copy_select(true, 1, n2IJ_vHS_dev, ComplexType(1.0), vIJ, ComplexType(0.0), v2d);
    nda::tensor::add(ComplexType(1.0),v_,"wsij",ComplexType(0.0),v,"swij");

    return v;
  }

  auto vHS_sparse(nda::MemoryArrayOfRank<2> auto const& X, double dt)
  {
    using nda::range;
    auto all = range::all;
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = X.extent(0);
    int nIJ = n2IJ.size();
    // sanity checks!
    utils::check(X.extent(1) == nCV, "Size mismatch");

    if( spvHS.extent(0) != nspin or
        spvHS(0).extent(0) != nwalk*npol*NMO or spvHS(0).extent(1) != nwalk*npol*NMO or 
        spvHS(nspin-1).extent(0) != nwalk*npol*NMO or spvHS(nspin-1).extent(1) != nwalk*npol*NMO )
        make_csr_vHS(nwalk);
    utils::check(spvHS(0).extent(0) == nwalk*npol*NMO and spvHS(0).extent(1) == nwalk*npol*NMO, "Size mismatch");
    utils::check(spvHS(nspin-1).extent(0) == nwalk*npol*NMO and spvHS(nspin-1).extent(1) == nwalk*npol*NMO, "Size mismatch");
    if(walker_type == COLLINEAR) {
      utils::check(spvHS(0).capacity() == nwalk*nIJ_first_beta, "Size mismatch");
      utils::check(spvHS(1).capacity() == nwalk*(nIJ-nIJ_first_beta), "Size mismatch");
    } else if(walker_type == NONCOLLINEAR) {
      utils::check(spvHS(0).capacity() == nwalk*nIJ, "Size mismatch");
    }

    memory::buffered_array<MEM,ComplexType,2> vIJ(nwalk, nIJ);
    vIJ() = ComplexType(0.0);
    for(int i=0; i<Hams.size(); i++)
      Hams[i].vHS(X(nda::range::all,field_ranges[i]), vIJ, dt);

    auto vals_up = nda::reshape(spvHS(0).values(), std::array<long,2>{nwalk,nIJ_first_beta});
    vals_up() = vIJ(all,range(nIJ_first_beta));
    if(walker_type == COLLINEAR) {
      auto vals_dn = nda::reshape(spvHS(1).values(), std::array<long,2>{nwalk,nIJ-nIJ_first_beta});
      vals_dn() = vIJ(all,range(nIJ_first_beta,nIJ));
    }

    return spvHS(); 
  }

  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto& v, double dt)
  {
    auto all = nda::range::all;
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int NMO   = PsiC.extent(2) / npol;
    int nwalk = v.extent(0);
    int nci = PsiC.extent(0); 
    int nIJ = n2IJ.size(); 
    int nelec = nel[0]+nel[1];
    utils::check(v.extent(1) == nCV, "Size mismatch");

    memory::array_view<MEM,const ComplexType,3> G3d(std::array<long,3>{nwalk,nelec,npol*NMO},G.data());
    memory::buffered_array<MEM,ComplexType,2> GIJ(nwalk, nIJ);
    GIJ() = ComplexType(0.0);

    if( nci == 1 ) {
      if( sparse_G_eval ) {
         getGIJ_for_vbias(G3d,GIJ);
      } else {
        // generate full G
        auto Gfull = getGFull(0,G3d);

        // B[:][n] = A[:][ I[n] ]
        nda::copy_select(false, 1, n2IJ_dev, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
      }
      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, v(all,field_ranges[i]), dt); 
    } else {
      // if nci > 1, we expect [...][nwalk]
      utils::check(G.shape() == std::array<long,2>{nwalk, nspin*npol*NMO*npol*NMO}, "Size mismatch");
      // B[:][n] = A[:][ I[n] ]
      nda::copy_select(false, 1, n2IJ_dev, ComplexType(1.0), G, ComplexType(0.0), GIJ);

      for(int i=0; i<Hams.size(); i++) 
        Hams[i].vbias(GIJ, v(all,field_ranges[i]), dt);
    }
  }

  std::tuple<int,int> vHS_dims() const {
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    return std::make_tuple(nspin,npol);
  }
  int number_of_cholesky_vectors() const { return nCV; }
  int number_of_ke_vectors() const { return ET.get_n2IJ().extent(0); }

  nda::array<ComplexType, 2> getHSPotentials()
  { return nda::array<ComplexType, 2>{}; }

private:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type = UNDEFINED_WALKER_TYPE;

  int nel[2];

  // (conjugated) orbital Matrix
  // This should be in node memory
  // PsiC(ndets,npin,i,a) = std::conj( PsiTrial(ndets,npin,i,a) )  
  memory::const_shared_array<MEM,ComplexType,4> PsiC;

  // One body Hamiltonian
  SparseEnergy<MEM,REAL> ET; 

  // list of ModelComponents
  // By convention, Hams[0] is a 1-body term.
  // All others (n>0) must be interacting terms. 
  std::vector<ModelComponent<MEM,REAL>> Hams;

  // csr_matrix matrix that will store vHS_sparse if needed
  nda::array<csrMat<ComplexType>,1> spvHS;

  // Vector with an ordered list of all IJ terms ( c^I cJ ) present in any term in the Hamiltonian,
  // n2IJ[n] = (I + si*NMO) * (npol * NMO) + J 
  // I/J are in range [0,NMO] for CLOSED/COLLINEAR and [0,2*NMO] for NONCOLLINEAR
  // only same spin blocks are present, consistent with the factorization of the coulomb 
  // e.g. no spin-orbit terms in the factorized pieces 
  nda::array<long,1> n2IJ;
  memory::array<MEM,long,1> n2IJ_dev;

  // first index of n2IJ of the spin=1 block. For NONCOLLINEAR, this should be equal to n2IJ.size() 
  int nIJ_first_beta;

  // n2IJ array, but with the index mapping consistent with the format of vHS
  // Notice that n2IJ is compatible with the Green's function indexing  
  memory::array<MEM,long,1> n2IJ_vHS_dev;

  std::vector<nda::range> field_ranges;
  std::vector<nda::range> ke_pos;

  // container for vHS in case vHS is requested in sparse form
  // sparse_vHS_t sparse_vHS;
  // sparse_vHS should be a csr_matrix in device memory. 

  int num_ke_vectors = 0;
  int nCV = 0;  

  // Use sparse/dense GIJ evaluation, default choice is true, adjusted at runtime
  bool sparse_G_eval = true;

  auto getGFull_for_energy(int idet, int ispin, nda::MemoryArrayOfRank<3> auto const& Gc) 
  {
    using nda::range;
    auto all = range::all;
    int npol   = walker_type == NONCOLLINEAR ? 2 : 1;
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[ispin],npol*NMO}, "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> Gfull(npol * NMO * npol * NMO, nwalk);
    auto Gfull_3d = nda::reshape(Gfull, std::array<long,3>{npol * NMO, npol * NMO, nwalk});

    // Empty spin sector (e.g. COLLINEAR with ndown==0): the sum over occupied
    // orbitals is empty, and the contraction below would be zero-sized.
    if(nel[ispin] == 0) {
      Gfull() = 0.0;
      return Gfull;
    }

    auto psi = PsiC()(idet,ispin,all,range(nel[ispin]));
    if constexpr (MEM==HOST_MEMORY) {
      memory::buffered_array<MEM,ComplexType,2> G_(npol * NMO, npol * NMO);
      auto Gfull_2d = nda::reshape(Gfull, std::array<long,2>{npol * NMO * npol * NMO, nwalk});
      for(int iw=0; iw<nwalk; ++iw) { 
        nda::blas::gemm(psi,Gc(iw,all,all),G_);
        Gfull_2d(all,iw) = nda::flatten(G_); 
      }
    } else {
      nda::tensor::contract(psi,"ia",Gc,"waj",Gfull_3d(all,all,all),"ijw");
    }
    return Gfull;
  }

  auto getGFull(int idet, nda::MemoryArrayOfRank<3> auto const& Gc)
  {
    using nda::range;
    auto all = range::all;
    int npol   = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin  = walker_type == COLLINEAR ? 2 : 1;
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    memory::buffered_array<MEM,ComplexType,2> Gfull(nwalk, nspin * npol * NMO * npol * NMO);
    auto Gfull_4d = nda::reshape(Gfull, std::array<long,4>{nwalk, nspin, npol * NMO, npol * NMO});

    // Empty beta sector (ndown==0): its block stays zero, the products below
    // would be zero-sized.
    bool empty_beta = (walker_type == COLLINEAR and nel[1] == 0);
    if(empty_beta) {
      Gfull() = 0.0;
    }

    if constexpr (MEM==HOST_MEMORY) {
      auto psi = PsiC()(idet,0,all,range(nel[0]));
      int n0 = 0;
      for(int iw=0; iw<nwalk; ++iw)
        nda::blas::gemm(psi,Gc(iw,range(n0,n0+nel[0]),all),Gfull_4d(iw,0,all,all));
      if( walker_type == COLLINEAR and not empty_beta ) {
        auto psi_dn = PsiC()(idet,1,all,range(nel[1]));
        n0 = nel[0];
        for(int iw=0; iw<nwalk; ++iw)
          nda::blas::gemm(psi_dn,Gc(iw,range(n0,n0+nel[1]),all),Gfull_4d(iw,1,all,all));
      }
    } else {
      auto psi = PsiC()(idet,0,all,range(nel[0]));
      int n0 = 0;
      nda::tensor::contract(psi,"ia",Gc(all,range(n0,n0+nel[0]),all),"waj",Gfull_4d(all,0,all,all),"wij");
      if( walker_type == COLLINEAR and not empty_beta ) {
        auto psi_dn = PsiC()(idet,1,all,range(nel[1]));
        n0 = nel[0];
        nda::tensor::contract(psi_dn,"ia",Gc(all,range(n0,n0+nel[1]),all),"waj",Gfull_4d(all,1,all,all),"wij");
      }
    }
    return Gfull;
  }

  auto getGIJ_for_energy(int idet, int ispin, nda::MemoryArrayOfRank<3> auto const& Gc)
  {
    using nda::range;
    auto all = range::all;
    int npol   = walker_type == NONCOLLINEAR ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[ispin],npol*NMO}, "Shape mismatch");
    auto ET_n2IJ = ET.get_n2IJ();
    int nIJ = ET_n2IJ.extent(0);
    memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);

    // Empty spin sector (e.g. COLLINEAR with ndown==0): the sum over occupied
    // orbitals is empty, and the work arrays below would be zero-sized.
    if(nel[ispin] == 0) {
      GIJ() = 0.0;
      return GIJ;
    }

    auto psi = PsiC()(idet,ispin,all,range(nel[ispin]));

    if constexpr (MEM==HOST_MEMORY) {
      //C[w][n] = sum_a psi[ I[n] ][a] Gc[w][a][ J[n] ]
      for(int n=0; n<nIJ; ++n) {
        int In = int(ET_n2IJ(n)/M);
        int Jn = int(ET_n2IJ(n)%M);
        nda::tensor::contract(psi(In,all),"a",Gc(all,all,Jn),"wa",GIJ(n,all),"w"); 
      }
    } else {
      // batched gemm (gemv), with transposed Gc 
      memory::buffered_array<MEM,ComplexType,3> Gt(npol*NMO,nel[ispin],nwalk);
      nda::tensor::assign(Gc,"waj",Gt,"jaw");
      using At = decltype(psi(range(0,1),all));
      using Bt = decltype(Gt(0,all,all));
      using Ct = decltype(GIJ(range(0,1),all));
      std::vector<At> vA; vA.reserve(nIJ);
      std::vector<Bt> vB; vB.reserve(nIJ);
      std::vector<Ct> vC; vC.reserve(nIJ);
      for(int n=0; n<nIJ; ++n) {
        int In = int(ET_n2IJ(n)/M);
        int Jn = int(ET_n2IJ(n)%M);
        vA.emplace_back(psi(range(In,In+1),all));
        vB.emplace_back(Gt(Jn,all,all));
        vC.emplace_back(GIJ(range(n,n+1),all));
      }
      nda::blas::gemm_batch<false>(ComplexType(1.0),vA,vB,ComplexType(0.0),vC);
    }

    return GIJ;
  }

  void getGIJ_for_vbias(nda::MemoryArrayOfRank<3> auto const& Gc, nda::MemoryMatrix auto && GIJ)
  {     
    using nda::range;
    auto all = range::all;
    int npol   = walker_type == NONCOLLINEAR ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nwalk  = Gc.extent(0);
    utils::check( Gc.shape() == std::array<long,3>{nwalk,nel[0]+nel[1],npol*NMO}, "Shape mismatch");
    int nIJ = n2IJ.extent(0);
    utils::check( GIJ.shape() == std::array<long,2>{nwalk,nIJ}, "Shape mismatch");

    // Empty beta sector (ndown==0): the beta IJ terms are all zero, the
    // products below would be zero-sized.
    bool empty_beta = (walker_type == COLLINEAR and nel[1] == 0);

    if constexpr (MEM==HOST_MEMORY) {
      //C[w][n] = sum_a psi[ I[n] ][a] Gwaj[w][a][ J[n] ]
      {
        auto psi = PsiC()(0,0,all,range(nel[0]));
        auto Gwaj = Gc(all,range(nel[0]),all);
        for(int n=0; n<nIJ_first_beta; ++n) {
          int In = int(n2IJ(n)/M);
          int Jn = int(n2IJ(n)%M);
          nda::tensor::contract(psi(In,all),"a",Gwaj(all,all,Jn),"wa",GIJ(all,n),"w");
        }
      }
      if(empty_beta) {
        GIJ(all,range(nIJ_first_beta,nIJ)) = 0.0;
      } else if(walker_type == COLLINEAR) {
        auto psi = PsiC()(0,1,all,range(nel[1]));
        auto Gwaj = Gc(all,range(nel[0],nel[0]+nel[1]),all);
        for(int n=nIJ_first_beta; n<nIJ; ++n) {
          int In = int(n2IJ(n)/M) - NMO;
          int Jn = int(n2IJ(n)%M);
          nda::tensor::contract(psi(In,all),"a",Gwaj(all,all,Jn),"wa",GIJ(all,n),"w");
        }
      }
    } else {
      // batched gemm (gemv), with transposed Gc
      memory::buffered_array<MEM,ComplexType,3> Gt(npol*NMO,nel[0]+nel[1],nwalk);
      memory::buffered_array<MEM,ComplexType,2> C(nIJ,nwalk);
      nda::tensor::assign(Gc,"waj",Gt,"jaw");
      auto _psi_ = PsiC()(0,0,all,range(nel[0]));
      auto _Gjaw_ = Gt(all,range(nel[0]),all);
      using At = decltype(_psi_(range(0,1),all));
      using Bt = decltype(_Gjaw_(0,all,all));
      using Ct = decltype(C(range(0,1),all));
      std::vector<At> vA; vA.reserve(nIJ);
      std::vector<Bt> vB; vB.reserve(nIJ);
      std::vector<Ct> vC; vC.reserve(nIJ);
      {
        auto psi = PsiC()(0,0,all,range(nel[0]));
        auto Gjaw = Gt(all,range(nel[0]),all);
        for(int n=0; n<nIJ_first_beta; ++n) {
          int In = int(n2IJ(n)/M);
          int Jn = int(n2IJ(n)%M);
          vA.emplace_back(psi(range(In,In+1),all));
          vB.emplace_back(Gjaw(Jn,all,all));
          vC.emplace_back(C(range(n,n+1),all));
        }
      }
      if(empty_beta) {
        C(range(nIJ_first_beta,nIJ),all) = 0.0;
      } else if(walker_type == COLLINEAR) {
        auto psi = PsiC()(0,1,all,range(nel[1]));
        auto Gjaw = Gt(all,range(nel[0],nel[0]+nel[1]),all);
        for(int n=nIJ_first_beta; n<nIJ; ++n) {
          int In = int(n2IJ(n)/M) - NMO;
          int Jn = int(n2IJ(n)%M);
          vA.emplace_back(psi(range(In,In+1),all));
          vB.emplace_back(Gjaw(Jn,all,all));
          vC.emplace_back(C(range(n,n+1),all));
        }
      }
      nda::blas::gemm_batch<false>(ComplexType(1.0),vA,vB,ComplexType(0.0),vC);
      nda::tensor::assign(C,"iw",GIJ,"wi");
    }
  }

  void make_csr_vHS(int nwalk) 
  {
    using nda::range;
    auto all = range::all;
    int nspin  = walker_type == COLLINEAR ? 2 : 1;
    int npol   = walker_type == NONCOLLINEAR ? 2 : 1;
    int M      = PsiC.extent(2);
    int NMO    = PsiC.extent(2) / npol;
    int nIJ = n2IJ.extent(0);

    spvHS.resize(nspin);
    
    nda::array<int,1> cnt(nspin*M,0);
    for( auto IJ: n2IJ ) cnt( IJ/M )++;
 
    { // alpha
      nda::array<int,2> nnz(nwalk,M);
      for(int w=0; w<nwalk; w++)
        nnz(w,all) = cnt(range(M));

      math::sparse::csr_matrix<ComplexType, HOST_MEMORY, int, int>
          v_h({nwalk*M,nwalk*M},nda::flatten(nnz));

      for( auto IJ: n2IJ(range(nIJ_first_beta)) ) {
        int I(int(IJ/M)), J(int(IJ%M));
        for(int w=0; w<nwalk; w++)
          v_h.emplace_back( { w*M + I, w*M + J }, 0.0 );
      }
      spvHS(0) = std::move(v_h);
    }
    if(walker_type == COLLINEAR) {
      nda::array<int,2> nnz(nwalk,NMO);
      for(int w=0; w<nwalk; w++)
        nnz(w,all) = cnt(range(NMO,2*NMO));

      // need to setup in host 
      math::sparse::csr_matrix<ComplexType, HOST_MEMORY, int, int>
          v_h({nwalk*NMO,nwalk*NMO},nda::flatten(nnz));
       
      for( auto IJ: n2IJ(range(nIJ_first_beta,nIJ)) ) {
        int I(int(IJ/M)), J(int(IJ%M));
        for(int w=0; w<nwalk; w++)
          v_h.emplace_back( { w*NMO + (I-NMO), w*NMO + J }, 0.0 );
      }
      spvHS(1) = std::move(v_h);
    }

  }

  /// Determine algorithm for evaluation of GIJ. Using energy workflow as benchmark 
  void find_alg(nda::MemoryArrayOfRank<3> auto const& G3d) 
  {
    using nda::range;
    auto all = range::all;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int nwalk = G3d.extent(0);

    {
      int nIJ(ET.get_n2IJ().extent(0));
      {
        for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) { 
          auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
          auto GIJ = getGIJ_for_energy(0,is,Gwaj);
        }
      }
   
      {
        // generate full G
        auto ET_n2IJ = ET.get_n2IJ_dev();
        memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);
        for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) {
          auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
          auto Gfull = getGFull_for_energy(0,is,Gwaj);
          // B[n][:] = A[ I[n] ][:] 
          nda::copy_select(false, 0, ET_n2IJ, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
        }
      }

    }

    // now resize allocators to remove allocation overhead. This will fail if
    // buffer space was allocated previous to calling this routine!!!
    utils::resize_nda_static_allocator();

    double t_sparse(0.0), t_gfull(0.0);
    {
      int nIJ(ET.get_n2IJ().extent(0));
      t_sparse = utils::function_timer([&] {
        for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) {
          auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
          auto GIJ = getGIJ_for_energy(0,is,Gwaj);
        }
      });

      t_gfull = utils::function_timer([&] {
        // generate full G
        auto ET_n2IJ = ET.get_n2IJ_dev();
        memory::buffered_array<MEM,ComplexType,2> GIJ(nIJ, nwalk);
        for(int is=0, n0=0; is<nspin; is++, n0+=nel[0] ) {
          auto Gwaj = G3d(all,range(n0,n0+nel[is]),all);
          auto Gfull = getGFull_for_energy(0,is,Gwaj);
          // B[n][:] = A[ I[n] ][:]
          nda::copy_select(false, 0, ET_n2IJ, ComplexType(1.0), Gfull, ComplexType(0.0), GIJ);
        }
      });
    }

    app_log(2,"Runtime optimization of Model Hamiltonian Operations:");
    app_log(2,"  - G full: {} s",t_gfull);
    app_log(2,"  - G sparse: {} s",t_sparse);
    sparse_G_eval = (t_gfull > t_sparse);
    if(sparse_G_eval)
      app_log(2, "Using sparse algorithm to evaluate GIJ in Model Hamiltonian Operations.");
    else
      app_log(2, "Using dense algorithm to evaluate GIJ in Model Hamiltonian Operations.");
  }

};

} // namespace afqmc

} // namespace sfqmc

