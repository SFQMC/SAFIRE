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
#include <boost/math/tools/roots.hpp>

#include "config.h" 
#include "AFQMC/config.h"
#include "utilities/check.hpp"
#include "utilities/check_shape.hpp"
#include "utilities/mpi_context.h"

#include "numerics/sparse/sparse.hpp"
#include "numerics/shared_array/const_shared_array.hpp"

namespace sfqmc
{
namespace afqmc
{

template<MEMORY_SPACE MEM, bool REAL>
class Discrete_GeneralUJ
{
  using ValueType     = typename std::conditional_t<REAL, RealType, ComplexType>;

  template<class T>
  using csrMat = math::sparse::csr_matrix<T, MEM, int, int>;

public:

  Discrete_GeneralUJ() {}

  Discrete_GeneralUJ(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi,
                     WALKER_TYPES type,
                     PropagatorTypes ptype,  
                     memory::const_shared_array<MEM,ComplexType,1>&& h0_,
                     math::sparse::CSRMatrix auto&& vn_,
                     math::sparse::CSRMatrix auto&& vnT_,
                     math::sparse::CSRMatrix auto&& u_,
                     bool shift_ = false,  
                     [[maybe_unused]] bool p_shift_ = false,
                     ComplexType e0 = 0
                    )
      : mpi(_mpi),
        walker_type(type),
        propg_type(ptype),
        nCV(0),
        shift_one_body_terms(shift_),
        E0(e0),
        h0(std::move(h0_)),
	U(std::move(u_)),
        SpVn(std::move(vn_)),
        SpVnT(std::move(vnT_))
  {
    utils::check((propg_type==DiscreteChargePropagator) or
                 (propg_type==DiscreteSpinPropagator), " Error: Wrong PropagatorTypes argument in Discrete_GeneralUJ. ");      
    nCV = SpVn.extent(1);
    utils::check(SpVn.extent(0) == SpVnT.extent(1), "Size mismatch");
    utils::check(SpVn.extent(1) == SpVnT.extent(0), "Size mismatch");
    utils::check(SpVn.extent(0) == h0.extent(0), "");
  }

  ~Discrete_GeneralUJ() {}

  Discrete_GeneralUJ(const Discrete_GeneralUJ<MEM,REAL>& other)            = default;
  Discrete_GeneralUJ& operator=(const Discrete_GeneralUJ<MEM,REAL>& other) = default;
  Discrete_GeneralUJ(Discrete_GeneralUJ<MEM,REAL>&& other)                 = default;
  Discrete_GeneralUJ& operator=(Discrete_GeneralUJ<MEM,REAL>&& other)      = default;

  /*
   * n2IJ maps an index in the ordering of the sparse structures to the ordering 
   * of H1 (the generic spin ordering of 1-body operators)
   */
  void addOneBodyPropagatorMatrix(nda::array<ComplexType,3> & H1, double dt,
                                  [[maybe_unused]] nda::array<ComplexType, 1> const& vMF,
                                  nda::array<long,1> const& n2IJ)
  {

    utils::check(initialized, " Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    int npol  = walker_type == NONCOLLINEAR ? 2 : 1;
    int nspin = walker_type == COLLINEAR ? 2 : 1;
    int NMO = H1.extent(1) / npol;

    utils::check(H1.shape() == std::array<long,3>{nspin, npol*NMO, npol*NMO}, "Shape mismatch");
    utils::check(n2IJ.extent(0) == h0.extent(0), "Size mismatch");
    utils::check(n2IJ.extent(0) == hMF.extent(0), "Size mismatch");

    // if shift_one_body_terms is false, both h0 and hMF are included in vHS
    if(shift_one_body_terms) { 
      auto H1d = nda::flatten(H1);
      auto h0_h = nda::to_host(h0());
      nda::copy_select(true, n2IJ, ComplexType(dt), h0_h, ComplexType(1.0), H1d);
      auto hMF_h = nda::to_host(hMF());
      nda::copy_select(true, n2IJ, ComplexType(dt), hMF_h, ComplexType(1.0), H1d);
    }  
  }

  void getFieldTypes(nda::MemoryVector auto && v) const {
    utils::check(v.size() == nCV, "Size mismatch");
    v() = int(propg_type);
  }

  template<class map_t>
  void update(double dt, nda::MemoryVector auto&& nI, nda::MemoryVector auto&& n2IJ, 
              map_t& IJ2n, nda::MemoryVector auto&& vMF, bool natural_shift)
  {
    // store dt and check?
    initialized = true; 
    setup_Vn_hmf(dt,nI,n2IJ,IJ2n,vMF,natural_shift);
  }

  // v(w,IJ) = sum_n Vn(IJ,n) X(w,n)
  void vHS(nda::MemoryArrayOfRank<2> auto const& X, nda::MemoryMatrix auto& v, double dt)
  {
    auto all = nda::range::all;
    utils::check(initialized," Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    utils::check(X.extent(0) == v.extent(0), "Size mismatch");
    utils::check(SpVn.extent(1) == X.extent(1), "Size mismatch");
    utils::check(SpVn.extent(0) == v.extent(1), "Size mismatch");
    utils::check(v.extent(1) == h0.extent(0), "Size mismatch");
    utils::check(v.extent(1) == hMF.extent(0), "Size mismatch");

    math::sparse::csrmm<'N'>(ComplexType(1.0), SpVn, nda::transpose(X), 
                             ComplexType(1.0), nda::transpose(v));

    if(shift_one_body_terms) return;
  
    // multiply by '-i' to compensate for factor of 'i' implicit in the propagator
    ComplexType ia(0.0, dt);

    // v(w,n) = v(w,n) + ia*h0(n);
    if constexpr (MEM==HOST_MEMORY){
      for(int iw=0; iw<v.extent(0); ++iw) v(iw,all) += ia*(h0()+hMF());     
    }
    else {
      //nda::tensor::add(ia,h0(),"i",ComplexType(1.0),v,"wi");
      //nda::tensor::add(ia,hMF(),"i",ComplexType(1.0),v,"wi");
      //FIX: need a better solution here
      for(int iw=0; iw<v.extent(0); ++iw){
        nda::tensor::add(ia,h0(),"i",ComplexType(1.0),v(iw,all),"i");
        nda::tensor::add(ia,hMF(),"i",ComplexType(1.0),v(iw,all),"i");
      }
    }
  }

  // v(w,n) = sum_IJ VnT(n,IJ) G(w,IJ)
  void vbias(nda::MemoryArrayOfRank<2> auto const& G, nda::MemoryArrayOfRank<2> auto&& v, double dt)
  {
    utils::check(initialized," Error: Using uninitialized Discrete_GeneralUJ object. Call update first."); 
    utils::check(SpVnT.extent(0) == v.extent(1), "Size mismatch");
    utils::check(SpVnT.extent(1) == G.extent(1), "Size mismatch");
    utils::check(G.extent(0) == v.extent(0), "Size mismatch");

    math::sparse::csrmm<'N'>(ComplexType(1.0), SpVnT, nda::transpose(G), 
                             ComplexType(1.0), nda::transpose(v));
  }

  template<class... Args>
  void generalizedFockMatrix([[maybe_unused]] Args&&... args)
  {
    utils::check(false," Error: generalizedFockMatrix not implemented for this hamiltonian.");
  }

  int number_of_ke_vectors() const { return nCV; }
  int number_of_cholesky_vectors() const { return nCV; }

private:

  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  WALKER_TYPES walker_type;

  PropagatorTypes propg_type;

  // this class unfortunately doesn't quite satisfy RAII, 
  // since it needs a timestep and on-site MF occupancies to setup properly.
  bool initialized = false;

  int nCV;
  // if shift_one_body_terms=true, h0 is added to the one body propagator.
  // otherwise it is added through vHS
  bool shift_one_body_terms = false;

  ComplexType E0 = 0.0;

  // All sparse matrices have a compact representation of IJ, as defined
  // by n2IJ in ModelHamOps. 

  // constant one-body term associated with the 
  // interacting term.
  memory::const_shared_array<MEM,ComplexType,1> h0;

  // 1-body part of MF subtraction. Not given by vMF in discrete case!!!
  // Empty until update() runs; rebuilt wholesale on every update.
  memory::const_shared_array<MEM,ComplexType,1> hMF;

  // need to keep a copy of the U matrix. Keeping on host memory.
  math::sparse::csr_matrix<ValueType,HOST_MEMORY,int,int> U;

  // HS operator 
  csrMat<ComplexType> SpVn;

  // transposed HS operator 
  csrMat<ComplexType> SpVnT;

  // HS operator and 1-body MF substraction terms depend on timestep
  // and on MF onsite occupations.    
  // follows addComponent from Hamiltonians/ModelHamOpsGenerator.icc, 
  // but uses correct prefactors, which are now functions of dt*U and nMFJ = <nI +- nJ>   
  // assumes nMF[i] = <c^{+}_i c_i>_MF is the mean-field site occupations. UHF convention for
  // spin ordering (e.g. all up, followed by all down).   
  // assuming onsite densities are real...
  // This is a collective call!
  template<class map_t>     
  void setup_Vn_hmf(double dt, nda::MemoryVector auto const& nMF, 
                    nda::MemoryVector auto const& n2IJ, map_t& IJ2n, 
                    nda::MemoryVector auto&& vMF, bool natural_shift)
  {
    // if dt == 0, it will mess up the sparsity structure below
    utils::check(dt > 0, "dt has to be positive");
    
    int NMO = U.extent(0) / 2;

// you also want a way to guarantee also in the cpu that the term already exists, 
// not being dynamically added. Add another version of += that works on gpus and that
// aborts if the term doesn't already exists...

    utils::check_shape(nMF, "nMF", 2 * NMO);

    int nIJ = n2IJ.extent(0);
    RealType sign = (propg_type == DiscreteChargePropagator) ? RealType(1.0) : RealType(-1.0);
    ComplexType scl = (propg_type == DiscreteChargePropagator?(1.0):ComplexType(0.0,-1.0));
    int M = NMO;	
    int M2 = walker_type == NONCOLLINEAR ? 2*M : M;
    int Madd = walker_type == NONCOLLINEAR ? M : 0;

    nda::array<ComplexType,1> hMF_h(nIJ, ComplexType(0.0));

    if(mpi->comm.root()) {

      math::sparse::csr_matrix<ComplexType, HOST_MEMORY, int, int> VnT({SpVnT.extent(0),SpVnT.extent(1)},4);
      // save parameters to avoid excessive recalculation
      std::vector<std::tuple<RealType,RealType,ComplexType,ComplexType>> params; 

      // opposite spin terms
      int nFields = 0;
      for(long i=0; i<M; ++i) {
        for(long p=U.row_begin(i); p<U.row_end(i); ++p) {
          auto Uij = U.values(p);
          long j = long(U.columns(p));
          if( std::abs(Uij) < 1e-6 ) continue;
          if( i > j ) continue;
          // add i up/j dn term 
          long I_ = i*M2+i;           // (i up, i up)
          auto nI = IJ2n[I_];
          long J_ = (j+M)*M2+j+Madd;  // (j dn, j dn)
          auto nJ = IJ2n[J_];
          auto nMF_ij = ComplexType(nMF[i]) + sign*ComplexType(nMF[j+M]); 
          auto [alpha,n_ij,nMF_ij_c] = get_parameters(dt,Uij,nMF_ij,natural_shift,params);
          vMF[nFields] = alpha * scl * nMF_ij_c; 
          hMF_h[nI] += sign*Uij*n_ij;
          hMF_h[nJ] += Uij*n_ij;
          VnT[nFields][nI] += alpha * scl; 
          VnT[nFields++][nJ] += sign * alpha * scl; 
          if( i != j ) {
          // add i dn/j up term 
          I_ = (i+M)*M2+i+Madd;      // (i dn, i dn)
          nI = IJ2n[I_];
          J_ = j*M2+j;               // (j up, j up)
          nJ = IJ2n[J_];
          nMF_ij = ComplexType(nMF[i+M]) + sign*ComplexType(nMF[j]); 
          std::tie(alpha,n_ij,nMF_ij_c) = get_parameters(dt,Uij,nMF_ij,natural_shift,params);
          vMF[nFields] = alpha * scl * nMF_ij_c; 
          hMF_h[nI] += sign*Uij*n_ij;
          hMF_h[nJ] += Uij*n_ij;
          VnT[nFields][nI] += alpha * scl; 
          VnT[nFields++][nJ] += sign * alpha * scl; 
          }
        }
      }
      // same spin terms
      for(long i=0; i<M; ++i) {
        for(long p=U.row_begin(i+M); p<U.row_end(i+M); ++p) {
          auto Uij = U.values(p);
          long j = long(U.columns(p));
          if( std::abs(Uij) < 1e-6 ) continue;
          if( i >= j ) continue;
          // add up/up term 
          long I_ = i*M2+i;      // (i up, i up)
          auto nI = IJ2n[I_];
          long J_ = j*M2+j;      // (j up, j up)
          auto nJ = IJ2n[J_];
          auto nMF_ij = ComplexType(nMF[i]) + sign*ComplexType(nMF[j+M]); 
          auto [alpha,n_ij,nMF_ij_c] = get_parameters(dt,Uij,nMF_ij,natural_shift,params);
          vMF[nFields] = alpha * scl * nMF_ij_c; 
          hMF_h[nI] += sign*Uij*n_ij; 
          hMF_h[nJ] += Uij*n_ij; 
          VnT[nFields][nI] += alpha * scl; 
          VnT[nFields++][nJ] += sign * alpha * scl; 
          // add dn/dn term 
          I_ = (i+M)*M2+i+Madd;      // (i dn, i dn)
          nI = IJ2n[I_];
          J_ = (j+M)*M2+j+Madd;      // (j dn, j dn)
          nJ = IJ2n[J_];
          nMF_ij = ComplexType(nMF[i+M]) + sign*ComplexType(nMF[j]); 
          std::tie(alpha,n_ij,nMF_ij_c) = get_parameters(dt,Uij,nMF_ij,natural_shift,params);
          vMF[nFields] = alpha * scl * nMF_ij_c; 
          hMF_h[nI] += sign*Uij*n_ij;
          hMF_h[nJ] += Uij*n_ij;
          VnT[nFields][nI] += alpha * scl; 
          VnT[nFields++][nJ] += sign * alpha * scl; 
        }
      }

      VnT.remove_empty_spaces();
      // sparse matrices should be compatible
      // if for some reason they are not, you need to communicate entire csr_matrix
      {
        utils::check(VnT.nnz() == SpVnT.nnz(), "Error: Contact developers."); 
        nda::range rng(VnT.nnz());
        auto col_h = nda::to_host(SpVnT.columns());
        utils::check(nda::sum(nda::abs(VnT.columns()(rng)-col_h(rng)))==0, "Error: Contact developers."); 
        SpVnT.values() = VnT.values();
      }
      {
        // now transpose  
        nda::array<ComplexType,2> Vn_array = math::sparse::to_array<'T'>(VnT);
        auto Vn = math::sparse::to_csr<HOST_MEMORY,int,int>(Vn_array);
        utils::check(Vn.nnz() == SpVn.nnz(), "Error: Contact developers.");
        nda::range rng(Vn.nnz());
        auto col_h = nda::to_host(SpVn.columns());
        utils::check(nda::sum(nda::abs(Vn.columns()(rng)-col_h(rng)))==0, "Error: Contact developers.");
        SpVn.values() = Vn.values();
      }

    } // mpi->comm.root()

    mpi->broadcast(vMF);
    mpi->broadcast(SpVn.values());
    mpi->broadcast(SpVnT.values());
    hMF = memory::share_from_root(*mpi, [&] { return memory::to_memory_space<MEM>(std::move(hMF_h)); });
  }

  // not sure if these equations hold for complex U (actual non-zero complex part...)
  template<typename T>
  std::tuple<ComplexType,ComplexType,ComplexType> get_parameters(double dt, ValueType U_, T nMF_, 
      bool natural_shift, std::vector<std::tuple<RealType,RealType,ComplexType,ComplexType>> &params)
  {
    using std::log;
    using std::cos;
    using std::acos;
    using std::cosh;
    using std::abs;
    using std::sqrt;
    using std::get;
    using boost::math::tools::bisect;
    auto tol = [](double min, double max) {
        return std::abs(max - min) < 1e-12; 
    };
    utils::check(abs(U_) > 1e-8, "Error in Discrete_GeneralUJ::get_parameters: U==0.");
    utils::check(abs(std::imag(U_)) < 1e-8, "Error in Discrete_GeneralUJ::get_parameters: imag(U) > 0 not yet allowed.");
    utils::check(abs(std::imag(nMF_)) < 1e-8, "Error in Discrete_GeneralUJ::get_parameters: imag(nMF) > 0. Should not happen.");

    RealType nMF = std::real(nMF_);
    if(propg_type == DiscreteChargePropagator) {
      if(natural_shift) nMF = 1.0;
      utils::check(abs(nMF) >= 0.0, "Error in Discrete_GeneralUJ(charge)::get_parameters: nMF<0.");
      utils::check(abs(nMF) <= 2.0, "Error in Discrete_GeneralUJ(charge)::get_parameters: nMF>2.");
    } else {
      if(natural_shift) nMF = 0.0;
      utils::check(abs(nMF) <= 1.0, "Error in Discrete_GeneralUJ(spin)::get_parameters: abs(nMF)>1.");
    }
    // just keep real part for now... Not sure what to do otherwise
    RealType Ud = std::real(U_);
    // look in table
    for( auto& v : params )
      if( (abs(Ud-get<0>(v)) < 1e-4) and (abs(nMF-get<1>(v)) < 1e-4) ) 
	      return std::make_tuple(get<2>(v),get<3>(v),ComplexType(nMF));

    ComplexType alpha(0.0),n(0.0);

    if(propg_type == DiscreteChargePropagator) {

      if( abs(nMF-1.0) < 1e-8 ) {
        // n=1 
        alpha = acos( exp( -ComplexType(dt)*0.5*ComplexType(Ud) ) );   	
      } else if((abs(nMF) < 1e-8) or (abs(nMF-2.0) < 1e-8)) {
        // n=0/2 
        alpha = acos( sqrt( 1.0 / ( 2.0 - exp( -ComplexType(dt)*ComplexType(Ud) ) ) ) );   	
      } else {	
        // generic case

        double Ud_ = double(Ud);
        ComplexType a0_c = acos( sqrt( 1.0 / ( 2.0 - exp( -dt*ComplexType(Ud_) ) ) ) );
        ComplexType a1_c = acos( exp( -dt*0.5*ComplexType(Ud_) ) );
        double n_ = (nMF < RealType(1.0)) ? double(nMF) : 2.0-double(nMF) ; 
        if(Ud < 0.0) {
          // alpha is imag 
          // just checking
          if( (std::abs(std::real(a0_c)) > 1e-8) or (std::abs(std::real(a1_c)) > 1e-8) )   	
            utils::check(false," Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = std::imag(a0_c); 
          double a1 = std::imag(a1_c); 
          auto f = [&](double a)
          {
            ComplexType c = cos(ComplexType(0.0,a)*(1.0-n_));
            return exp(-1.0*dt*Ud_) - std::real(cos(ComplexType(0.0,a)*(2.0-n_)) * cos(ComplexType(0.0,a)*n_) / (c*c));
          };
          // [a1,a0] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 )
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),tol,miter);
          if(miter == 300)
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          alpha = ComplexType(0.0,RealType(root.first));	     
        } else {
          // alpha is real
          if( (std::abs(std::imag(a0_c)) > 1e-8) or (std::abs(std::imag(a1_c)) > 1e-8) )   	
            utils::check(false," Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = std::real(a0_c); 
          double a1 = std::real(a1_c); 
          auto f = [&](double a)
          {
            double c = cos(a*(1.0-n_));
                  return exp(-1.0*dt*Ud_) - cos(a*(2.0-n_)) * cos(a*n_) / (c*c);
          }; 
          // [a0,a1] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 ) 
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),tol,miter);
          if(miter == 300)
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
                alpha = ComplexType(RealType(root.first));
        }
      }	
        utils::check(abs(cos(alpha*(1.0-nMF))) > 1e-8, 
                    "Error in Discrete_GeneralUJ::get_parameters: cosh(a*(1-n) == 0.0)");
        n = 1.0 - log( cos(alpha*nMF) / cos(alpha*(2.0-nMF)) ) / (2.0*ComplexType(RealType(dt))*ComplexType(Ud)); 

    } else {

      if( (abs(nMF-1.0) < 1e-8) or (abs(nMF+1.0) < 1e-8) ) {
        // n=1/-1 
        alpha = acosh( sqrt( 1.0 / ( 2.0 - exp( ComplexType(RealType(dt))*ComplexType(Ud) ) ) ) );   	
      } else if(abs(nMF) < 1e-8) {
        // n=0 
        alpha = acosh( exp( ComplexType(RealType(dt))*0.5*ComplexType(Ud) ) );  
      } else {	
	      // generic case
        double Ud_ = double(Ud);
        ComplexType a0_c = acosh( sqrt( 1.0 / ( 2.0 - exp( dt*ComplexType(Ud_) ) ) ) );
        ComplexType a1_c = acosh( exp( dt*0.5*ComplexType(Ud_) ) );
	      double n_ = double(abs(nMF)); 
        if(Ud < 0.0) {
	        // alpha is imag 
	        // just checking
          if( (std::abs(std::real(a0_c)) > 1e-8) or (std::abs(std::real(a1_c)) > 1e-8) )   	
            utils::check(false," Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = std::imag(a0_c); 
          double a1 = std::imag(a1_c); 
          auto f = [&](double a)
          {
            ComplexType c = cosh(ComplexType(0.0,a)*n_);
            return exp(dt*Ud_) - std::real(cosh(ComplexType(0.0,a)*(1.0-n_)) * cosh(ComplexType(0.0,a)*(1.0+n_)) / (c*c));
          };
          // [a1,a0] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 )
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),tol,miter);
          if(miter == 300)
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");
          alpha = ComplexType(0.0,RealType(root.first));	     
        } else {
          // alpha is real
          if( (std::abs(std::imag(a0_c)) > 1e-8) or (std::abs(std::imag(a1_c)) > 1e-8) )   	
            utils::check(false," Error in Discrete_GeneralUJ::get_parameters: Unexpected value. Report problem.");
          double a0 = std::real(a0_c); 
          double a1 = std::real(a1_c); 
          auto f = [&](double a)
	        {
            double c = cosh(a*n_);
                  return exp(dt*Ud_) - cosh(a*(1.0-n_)) * cosh(a*(1.0+n_)) / (c*c);
          }; 
          // [a0,a1] should bracket the root and f(a) should be monotonically increasing
          if( f(a0)*f(a1) > 0.0 ) 
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
          std::uintmax_t miter(300);
          auto root = bisect(f,std::min(a0,a1),std::max(a0,a1),tol,miter);
          if(miter == 300)
            utils::check(false,"Error in Discrete_GeneralUJ::get_parameters: Problems bracketing root.");		
                alpha = ComplexType(RealType(root.first));
        }

      }	
      utils::check(abs(cosh(alpha*(1.0-nMF))) > 1e-8, 
                   "Error in Discrete_GeneralUJ::get_parameters: cosh(a*(1-n) == 0.0)");
      n = log(cosh(alpha*(1.0+nMF)) / cosh(alpha*(1.0-nMF))) / (2.0*ComplexType(RealType(dt))*ComplexType(Ud)); 

    }	
    app_log(1," Discrete Propagator parameter: dt={}, U={}, nMF={}, alpha={}, n={}",dt,double(Ud),
		double(nMF),alpha,n);
    params.emplace_back(std::make_tuple(Ud,nMF,alpha,n));	
    return std::make_tuple(alpha,n,ComplexType(nMF));
  }
};

} // namespace afqmc

} // namespace sfqmc

