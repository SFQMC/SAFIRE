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


#include "config.h"
#include "AFQMC/config.h"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "AFQMC/Propagators/AFQMCBasePropagator.h"
#include "AFQMC/Propagators/construct_X.hpp"
#include "AFQMC/SlaterDeterminantOperations/orthogonalize.hpp"
#include "AFQMC/SlaterDeterminantOperations/propagate.hpp"
#include "AFQMC/Utilities/AFQMCTimer.h"
#include "AFQMC/Walkers/WalkerConfig.hpp"
#include "numerics/shared_array/const_shared_array.hpp"
#include "numerics/operations/exp.hpp"
#include "numerics/operations/tensor.hpp"

namespace sfqmc
{
namespace afqmc
{

/*
 * Constructs the various 1-body propagators for the timestep of this propagator.
 * The forward-direction propagators are always constructed. If Pinv is true, the
 * backward-direction propagators, P1x_inv, are constructed as well.
 */
template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::generateP1(bool Pinv)
{
  using nda::range;
  auto all = range::all;
  bool build_inv = Pinv or ( P1s_inv.size() > 0 ? (P1s_inv(0).capacity() > 0) : false );
  const int NMO = wfn->getNMO();
  const double dt = timestep;
  const WALKER_TYPES walker_type = wfn->getWalkerType();

  app_log(1, "\n  - Generating a new 1-body propagator with timestep: {}",dt);

  // update hamiltonian factorization parameters if needed (e.g. in discrete factorization)
  bool discrete_propg = false;
  for ( int i=0; i<FieldTypes.size(); i++ ) {
    int v(FieldTypes[i]);
    if( (PropagatorTypes(v) == DiscreteChargePropagator) or
        (PropagatorTypes(v) == DiscreteSpinPropagator) ) {
      discrete_propg = true;
      break;
    }
  }
  long nCV = wfn->number_of_cholesky_vectors();

  // discrete propagators setup their own vMF
  memory::buffered_array<MEM,ComplexType,1> vMF_discrete(nCV);
  if(discrete_propg) {
    int npol         = walker_type == NONCOLLINEAR ? 2 : 1;
    nda::array<ComplexType,1> nMF(2*NMO, ComplexType(0.0));
    // setup sparse vector to generate <nI>
    auto Gmf_shm = wfn->G_MF();
    if(mpi->comm.root()) {
      auto Gmf = nda::to_host(Gmf_shm()); 
      for(int i=0; i<npol*NMO; i++)
        nMF(i) = Gmf(0,i,i);
      if(walker_type == COLLINEAR)
        for(int i=0; i<NMO; i++)
          nMF(i+NMO) = Gmf(1,i,i);
    }
    mpi->broadcast(nMF);
    wfn->update_potentials(dt,nMF,vMF_discrete,natural_shift);
  }

  // calculate vMF for the current time step
  memory::buffered_array<MEM,ComplexType,1> vt(nCV);
  vt() = ComplexType(0.0);
  if (subtractMF)
  {
    auto hamtype(wfn->getHamType());
    // collective call
    wfn->vMF(vt, dt);
    if(hamtype == HamiltonianType::model_hamiltonian) {
      // depending on charge/spin, you should also set imag/real parts to zero
      // overwrite vMF if needed
      if(discrete_propg) {
        for ( int i=0; i<FieldTypes.size(); i++ ) {
          int v(FieldTypes(i));
          if( (PropagatorTypes(v) == DiscreteChargePropagator) or
              (PropagatorTypes(v) == DiscreteSpinPropagator) ) {
            vt(nda::range(i,i+1)) = vMF_discrete(nda::range(i,i+1));
          }
        }
      }
    } else {
      // continuous propagator, charge decomposition. vt should be real
      math::zero_imag(vt);
    }
  }
  vMF = memory::share_from_root(*mpi, [&] { return memory::array<MEM,ComplexType,1>(vt()); });

  if(mpi->comm.root()) {
    auto v_h = nda::to_host(vMF());
    RealType vmax = 0;
    for (int i = 0; i < v_h.size(); i++)
      vmax = std::max(vmax, std::abs(v_h(i)));
    app_log(1," Largest component of Mean-field subtraction potential: {}",vmax);
    if (vmax > vbias_bound) {
      app_warning(" WARNING: Mean-field subtraction potential has components outside vbias_bound.");
      app_warning("          Consider increasing vbias_bound. max(vMF[n])={}, vbias_bound={} ",
  		    vmax,vbias_bound);
    }
  }
  mpi->comm.barrier();

  // assemble H1(i,j) = dt * (h(i,j) + vn0(i,j) + sum_n vMF[n]*vn(i,j,n))
  // H1 should have the same spin structure as walker_type
  int nspin = walker_type == COLLINEAR ? 2 : 1;
  int npol  = walker_type == NONCOLLINEAR ? 2 : 1;

  // H1 is in host
  nda::array<ComplexType,3> H1;
  if(mpi->comm.root()) {
    auto vMF_h = nda::to_host(vMF());
    H1 = wfn->getOneBodyPropagatorMatrix(dt, vMF_h);

    utils::check(H1.shape() == std::array<long,3>{nspin,npol*NMO,npol*NMO}, "Shape mismatch.");

    memory::buffered_array<HOST_MEMORY, ComplexType, 3> H1tmp{H1};

    math::hermitize(H1);

    nda::tensor::add(1, H1, -1, H1tmp);
    if(nda::linalg::norm(nda::flatten(H1tmp())) > 1e-5) {
      app_warning("H1 is not hermitian!");
    }

    nda::tensor::scale(ComplexType(-0.5),H1);
  }
  auto exp_H1 = [&] {
    nda::array<ComplexType,3> P(nspin,npol*NMO,npol*NMO);
    for(int i=0; i<nspin; ++i) {
      P(i,all,all) = math::exp_hermitian(H1(i,all,all), printP1eV);
    }
    return memory::to_memory_space<MEM>(std::move(P));
  };
  P1d = memory::share_from_root(*mpi, exp_H1);
  if(build_inv) {
    if(mpi->comm.root()) {
      nda::tensor::scale(ComplexType(-1.0),H1);
    }
    P1d_inv = memory::share_from_root(*mpi, exp_H1);
  }

  if(P1s.size() != nspin) P1s.resize(nspin); 
  for(int i=0; i<nspin; i++) 
    P1s(i) = math::sparse::to_csr<MEM>(P1d()(i,all,all),1e-8); 
  if(build_inv) { 
    if(P1s_inv.size() != nspin) P1s_inv.resize(nspin); 
    for(int i=0; i<nspin; i++) 
      P1s_inv(i) = math::sparse::to_csr<MEM>(P1d_inv()(i,all,all),1e-8); 
  }
  mpi->comm.barrier();

}

template<MEMORY_SPACE MEM>
template<char TA, typename VHS_t>
void AFQMCBasePropagator<MEM>::apply_propagators(WalkerSet<MEM>& wset, VHS_t const& v, bool P1inv)
{
  if(P1inv) {
    if(denseP1) {
      det_ops::Propagate<MEM,TA>(wset,P1d_inv(),v,order);
    } else {
      det_ops::Propagate<MEM,TA>(wset,P1s_inv(),v,order);
    }
  } else {
    if(denseP1) {
      det_ops::Propagate<MEM,TA>(wset,P1d(),v,order);
    } else {
      det_ops::Propagate<MEM,TA>(wset,P1s(),v,order);
    }
  }
}

template<MEMORY_SPACE MEM>
template<char TA, typename VHS_t, nda::MemoryArrayOfRank<3> Mat3>
void AFQMCBasePropagator<MEM>::apply_propagators(WALKER_TYPES wtype, int npol,
                                                 Mat3&& Xa, Mat3&& Xb,
                                                 VHS_t const& v, bool P1inv)
{
  if(P1inv) {
    if(denseP1) {
      if(wtype == COLLINEAR)
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1d_inv(),v,order);
      else
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1d_inv(),v,order);
    } else {
      if(wtype == COLLINEAR)
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1s_inv(),v,order);
      else
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1s_inv(),v,order);
    }
  } else {
    if(denseP1) {
      if(wtype == COLLINEAR)
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1d(),v,order);
      else
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1d(),v,order);
    } else {
      if(wtype == COLLINEAR)
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,Xb,P1s(),v,order);
      else
        det_ops::Propagate<MEM,TA>(wtype,npol,Xa,P1s(),v,order);
    }
  }
}

/*
 * Propagates the walker population nsteps forward.
 */
template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::Propagate(WalkerSet<MEM>& wset, RealType Eshift, int nt)
{
  auto setup_timer = timers.setup.start();
  const RealType dt = timestep;
  int nwalk        = wset.size();
  int nCV     = wfn->number_of_cholesky_vectors();
  bool ft = wset.isFiniteTemperature();

  memory::buffered_array<MEM,ComplexType,1> XvMF(nwalk);
  memory::buffered_array<MEM,ComplexType,1> hybrid_weight(nwalk, 0.0);
  memory::buffered_array<MEM,ComplexType,1> new_overlaps(nwalk, 0.0);
  memory::buffered_array<MEM,ComplexType,2> new_energies(nwalk, 3);
  new_energies() = ComplexType(0.0);

  setup_timer.stop();

  // store current time-slice for finite-T walkers
  if (ft)
    wset.setTauStep(nt);

  memory::buffered_array<MEM,ComplexType,2> X(nwalk, nCV);
  // 1. Calculate vbias for initial configuration
  if(free_projection) {
    X() = ComplexType(0.0);
  } else {
    if(!ft) {
      wfn->vbias(wset, X, dt);
    } else {
      wfn->vbias(wset, X, dt, nt-1);
    }
  }

  // 2. Assemble X(nwalk, nCV)
  auto assemble_X_time = timers.assemble_X.start();
  assemble_X(X, hybrid_weight);
  // XvMF[iw] = sum_m ( im * X[iw,m] * vMF[m] ); W_MSsub = exp(-XvMF), careful with sign convention
  ComplexType im(0.0,1.0);
  nda::blas::gemv(im, X(), vMF(), 0.0, XvMF());
  assemble_X_time.stop();

  //std::cout<<" X: " <<nda::sum(nda::to_host(X)) <<std::endl;

  // Store X in wset if back propagating
  if(wset.NumBackProp() > 0) {
    wset.storeFields(wset.getHistoryPos() % wset.NumBackProp(), X);
  }

  if(denseP2) {
    // 3. Calculate vHS
    auto vHS_time = timers.vHS.start();
    auto vHS = wfn->vHS(X, dt);
    vHS_time.stop();

    // 4. Propagate walkers
    auto propagate_time = timers.propagate.start();
    apply_propagators<'N'>(wset,vHS);
    propagate_time.stop();
  } else {
    // 3. Calculate vHS
    auto vHS_time = timers.vHS.start();
    auto vHS = wfn->vHS_sparse(X, dt);
    vHS_time.stop();

    // 4. Propagate walkers
    auto propagate_time = timers.propagate.start();
    apply_propagators<'N'>(wset,vHS);
    propagate_time.stop();
  }

  // 5. Calculate local energy/overlap
  auto pseudo_energy_time = timers.pseudo_energy.start();
  if (hybrid)
    wfn->Log_Overlap(wset, new_overlaps);
  else
    wfn->Energy(wset, new_energies, new_overlaps);
  pseudo_energy_time.stop();


  // 6. update weights/energy/etc, apply constrains/bounds/etc
  auto extra_time = timers.extra.start();
  if (free_projection) {
    free_projection_walker_update(wset, dt, new_overlaps, XvMF, Eshift, wfn->energy_offset(),
                                  hybrid_weight,debug_verbosity);
  } else {
    if (hybrid) {
      hybrid_walker_update(wset, dt, apply_constraint, importance_sampling, Eshift,
                           wfn->energy_offset(), new_overlaps, XvMF,
                           hybrid_weight, lower_cutoff_scale, upper_cutoff_scale, debug_verbosity,
                           use_cp_constraint, eloc_bound_stats);
    } else {
      local_energy_walker_update(wset, dt, apply_constraint, Eshift, new_overlaps, new_energies, XvMF,
                                 lower_cutoff_scale, upper_cutoff_scale, eloc_bound_stats);
    }
  }

  // 7. bound the weights, so that no single walker can dominate the population before the
  //    next branching
  bound_walker_weights(wset, weight_bound_floor, weight_bound_fraction, weight_bound_stats);

  if(wset.NumBackProp() > 0) {
    wset.advanceHistoryPos();
  }
  extra_time.stop();
}

/*
 * This routine assumes that the 1 body propagator does not need updating
 * Input:
 *  - nbsteps: # of back propagation steps to perform.
 *  - nStabalize: # of steps between ortogonalization
 *  - wset: walker set, contains field history
 *  - Refs(nwalk, number_of_references, npol*NMO, nel)
 *  - logdetR(nwalk, number_of_references)
 */
template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::BackPropagate(int nbpsteps, int nStabalize, WalkerSet<MEM>& wset,
        memory::array_view<MEM,ComplexType,4> Refs, memory::array_view<MEM,ComplexType,2> logdetR)
{
  using nda::range;
  auto all = range::all;
  auto walker_type = wset.getWalkerType();
  int npol         = (walker_type == NONCOLLINEAR) ? 2 : 1;
  int nspin        = (walker_type == COLLINEAR) ? 2 : 1;
  int nwalk        = wset.size();
  int nCV          = wfn->number_of_cholesky_vectors();
  int nrefs        = Refs.extent(1);
  // Dimensions inferred from the walker Slater matrices (nwalk, npol*NMO, naea/naeb).
  const int NMO    = int(wset.SlaterMatrices(Alpha).extent(1)) / npol;
  const int nup    = int(wset.SlaterMatrices(Alpha).extent(2));
  const int ndown  = (walker_type == COLLINEAR) ? int(wset.SlaterMatrices(Beta).extent(2)) : 0;
  int nel          = ( walker_type==COLLINEAR ? nup+ndown : nup );

  utils::check(Refs.shape() == std::array<long,4>{nwalk,nrefs,npol*NMO,nel},
               "Size mismatch");
  utils::check(logdetR.shape() == std::array<long,2>{nwalk,nrefs}, "Size mismatch");

  // Fields(nwalk, steps, nCV)
  auto Fields = wset.getFields();
  utils::check(nbpsteps > 0, "Error in BackPropagate: nbpsteps:{}",nbpsteps);
  utils::check(Fields.extent(1) >= nbpsteps, "Size mismatch");
  utils::check(Fields.extent(0) == nwalk and Fields.extent(2) == nCV, "Size mismatch");

  memory::buffered_array<MEM,ComplexType,3> SMA(nwalk, npol*NMO, nup);
  memory::buffered_array<MEM,ComplexType,3> SMB(nwalk, npol*NMO,
                                                (walker_type==COLLINEAR ? ndown : 0));
  memory::buffered_array<MEM,ComplexType,2> X(nwalk, nCV);
  memory::buffered_array<MEM,ComplexType,1> ldet(nwalk);
  logdetR() = ComplexType(0.0);

  // 0. copy SlaterMatrix to SMA/SMB
  if(nup > 0) {
    SMA() = wset.SlaterMatrices(Alpha);
  }
  if(walker_type==COLLINEAR && ndown > 0) {
    SMB() = wset.SlaterMatrices(Beta);
  }

  int nfields = wset.NumBackProp();
  int slot    = wset.getHistoryPos() % nfields;
  for (int ni = nbpsteps - 1; ni >= 0; --ni)
  {
    // walk the field ring backwards from the newest entry; the cursor is already
    // advanced for the next step, so step back before reading
    slot = ((slot == 0) ? nfields - 1 : slot - 1);
    // 1. Get X(nwalk,nCV) from wset
    X() = Fields(all,slot,all);

    if(denseP2) {

      // 2. Calculate vHS
      auto vHS = wfn->vHS(X, timestep);
      utils::check(vHS.shape() == std::array<long,4>{nspins_in_vHS,nwalk,npol_in_vHS*NMO,NMO},
                 "Size mismatch");

      // MAM: can do all references together in principle,
      //      would consume more memory but be very efficient in GPU!
      for (int nr = 0; nr < nrefs; ++nr)
      {
        // 3. copy reference to SlaterMatrix
        if(nup > 0) {
          math::copy(Refs(all,nr,all,range(nup)), wset.SlaterMatrices(Alpha));
        }
        if(walker_type==COLLINEAR && ndown > 0) {
          math::copy(Refs(all,nr,all,range(nup,nup+ndown)), wset.SlaterMatrices(Beta));
        }

        // 4. Propagate walkers
        apply_propagators<'H'>(wset,vHS);

        // always end (n==0) with an orthogonalization
        if (ni == 0 || ni % nStabalize == 0) {
          ldet() = ComplexType(0.0);
          det_ops::orthogonalize(wset, ldet);
          nda::tensor::add(ComplexType(1.0),ldet,"w",ComplexType(1.0),logdetR(all,nr),"w");
        }

        // 5. copy reference to back
        if(nup > 0) {
          math::copy(wset.SlaterMatrices(Alpha), Refs(all,nr,all,range(nup)));
        }
        if(walker_type==COLLINEAR && ndown > 0)
          math::copy(wset.SlaterMatrices(Beta), Refs(all,nr,all,range(nup,nup+ndown)));
      }

    } else {

      // 2. Calculate vHS
      auto vHS = wfn->vHS_sparse(X, timestep);
      utils::check(vHS.extent(0) == nspin, "Size mismatch");
      utils::check((vHS(0).shape()==std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}) and
               (vHS(nspin-1).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}),
               "Size mismatch");

      for (int nr = 0; nr < nrefs; ++nr)
      {
        // 3. copy reference to SlaterMatrix
        math::copy(Refs(all,nr,all,range(nup)),wset.SlaterMatrices(Alpha));
        if(walker_type==COLLINEAR)
          math::copy(Refs(all,nr,all,range(nup,nup+ndown)),wset.SlaterMatrices(Beta));

        // 4. Propagate walkers
        apply_propagators<'H'>(wset,vHS);

        // always end (n==0) with an orthogonalization
        if (ni == 0 || ni % nStabalize == 0) {
          ldet() = ComplexType(0.0);
          det_ops::orthogonalize(wset, ldet);
          nda::tensor::add(ComplexType(1.0),ldet,"w",ComplexType(1.0),logdetR(all,nr),"w");
        }

        // 5. copy reference to back
        math::copy(wset.SlaterMatrices(Alpha), Refs(all,nr,all,range(nup)));
        if(walker_type==COLLINEAR)
          math::copy(wset.SlaterMatrices(Beta), Refs(all,nr,all,range(nup,nup+ndown)));
      }

    }
  }

  // 6. restore the Slater Matrix
  wset.SlaterMatrices(Alpha) = SMA();
  if(walker_type==COLLINEAR)
    wset.SlaterMatrices(Beta) = SMB();

  mpi->comm.barrier();
}

/*
 * Propagate "operator states" forward in time using stored fields.
 */
template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::PropagateOperators(int nsteps, WalkerSet<MEM>& wset,
        memory::array_view<MEM,ComplexType,4> X, memory::array_view<MEM,ComplexType,4> Y,
        memory::array_view<MEM,ComplexType,4> M)
{
  using nda::range;
  auto all = range::all;
  auto walker_type = wset.getWalkerType();
  int npol         = (walker_type == NONCOLLINEAR) ? 2 : 1;
  int nspin        = (walker_type == COLLINEAR) ? 2 : 1;
  int nwalk        = wset.size();
  int nCV          = wfn->number_of_cholesky_vectors();
  ComplexType one(1.0,0.0);
  // Dimensions inferred from the walker Slater matrices (nwalk, npol*NMO, naea/naeb).
  const int NMO    = int(wset.SlaterMatrices(Alpha).extent(1)) / npol;
  const int nup    = int(wset.SlaterMatrices(Alpha).extent(2));
  const int ndown  = (walker_type == COLLINEAR) ? int(wset.SlaterMatrices(Beta).extent(2)) : 0;

  // Fields(nwalk, steps, nCV)
  auto Fields = wset.getFields();
  utils::check(Fields.extent(1) >= nsteps, "Size mismatch");
  utils::check(Fields.extent(0) == nwalk and Fields.extent(2) == nCV, "Size mismatch");
  utils::check(nsteps > 0, "Error in PropagateOperators: nsteps:{}",nsteps);

  // generate P1x_inv if not yet constructed
  if(P1s_inv.size() != nspin)
    generateP1(true);

  {
    memory::buffered_array<MEM,ComplexType,2> Xfield(nwalk, nCV);
    // MAM: Regularization is done after all nsteps are applied.
    //      If this leads to instabilities (e.g. for large nsteps),
    //      apply regularization inside this loop with some frequency
    // replay the nsteps most recent slots of the ring, oldest first
    int nfields = wset.NumBackProp();
    int slot    = (wset.getHistoryPos() % nfields + nfields - nsteps) % nfields;
    for (int i = 0; i < nsteps; ++i, slot = (slot + 1) % nfields)
    {

      // 1. Get Xfield(nwalk,nCV) from wset
      Xfield() = Fields(all,slot,all);

      if(denseP2) {

        // 2. Calculate vHS
        auto vHS = wfn->vHS(Xfield, timestep);
        utils::check(vHS.shape() == std::array<long,4>{nspins_in_vHS,nwalk,npol_in_vHS*NMO,NMO},
                   "Size mismatch");

        // 4. Propagate walkers
        apply_propagators<'N'>(walker_type,npol,X(all,0,all,all),X(all,nspin-1,all,all),vHS);
        // scale vHS by -1, since we need ma::H( B^(-1) )
        nda::tensor::scale(ComplexType(-1.0),vHS);
        apply_propagators<'H'>(walker_type,npol,Y(all,0,all,all),Y(all,nspin-1,all,all),vHS,true);
      } else {

        // 2. Calculate vHS
        auto vHS = wfn->vHS_sparse(Xfield, timestep);
        utils::check(vHS.extent(0) == nspin, "Size mismatch");
        utils::check((vHS(0).shape()==std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}) and
                 (vHS(nspin-1).shape() == std::array<long,2>{nwalk*npol*NMO,nwalk*npol*NMO}),
                 "Size mismatch");

        // 4. Propagate walkers
        apply_propagators<'N'>(walker_type,npol,X(all,0,all,all),X(all,nspin-1,all,all),vHS);
        // scale vHS by -1, since we need ma::H( B^(-1) )
        nda::tensor::scale(ComplexType(-1.0),vHS(0).values());
        if(nspin==2)
          nda::tensor::scale(ComplexType(-1.0),vHS(1).values());
        apply_propagators<'H'>(walker_type,npol,Y(all,0,all,all),Y(all,nspin-1,all,all),vHS,true);

      }

    }

  }

  // 4.Regularize states using the current walker's Slater Matrix (which should be orthonormal)
  // Ynew = S * H(S) * Y
  // M += H(Ynew) * X
  // Xnew = X - S * H(S) * X
  // where S is the walker's Slater Matrix.
  SpinTypes stype[2] = {Alpha, Beta};
  int nel[2] = {nup, ndown};

  if constexpr (MEM==HOST_MEMORY) {

    for(int ispin=0; ispin<nspin; ispin++) {

      auto SM = wset.SlaterMatrices(stype[ispin]);
      memory::buffered_array<MEM,ComplexType,2> SY(nel[ispin], npol*NMO);

      for(int iw=0; iw<nwalk; ++iw) {

        auto SM_iw  = SM(iw,all,all);
        auto X_iw  = X(iw,ispin,all,all);
        auto Y_iw  = Y(iw,ispin,all,all);
        auto M_iw  = M(iw,ispin,all,all);

        // SY = H(S) * Y
        nda::blas::gemm(nda::dagger(SM_iw),Y_iw,SY);

        // Ynew = S * SY
        nda::blas::gemm(SM_iw,SY,Y_iw);

        // M += H(Ynew) * X
        nda::blas::gemm(one,nda::dagger(Y_iw),X_iw,one,M_iw);

        // SY = H(S) * X
        nda::blas::gemm(nda::dagger(SM_iw),X_iw,SY);

        // Xnew = X - S * SY
        nda::blas::gemm(-one,SM_iw,SY,one,X_iw);

      } // iw

    } // ispin

  } else {

    for(int ispin=0; ispin<nspin; ispin++) {

      memory::buffered_array<MEM,ComplexType,3> SY(nwalk, nel[ispin], npol*NMO);

      // SY = H(S) * Y
      nda::tensor::contract(nda::conj(wset.SlaterMatrices(stype[ispin])), "nki",
                       Y(all,ispin,all,all), "nkj", SY, "nij");

      // Ynew = S * SY
      nda::tensor::contract(wset.SlaterMatrices(stype[ispin]), "nik",
                       SY, "nkj", Y(all,ispin,all,all), "nij");

      // M += H(Ynew) * X
      nda::tensor::contract(one, nda::conj(Y(all,ispin,all,all)), "nki",
                                 X(all,ispin,all,all), "nkj",
                            one, M(all,ispin, all, all), "nij");

      // SY = H(S) * X
      nda::tensor::contract(nda::conj(wset.SlaterMatrices(stype[ispin])), "nki",
                       X(all,ispin,all,all), "nkj", SY, "nij");

      // Xnew = X - S * SY
      nda::tensor::contract(-one, wset.SlaterMatrices(stype[ispin]), "nik", SY, "nkj",
                            one, X(all,ispin,all, all), "nij");

    } // ispin

  } // MEM
  mpi->comm.barrier();
}

template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::assemble_X(memory::array_view<MEM,ComplexType,2> X,
                                          memory::array_view<MEM,ComplexType,1> HWs,
                                          bool addRAND)
{
  // On entry, X(iw,m) = vbias(iw,m);
  // remember to call vbi = apply_bound_vbias(*vb);

  auto [nwalk,nCV] = X.shape();

  // Count force-bias (vbias) clamp hits for the end-of-run report. On entry X(iw,m) still holds
  // the raw vbias, so re-evaluate the exact magnitude condition construct_X uses to clamp it.
  // Host-only (X is on device for DEVICE_MEMORY); device counting is not supported.
  if constexpr (MEM == HOST_MEMORY) {
    auto Xv = X();
    for (long iw = 0; iw < nwalk; ++iw)
      for (long m = 0; m < nCV; ++m) {
        ++vbias_bound_stats.total;
        if (std::abs(Xv(iw, m)) > vbias_bound) ++vbias_bound_stats.upper;
      }
  }
  // generate random numbers
  utils::check( rng_block_size >= nCV, "Error in assemble_X: rng_block_size < nCV.");
  memory::buffered_array<MEM,RealType,2> RNGbuff(nwalk,rng_block_size);
  RNGbuff() = RealType(0.0);
  if (addRAND)
  {
    // always generate rng_block_size per walker, to keep generators synchronized in
    // correlated sampling calculations
    rng->sampleUniformFields(nda::flatten(RNGbuff));
  }


  // The mean-field subtraction splits 0.5*sum_n v_n^2 into 0.5*sum_n (v_n - vMF_n)^2 (applied by
  // vHS) plus sum_n vMF_n*v_n (folded into the 1-body propagator) minus the c-number
  // 0.5*sum_n vMF_n^2, which nothing applies. Seeding the hybrid weight with it puts that
  // constant back into the reconstructed local energy, since eloc subtracts HWs and divides by
  // dt.
  HWs() = 0.5 * nda::blas::dot(vMF(), vMF());

  if constexpr (MEM==DEVICE_MEMORY) {
#if defined(ENABLE_DEVICE)
    construct_X(use_cp_constraint or project_force_bias, free_projection, vbias_bound, FieldTypes_dev, vMF, HWs, RNGbuff, X);
#endif
  } else {
    detail::construct_X_impl f{use_cp_constraint || project_force_bias, free_projection, vbias_bound, FieldTypes, vMF(), HWs(), RNGbuff(), X()};
    for(long iw = 0; iw < nwalk; ++iw) {
      for(long m = 0; m < nCV; ++m) {
        f(iw, m);
      }
    }
  }
}

template<MEMORY_SPACE MEM>
void AFQMCBasePropagator<MEM>::Orthogonalize(WalkerSet<MEM>& wset)
{
  memory::buffered_array<MEM,ComplexType,1> ldet(wset.size(),0.0);
  det_ops::orthogonalize(wset,ldet);
  wfn->Log_Overlap(wset);
}

template class AFQMCBasePropagator<HOST_MEMORY>;

#if defined(ENABLE_DEVICE)
template class AFQMCBasePropagator<DEVICE_MEMORY>;
#endif

} // namespace afqmc


} // namespace sfqmc
