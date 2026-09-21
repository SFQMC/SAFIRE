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


#include "AFQMC/config.h"
#include "AFQMC/Wavefunctions/Wavefunction.hpp"
#include "AFQMC/Hamiltonians/hdf5_helpers.hpp"
#include "AFQMC/Utilities/readWfn.h"
#include "IO/banner.hpp"

// The dispatch lives here so that the NOMSD/PHMSD/NOMSD_FT bodies are instantiated
// once, in this TU, instead of in every TU that calls into Wavefunction.

namespace sfqmc {
namespace afqmc {

namespace {

// Convert an array of numbers of electrons per flavor ({nup, ndown}) from one walker type to another.
// Most of the time, this is is an identity. However, in the noncollinear case, only
// the first component contains all electrons and the rest is supposed to be zero.
template<std::size_t N = 2>
auto broadcast_number_of_electrons(const std::array<int, N> &nel, WALKER_TYPES from, WALKER_TYPES to) {
  static_assert(N > 0, "Cannot have no electron flavors");
  utils::check(walkerTypeIsConvertible(from, to), "Cannot convert {} wavefunction to {} walker type", walkerTypeToString(from), walkerTypeToString(to));
  if(from == CLOSED) {
    utils::check(
      !nel.empty() && std::all_of(nel.begin(), nel.end(), [&](auto n) { return n == nel.front(); }),
      "Closed wavefunction does not have uniform number of electrons: {}", nel);
  }
  
  if(to == NONCOLLINEAR) {
    std::array<int, N> result{};
    result[0] = std::accumulate(nel.begin(), nel.end(), 0);
    return result;
  }
  return nel;
}

/// Everything read out of the trial file before it is known which kind of trial it holds.
/// Each builder opens its own subgroup of `wgrp`.
struct TrialFile {
  h5::group wgrp;
  int NMO;
  int nup;   ///< as recorded in the file, before any walker type conversion
  int ndown;
};

/// The wavefunction type recorded in the trial file. Only the root touches the file.
WAVEFUNCTION_TYPES peek_wavefunction_type(const std::string& filename,
                                          utils::mpi_context_t<boost::mpi3::communicator>& mpi) {
  int wfn_type{};
  if(mpi.comm.root()) {
    wfn_type = int(afqmc::getWavefunctionType(filename));
  }
  mpi.comm.broadcast_n(&wfn_type, 1, 0);
  return WAVEFUNCTION_TYPES(wfn_type);
}

/// Densifies every determinant of a sparse trial into node-shared storage.
template<MEMORY_SPACE MEM, int Rank>
auto to_dense_shared(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                     const nda::array<PsiT_Matrix<MEM>, Rank>& PsiT) {
  nda::array<memory::const_shared_array<MEM,ComplexType,2>, Rank> dense(PsiT.shape());
  for(long i = 0; i < PsiT.size(); ++i) {
    dense.data()[i] = memory::share_from_root(mpi, [&] {
      return memory::to_memory_space<MEM>(math::sparse::to_array<'N'>(PsiT.data()[i]));
    });
  }
  return dense;
}

/// The walker type the initial guess in `grp` was written for, checked against the walkers it
/// is about to fill, together with the dimensions recorded next to it.
std::pair<WALKER_TYPES, nda::array<int,1>> peek_guess_type(h5::group grp, WALKER_TYPES walker_type) {
  nda::array<int,1> dims(5);
  nda::h5_read(grp, "dims", dims);

  WALKER_TYPES wtype(initWALKER_TYPES(dims[3]));
  utils::check(walkerTypeIsConvertible(wtype, walker_type),
               "Initial guess ({}) not convertible to walker_type {}",
               walkerTypeToString(wtype), walkerTypeToString(walker_type));
  return {wtype, std::move(dims)};
}

/// Reads the initial walker Slater matrices, resized to the walkers that will hold them.
WalkerSetInitialGuess read_initial_guess(h5::group grp, WALKER_TYPES walker_type,
                                         int NMO, int nup, int ndown) {
  using nda::range;
  auto all = range::all;

  auto const [wtype, dims] = peek_guess_type(grp, walker_type);
  auto nel_in_guess = std::to_array({dims[1], dims[2]});
  auto [nspin_in_guess, npol_in_guess] = walkerTypeToDims(wtype);

  // Read the trial's per-spin orbital matrices at their true (in-file) widths.
  std::array<std::string,2> dataset_names{{"Psi0_alpha", "Psi0_beta"}};
  std::vector<nda::matrix<ComplexType>> Min;
  Min.reserve(nspin_in_guess);
  for(int is = 0; is < nspin_in_guess; is++) {
    utils::check(nup >= nel_in_guess[is], "initial guess contains more electrons of spin {} than walker nup ({})", nel_in_guess[is], nup);
    nda::matrix<ComplexType> m(npol_in_guess * NMO, nel_in_guess[is]);
    m() = 0.0;
    utils::h5_read(grp, dataset_names[is], m);
    Min.push_back(std::move(m));
  }

  auto [nspin, npol] = walkerTypeToDims(walker_type);
  // Walker-sized per-spin widths: alpha=nup, beta=ndown (collinear). Kept exact
  // (no max-padding) so naeb is recoverable from the beta matrix's width.
  std::array<int,2> out_width{{nup, ndown}};

  std::vector<nda::matrix<ComplexType>> M;
  M.reserve(nspin);
  if(walker_type == NONCOLLINEAR && wtype != NONCOLLINEAR) {
    // Interleave the (NMO-row) spin channels into one 2*NMO-row matrix.
    nda::matrix<ComplexType> a(npol * NMO, nup);
    a() = 0.0;
    auto a3 = reshape(a, npol, NMO, nup);
    int offset = 0;
    for(int ip = 0; ip < npol; ip++) {
      a3(ip, all, range(offset, offset + nel_in_guess[ip])) =
          Min[ip % nspin_in_guess](all, range(nel_in_guess[ip]));
      offset += nel_in_guess[ip];
    }
    M.push_back(std::move(a));
  } else {
    for(int is = 0; is < nspin; is++) {
      nda::matrix<ComplexType> m(npol * NMO, out_width[is]);
      m() = 0.0;
      int src = is % nspin_in_guess;
      int nc  = std::min<int>(out_width[is], nel_in_guess[src]);
      m(all, range(nc)) = Min[src](all, range(nc));
      M.push_back(std::move(m));
    }
  }

  return {.walker_type = walker_type, .payload = std::move(M)};
}

/// Reads the finite-temperature UDV initial guess. Collective on mpi.comm, since the guess
/// is allocated in shared memory.
WalkerSetInitialGuess read_initial_guess_ft(h5::group grp,
                                            utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                            WALKER_TYPES walker_type, int NMO) {
  auto const [wtype, dims] = peek_guess_type(grp, walker_type);
  auto [nspin, npol] = walkerTypeToDims(walker_type);

  return {.walker_type = walker_type,
          .payload = memory::share_from_root(mpi, [&] {
            nda::array<ComplexType,4> M(3, nspin, npol * NMO, NMO);
            M() = 0.0;
            auto URup = M(0,0,nda::ellipsis{});
            utils::h5_read(grp,"UR_alpha",URup);
            auto DRup = M(1,0,nda::ellipsis{});
            utils::h5_read(grp,"DR_alpha",DRup);
            auto VRup = M(2,0,nda::ellipsis{});
            utils::h5_read(grp,"VR_alpha",VRup);
            if(walker_type == COLLINEAR) {
              if(wtype == COLLINEAR) {
                auto URdn = M(0,1,nda::range::all,nda::range(NMO));
                utils::h5_read(grp,"UR_beta",URdn);
                auto DRdn = M(1,1,nda::range::all,nda::range(NMO));
                utils::h5_read(grp,"DR_beta",DRdn);
                auto VRdn = M(2,1,nda::range::all,nda::range(NMO));
                utils::h5_read(grp,"VR_beta",VRdn);
              } else if(wtype == CLOSED) {
                M(0,1,nda::ellipsis{}) = URup();
                M(1,1,nda::ellipsis{}) = DRup();
                M(2,1,nda::ellipsis{}) = VRup();
              } else {
                utils::check(false,"Error: Unknown wtype. ");
              }
            }
            return M;
          })};
}

/// Builds the reference determinant(s) a PHMSD expansion is defined against, for a file that
/// only stores occupation strings. Occupations (and the coefficient signs that go with them)
/// are rewritten in place to refer to the reference that comes out.
void build_PsiT_MO_phmsd(WALKER_TYPES walker_type, int npol, int NMO, int nup,
      int ndown, int ndets, nda::array<ComplexType,1>& coeffs,
      nda::array<int,2>& occs, nda::array<PsiT_Matrix<HOST_MEMORY>,1>& PsiT_MO)
{
  using nda::range;
  auto all = range::all;
  ComplexType one(1.0, 0.0);
  bool trivial_ref = true;
  for(int i=0; i<nup; i++)
    if(occs(0,i) != i) {
      trivial_ref = false;
      break;
    }
  for(int i=0; i<ndown; i++)
    if(occs(0,nup+i) != NMO+i) {
      trivial_ref = false;
      break;
    }

  auto sort_with_sign = [](int NE, auto&& Iwork) {
    RealType ci=1.0;
    for (int i = 0; i < NE; i++)
      for (int j = i + 1; j < NE; j++)
      {
        if (Iwork[j] < Iwork[i])
        {
          ci *= RealType(-1.0);
          std::swap(Iwork[i], Iwork[j]);
        }
      }
    return ci;
  };

  if(trivial_ref) {

    PsiT_MO.resize(1);
    PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({npol*NMO,npol*NMO},1);

    // makes sense to move the reordering of non-compact excitations to here!
    for (int k = 0; k < npol*NMO; k++)
      PsiT_MO(0).emplace_back({k, k}, one);

  } else {

    // reference determinant is non-trivial (occupy bottom nalpha/nbeta states...)
    // build non-trivial reference and redefine excitations with respect to this new reference...
    app_log(1,"Found non-trivial reference determinant. Constructing appropriate reference state.");

    // if beta reference configuration has singly occupied states,
    // you will need separate references
    bool separate_references = false;
    if(walker_type == COLLINEAR)
      for(int i=0; i<ndown; ++i)
        if( *std::find(occs.begin(), occs.begin() + nup, occs(0,nup+i)-NMO) !=
          occs(0,nup+i)-NMO ) {
          separate_references = true;
          break;
        }

    if(separate_references) { // only if collinear and can't find a unique reference
      PsiT_MO.resize(2);
      PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({NMO,NMO},1);
      PsiT_MO(1) = PsiT_Matrix<HOST_MEMORY>({NMO,NMO},1);

      // each spin has its own reference
      int dN_[2] = {0,NMO};
      int E0_[2] = {0,nup};
      int E1_[2] = {nup,nup+ndown};
      for(int is=0; is<2; is++) {
        int dN = dN_[is];
        int E0 = E0_[is];
        int E1 = E1_[is];
        std::vector<int> m(NMO,-1);
        std::vector<int> im(NMO,-1);
        int norbs=0;  // number of states found so far
        // doubly occupied first, and since we checked all beta are doubly occp
        for(int i=E0; i<E1; i++) {
          im[occs(0,i)-dN] = norbs;
          m[norbs++] = occs(0,i)-dN;
        }
        // now add all remaining states
        for(int n=1; n<occs.extent(0); ++n) {
          for(int i=E0; i<E1; i++)
            if(im[occs(n,i)-dN] < 0) {
              im[occs(n,i)-dN] = norbs;
              m[norbs++] = occs(n,i)-dN;
            }
        }
        // now change occupation strings according to the generated map
        for(int n=0; n<occs.extent(0); ++n) {
          for(int i=E0; i<E1; i++) occs(n,i) = im[occs(n,i)-dN]+dN;
          coeffs[n] *= sort_with_sign(E1-E0,occs(n,range(E0,E1)));
        }
        for (int k = 0; k < norbs; k++)
          PsiT_MO(is).emplace_back({k, m[k]}, one);
      } // is
    } else { // separate_references

      PsiT_MO.resize(1);
      PsiT_MO(0) = PsiT_Matrix<HOST_MEMORY>({npol*NMO,npol*NMO},1);

      {

        std::vector<int> m(npol*NMO,-1);
        std::vector<int> im(npol*NMO,-1);
        int norbs=0;  // number of states found so far
        if(walker_type == NONCOLLINEAR) {
          for(int i=0; i<nup; i++) {
	    im[occs(0,i)] = norbs;
	    m[norbs++] = occs(0,i);
	  }
          // now add all remaining states
          for(int n=1; n<occs.extent(0); ++n) {
            for(int i=0; i<nup; i++)
              if(im[occs(n,i)] < 0) {
                im[occs(n,i)] = norbs;
                m[norbs++] = occs(n,i);
              }
          }
        } else {
          // doubly occupied first, and since we checked all beta are doubly occp
          for(int i=0; i<ndown; i++) {
            im[occs(0,nup+i)-NMO] = norbs;
            m[norbs++] = occs(0,nup+i)-NMO;
          }
          // singly occupied now
          for(int i=0; i<nup; ++i)
            if( *std::find(occs.begin()+nup, occs.begin()+nup+ndown,
		occs(0,i)+NMO) != occs(0,i)+NMO ) {
              im[occs(0,i)] = norbs;
              m[norbs++] = occs(0,i);
	    }
          // now add all remaining states
          for(int n=1; n<occs.extent(0); ++n) {
            for(int i=0; i<nup; i++)
              if(im[occs(n,i)] < 0) {
                im[occs(n,i)] = norbs;
                m[norbs++] = occs(n,i);
              }
            for(int i=nup; i<nup+ndown; i++)
              if(im[occs(n,i)-NMO] < 0) {
                im[occs(n,i)-NMO] = norbs;
                m[norbs++] = occs(n,i)-NMO;
              }
          }
        }
        // now change occupation strings according to the generated map
        for(int n=0; n<occs.extent(0); ++n) {
          for(int i=0; i<nup; i++) occs(n,i) = im[occs(n,i)];
          if(walker_type == COLLINEAR)
            for(int i=nup; i<nup+ndown; i++) occs(n,i) = im[occs(n,i)-NMO]+NMO;
          coeffs[n] *= sort_with_sign(nup+ndown,occs(n,all));
        }
        for (int k = 0; k < norbs; k++)
          PsiT_MO(0).emplace_back({k, m[k]}, one);

      }

    } // separate_references

  } // trivial_ref
  // generate compact form
  for(int i=0; i<PsiT_MO.size(); ++i)
    PsiT_MO(i).remove_empty_spaces();
}

template<MEMORY_SPACE MEM>
Wavefunction<MEM> nomsd_from_params(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                    const WavefunctionParameters& params, const TrialFile& trial,
                                    WALKER_TYPES walker_type, Hamiltonian& h, int targetNW) {
  h5::group ngrp = trial.wgrp.open_group("NOMSD");

  // Read common trial wavefunction input options.
  int ndets_to_read = params.ndets_to_read;
  nda::array<ComplexType,1> ci;
  WALKER_TYPES input_wtype{};
  getCommonInput(ngrp, ndets_to_read, ci, input_wtype);
  utils::check(input_wtype != NONCOLLINEAR || walker_type == NONCOLLINEAR,
      "Error: Trial wavefunction is NONCOLLINEAR and requires NONCOLLINEAR walkers. walker_type: {}", walkerTypeToString(walker_type));

  auto [nup, ndown] = broadcast_number_of_electrons({trial.nup, trial.ndown}, input_wtype, walker_type);

  // Create Trial wavefunction.
  auto PsiT = read_nomsd_wavefunction<MEM>(ngrp, ndets_to_read, walker_type, trial.NMO, nup, ndown);

  // Set initial walker's Slater matrix.
  auto guess = read_initial_guess(ngrp, walker_type, trial.NMO, nup, ndown);

  auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, PsiT);

  auto build = [&](auto&& orbs) {
    using OrbType = typename std::remove_cvref_t<decltype(orbs)>::value_type;
    return Wavefunction<MEM>(NOMSD<MEM,OrbType>(params, trial.NMO, nup, ndown, walker_type, mpi,
                                                std::move(HOps), std::move(ci), std::move(orbs), targetNW),
                             std::move(guess));
  };

  if(resolved(params.dense_trial, "dense_trial")) {
    return build(to_dense_shared<MEM>(*mpi, PsiT));
  }
  return build(std::move(PsiT));
}

template<MEMORY_SPACE MEM>
Wavefunction<MEM> nomsd_ft_from_params(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                       const WavefunctionParameters& params, const TrialFile& trial,
                                       WALKER_TYPES walker_type, Hamiltonian& h, int targetNW) {
  h5::group ngrp = trial.wgrp.open_group("NOMSD");

  // Read common trial wavefunction input options.
  int ndets_to_read = params.ndets_to_read;
  nda::array<ComplexType,1> ci;
  WALKER_TYPES input_wtype{};
  getCommonInput(ngrp, ndets_to_read, ci, input_wtype);
  utils::check(input_wtype != NONCOLLINEAR || walker_type == NONCOLLINEAR,
      "Error: Trial wavefunction is NONCOLLINEAR and requires NONCOLLINEAR walkers. walker_type: {}", walkerTypeToString(walker_type));

  int ntau = trial.nup;
  utils::check(trial.ndown == 0, "expected ndown dimension to be 0 at finite temperature");

  // Create Trial wavefunction.
  auto PsiT = read_nomsd_wavefunction<MEM>(ngrp, ndets_to_read, walker_type, trial.NMO, ntau);

  // Set initial walker's Slater matrix.
  auto guess = read_initial_guess_ft(ngrp, *mpi, walker_type, trial.NMO);

  // at finite temperature the operators are built against the identity rather than the
  // trial, so they are not half-rotated
  nda::array<PsiT_Matrix<MEM>, 2> IMat(PsiT.extent(0), PsiT.extent(1));
  int dim = PsiT(0,0,0).extent(1); // dim = NMO
  for(int i = 0; i < IMat.extent(0); ++i) {
    for(int s = 0; s < IMat.extent(1); ++s) {
      IMat(i,s) = math::sparse::identity<ComplexType>(dim);
    }
  }
  auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, IMat);

  auto build = [&](auto&& orbs) {
    using OrbType = typename std::remove_cvref_t<decltype(orbs)>::value_type;
    return Wavefunction<MEM>(NOMSD_FT<MEM,OrbType>(params, trial.NMO, ntau, walker_type, mpi,
                                                   std::move(HOps), std::move(ci), std::move(orbs), targetNW),
                             std::move(guess));
  };

  if(resolved(params.dense_trial, "dense_trial")) {
    return build(to_dense_shared<MEM>(*mpi, PsiT));
  }
  return build(std::move(PsiT));
}

template<MEMORY_SPACE MEM>
Wavefunction<MEM> phmsd_from_params(std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
                                    const WavefunctionParameters& params, const TrialFile& trial,
                                    WALKER_TYPES walker_type, Hamiltonian& h, int targetNW) {
  // Implementation notes:
  //  - PsiT_MO: [Nact, NMO] where Nact is the number of active space orbitals,
  //                     those that participate in the ci expansion
  //  - The half rotation is done with respect to the supermatrix PsiT_MO
  //  - Need to calculate Nact and create a mapping from orbital index to active space index.
  //    Those orbitals in the corresponding virtual space (not in active) map to -1 as a precaution.

  auto [nspin, npol] = walkerTypeToDims(walker_type);
  int const NMO = trial.NMO;
  // phmsd does not support conversion to noncollinear
  int const nup = trial.nup;
  int const ndown = trial.ndown;

  h5::group ngrp = trial.wgrp.open_group("PHMSD");
  int ndets_to_read = params.ndets_to_read;

  // 1. Read occupancies and coefficients.
  nda::array<PsiT_Matrix<HOST_MEMORY>, 1> PsiT_MO;
  nda::array<int,2> occs;
  nda::array<ComplexType,1> coeffs;
  std::string orb_type;
  app_log(1,"Reading PHMSD wavefunction from {}", params.filename);
  read_ph_wavefunction_hdf(ngrp, coeffs, occs, ndets_to_read, walker_type,
                           NMO, nup, ndown, PsiT_MO, orb_type);
  utils::check_shape(occs, "occs", ndets_to_read, nup + ndown);
  app_log(1,"Finished reading PHMSD wavefunction ");

  // 2. Build reference MOs (PsiT_MO) if needed.
  utils::check(orb_type == "occ" || orb_type == "mixed", "Invalid wavefunction type:{}",orb_type);
  if(orb_type == "occ") {
    build_PsiT_MO_phmsd(walker_type,npol,NMO,nup,ndown,ndets_to_read,coeffs,occs,PsiT_MO);
  }

  // 3. Construct Structures.
  ph_excitations<int, ComplexType, MEM> abij = build_ph_struct<MEM>(coeffs, occs, ndets_to_read, npol*NMO, nup, ndown);

  // returns the number of times a given orbital appears in the ci expansion
  auto orb_counts = find_active_space(walker_type, abij, NMO, nup, ndown);

  // mapping from old to new occupation indexes. Orbitals in the virtual space keep -1.
  std::map<int, int> mo2active;
  for(int i = 0; i < 2 * NMO; i++) {
    mo2active[i] = -1;
  }
  if(PsiT_MO.extent(0) == 1) {
    // RHF/GHF: one reference serves both spins, so an orbital occupied in either of them
    // takes the same active index
    int nactive = 0;
    for(int i = 0; i < npol*NMO; i++) {
      if(walker_type == COLLINEAR) {
        if(orb_counts[i] >= 0 || orb_counts[i + NMO] >= 0) {
          if(orb_counts[i] >= 0) {
            mo2active[i] = nactive;
          }
          if(orb_counts[i + NMO] >= 0) {
            mo2active[i + NMO] = nactive;
          }
          nactive++;
        }
      } else if(orb_counts[i] >= 0) {
        mo2active[i] = nactive++;
      }
    }
  } else {
    // UHF: each spin numbers its own active orbitals
    int nactive_alpha = 0;
    int nactive_beta = 0;
    for(int i = 0; i < npol*NMO; i++) {
      if(orb_counts[i] >= 0) {
        mo2active[i] = nactive_alpha++;
      }
      if(orb_counts[i + NMO] >= 0) {
        mo2active[i + NMO] = nactive_beta++;
      }
    }
  }

  // now that mappings have been constructed, map indexes of excited state orbitals
  // to the corresponding active space indexes
  {
    // map reference
    auto refc = abij.reference_configuration();
    for (int i = 0; i < nup + ndown; i++, ++refc)
      *refc = mo2active[*refc];
    for (int n = 1; n < abij.maximum_excitation_number()[0]; n++)
    {
      auto it  = abij.alpha_begin(n);
      auto ite = abij.alpha_end(n);
      for (; it < ite; ++it)
      {
        auto exct = (*it) + n; // only need to map excited state indexes
        for (int np = 0; np < n; ++np, ++exct)
          *exct = mo2active[*exct];
      }
    }
    for (int n = 1; n < abij.maximum_excitation_number()[1]; n++)
    {
      auto it  = abij.beta_begin(n);
      auto ite = abij.beta_end(n);
      for (; it < ite; ++it)
      {
        auto exct = (*it) + n; // only need to map excited state indexes
        for (int np = 0; np < n; ++np, ++exct)
          *exct = mo2active[*exct];
      }
    }
  }

  auto guess = read_initial_guess(ngrp, walker_type, NMO, nup, ndown);

  auto n_unique(abij.number_of_unique_excitations());
  app_log(1,"Number of unique determinants per spin channel: {} {} ",
              n_unique[0],n_unique[1]);
  nda::array<int,1> counts_alpha(n_unique[0],0);
  nda::array<int,1> counts_beta(n_unique[1],0);
  for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
  {
    ++counts_alpha(std::get<0>(*it));
    ++counts_beta(std::get<1>(*it));
  }

  using ucsr_mat_t = math::sparse::ucsr_matrix<ComplexType, HOST_MEMORY, int, int>;
  std::vector<ucsr_mat_t> unsorted_det_coupling;
  unsorted_det_coupling.reserve(2);
  unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[0],n_unique[1]}, counts_alpha));
  unsorted_det_coupling.emplace_back(ucsr_mat_t({n_unique[1],n_unique[0]}, counts_beta));

  for (auto it = abij.configurations_begin(); it < abij.configurations_end(); ++it)
  {
    // sparse matrix
    unsorted_det_coupling[0].emplace({std::get<0>(*it),std::get<1>(*it)},
                                   std::conj(std::get<2>(*it)));
    unsorted_det_coupling[1].emplace({std::get<1>(*it),std::get<0>(*it)},
                                   std::conj(std::get<2>(*it)));
  }

  // csr matrix with determinant couplings in COLLINEAR case
  nda::array<PsiT_Matrix<MEM>,1> det_coupling_matrix;
  det_coupling_matrix.resize(2);
  det_coupling_matrix(0) = unsorted_det_coupling[0];
  det_coupling_matrix(1) = unsorted_det_coupling[1];

  // PsiT_MO carries n_ref reference(s): 1 for RHF/GHF (combined), 2 for UHF.
  // The Hamiltonian expects one entry per spin channel (nspin), so for a single
  // combined reference under COLLINEAR we duplicate it across both spins -- the
  // same walker-type conversion every other wavefunction path performs on read.
  int const n_ref = PsiT_MO.extent(0);

  // 2d version just for getHamiltonianOperations (copy, so PsiT_MO survives)
  nda::array<PsiT_Matrix<MEM>, 2> PsiT_2d(1, nspin);
  for(int i=0; i<nspin; i++) {
    PsiT_2d(0,i) = PsiT_MO(i % n_ref);
  }
  auto HOps = h.getHamiltonianOperations<MEM>(walker_type, mpi, PsiT_2d);

  // 1-d array for PHMSD keeps the original reference count (1 or 2)
  nda::array<PsiT_Matrix<MEM>, 1> PsiT_1d(n_ref);
  for(int i=0; i<n_ref; i++) {
    PsiT_1d(i) = std::move(PsiT_MO(i));
  }

  return Wavefunction<MEM>(PHMSD<MEM>(params, walker_type, NMO, nup, ndown, mpi, std::move(HOps),
                                      std::move(abij), std::move(det_coupling_matrix),
                                      std::move(PsiT_1d), targetNW),
                           std::move(guess));
}

}

template<MEMORY_SPACE MEM>
Wavefunction<MEM> Wavefunction<MEM>::from_params(
    std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> mpi,
    const WavefunctionParameters& params, WALKER_TYPES walker_type,
    bool finiteT, Hamiltonian& h, int targetNW) {
  app_log(1, section(std::format("Initializing Wavefunction \"{}\"", params.name)));

  utils::check(not params.name.empty(), "Missing required input: name");
  utils::check(not params.filename.empty(), "Missing required input: filename");

  auto const [NMO, nup_in_wfn, ndown_in_wfn] = read_info_from_wfn(params.filename, "any");
  utils::check(ndown_in_wfn <= nup_in_wfn,
               "Error nup < ndown: Up spin must be the majority spin. nup: {}, ndown: {}",
               nup_in_wfn, ndown_in_wfn);

  WAVEFUNCTION_TYPES const wfn_type = peek_wavefunction_type(params.filename, *mpi);

  // everyone reading for now, change it problematic
  h5::file file(params.filename,'r');
  h5::group grp(file);
  TrialFile const trial{.wgrp = grp.open_group("Wavefunction"),
                        .NMO = NMO, .nup = nup_in_wfn, .ndown = ndown_in_wfn};

  if(wfn_type == NOMSD_WFN) {
    app_log(1,"Wavefunction type: NOMSD");
    if(finiteT) {
      return nomsd_ft_from_params<MEM>(mpi, params, trial, walker_type, h, targetNW);
    }
    return nomsd_from_params<MEM>(mpi, params, trial, walker_type, h, targetNW);
  }
  if(wfn_type == PHMSD_WFN) {
    app_log(1,"Wavefunction type: PHMSD");
    return phmsd_from_params<MEM>(mpi, params, trial, walker_type, h, targetNW);
  }
  APP_ABORT("Error: Unknown wave-function wfn_type: {}", wfn_type);
}

template<MEMORY_SPACE MEM>
MEMORY_SPACE Wavefunction<MEM>::get_memory_space() const {
  return std::visit([&](auto&& a) { return a.get_memory_space(); }, var);
}

template<MEMORY_SPACE MEM>
int Wavefunction<MEM>::number_of_cholesky_vectors() const {
  return std::visit([&](auto&& a) { return a.number_of_cholesky_vectors(); }, var);
}

template<MEMORY_SPACE MEM>
RealType Wavefunction<MEM>::energy_offset() const {
  return std::visit([&](auto&& a) { return a.energy_offset(); }, var);
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
ComplexType Wavefunction<MEM>::getReferenceWeight(int i) const {
  return std::visit([&](auto&& a) { return a.getReferenceWeight(i); }, var);
}

template<MEMORY_SPACE MEM>
void Wavefunction<MEM>::getReferences(memory::buffered_array<MEM,ComplexType,3>& Refs) {
  std::visit([&](auto&& a) { a.getReferences(Refs); }, var);
}

template<MEMORY_SPACE MEM>
HamiltonianType Wavefunction<MEM>::getHamType() const {
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
