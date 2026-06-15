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

#pragma once

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <ctype.h>

#include "AFQMC/config.h"

#include "nda/nda.hpp"
#include "numerics/sparse/sparse.hpp"

namespace sfqmc::afqmc
{

// For each (spin, kpoint), returns the list of global PsiT row indices (occupied
// orbitals) whose Bloch orbital lives at that kpoint. The rows of each list are in
// increasing global-row order. This does NOT assume the PsiT rows are sorted
// block-diagonally by kpoint; it only requires that each orbital is confined to a
// single kpoint block (a single-kpoint Bloch state). For an already-sorted
// wavefunction, list (is,ik) is exactly the contiguous range that the previous
// count-based implementation implied.
template<MEMORY_SPACE MEM>
auto nocc_per_kpoint(WALKER_TYPES type, int nkpts, nda::array<PsiT_Matrix<MEM>,2> const& PsiT)
{
  int nspin = (type == COLLINEAR?2:1);
  int npol = (type == NONCOLLINEAR?2:1);
  int ndet = PsiT.extent(0);
  int nspin_in_PsiT = PsiT.extent(1);
  int NMO = PsiT(0,0).extent(1)/npol;
  int nbnd = NMO/nkpts;
  utils::check(NMO%nkpts==0, "NMO%nkpts != 0");
  utils::check(nspin_in_PsiT==1 or nspin_in_PsiT==nspin, "Shape mismatch");

  // builds the per-(spin,kpoint) row-index lists for determinant id
  auto rows_per_kp = [&](int id) {
    nda::array<nda::array<int,1>,2> occ(nspin,nkpts);
    for(int is=0; is<nspin; is++) {
      auto cols = nda::to_host(PsiT(id,is%nspin_in_PsiT).columns());
      auto row_begin = nda::to_host(PsiT(id,is%nspin_in_PsiT).row_begin());
      auto row_end = nda::to_host(PsiT(id,is%nspin_in_PsiT).row_end());
      int nel = PsiT(id,is%nspin_in_PsiT).extent(0);
      // first pass: count occupied orbitals per kpoint
      nda::array<int,1> cnt(nkpts);
      cnt() = 0;
      std::vector<int> kpt(nel);
      for(int ia=0; ia<nel; ++ia) {
        int k0 = (cols(row_begin(ia))/nbnd)%nkpts;
        int k1 = (cols(row_end(ia)-1)/nbnd)%nkpts;
        utils::check( k0==k1, "Error: orbital row {} spans multiple kpoint blocks (not a single-kpoint Bloch state).",ia);
        kpt[ia] = k0;
        cnt(k0)++;
      }
      // second pass: allocate and fill, preserving increasing global-row order
      for(int ik=0; ik<nkpts; ++ik) occ(is,ik) = nda::array<int,1>(cnt(ik));
      nda::array<int,1> pos(nkpts);
      pos() = 0;
      for(int ia=0; ia<nel; ++ia) {
        int k0 = kpt[ia];
        occ(is,k0)(pos(k0)++) = ia;
      }
    }
    return occ;
  };

  // compute based on first determinant, check all others have matching occupations
  auto nocc_per_kp = rows_per_kp(0);
  for(int id=1; id<ndet; ++id) {
    auto nocc = rows_per_kp(id);
    for(int is=0; is<nspin; is++)
      for(int ik=0; ik<nkpts; ++ik)
        utils::check(nocc(is,ik).size() == nocc_per_kp(is,ik).size(),
          "Error: Occupation numbers for each kpoint differ between determinants, idet:{}",id);
  }
  return nocc_per_kp;
}

// Total number of occupied orbitals for spin is across all kpoints.
inline int nelec_for_spin(nda::array<nda::array<int,1>,2> const& nocc, int is)
{
  int n = 0;
  for(int ik=0; ik<nocc.extent(1); ++ik) n += int(nocc(is,ik).size());
  return n;
}

// Maximum occupation over all (spin,kpoint) blocks.
inline int max_nocc_per_kpoint(nda::array<nda::array<int,1>,2> const& nocc)
{
  int n = 0;
  for(int is=0; is<nocc.extent(0); ++is)
    for(int ik=0; ik<nocc.extent(1); ++ik)
      n = std::max(n, int(nocc(is,ik).size()));
  return n;
}

// True if a row-index list is a contiguous ascending run (rows(a)==rows(0)+a).
// This holds for an already kpoint-sorted wavefunction, enabling a single slice
// copy; otherwise the rows must be gathered individually.
inline bool contiguous_rows(nda::array<int,1> const& rows)
{
  for(int a=1; a<int(rows.size()); ++a)
    if(rows(a) != rows(a-1)+1) return false;
  return true;
}

} // namespace sfqmc::afqmc
