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

#include <cstdlib>
#include <tuple>
#include <algorithm>

#pragma once

namespace sfqmc
{

template<typename IType>
inline std::tuple<IType, IType> FairDivideBoundary(IType me, IType ntot, IType npart)
{
  IType bat     = ntot / npart;
  IType residue = ntot % npart;
  if (me < residue)
    return std::make_tuple(me * (bat + 1), (me + 1) * (bat + 1));
  else
    return std::make_tuple(me * bat + residue, (me + 1) * bat + residue);
}

template<class IVec>
inline void FairDivide(int ntot, int npart, IVec& adist)
{
  if (adist.size() != npart + 1)
    adist.resize(npart + 1);
  int bat     = ntot / npart;
  int residue = ntot % npart;
  adist[0]    = 0;
  for (int i = 1; i < npart; i++)
  {
    if (i < residue)
      adist[i] = adist[i - 1] + bat + 1;
    else
      adist[i] = adist[i - 1] + bat;
  }
  adist[npart] = ntot;
}

} // sfqmc
