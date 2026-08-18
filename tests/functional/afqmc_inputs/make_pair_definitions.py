#!/usr/bin/env python3
# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

# TODO: Merge this with other input scripts!

from pathlib import Path

import numpy as np

from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.io import write_pair_correlators

ROOT = Path(__file__).resolve().parent

# One entry per system directory. `directions` are passed to Lattice.get_directed_pairs, and
# must all be periodic for the lattice at hand: a direction that leaves the cell yields a
# negative entry, and the estimator skips those without advancing its flat index, which
# shifts every element after it.
SYSTEMS = {
    "square_4x4_hubbard_nup5_ndn5": dict(
        lattice=dict(L1=4, L2=4, boundary1="PBC", boundary2="PBC"),
        directions=["s", "+x", "+y"],
        nbands=1,
        nmo=16,
    ),
    "square_6x1_hubbard_kanamori_nup6_ndn6": dict(
        lattice=dict(L1=6, L2=1, boundary1="PBC"),
        directions=["s", "+x"],
        nbands=2,
        nmo=12,
    ),
}


def orbital_maps(lattice_params, directions, nbands):
    """The index offsets of `directions`, in basis space, as int32 arrays."""
    lattice = get_lattice(params=lattice_params)
    site_maps = lattice.get_directed_pairs(directions=directions)
    # Lattice.sites is appended in _index_map order, so site_map is indexed by site index.
    return {
        direction: np.array([ibar * nbands + band for ibar in site_map
                             for band in range(nbands)], dtype=np.int32)
        for direction, site_map in site_maps.items()
    }


def main():
    for data_dir, spec in SYSTEMS.items():
        maps = orbital_maps(spec["lattice"], spec["directions"], spec["nbands"])
        for direction, offsets in maps.items():
            if offsets.shape != (spec["nmo"],):
                raise ValueError(f"{data_dir}/{direction}: got {offsets.shape[0]} offsets, "
                                 f"expected NMO = {spec['nmo']}")
            if offsets.min() < 0:
                raise ValueError(f"{data_dir}/{direction}: leaves the cell; only periodic "
                                 f"directions are usable")

        out = ROOT / data_dir / "pair_correlators.h5"
        out.unlink(missing_ok=True)
        write_pair_correlators(out, maps)
        print(f"wrote {out}: {', '.join(maps)}")


if __name__ == "__main__":
    main()
