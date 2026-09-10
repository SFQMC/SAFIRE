# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Python tooling for SAFIRE.

The public API is re-exported flat here, so callers do not need to know the
module tree. This surface is filled in as the package is built out; see
DESIGN.md for the intended final set.

**This list defines what is user-facing.** A name re-exported here is public;
anything reachable only by a deeper path is an implementation detail. See
"The top-level import surface defines what is user-facing" in DESIGN.md.
"""

from safiretools.hamiltonian.base import Hamiltonian
from safiretools.hamiltonian.model.builder import HamiltonianBuilder
from safiretools.hamiltonian.model.lattice import Lattice
from safiretools.hamiltonian.model.lattice_hamiltonian import LatticeHamiltonian
from safiretools.hamiltonian.molecular import MolecularHamiltonian
from safiretools.hamiltonian.periodic import PeriodicHamiltonian
from safiretools.types import SpinSymm
from safiretools.wavefunction.base import Wavefunction
from safiretools.wavefunction.nomsd import NOMSDWavefunction
from safiretools.wavefunction.phmsd import PHMSDWavefunction

__all__ = [
    'Hamiltonian',
    'HamiltonianBuilder',
    'Lattice',
    'LatticeHamiltonian',
    'MolecularHamiltonian',
    'NOMSDWavefunction',
    'PHMSDWavefunction',
    'PeriodicHamiltonian',
    'SpinSymm',
    'Wavefunction',
]
