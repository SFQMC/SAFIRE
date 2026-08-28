# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from enum import IntEnum


class SpinSymm(IntEnum):
    """
    Spin symmetry of a Hamiltonian, Wavefunction, or Walker ordered by increasing
    generality. Values match WALKER_TYPES in src/AFQMC/config.h exactly.
    """

    CLOSED = 1
    COLLINEAR = 2
    NONCOLLINEAR = 3
