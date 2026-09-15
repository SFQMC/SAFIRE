# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from safiretools.types import SpinSymm


def test_values_match_walker_types():
    """
    SpinSymm's integer values must match WALKER_TYPES in src/AFQMC/config.h
    (CLOSED=1, COLLINEAR=2, NONCOLLINEAR=3) since they're used as the C++
    wire format directly.
    """
    assert SpinSymm.CLOSED == 1
    assert SpinSymm.COLLINEAR == 2
    assert SpinSymm.NONCOLLINEAR == 3
