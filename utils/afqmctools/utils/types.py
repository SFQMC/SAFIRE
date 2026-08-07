# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from enum import IntFlag

class SpinSymm(IntFlag):
    """
    An enumerated class/type in order to
        document the different types of
        spin_symmetry.

    The values are meaningful!! a larger
        value implies less symmetry. This
        convention is used to match the
        internal conventions in the C++
    """

    CLOSED=1
    COLLINEAR=2
    NONCOLLINEAR=3


def get_spin_symm_enum(spin_symm):
    if isinstance(spin_symm,str):
        spin_symm = spin_symm.lower()

    if spin_symm in ('rhf','closed',SpinSymm.CLOSED,1):
        return SpinSymm.CLOSED
    elif spin_symm in ('uhf','collinear',"col",SpinSymm.COLLINEAR,2):
        return SpinSymm.COLLINEAR
    elif spin_symm in ('ghf','noncollinear',"nc",SpinSymm.NONCOLLINEAR,3):
        return SpinSymm.NONCOLLINEAR
    else:
        raise ValueError(f"Unknown Spin symmetry: {spin_symm}")
