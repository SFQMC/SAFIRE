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
"""

from safiretools.hamiltonian.model.lattice import Lattice

__all__ = [
    'Lattice',
]
