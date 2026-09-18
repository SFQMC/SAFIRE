# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""Reading a results.h5 and turning its bins into averages with error bars."""

from safiretools.results.results import Results
from safiretools.results.stats import jackknife, rebinning_analysis, standard_error

__all__ = [
    'Results',
    'jackknife',
    'rebinning_analysis',
    'standard_error',
]
