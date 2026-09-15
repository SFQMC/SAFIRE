# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
    Test the statistics tools for averaging the scalar
       table results form the AFQMC code.

    Desired Behavior:
    1. gets correct average/errorbars on known test data
    2. 
"""
import sys
from pathlib import Path

import pytest

import cli.scalar_tab as st
from stats.scalar_dat import analyze_scalar_data

TEST_ROOT = Path(__file__).resolve().parent


@pytest.fixture
def arguments(monkeypatch):
    """
    Fixture to set command line arguments
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            'scalar_stats',
            (TEST_ROOT/'data'/'qmc.s000.scalar.dat').as_posix(),
            '-s','time','-e','5.0'
        ]
    )
    return st.parse_args()

def test_scalar_tab_energy(arguments,capsys):
    '''
    Test the statistics tools for averaging the scalar
       table results form the AFQMC code.

    Desired Behavior:
    1. gets correct average/errorbars on known test data
    2. 
    '''

    correct_output = "-15.801084 +/-   0.001025 6.43  5.0/10.0"
    analyze_scalar_data(arguments)
    assert correct_output in capsys.readouterr().out
