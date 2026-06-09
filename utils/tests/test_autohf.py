# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import pytest
# for Dev
import numpy as np
import scipy.sparse as sps

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.director import HamiltonianDirector
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import SpinSymm

pytest.importorskip("autohf")
from autohf.solver import lattice_hf # TODO: move partially to afqmctools

class TestAutoHFConverters:
    """
    use 4x4 square lattice as test case.

    Test that the energy of a somewhat arbitrary state is correct
    Useful for modifying the energy routines
    """

    @pytest.mark.dev
    def test_square_lattice(self):
      lattice = get_lattice(
            params={
                'L1' : 4,
                'L2' : 4,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
            }
        )
      builder = HamiltonianBuilder(
                lattice=lattice,
                spin_symm=SpinSymm.COLLINEAR
                    )
      # add standard Hubbard terms
      builder.nth_neighbor_hopping([1.0,0.2])
      builder.onsite_hubbard(1.23)
      builder.hubbard_U1_density_density(0.3,nth_neighbor=1)
      builder.hubbard_U2_spin_spin(0.25,nth_neighbor=1)
      builder.hubbard_Jij(0.124,nth_neighbor=1)
      builder.finalize()
      Ne = 5
      hf_settings = dict(
          numSteps = -1,
          output = None,
          opt_method="lbfgs",
          ansatz="SD_ROT",
          nelec = [Ne,Ne],
          numTrials = 1,
          seed = 1,
          noncollinear = False
      )
      orbs = np.arange(lattice.N_sites*Ne*2).reshape(2,lattice.N_sites,Ne)
      orbs = np.linalg.qr(orbs)[0]
      data = lattice_hf(
          hamiltonian=builder.hamiltonian,
          lattice=lattice,
          settings=hf_settings,
        initial_guess=np.hstack(orbs),
      )
      assert np.allclose(data["E_final"],-5.161777352659391,atol=1e-5)
    
    @pytest.mark.dev
    def test_square_lattice_soc(self):
      lattice = get_lattice(
            params={
                'L1' : 4,
                'L2' : 4,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
            }
        )
      builder = HamiltonianBuilder(
                lattice=lattice,
                spin_symm=SpinSymm.NONCOLLINEAR
                    )
      # add standard Hubbard terms
      builder.nth_neighbor_hopping(1.0)
      builder.rashba_soc(0.7)
      builder.onsite_hubbard(1.23)
      builder.hubbard_U1_density_density(0.3,nth_neighbor=1)
      builder.hubbard_U2_spin_spin(0.25,nth_neighbor=1)
      builder.hubbard_Jij(0.124,nth_neighbor=1)
      builder.finalize()
      Ne = 5
      hf_settings = dict(
          numSteps = -1,
          output = None,
          opt_method="lbfgs",
          ansatz="SD_ROT",
          nelec = [Ne,Ne],
          numTrials = 1,
          seed = 1,
          noncollinear = True
      )
      orbs = np.arange(lattice.N_sites*Ne*2).reshape(2,lattice.N_sites,Ne)
      orbs = np.linalg.qr(orbs)[0]
      orbs = np.block([[orbs[0],np.zeros_like(orbs[1])],
                         [np.zeros_like(orbs[0]),orbs[1]]])
      data = lattice_hf(
          hamiltonian=builder.hamiltonian,
          lattice=lattice,
          settings=hf_settings,
        initial_guess=orbs,
      )
      assert np.allclose(data["E_final"],-4.631279114235719,atol=1e-5)

