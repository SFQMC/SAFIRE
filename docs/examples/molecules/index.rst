We provide the following worked examples in which we
go through the entire workflow. 
The quantum chemistry workflow requires some external 
quantum chemistry code to generate integrals and a trial wavefunction.
For the convenience, we use
PySCF since it can be directly invoked within
interactive Python notebooks; however, using the information
in the tutorials, these same calculations can be performed starting
from other quantum chemistry codes so long as they can write a FCIDUMP,
and can print wavefunction information. Both of these features are
ubiquitous in modern quantum chemistry codes.

.. use hidden toctree to compile the jupyter notebooks in the examples/molecules folder

.. toctree::
   :hidden:

   02_B_atom_SHCI_trial_wfn/06_SHCI_trial_wavefunction.ipynb
   03_n2-phmsd/03_nitrogen_dimer_pec.ipynb
   06_Pb-spin-orbit/06_Electron_affinity_of_Pb.ipynb
   08_local_embedding/08_local_embedding.ipynb

1. :ref:`Oxygen atom - simple AFQMC run <example_molecule_o_atom>`
2. :doc:`B atom - SHCI trial wavefunction </examples/molecules/02_B_atom_SHCI_trial_wfn/06_SHCI_trial_wavefunction>`
3. :doc:`N2 molecule PEC using ph-MSD trial wavefunction </examples/molecules/03_n2-phmsd/03_nitrogen_dimer_pec>`
4. :ref:`Vanadium atom - fully polarized AFQMC <example_molecule_v_fully_polarized>`
5. :ref:`Li2 molecule with frozen core approximation <example_molecule_li2_frozen_core>`
6. :doc:`Lead atom with spin-orbit coupling </examples/molecules/06_Pb-spin-orbit/06_Electron_affinity_of_Pb>`
7. :ref:`3d Transition Metal Oxides benchmark <example_molecule_3d_tmo_benchmark>`
8. :doc:`Local embedding </examples/molecules/08_local_embedding/08_local_embedding>`

.. 3. `Charge density of the water molecule <https://colab.research.google.com/drive/1uiF2R6CE_cSQ7m9uj2R2aS0KB1Un3Bn_>`_ 


