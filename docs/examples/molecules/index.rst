.. _examples_molecules:

Quantum Chemistry
~~~~~~~~~~~~~~~~~

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

.. toctree::
   :numbered:
   :glob:
   :maxdepth: 1

   /examples/molecules/0*/*

.. 3. :doc:`Charge density of the water molecule </examples/molecules/tbd_H2O_charge_density/index>`
