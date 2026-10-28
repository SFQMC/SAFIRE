.. _user_manual:

User Manual
===========


This user manual is a reference description of the SAFIRE code, its components and their input and output formats. If you are a practitioner interested in running a standard calculation, refer to the :ref:`tutorials` or :ref:`user_examples` first.

Running an :ref:`afqmc` calculation requires several input files

.. code-block::

  $ ls tutorials/solids/01_hello_safire/files/

  afqmc.json
  hamiltonian.h5
  wavefunction.h5

- ``afqmc.json`` is the :ref:`Job File <input_afqmc>`
- ``hamiltonian.h5`` is a :ref:`Hamiltonian File <Hamiltonian-classes>`
- ``wavefunction.h5`` is a :ref:`Wavefunction File <Wavefunction-classes>`

Running SAFIRE

.. code-block::

  $ safire afqmc.json

writes ``afqmc.results.h5``, the :ref:`output_afqmc` containing all the measurements performed by the :ref:`estimators` during the simulation.

The following pages go into detail about each of the components.  

.. toctree::
   :maxdepth: 1
   
   user_manual/input_description_afqmc      
   user_manual/hamiltonians
   user_manual/wavefunctions
   user_manual/estimators
   user_manual/walkers
   user_manual/output


