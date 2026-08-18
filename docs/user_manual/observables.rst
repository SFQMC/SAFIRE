
.. _observables:

Observables
===========

This section describes the observables available in SAFIRE. 
Some observables only make sense in certain application areas.
Here, we provide a list of the currently implemented observables,
along with a brief description, and the mathematical form of each observable.
It is the user's responsibility to ensure that the observable is appropriate 
for their specific use case.
Below is a summary of all observables currently implemented in SAFIRE.

Full 1-RDM
----------

Computes the full one-particle reduced density matrix (1-RDM).

.. math::
	
    \hat{O} = \hat{\Gamma^1} = \sum_{ij\sigma\sigma'} \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma'}


Sample Input Block
~~~~~~~~~~~~~~~~~~

.. code-block:: json

	{
	"estimator": {
		"name" : "mixed",
		"onerdm" : {
			"name" : "my_one_rdm"
		}
	}
	}

Wavefunction Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with NOMSD Wavefunctions only.

Hamiltonian Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with all Hamiltonian classes.

Walker Type Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with CLOSED, COLLINEAR, and NONCOLLINEAR walkers.




Two-body reduced density matrix (2-RDM)
---------------------------------------

Calculates the full two-particle reduced density matrix, including all spin blocks.

.. math::

	\hat{O} = \hat{\Gamma}^2 = \sum_{ijkl} \sum_{\sigma\sigma'} \hat{c}_{i\sigma}^\dagger \hat{c}_{j\sigma'}^\dagger \hat{c}_{k\sigma'} \hat{c}_{l\sigma}

.. caution::

	Computing and storing the full 2-RDM can be very memory intensive and greatly increases the run time.
	Use only when absolutely necessary, and restrict to small systems.

Sample Input Block
~~~~~~~~~~~~~~~~~~

.. code-block:: json

	{
	"estimator": {
		"name" : "mixed",
		"twordm" : {
			"name" : "my_two_rdm"
		}
	}
	}


Wavefunction Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with NOMSD Wavefunctions only.

Hamiltonian Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with all Hamiltonian classes.

Walker Type Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with CLOSED, and COLLINEAR walkers.
Not currently implemented for NONCOLLINEAR walkers.



Diagonal Two-body reduced density matrix
----------------------------------------

Computes the diagonal elements of the two-particle reduced density matrix (2-RDM).

.. math::

	\hat{O} = \sum_{ij} \sum_{\sigma \sigma'} \hat{c}_{i\sigma}^\dagger \hat{c}^\dagger_{j\sigma'} \hat{c}_{j\sigma'} \hat{c}_{i\sigma} 

Sample Input Block
~~~~~~~~~~~~~~~~~~
.. code-block:: json

	{
	"estimator": {
		"name" : "mixed",
		"diag2rdm" : {
			"name" : "my_diag_two_rdm"
		}
	}
	}


Wavefunction Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with NOMSD Wavefunctions only.

Hamiltonian Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with all Hamiltonian classes.

Walker Type Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with CLOSED, COLLINEAR, and NONCOLLINEAR walkers.



Spin-Spin Correlation
---------------------

Computes walker-averaged spin-spin correlation functions.

.. math::
	\hat{O} = \hat{S}_i \cdot \hat{S}_j

where

.. math::
	
    \hat{S}_i = \frac{1}{2} \sum_{\sigma,\sigma'} \hat{c}_{i\sigma}^\dagger \vec{\sigma}_{\sigma\sigma'} c_{i\sigma'}


Sample Input Block
~~~~~~~~~~~~~~~~~~
.. code-block:: json

	{
	"estimator": {
		"name" : "mixed",
		"spinspin" : {
			"name" : "my_spinspin"
		}
	}
	}


Wavefunction Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with NOMSD Wavefunctions only.

Hamiltonian Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with all Hamiltonian classes.

Walker Type Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with CLOSED, COLLINEAR, and NONCOLLINEAR walkers.


Pair Correlation Functions
--------------------------

.. important::

	Pair correlation functions are an experimental feature.

Calculates walker-averaged pair correlation functions based on the pair definitions in the input HDF5 file.
By pair correlation functions, we mean

.. math::

	P^{\alpha \beta}_{ij} ≡ ⟨ Δ^\dagger_{i,\alpha}\Delta_{j,\beta} ⟩,

where  :math:`i,j` are basis set indices,
:math:`\alpha, \beta` are spatial offset indices, and

.. math::

	\Delta^\dagger_{i,\alpha} \equiv \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{i + e_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{i + e_\alpha \uparrow} \right),

where :math:`e_\alpha` is the *index offset* associated with the spatial offset.
For example, :math:`e_\alpha` may correspond to a :math:`+x` offset.
For convenience, we define,

.. math::

	\bar{i}_\alpha \equiv i + e_\alpha,


such that,

.. math::

	\Delta^\dagger_{i,\alpha} \equiv \frac{1}{\sqrt{2}} \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{\bar{i}_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{\bar{i}_\alpha \uparrow} \right),


and

.. math::

	\Delta_{j,\beta} \equiv \frac{1}{\sqrt{2}} \left( c_{\bar{j}_\beta\downarrow} c_{j \uparrow}  - c_{\bar{j}_\beta \uparrow} c_{j \downarrow} \right),


Using this language, the pair correlation function can be written as,

.. math::

	P^{\alpha \beta}_{ij} ≡ \frac{1}{2} ⟨ \left(c^{\dagger}_{i\uparrow} c^{\dagger}_{\bar{i}_\alpha \downarrow} -  c^{\dagger}_{i\downarrow} c^{\dagger}_{\bar{i}_\alpha \uparrow} \right)  \left( c_{\bar{j}_\beta\downarrow} c_{j \uparrow}  - c_{\bar{j}_\beta \uparrow} c_{j \downarrow} \right) ⟩.

And, so

.. math::

	\hat{O} = \hat{P}^{\alpha \beta}

SAFIRE requires a set of :math:`\bar{i}_\alpha = i + e_\alpha`, in the HDF5 input file.
Specific pairings may be selected from those which are defined in the json input file.
This allows the user to skip some of the offsets saved in the HDF5 file if desired.
In the sample input block below, we select the pair types :math:`s`, :math:`+x`, :math:`-x`, :math:`+y`, :math:`-y`, and :math:`xy` from the HDF5 file.

Sample Input Block
~~~~~~~~~~~~~~~~~~
.. code-block:: json

	{
	"estimator": {
		"name" : "mixed",
		"pair_correlators" : {
			"name" : "my_pair_correlators",
        	"filename" : "pair_definitions.h5",
        	"pair_type" : ["s","+x","-x","+y","-y","xy"]
		}
	}
	}


Wavefunction Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with NOMSD Wavefunctions only.

Hamiltonian Class Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The pair correlation function estimator is most appropriate for lattice model Hamiltonians;
however, it can be used with all Hamiltonian classes assuming that pair definitions are provided in the input HDF5 file.

Walker Type Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~

Compatible with COLLINEAR, and NONCOLLINEAR walkers.
Not implemented for CLOSED walkers.

