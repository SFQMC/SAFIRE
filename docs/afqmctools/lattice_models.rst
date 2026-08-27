.. _lattice_models_py:

Lattice Model Tools for AFQMC
=============================


Overview
--------

A lattice model Hamiltonian consists of a series of Hamiltonian terms (i.e. a hopping term, 
hubbard on-site interaction, etc.) which are defined in terms of a generic lattice.
In practical calculations, a specific lattice must be used and some set of Hamiltonian terms 
are constructed on the specific lattice.
In the AFQMC lattice model tools, the creation of a `Lattice` class instance and of a `Hamiltonian` class instance 
are separated. 
First, the `Lattice` instance is generated 
The lattice is an object which is aware of geometrical properties such as the positions of lattice sites,
neighbors, boundary conditions, etc.


Lattice Class
-------------


Hamiltonian Builder
-------------------



Model Hamiltonian Builder
=========================

TODO: explain the combined (lattice,band) indices

The afqmctools Python module includes tooling to build general,
multi-band Hubbard-Kanamori Hamiltonians of the type:

.. math::

   \hat{H} = \sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'} 
   + \sum_{i} U_i \hat{n}_{i\uparrow} \hat{n}_{i\downarrow} \\
   + \sum_{i<j} U_{ij}^1 (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} ) 
   + \sum_{i<j} U_{ij}^2 (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} ) \\
   + \sum_{i<j} J_{ij} ( 
     \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
     +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} 
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
     +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
     ),

where :math:`i`,\ :math:`j` are combined lattice and band indices,
:math:`\hat{c}^\dagger_{i\sigma}`, :math:`\hat{c}_{j\sigma'}`
create/annihilate and electron on the site (and band) corresponding to
:math:`i`/:math:`j` with spin :math:`\sigma`/:math:`\sigma'`,
:math:`\hat{n}_{i\sigma}` is the number operator,
:math:`t^{\sigma \sigma'}_{ij}` includes all one-body terms
(:math:`n^{th}`-order neighbor hopping, spin-orbit coupling, etc.),
:math:`U` is the traditional on-site hubbard interaction, :math:`U^1` is
a density-density interaction, :math:`U^2` is
a spin-spin interaction, and :math:`J` is a Hund's coupling term.
:math:`U^1`, :math:`U^2`, and :math:`J` are typically only non-zero between
bands on the same lattice site, but are by no means limited in this way.

Currently, SAFIRE can use any form of this Hamiltonian. For
convenience, we supply tooling to build the most common forms of the
Hamiltonian described above. Only on-site, but inter-band,
:math:`U`,\ :math:`U^1`,\ :math:`U^2`, and :math:`J` are implemented;
however, a motivated user can build a custom Hamiltonian and save it in
the format described below in the section, " input file
format”

Building a Model Hamiltonian
----------------------------

This section explains how to build a model Hamiltonian using the
included Python tooling. Input format/conventions are described in more
detail in the section `Model Hamiltonian Builder Input
Conventions <#model-hamiltonian-builder-input-conventions>`__

A Hamiltonian may be built from a Python ``dict`` as follows:

.. code:: python

   from afqmctools.hamiltonian.model.builder import HamiltonianBuilder

   params = {
       'hamiltonian' : {
           't' : 1.0,
           'U' : 2.0,
           'U1' : 1.5,
           'U2' : 1.0,
           'J' : 0.5
     },
     'lattice' : {
         'L1' : 6,
         'L2' : 1,
         'boundary1' : 'PBC',
     }
   }

   hamiltonian = HamiltonianBuilder.from_input(
       source = params
   ).hamiltonian

   ...

the Hamiltonian can then be saved in the SAFIRE format with:

.. code:: python

   ...

   from afqmctools.utils.io import write_model_hamiltion

   write_model_hamiltion(
       hamiltonian=hamiltonian,
           fname='afqmc.h5',
           nelec=(6,6)
   )

Alternatively, the cli includes a tool to automatically build and save a
Hamiltonian from parameters saved in an input file (in toml format). 
If ``afqmctools`` was installed as described in
`Installation <#installation>`__, a model Hamiltonian can be built
using:

.. code:: bash

   $ make_model_ham -i model_ham_params.toml -o afqmc.h5 

For example, the following input file will make the same
Hamiltonian as above.

In TOML:

.. code:: toml

   [hamiltonian]
   nbands = 2
   t = 1.0
   U = 2.0
   U1 = 1.5
   U2 = 1.0
   J = 0.5

   [lattice]
   L1 = 6
   L2 = 1
   boundary1 = "PBC"

   [cli_params]
   nelec = [6,6]


Model Hamiltonian Builder Input Conventions
-------------------------------------------

Model Hamiltonian parameters can be specified either directly as a
Python dict, or via an input file in toml format. 
both cases share the following conventions.

The model Hamiltonian builder will reference two different input blocks;
The ``hamiltonian`` block - where
Hamiltonian parameters are specified - and a ``lattice`` block - where
the lattice that underlies the Hamiltonian is defined.
Any supported Hamiltonian can be constructed on any supported 
lattice.

`hamiltonian` block
~~~~~~~~~~~~~~~~~~~

The ``hamiltonian`` block consists of the following fields. In the
following, :math:`i,j` are combined indices including both lattice
indices (:math:`\mu`,\ :math:`\nu`), and band indices
(:math:`m`,\ :math:`m'`)

TODO: This is a little too busy. Try making each parameter a sub-sub-section (whatever level is appropriate), and space out each case.

-  ``t`` : specifies hopping terms,
   :math:`\sum_{ij,\sigma \sigma'}t^{\sigma \sigma'}_{ij}\hat{c}^\dagger_{i\sigma}\hat{c}_{j\sigma'}`.
   ``t`` may be any of:

   -  None : no hopping : :math:`t^{\sigma \sigma'}_{ij} = 0`
   -  a single number : interpreted as the nearest-neighbor hopping.
      Assumes no hopping between band :
      :math:`t^{\sigma \sigma'}_{ij} = t\delta_{mm'}\delta_{\nu,\mu\pm1}\delta_{\sigma\sigma'}`
   -  a 1-D array of length :math:`n-1` : interpreted as multiple
      hopping terms where ``t[n-1]`` is the hopping strength between
      :math:`n^{th}`-order neighbors. Assumes no hopping between bands :
      :math:`t^{\sigma \sigma'}_{ij} = \sum_{n=1} t_n \delta_{mm'}\delta_{\nu,\mu\pm n}\delta_{\sigma\sigma'}`
   -  a 2-D array with shape :math:`N_{bands}` x :math:`N_{bands}` :
      defines hopping between bands and/or allows different bands to use
      different hopping strengths. In this case, the same ``t`` array is
      used for all lattice sites.
      :math:`t^{\sigma \sigma'}_{ij} = t_{mm'} \delta_{\nu,\nu\pm 1}\delta_{\sigma\sigma'}`

-  ``U`` : specifies the on-site Hubbard interaction. May be
   any of:

   -  None : no hubbard U : :math:`U_i = 0`
   -  a single number : interpreted as a constant U across lattice
      sites and bands :math:`U_i = U \delta_{mm'} \delta_{\mu\nu}`
   -  a 1-D array of length :math:`N_{bands}` : interpreted as
      :math:`U_m` where :math:`m` is the band index, but otherwise
      constant in lattice index.  :math:`U_i = U_m \delta_{\mu\nu}`
   -  a 1-D array of length :math:`N_{sites}` : interpreted as
      :math:`U_\mu` where :math:`\mu` is the lattice site index, but
      otherwise constant in band index. :math:`U_i = U_{\mu} \delta_{mm'}`

TODO: update the description of ``U1``, ``U2``, and ``J`` to include the intersite case.

-  ``U1`` : specifies the on-site, density-density Hubbard
   interaction.
   :math:`\sum_{i<j} U_{ij}^1 (\hat{n}_{i\uparrow} \hat{n}_{j\downarrow} + \hat{n}_{i\downarrow} \hat{n}_{j\uparrow} )`
   May be any of:

   -  None : no density-density Hubbard interaction :math:`U_{ij}^1 = 0`
   -  a single number : interpreted as a constant U1 for all valid
      interband interactions, on the same site.
      :math:`U_{ij}^1 = U^1 \delta_{\mu\nu} \forall m,m'`
   -  a 2-D array with shape :math:`N_{bands}` x :math:`N_{bands}` :
      interpreted as the onsite, interband :math:`U^1` interaction
      which is the same for all sites. The diagonal is ignored since it
      should be specified via by the parameter ``U``.
      :math:`U_{ij}^1 = U^1_{mm'} \delta_{\mu\nu}`

-  ``U2`` : specifies the interband, on-site spin-spin Hubbard
   interaction. It follows similar input conventions as ``U1``, but is
   understood to describe interactions between electrons of the same
   spin. See ``U1`` for more.
   :math:`\sum_{i<j} U_{ij}^2 (\hat{n}_{i\uparrow} \hat{n}_{j\uparrow} + \hat{n}_{i\downarrow} \hat{n}_{j\downarrow} )`
-  ``J`` : specifies the on-site Hund’s interaction. It follows similar
   input conventions as ``U1``, but is understood to describe
   interactions between electrons of the same spin. See ``U1`` for more.
   :math:`\sum_{i<j} J_{ij} (\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow} +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow} +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow} +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow})`

`lattice` block
~~~~~~~~~~~~~~~~~~~

The ``lattice`` block consists of the following fields. In the
following, :math:`\hat{a}_1` and :math:`\hat{a}_2` are the lattice
vectors.

-  ``L1`` : the number of lattice sites along :math:`\hat{a}_1`.
-  ``L2`` : the number of lattice sites along :math:`\hat{a}_2`.
-  (optional) ``type`` : either ``square`` or ``triangular`` : use
   either a square lattice or a triangular lattice.
-  (optional) ``boundary1`` : either ``'open'`` or ``'pbc'``. the type
   of boundary to use in the direction perpendicular to
   :math:`\hat{a}_2`. ``open`` is the default.
-  (optional) ``boundary2`` : either ``'open'`` or ``'pbc'``. the type
   of boundary to use in the direction perpendicular to
   :math:`\hat{a}_1`. ``open`` is the default.
