.. _Wavefunction-classes:

Wavefunction File formats
-------------------------

The trial wavefunction in AFQMC is typically a linear combination of Slater determinants,

.. math::
  |\Psi_\mathrm{T} \rangle = \sum^{N_\mathrm{det}}_n C_n | \Phi_n \rangle

where :math:`C_n` is a complex-valued coefficient, and :math:`|\Phi_n\rangle` are Slater determinants which are not necessarily orthogonal to each other. Of course, each Slater determinant consists of some set of single-particle orbitals, :math:`\{ \psi_p \}`, such that,

.. math::
  \psi_{p} = \sum_i \bar{C}_{ip} \phi_i

where :math:`\{\phi_i\}` are the chosen orthonormal basis set orbitals. Slater determinants can either be represented explicitly as a Slater matrix,

.. math::
  \Phi = \begin{bmatrix}
    \bar{C}_{00} &\bar{C}_{01} & \bar{C}_{02} & \dots  & \bar{C}_{0N} \\
    \bar{C}_{10} & \bar{C}_{11} & \bar{C}_{12} & \dots  & \bar{C}_{1N} \\
    \vdots & \vdots & \vdots & \ddots & \vdots \\
    \bar{C}_{M0} & \bar{C}_{M1} & \bar{C}_{M2} & \dots  & \bar{C}_{MN}
  \end{bmatrix}

where :math:`M,N` are the number of basis functions and electrons, respectively, or in terms of an "occupation vector", which is simply a list of orbital indices which should be occupied.
See the :ref:`SAFIRE tutorials <tutorials>` for tutorials and examples on generating and writing trial wavefunctions.

SAFIRE implements two basic types of trial wavefunction:

1. **"particle-hole" multi-Slater determinant (ph-mSD) trial wavefunctions** which is a configuration interaction-like wavefunction where :math:`\langle \Phi_n |\Phi_m\rangle = \delta_{nm} \forall n,m`.
   In this case, the Slater determinants are specified in terms of a set of orbitals, and a list of :math:`C_n` and corresponding occupancy vectors. See the HDF5 file format description in :ref:`PHMSD <phmsd_wavefunction>` for more.

2. **non-orthogonal multi-slater Determinant (NOMSD) trial wavefunction** where *strictly* :math:`\langle \Phi_n |\Phi_m\rangle \neq 0 \forall n,m`. In this case, the Slater determinants are specified as a list of :math:`C_n` and corresponding Slater matrices. See the HDF5 file format description in :ref:`NOMSD <nomsd_wavefunction>` for more.

We'll explore each of these wavefunction types in more detail below.

.. note::
   Using the ph-mSD trial wavefunction is recommended when using a large number of determinants since it is significantly more memory efficient than the NOMSD trial wavefunction,
   and since, in some cases, fast Woodbury updates can be applied.

.. warning::
  When using NOMSD trial wavefunctions, it is important that each Slater determinant has a non-zero overlap with all other Slater determinants to avoid singular values. Use ph-mSD for orthogonal expansions of Slater determinants.

AFQMC allows for two types of multi-determinant trial wavefunctions: non-orthogonal multi
Slater determinants (NOMSD) or SHCI/CASSCF style particle-hole multi Slater determinants
(PHMSD).


.. _nomsd_wavefunction:

NOMSD
~~~~~

.. code-block:: text

    h5dump -n wfn.h5

    HDF5 "wfn.h5" {
        FILE_CONTENTS {
            group      /
            group      /Wavefunction
            group      /Wavefunction/NOMSD
            group      /Wavefunction/NOMSD/PsiT_0
            dataset    /Wavefunction/NOMSD/PsiT_0/data_
            dataset    /Wavefunction/NOMSD/PsiT_0/dims
            dataset    /Wavefunction/NOMSD/PsiT_0/jdata_
            dataset    /Wavefunction/NOMSD/PsiT_0/pointers_begin_
            dataset    /Wavefunction/NOMSD/PsiT_0/pointers_end_
            group      /Wavefunction/NOMSD/PsiT_1
            dataset    /Wavefunction/NOMSD/PsiT_1/data_
            dataset    /Wavefunction/NOMSD/PsiT_1/dims
            dataset    /Wavefunction/NOMSD/PsiT_1/jdata_
            dataset    /Wavefunction/NOMSD/PsiT_1/pointers_begin_
            dataset    /Wavefunction/NOMSD/PsiT_1/pointers_end_
            dataset    /Wavefunction/NOMSD/ci_coeffs
        }
    }

Note that the :math:`\alpha` components of the trial wavefunction are stored under
``PsiT_{2n}`` and the :math:`\beta` components are stored under ``PsiT_{2n+1}``.

-  ``/Wavefunction/NOMSD/PsiT_{2n}/data_`` Array of length :math:`nnz` containing non-zero
   elements of :math:`n`-th :math:`\alpha` component of trial wavefunction walker
   wavefunction. Note the **conjugate transpose** of the Slater matrix is stored.
-  ``/Wavefunction/NOMSD/PsiT_{2n}/dims`` Array of length 3 containing
   :math:`[M,N_{\alpha},nnz]` where :math:`nnz` is the number of non-zero elements of this
   Slater matrix
-  ``/Wavefunction/NOMSD/PsiT_{2n}/jdata_`` CSR indices array.
-  ``/Wavefunction/NOMSD/PsiT_{2n}/pointers_begin_`` CSR format begin index pointer array.
-  ``/Wavefunction/NOMSD/PsiT_{2n}/pointers_end_`` CSR format end index pointer array.
-  ``/Wavefunction/NOMSD/ci_coeffs`` :math:`N_D` length array of ci coefficients. Stored
   as complex numbers.
-  ``spin_type`` attribute of ``/Wavefunction/NOMSD``: ``"closed"``, ``"collinear"`` or
   ``"noncollinear"``.

No sizes are stored separately. :math:`N_D` is the length of ``ci_coeffs``, :math:`M` and
:math:`N_\alpha` are the columns and rows of ``PsiT_0``, and :math:`N_\beta` the rows of
``PsiT_1``.


.. _phmsd_wavefunction:

PHMSD
~~~~~

.. code-block:: text

    h5dump -n wfn.h5

    HDF5 "wfn.h5" {
        FILE_CONTENTS {
            group      /
            group      /Wavefunction
            group      /Wavefunction/PHMSD
            dataset    /Wavefunction/PHMSD/ci_coeffs
            dataset    /Wavefunction/PHMSD/occa
            dataset    /Wavefunction/PHMSD/occb
            dataset    /Wavefunction/PHMSD/type
        }
    }

-  ``/Wavefunction/PHMSD/ci_coeffs`` :math:`N_D` length array of ci coefficients. Stored
   as complex numbers.
-  ``spin_type`` attribute of ``/Wavefunction/PHMSD``: ``"collinear"``.
-  ``number_of_orbitals`` attribute of ``/Wavefunction/PHMSD``: the number of spatial
   orbitals :math:`M`.
-  ``/Wavefunction/PHMSD/occa``, ``/Wavefunction/PHMSD/occb`` Integer arrays of shape
   :math:`[N_D,N_\alpha]` and :math:`[N_D,N_\beta]` describing the determinant occupancies.
   For example if :math:`(N_\alpha=N_\beta=2)`, :math:`N_D=2`, :math:`M=4`, and
   :math:`|\Psi_\mathrm{T}\rangle = |0,1\rangle|0,1\rangle + |0,1\rangle|0,2\rangle` then
   occa = :math:`[[0, 1], [0, 1]]` and occb = :math:`[[0, 1], [0, 2]]`.
-  ``/Wavefunction/PHMSD/type`` integer 0/1. 1 implies trial wavefunction is written in
   different basis than the underlying basis used for the integrals. If so a matrix of
   orbital coefficients is required to be written in the NOMSD format. If 0 then assume
   wavefunction is in same basis as integrals.

:math:`N_D` is the length of ``ci_coeffs``, and :math:`N_\alpha` and :math:`N_\beta` the
widths of ``occa`` and ``occb``. :math:`M` is the one size that is stored explicitly, since the
occupancies alone do not span the orbitals.


.. _initial_walkers:

Initial walkers
~~~~~~~~~~~~~~~

A trial wavefunction file does not store an initial walker. Every walker of a walker set starts
out as a determinant derived from a wavefunction instead: the determinant with the largest
coefficient of a NOMSD wavefunction, or the reference determinant of a PHMSD one. By default
that is the wavefunction of the execute block that introduces the walker set; the ``from``
parameter of the walker set names a different one, either by name or as an inline block:

.. code-block:: json

    {
      "wavefunction": [
        {"name": "uhf", "filename": "wfn_uhf.h5"},
        {"name": "rohf", "filename": "wfn_rohf.h5"}
      ],
      "execute": [
        {
          "wavefunction": "uhf",
          "walker_set": {"from": {"wavefunction": "rohf"}}
        }
      ]
    }

Starting from an ROHF determinant rather than the UHF trial itself can speed up equilibration
considerably. Only the first execute block falls back to a default walker set; a later execute
block without a ``walker_set`` continues with the walker set of the one before it.

