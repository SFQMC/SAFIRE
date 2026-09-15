# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import numpy as np
import scipy.sparse as sps

from afqmctools.utils.types import SpinSymm,get_spin_symm_enum

import logging
logger = logging.getLogger(__name__)

class Hamiltonian:
    """
    A container for Hamiltonian terms. Terms are stored
    by the naming conventions used in the afqmc code.
        
    An *example* of the internal structure is as follows::

        terms = {
            'tij' : [tij_csr_1,tij_csr_2,...],
            'Ui' : [U_csr],
            'U1ij' : [U1ij_csr],
            'U2ij' : [U2ij_csr],
            'Jij' : [Jij_csr_1,Jij_csr_2]
        }

    A dictionary with keys corresponding to the standard
    generalized Hamiltonian term names stores a *list*
    of terms of the same type. The list of terms can be
    collapsed into a single term if desired; however,
    keeping terms separate provides more flexibility,
    especially regarding the Hubbard-Stratonovich
    transformation. Each entry in the list should be
    a `scipy.sparse.csr_array` instance.

    By default, assigning terms to the Hamiltonian
    will append an additional term to the relevant list.
    
    Design notes for Developers:

    - should serve only as a container for Hamiltonian terms
        which are built via the HamiltonianBuilder.
    - this class should not directly reference the underlying lattice.
    """

    def __init__(self) -> None:
        self.terms = dict()
        self.num_components = 0
        
        self._nbands = 0
        self._spin_symm = SpinSymm.CLOSED
        
        self.real_valued = True
        self.twist = None

        self.nsites = None

    def __getitem__(self,key):
        return self.terms[key]

    def __setitem__(self,key,value,term_index=None):
        """
        Set the internal 'terms' dictionary using key-value pair.
        By default, will append a new term to the dictionary (unless
        and index `term_index` is given, in which case the term at index
        `term_index` is *overwritten*)

        For one-body terms, we want to collapse all components into
            a single component (no advantage to keeping them separate)
        """
        if key not in self.keys():
            self.terms[key] = list()

        if self.real_valued and not value._real_valued:
            self.real_valued = False

        if term_index is None:
             self.terms[key].append(value)
        else:
            self.terms[key][term_index] = value

        self.num_components = self.num_components + 1

    def _pop_term(self,key):
        """
        WARNING: this method is destructive! Only use
            if you are sure this is correct.
        
        Clear the terms in self.terms with key `key`
        """
        self.num_components = self.num_components - 1
        return self.terms.pop(key)

    def recount_components(self):
        self.num_components = sum(
            [ len(comps) for comps in self.terms.values() ]
        )

    def keys(self):
        return self.terms.keys()

    def get(self,value,default=None):
        return self.terms.get(value,default)

    @property
    def nbands(self):
        return self._nbands
    

    @nbands.setter
    def nbands(self,value):
        self._nbands = value


    @property
    def spin_symm(self):
        return self._spin_symm


    @spin_symm.setter
    def spin_symm(self,new_spin_symm:SpinSymm):
        """
        Can only move from High-symmetry to Low-Symmetry.

        If we really insist on going form low-to-high symmetry,
          we can first delete the existing spin-symmetry value.
        """
        new_spin_symm = get_spin_symm_enum(new_spin_symm)
        
        if (self._spin_symm is None) or (self._spin_symm.value <= new_spin_symm.value ):
            self._spin_symm = new_spin_symm
        else:
            raise ValueError("Tried to change Hamiltonian from low spin symmetry to high spin symmetry")


    @spin_symm.deleter
    def spin_symm(self):
        """
        Explicitly invoking this deleter via (for ex)
        
        ```python
        ham = Hamiltonian()
        Hamiltonian.spin_symm = SpinSymm.CLOSED
        Hamiltonian.spin_symm = SpinSymm.COLLINEAR # allowed!
        Hamiltonian.spin_symm = SpinSymm.CLOSED # ValueError!!

        del Hamiltonian.spin_symm
        Hamiltonian.spin_symm = SpinSymm.CLOSED # now allowed!
        ```
        
          is considered
            an advanced use case.
        """
        del self._spin_symm
        self._spin_symm = None

    def get_one_body(self):
        """
        return the one-body part of the Hamiltonian as a single scipy sparse
        csr matrix
        """
        one_body_terms = self.get("tij")
        if one_body_terms is not None:
            return sum(one_body_terms).csr_array
        else:
            return None # TODO: just return sps.csr_array(shape); and follow it through the autohf solver.

    def get_U(self):
        """
        return the U part of the Hamiltonian as a single scipy sparse
        csr matrix. This includes U, U1, and U2
        
        Uij is separated into U+U1, and U2 to
            avoid issues with shape consistency while summing below.
        """
        U_interaction_terms = self.get("Uij",[])

        M = self.nsites * self.nbands

        U1_matricies = []
        U2_matricies = []
        for term in U_interaction_terms:
            if term.shape == (M,M):
                U1_matricies.append(term.csr_array)
            elif term.shape == (2*M,M):
                U2_matricies.append(term.csr_array)
            else:
                logger.dev(f"Invalid U term shape: {term.shape} with M = {M}")
                raise ValueError("[for developers] invalid U matrix shape found")

        U1 = sum(U1_matricies)
        U2 = sum(U2_matricies)

        # this is necessary U1 may not be present in the Hamiltonian
        U1_shape = getattr(U1,"shape",(0,0))
        U2_shape = getattr(U2,"shape",(0,0))
        
        shape = max(U1_shape,U2_shape)
        U = sps.lil_matrix(shape)

        U[:U1_shape[0],:U1_shape[1]] = U1
        U[:U2_shape[0],:U2_shape[1]] += U2

        return U.tocsr()

    def get_J(self):
        """
        return the J part of the Hamiltonian as a single scipy sparse
        csr matrix.
        """
        J_interaction_terms = self.get("Jij",[])
        return sum( term.csr_array for term in J_interaction_terms )
    
    def get_heisenberg(self):
        """
        return the get_heisenberg part of the Hamiltonian as a single scipy sparse
        csr matrix.
        """
        J_interaction_terms = self.get("J_heisenberg",[])
        return sum( term.csr_array for term in J_interaction_terms )

class HamiltonianComponent:
    """
    Class to encapsulate the CSR representation,
        and metadata corresponding
        to individual Hamiltonian components.
    """

    def __init__(
            self,
            csr_array:sps.csr_array,
            model_type,
            spin_symm=SpinSymm.CLOSED,
            **kwargs
        ) -> None:
        
        if not isinstance(csr_array, sps.sparray):
          csr_array = sps.csr_array(csr_array)
        self.csr_array = csr_array
        self.model_type = model_type

        self.spin_symm = spin_symm

        self.metadata = kwargs
        self._real_valued = np.iscomplexobj(csr_array)

        self.max_nnz = np.max(csr_array.indptr[1:]-csr_array.indptr[:-1])

        if "hst_type" in self.metadata.keys():
            self.hubbard_strat_type = self.metadata["hst_type"]
        else:
            self.hubbard_strat_type = None

    def __add__(self,other):
        """
        *Non-commutative* Addition of Hamiltonian terms require that both 'self' and 'other'
            be the same type of model term - i.e. 'tij', 'Ui', etc.
    
        Addition is *not commutative* as defined here! We take metadata from `self`

        spin_symmetry of the result is taken as the lowest symmetry between `self`
            and `other`.
        """
        if isinstance(other,HamiltonianComponent) and self.model_type == other.model_type:
            
            if (self.hubbard_strat_type != other.hubbard_strat_type):
                raise ValueError(
                    "[Error For Developers] "
                    "Addition between Hamiltoninan components with "
                    "differing Hubbard-Stratonovich types is not allowed."
                )

            # Note: larger spin_symm VALUE is less symmetry
            if self.spin_symm.value > other.spin_symm.value:
                spin_symm = self.spin_symm
            else:
                spin_symm = other.spin_symm
            
            return HamiltonianComponent(
                csr_array=self.csr_array+other.csr_array,
                model_type=self.model_type,
                spin_symm=spin_symm,
                **self.metadata
            )
    
        elif other is None or other == 0:
            return self
        else:
            raise ValueError(f"Can't add HamiltonianComponent and {other} woth type {type(other)}")

    def __radd__(self,other):
        return self.__add__(other)

    def __str__(self) -> str:
        return f"{self.csr_array}"

    @property
    def shape(self):
        return self.csr_array.shape

    def toarray(self):
        return self.csr_array.toarray()

def _ZeroComponent(*args,**kwargs):
    print("[WARNING] invoked _ZeroComponent which is deprecated and will be removed!")
    return None
