# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from pathlib import Path

import numpy as np

from afqmctools.systems.lattice import Lattice,get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import SpinSymm, Hamiltonian
import afqmctools.utils.io as afqmc_io


class _HamiltonianDirector:
    """
    Director base class for invoking specific sets of build steps via a 
    `_HamiltonianBuilder`

    Parameters
    ----------
    lattice : afqmctools.systems.lattice.Lattice, optional
        the Lattice instance which describes the geometry of the lattice. 
        If not specified, the derived class is responsible for generating 
        a Lattice instance.
    builder : afqmctools.hamiltonian.model.builder.HamiltonianBuilder, optional
        the builder that will be directed by HamiltonianDirector. If not 
        specified, a default HamiltonianBuilder will be used (this is what 
        almost all users will want).

    Attributes
    ----------
    builder : afqmctools.hamiltonian.model.builder.HamiltonianBuilder
        the HamiltonianBulder instance which will be directored by 
        HamiltonianDirector
    
    Methods
    -------
    build(*args,**kwargs)
        builds and return a Hamilton instance. args and kwargs are 
        forwarded to the derived Director class which implements
        build steps.

    """
    def __init__(
            self,
            lattice:Lattice=None,
            builder:HamiltonianBuilder=None,
            **kwargs
        ) -> None:
        if builder is None:
            self.builder = HamiltonianBuilder(lattice=lattice,**kwargs)

        if not isinstance(self.builder,HamiltonianBuilder):
            raise TypeError(
                "builder must be an instance of HamiltonianBuilder"
            )

        self._active_builder = True
        self._built = False

    def build(self,*args,**kwargs) -> Hamiltonian:
        '''Build and return a Hamiltonian instance

        Notes
        -----
        args, and kwargs are forwarded to the derived class's
            implementation of build steps.
        '''
        if self._active_builder:
            return self._build(*args,**kwargs)
        else:
            raise RuntimeError(
                "No active builder instance. Either bind a new "
                "HamiltonianBuilder using .bind_builder() or create a new HamiltonianDirector"
            )
    
    def release_builder(self) -> HamiltonianBuilder:
        '''Release the builder instance to the caller

        Returns
        -------
        builder : afqmctools.hamiltonian.model.builder.HamiltonianBuilder
            the HamiltonianBuilder instance that was directed by the
            HamiltonianDirector.

        Examples
        --------
        >>> director = HamiltonianDirector("input.toml")
        >>> builder = director.release_builder()
        >>> builder.custom_one_body(scipy.sparse.csr_array(numpy.diag([1.0]*12)))
        >>> director.bind_builder(builder)
        >>> hamiltonian = director.build()
        '''
        builder = self.builder
        self.builder = None
        self._active_builder = False
        return builder

    def bind_builder(self, builder:HamiltonianBuilder,force=False) -> None:
        '''Bind a builder to the director

        Parameters
        ----------
        builder : afqmctools.hamiltonian.model.builder.HamiltonianBuilder
            the HamiltonianBuilder instance that will be directed by the
            HamiltonianDirector.

        Examples
        --------
        Add a custom one-body term to the HamiltonianBuilder, while using the 
        HamiltonianDirector to manage the built-in build steps.
        >>> director = HamiltonianDirector("input.toml")
        >>> builder = director.release_builder()
        >>> builder.custom_one_body(scipy.sparse.csr_array(numpy.diag([1.0]*12)))
        >>> director.bind_builder(builder)
        >>> hamiltonian = director.build()
        '''
        if self.builder is not None:
            raise RuntimeError("HamiltonianBuilder already has a HamiltonianBuilder. "
                               "to force binding a new builder and overwritting the "
                               "old builder set force=True")
        self.builder = builder
        self._active_builder = True


def _get_builder_kwargs(**kwargs):
    
    _builder_keys = [
        'nbands',
        'spin_symm'
        ]

    builder_kwargs = {}

    if 'spin_symm' not in kwargs:
        kwargs['spin_symm'] = SpinSymm.COLLINEAR

    for key in _builder_keys:
        if key in kwargs.keys():
            builder_kwargs[key] = kwargs[key]
    
    return builder_kwargs


class HamiltonianDirector(_HamiltonianDirector):
    r"""Directs building a model Hamiltonian based on a source input file or dict. 

    Parameters
    ----------
    source : str | dict
        a description of the Hamiltonian (and possibly lattice) parameters.
        See Notes for details on Hamiltonian parameter syntax and conventions.
        If a str is given, it is interpreted as the name of an input file in .toml
        format and parameters are read from that file.
        If a dict is given, then parameters are read directly from the dict.
    lattice : afqmctools.systems.lattice.Lattice, optional
        the Lattice instance which describes the geometry of the lattice. 
        If not specified, the lattice will be built from the source input.
    builder : afqmctools.hamiltonian.model.builder.HamiltonianBuilder, optional
        the builder that will be directed by HamiltonianDirector. If not
        specified, a default HamiltonianBuilder will be used (this is what
        almost all users will want).
        

    Methods
    -------
    build()
        builds and returns a Hamilton instance based on the build steps
        defined in the source input.
    release_builder()
        releases the HamiltonianBuilder instance to the caller. This disables
        the build() method until a HamiltonianBuilder is bound to the director.
        A HamiltonianBuilder can be bound to the director using the bind_builder()
        method.
    bind_builder(builder:HamiltonianBuilder,force=False)
        binds a HamiltonianBuilder to the director. If force is True, then the
        current HamiltonianBuilder is replaced with the new builder. If force is
        False, then an error is raised if a HamiltonianBuilder is already bound to
        the director.

    
    Notes
    -----
    The Hamiltonian is built by a sequence of build steps which are specified
    in the source input. Build steps are specified as key-value pairs in the 
    'hamiltonian' section of the input file or in the source dict.
    By default, nearest-neighbor hopping is included in the Hamiltonian; but
    all interaction terms must be explicitly included in the input.

    The following build steps are supported, and are listed as `key : description`.
    Also see the `HamiltonianBuilder` class for more details on the build steps,
    and for finer control over Hamiltonian construction. The acceptable input dimensions
    are also listed for each build step; other input dimensions will raise an error.
    
    - t : nth-order neighbor hopping term; input convention are as follows:
      
      - if t is a scalar, then nerest-neighbor hopping is included with strength t
      - if t is a 1-dimensional list (i.e. [t1,t2,...,tn]), then up to nth-order hopping is included
      - if t is a 2-dimensional with shape (nbands, nbands), then
        the t is interpreted as an on-site inter-band hopping matrix
      - if t is a list of 2-dimensional arrays, then each element is interpreted as an on-site 
        inter-band hopping matrix. This is functionally the same as :math:`t_{mn} = \sum t^{(i)}_{mn}`
        where :math:`t^{(i)}` is the ith element of the list and :math:`m,n` are band indices.

    - U : onsite Hubbard interaction term; input convention are as follows:
      
      - if U is a scalar, then the onsite Hubbard interaction term is included with strength U
        and applied to all sites and bands.
      - if U is 1-dimensional with length nbands (i.e. [U1,U2,...,Um] where for an m-band model),
        then the onsite Hubbard interaction term is included with strength U_i applied to band i
        and uniformaly across sites.
      - if U is 1-dimensional with length nsites (i.e. [U1,U2,...,Un] where n is the number of sites), 
        then the onsite Hubbard interaction term is included with strength U_i applied to site i
        and uniformly across bands.

    - U1 : Hubbard density-density interaction term. Input convention are as follows:
      
      - if U1 is a scalar, then the Hubbard density-density interaction term is included with strength U1
        and is applied to all sites, and is unfirom across bands.
      - if U1 is a 2-dimensional with shape (nbands, nbands), then U1 is used as an intrasite density-density 
        interaction, and is applied uniformly accross sites.

    - U2 : Hubbard spin-spin interaction term. Input convention are as follows:
      
      - if U2 is a scalar, then the Hubbard spin-spin interaction term is included with strength U2
        and is applied to all sites, and is unfirom across bands.
      - if U2 is a 2-dimensional with shape (nbands, nbands), then U2 is used as an intrasite spin-spin
        interaction, and is applied uniformly accross sites.

    - J : Hund's coupling interaction term. Input convention are as follows:
      
      - if J is a scalar, then the Hund's coupling interaction term is included with strength J
        and is applied to all sites, and is unfirom across bands.
      - if J is a 2-dimensional with shape (nbands, nbands), then J is used as an intrasite 
        Hund's coupling interaction, and is applied uniformly accross sites.
        
    """
    def __init__(
        self, 
        source:str|Path|dict=None,
        lattice:Lattice=None,
        builder:HamiltonianBuilder=None
    ) -> None:
        
        self._ham_params = list()
        self._build_steps = list()

        # expert mode: overrides the default Hubbard-Stratonovich Types
        self._hst_type_overrides = {}

        if isinstance(source,str):
            source_dict = afqmc_io.read_input_params(source)
        elif isinstance(source,dict):
            source_dict = source
        else:
            raise ValueError(
                "Invalid parameter source."
                " source must be either a dict, or a str"
                " containing an input file name (json, toml,"
                " or yaml)"
            )
    
        self._parse_input(source_dict['hamiltonian'])

        if lattice is None:
            print("No lattice instance supplied: building from parameters")
            lattice = get_lattice(params=source_dict['lattice'])
        
        super().__init__(
            lattice,
            builder,
            **_get_builder_kwargs(**source_dict['hamiltonian'])
        )

    def _parse_input(self,input=None):
        """
        Parse input in order to build a sequance
          of build steps.
        """

        input = dict(input)

        _known_params = {
            'nbands' : 1,
            'spindep' : 0 ,
            'twist' : None,
            'afm_pin_type': "staggered",
            'fm_pin_type': "staggered",
        }

        _supported_steps = {
            'nth_neighbor_hopping' : 't',
            'custom_one_body' : 'custom_one_body',
            'onsite_hubbard' : 'U',
            'hubbard_U1_density_density' : 'U1',
            'hubbard_U2_spin_spin' : 'U2',
            'hubbard_Jij' : 'J',
            'heisenberg_J' : 'J_heisenberg',
            'nth_order_hubbard_Vij' : "V",
            'afm_pinning' : 'h_afm_pin',
            'fm_pinning' : 'h_fm_pin',
            'charge_pinning' : 'h_charge_pin',
        }

        _hst_type_overrides = input.get("hst_types",None)

        # TODO: consider how we want to handle defaulting to 1.0 AND allowing there
        #        to be no hopping!
        if input.get('t') is None or 't' not in input.keys():
            input['t'] = 1.0

        for param,default in _known_params.items():
            if param in input.keys():
                value = input[param]
            else:
                value = default
            self._ham_params.append((param,value))

        for step,key in _supported_steps.items():
            if key in input.keys():
                args = [input[key]]
                # look for hst override
                if _hst_type_overrides is not None and key in _hst_type_overrides:
                    args.append(_hst_type_overrides[key])
                self._build_steps.append((step,args))

        # using a generator to enfore that build steps happen exactly once
        self._build_step_iter = (build_step for build_step in self._build_steps)

    def _build(self):

        if self._built:
            print("Hamiltonian already built. Returning existing Hamiltonian")
            return self.builder.hamiltonian
        
        for param,value in self._ham_params:
            setattr(self.builder.hamiltonian,param,value)

        for step,arg in self._build_step_iter:
            print(f"running build step {step}({*arg,})")
            getattr(self.builder,step)(*arg)

        self.builder.finalize()

        if any( [ hst_tpye_override is not None for hst_tpye_override in self._hst_type_overrides.values()] ):
            self._override_hst_types()

        self._built = True

        return self.builder.hamiltonian

# Temporary alias
InputFileDirector = HamiltonianDirector
