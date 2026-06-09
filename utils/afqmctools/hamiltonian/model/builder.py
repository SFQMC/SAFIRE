# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from warnings import warn
import itertools
import functools

import numpy as np
import scipy.sparse as sps

from afqmctools.systems.lattice import Lattice
import afqmctools.hamiltonian.model.ham_class as model
from afqmctools.utils.matrix import force_herm, is_hermitian

def skip_empty_params(func):
    """
    Decorator to skip over a function if the parameters are all zero. None is interpreted as zero.

    Parameters
    ----------
    func : function
        the function to decorate

    Notes
    -----
    This decorator is useful for functions that would attempt to build
       an empty Hamiltonian component if the parameters are all zero. None is interpreted as zero.
    This decorator preserves the original docstring of the decorated function.
    """

    @functools.wraps(func)
    def wrapper(self, params, *args, **kwargs):
        """
        Wrapper function to check if all parameters are zero before calling. None is interpreted as zero.
        the decorated function.

        Parameters
        ----------
        self : object
            the object to call the function from
        params : iterable
            the parameters to check
        args : list
            additional arguments to pass to the function
        kwargs : dict
            additional keyword arguments to pass to the function
        """
        params = np.array(params)
        # The following line looks wrong, but is correct!
        #   numpy properly handles the comparison with None behind the scenes!
        params[params == None] = 0.0
        if not np.allclose(params,0.0):
            func(self,params,*args,**kwargs)
        elif kwargs.get("verbose",False):
            print(f"skipping empty param in {func.__name__}")

    return wrapper

def iterate_nth_order(start_n=1):
    """
    Decorator to iterate over a list of parameters, calling the method of the object, `obj`,
        with each parameter in `params`.
    
    decorates functions with the signature:
        func(self,params,nth_neighbor=n,*args,**kwargs)

    where `params` is an iterable of parameters to iterate over or a single parameter,
    args are additional arguments to pass to the method, and kwargs are additional keyword
    arguments to pass to the func.

    Note specifically that only functions that accept the nth_neighbor keyword arguments can
    be decorated with this function.

    Parameters
    ----------
    func : function 
        the function to decorate
    start_n : int, optional, default=1
        the starting index for the nth_neighbor parameter
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self,params,*args,**kwargs):
            params = np.array(params)
            if len(params.shape) in {0,2}:
                kwargs["nth_neighbor"] = kwargs.get("nth_neighbor",start_n)
                func(self,params,*args,**kwargs)
            elif len(params.shape) in {1,3}:
                for n,param in enumerate(params,start=start_n):
                    print(f"Calling {func.__name__} with {param} for nth_neighbor={n}")
                    kwargs["nth_neighbor"] = n
                    func(self,param,*args,**kwargs)
            else:
                raise ValueError(
                    f"could not iterate over {func.__name__} with params = {params}"
                     " np.array(params) must have dimension 0, 1, 2, or 3")
        return wrapper
    return decorator


class HamiltonianBuilder:
    """Builder class for constructing a Hamiltonian.

    Parameters
    ----------
    lattice : afqmctools.systems.lattice.Lattice
        the Lattice instance which describes the geometry of the lattice. 
        If not specified, the derived class is responsible for generating 
        a Lattice instance.
    nbands : int, optional, default=1
        the number of bands to use in the Hamiltonian.
    spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional, default=None
        the spin symmetry to use in the Hamiltonian. Possible values are:
        SpinSymm.CLOSED, SpinSymm.COLLINEAR, or SpinSymm.NONCOLLINEAR.
    
    Attributes
    ----------
    lattice : afqmctools.systems.lattice.Lattice
        a reference to the Lattice instance which the Hamiltonian is defined on
    hamiltonian : afqmctools.hamiltonian.model.ham_class.Hamiltonian
        the Hamiltonian which is built.
    
    Methods
    -------
    nth_neighbor_hopping(t=1.0,n:int=1)
        adds an nth-order neighbor hopping term to the Hamiltonian
    afm_pinning(h_afm_pin)
        adds edge antiferromagnetic (AFM) pinning to the Hamiltonian
        with strength h_afm_pin
    fm_pinning(h_fm_pin)
        adds edge ferromagnetic (FM) pinning to the Hamiltonian
        with strength h_fm_pin
    charge_pinning(h_charge_pin)
        adds edge charge pinning to the Hamiltonian with
        strength h_charge_pin
    edge_pinning(pinning_func,h_pin,...)
        adds edge pinning based on general pinning function
        with strength h_pins
    onsite_hubbard(U)
        adds onsite Hubbard U to the Hamiltonian with 
        possibly band-dependent amplitude U
    hubbard_U1_density_density(U1)
        adds U1 density-denisty term to the Hamiltonian with 
        possibly band-dependent amplitude U1
    hubbard_U2_spin_spin(U2)
        adds U2 spin-spin term to the Hamiltonian with 
        possibly band-dependent amplitude U2
    hubbard_Jij(J)
        adds Hund's J term to the Hamiltonian with 
        possibly band-dependent amplitude J
    heisenberg_J(J)
        adds an isotropic Heisenberg J term to the 
        Hamiltonian.
    nth_order_hubbard_Vij(V,n=1)
        adds nth-order neighbor extended Hubbard interactions
        to the Hamiltonian with possible band-dependent V
    print_components()
        print the components (i.e. terms) currently in the 
        Hamiltonian
    finalize()
        will combine terms whereever possible, keeping terms
        with different `hst_type` separate.
        
    Raises
    ------
    ValueError
        raised when any build step is invoked using an invalid amplitude
    ValueError
        raised when no hst_type can be unambiguously infered
        
    Notes
    -----
    see the User Documentation for more information on the definition
    an input conventions for each Hamiltonian term.
    """

    def __init__(
            self,
            lattice:Lattice=None,
            **kwargs
        ) -> None:
        
        if lattice is None:
            raise ValueError("Hamiltonian must be defined on 'Lattice' instance.")

        self.lattice = lattice
        self.hamiltonian = model.Hamiltonian()
        self.hamiltonian.nsites = self.lattice.N_sites

        _known_params = {
            'nbands' : 1,
            'afm_pin_type': "staggered",
            'fm_pin_type': "staggered",
            #'twist' : None,
            #'spindep' : 0 ,
        }
        for param,default in _known_params.items():
            if param in kwargs.keys():
                value = kwargs[param]
            else:
                value = default
            setattr(self.hamiltonian,param,value)

        if "spin_symm" in kwargs.keys():
            self.hamiltonian.spin_symm = kwargs["spin_symm"]


    def _add_term(self,key,term:model.HamiltonianComponent):
        """
        Add term to Hamiltonian.
        """
        self.hamiltonian[key] = term


    def _find_max_spin_symm(self):
        """
        determine maximum spin symmetry from the
            Hamiltonian components.
        """
        if self.hamiltonian.spin_symm is None:
            self.hamiltonian.spin_symm = model.SpinSymm.CLOSED

        current = self.hamiltonian.spin_symm

        for term in self.hamiltonian.terms.values():
            for component in term:

                if component.spin_symm.value > current.value:
                    current = component.spin_symm
                
                if current is model.SpinSymm.NONCOLLINEAR:
                    break
        
        self.hamiltonian.spin_symm = current
        print("Max spin symmetry is ", self.hamiltonian.spin_symm)


    def _index_map(self,lattice_i,band_m):
        """
        returns basis index given the lattice and band indices:
        mu = (i,m) where m is the 'fast' index.
        """
        return lattice_i*self.hamiltonian.nbands + band_m


    def _is_valid_band_matrix(self,input:np.ndarray):
        """
        Checks if `input` can be interpreted as 2-d matrix,
            with dimensions self.hamiltonian.nbands x self.hamiltonian.nbands
        """
        return input.shape == (self.hamiltonian.nbands,self.hamiltonian.nbands)
    

    def _upgrade_one_body_shape(self, in_mat:np.ndarray, target_spin_symm, nbasis:int) -> np.ndarray:
        """
        Upgrade the shape of a one-body matrix to match the target spin symmetry.
        
        Performs automatic shape conversions:
        - CLOSED (nbasis, nbasis) → COLLINEAR (2*nbasis, nbasis) by vstacking
        - CLOSED (nbasis, nbasis) → NONCOLLINEAR (2*nbasis, 2*nbasis) as block diagonal
        - COLLINEAR (2*nbasis, nbasis) → NONCOLLINEAR (2*nbasis, 2*nbasis) by splitting
          up/down sectors into diagonal blocks
        
        Parameters
        ----------
        in_mat : numpy.ndarray
            the input one-body matrix to upgrade
        target_spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm
            the target spin symmetry to upgrade to
        nbasis : int
            the basis size (nbands * N_sites)
        
        Returns
        -------
        numpy.ndarray
            the upgraded matrix with the appropriate shape for target_spin_symm
        
        Raises
        ------
        ValueError
            when the input matrix has an invalid shape for the target spin symmetry
        """
        if target_spin_symm == model.SpinSymm.NONCOLLINEAR:
            if in_mat.shape == (2*nbasis, 2*nbasis):
                # Already correct shape for NONCOLLINEAR
                return in_mat
            elif in_mat.shape == (nbasis, nbasis):
                # Upgrade from CLOSED to NONCOLLINEAR: block diagonal
                return sps.block_diag([in_mat, in_mat], format='csr').toarray()
            elif in_mat.shape == (2*nbasis, nbasis):
                # Upgrade from COLLINEAR to NONCOLLINEAR: split up/down into diagonal blocks
                up_block = in_mat[:nbasis, :]
                down_block = in_mat[nbasis:, :]
                return sps.block_diag([up_block, down_block], format='csr').toarray()
            else:
                raise ValueError(
                    "custom one-body in_mat has invalid shape. "
                    f"Must have shape (nbasis,nbasis), (2*nbasis,nbasis), or (2*nbasis,2*nbasis) "
                    f"for NONCOLLINEAR spin symmetry, but has shape {in_mat.shape}."
                )
        elif target_spin_symm == model.SpinSymm.COLLINEAR:
            if in_mat.shape == (2*nbasis, nbasis):
                # Already correct shape for COLLINEAR
                return in_mat
            elif in_mat.shape == (nbasis, nbasis):
                # Upgrade from CLOSED to COLLINEAR
                return np.vstack([in_mat, in_mat])
            else:
                raise ValueError(
                    "custom one-body in_mat has invalid shape. "
                    f"Must have shape (nbasis,nbasis) or (2*nbasis,nbasis) "
                    f"for COLLINEAR spin symmetry, but has shape {in_mat.shape}."
                )
        elif target_spin_symm == model.SpinSymm.CLOSED:
            if in_mat.shape == (nbasis, nbasis):
                # Already correct shape for CLOSED
                return in_mat
            else:
                raise ValueError(
                    "custom one-body in_mat has invalid shape. "
                    f"Must have shape (nbasis,nbasis) for CLOSED spin symmetry, "
                    f"but has shape {in_mat.shape}."
                )
        else:
            raise ValueError(
                "invalid spin symmetry. Only CLOSED, COLLINEAR and NONCOLLINEAR are supported"
            )


    def _wrap_one_body_spin_structure(self, mat_up, target_spin_symm, mat_down=None):
        """
        Wrap up and down spin sector matrices into the appropriate spin structure.
        
        Parameters
        ----------
        mat_up : sparse or dense array
            the spin-up sector matrix (nbasis x nbasis)
        target_spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm
            the target spin symmetry
        mat_down : sparse or dense array, optional
            the spin-down sector matrix (nbasis x nbasis). If None, uses mat_up.
        
        Returns
        -------
        sparse array
            the wrapped matrix in the appropriate format for target_spin_symm
        """
        if mat_down is None:
            mat_down = mat_up
        
        if target_spin_symm == model.SpinSymm.COLLINEAR:
            return sps.bmat(
                blocks=[
                    [mat_up],
                    [mat_down]
                ],
                format='csr'
            )
        elif target_spin_symm == model.SpinSymm.NONCOLLINEAR:
            return sps.block_diag(
                mats=[mat_up, mat_down],
                format='csr'
            )
        else:  # CLOSED
            return mat_up


    @iterate_nth_order(1)
    @skip_empty_params
    def nth_neighbor_hopping(self,t=1.0,nth_neighbor:int=1,spin_symm=None,opposite_twists=False):
        r"""adds an nth-order neighbor hopping term to the Hamiltonian

        .. math:: \sum_{\langle ij\rangle^n} (-t) \hat{c}^\dagger_i \hat{c}_j

        Parameters
        ----------
        t : float | numpy.ndarray, default: 1.0
            the hopping strength. By convention a minus sign (-) is applied to the hopping.
            i.e. :math:`\sum_{<ij>} (-t) \hat{c}^\dagger_i \hat{c}_j`
            If t is a float, hoping between sites but not between bands, is used. If t is an 
            numpy.ndarray, it must have shape nbands x nbands and is interpreted as a band 
            dependent hopping.
        nth_neighbor : int, optional, default: 1
            the order of neighbor to use for hopping. `nth_neighbor` equal to 1 is nearest-neighbors. 
            If `nth_neighbor` is not given, and `t` is an iterable, `t` is iterated over and `nth_neighbor` is 
            inferred from the array index, `i`, of each entry in `t` as `nth_neighbor=i+1`
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerated type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
        opposite_twists : bool, optional, default: False
            if True, the hopping matrix is constructed using opposite twists, for the up and down
            spins (i.e. twist_down = -twist_up )  If False, the hopping matrix is constructed using the same twist for both spins.
            
            
        Raises
        ------
        ValueError
            when the hopping matrxi can't be constructed for the combination of `t` and `nth_neighbor`

        Examples
        --------
        adds default hopping (nearest-neighbor with t=1.0)

        >>> HamiltonianBuilder.nth_neighbor_hopping()
        
        add nearest-, and next-nearest-neighbor hopping

        >>> HamiltonianBuilder.nth_neighbor_hopping(t=[1.0,0.5])
        """
        if spin_symm is None:
            spin_symm = self.hamiltonian.spin_symm

        _key = 'tij'

        # NOTE: if we insist on a list, we could use a mixture of floats and 2d-arrays
        t = np.array(t)

        if np.allclose(t,0.0):
            print("no hopping t provided, skipping build")
            return

        graph_shape = (self.lattice.N_sites,self.lattice.N_sites)
        basis_size = self.hamiltonian.nbands * self.lattice.N_sites

        row = list()
        column = list()
        data = list()

        def add_pair(pair):
            """
            add pair of indices to row,column,data,
                accounting for the number of bands
            """
            #for m in range(self.hamiltonian.nbands):
            #    row.append(self._index_map(pair.i,m))
            #    column.append(self._index_map(pair.j,m))
            if pair.phase != 0.0:
                hopping = -1*np.exp(-1j*pair.phase)
            else:
                hopping = -1
            data.append(hopping)
            row.append(pair.i)
            column.append(pair.j)

        # make the neighbor graph including the twist
        for pair in self.lattice.get_nth_neighbors(n=nth_neighbor,twist=self.hamiltonian.twist):
            add_pair(pair)

        neighbor_graph = sps.csr_array(
            (data,(row,column)),
            shape=graph_shape
        )

        if self._is_valid_band_matrix(t):
            tband = t
            if not is_hermitian(tband):
                print("Warning: band hopping is not hermitian")
                if force_herm:
                    print("Forcing hermitian band hopping")
                    tband = force_herm(tband, method='upper_triangular')
                else:
                    raise ValueError(
                        "band hopping is not hermitian, and force_herm is False. "
                        "Rerun with force_herm=True to continue."
                    )
        elif t.shape == ():
            tband = t*np.eye(self.hamiltonian.nbands)
        else:
            raise ValueError(f"could not-build {nth_neighbor}th-neighbor hopping from t = {t}")
      
        Hhop = sps.kron(
            A=neighbor_graph,
            B=tband,
            format='csr'
        )

        if spin_symm == model.SpinSymm.CLOSED and opposite_twists:
            warn(
                "Requested opposite twists for each spin sector and closed spin symmetry 1-body term."
                " Only the direct angle will be used. To use opposite twists, set spin_symm to "
                "COLLINEAR or NONCOLLINEAR."
            )

        if spin_symm == model.SpinSymm.CLOSED and opposite_twists:
            warn(
                "Requested opposite twists for each spin sector and closed spin symmetry 1-body term."
                " Only the direct angle will be used. To use opposite twists, set spin_symm to "
                "COLLINEAR or NONCOLLINEAR."
            )

        if opposite_twists:
            print("Using opposite twists for up and down spins")
            Hhop_down = Hhop.conj().T
        else:
            print("Using same twists for up and down spins")
            Hhop_down = Hhop

        Hhop_up = Hhop

        Hhop = self._wrap_one_body_spin_structure(Hhop_up, spin_symm, Hhop_down)

        component = model.HamiltonianComponent(
            csr_array=Hhop,
            model_type='one_body',
            spin_symm=spin_symm
        )

        self._add_term(_key,component)


    def custom_one_body(self,in_mat:np.ndarray,spin_symm=None):
        """add a custom one-body term to the Hamiltonian

        Parameters
        ----------
        in_mat : numpy.ndarray
            the one-body term to add to the Hamiltonian
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerated type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
        
        Notes
        -----
        The `in_mat` must have the correct shape for the Hamiltonian and spin symmetry. For
            COLLINEAR spin symmetry, the matrix must have shape (nbasis,nbasis) or (2*nbasis, 
            nbasis), and for NONCOLLINEAR spin symmetry, the matrix must have shape (2*nbasis,
            2*nbasis).

        Examples
        --------

        >>> import numpy as np
            from afqmctools.systems.lattice import get_lattice
            from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
            lattice = get_lattice(
            params=dict(
                    L1 = 3,
                    L2 = 2,
                    boundary1 = "PBC",
                    boundary2 = "PBC"
                )
            )
            builder = HamiltonianBuilder(lattice=lattice)
            # make a custom one-body term - random symetrix noise
            one_body_matrix = 0.0001*np.random.rand(nbasis,nbasis)
            one_body_matrix = 0.5*(one_body_matrix + one_body_matrix.T)
            builder.custom_one_body(one_body_matrix)
            builder.finalize()
        """
        if spin_symm is None:
            spin_symm = self.hamiltonian.spin_symm

        nbasis = self.hamiltonian.nbands * self.lattice.N_sites

        # Upgrade matrix shape to match target spin symmetry
        upgraded_mat = self._upgrade_one_body_shape(in_mat, spin_symm, nbasis)
    
        out_mat = sps.csr_array(upgraded_mat)

        component = model.HamiltonianComponent(
            csr_array=out_mat,
            model_type='one_body',
            spin_symm=spin_symm
        )
        self._add_term('tij',component)


    @skip_empty_params
    def onebody_onsite(self,epsilon,spin_symm=None):
        r"""
        Adds an onsite one-body term to the Hamiltonian (for example, a chemical potential,
            band energies, interband hopping, etc.)

        .. math::
           
            \hat{H}_{onsite} = \sum_i \sum_{m m'} \epsilon_{m m'} \hat{c}^\dagger_{i,m} \hat{c}_{i,m'}

        where :math:`\hat{c}^\dagger_{i,m}`/:math:`\hat{c}_{i,m}` are the creation / annihilation operator 
        for site i and band m.

        Parameters
        ----------
        epsilon : float | numpy.ndarray
            the onsite energy. If epsilon is a float, it is interpreted as a band-independent onsite energy.
            If epsilon is a numpy.ndarray, it must have shape (nbands,nbands) and is interpreted as a band-dependent
            onsite energy. In all cases, epsilon is applied uniformly to all sites.
        
        """

        if spin_symm is None:
            spin_symm = self.hamiltonian.spin_symm

        _key = 'tij'

        epsilon = np.array(epsilon)

        if self._is_valid_band_matrix(epsilon):
            epsilon_band = epsilon
            if not is_hermitian(epsilon_band):
                print("Warning: onsite one-body epsilon is not hermitian")
                if force_herm:
                    print("Forcing hermitian onsite one-body epsilon")
                    epsilon_band = force_herm(epsilon_band, method='upper_triangular')
                else:
                    raise ValueError(
                        "nsite one-body epsilon is not hermitian, and force_herm is False. "
                        "Rerun with force_herm=True to continue."
                    )
        elif epsilon.shape == ():
            epsilon_band = epsilon*np.eye(self.hamiltonian.nbands)
        else:
            raise ValueError(f"could not-build onsite one-body from epsilon = {epsilon}")
    
        epsilon_up = sps.kron(
            A=sps.eye(self.lattice.N_sites),
            B=epsilon_band,
            format='csr'
        )

        # for now, assume that the band energies are the same in the up and down sectors
        epsilon_mat = self._wrap_one_body_spin_structure(epsilon_up, spin_symm)

        component = model.HamiltonianComponent(
            csr_array=epsilon_mat,
            model_type='one_body',
            spin_symm=spin_symm
        )
        self._add_term(_key,component)


    @skip_empty_params
    def rashba_soc(self,rashba_lambda:float=1.0,t:float|np.ndarray=1.0,n:int=1,spin_symm=None):
        r"""
        Builds a Rashba spin-orbit coupling term of the form:
        
        .. math::
            
            \hat{H}_{rashba SOC} = i \lambda_{rashba} \sum_{ij,\sigma\sigma'} t_{ij} 
                                    (\vec{\sigma} \times \hat{r}_{ij})^{\sigma\sigma'}_z 
                                    \hat{c}^\dagger_{i\sigma} \hat{c}_{j\sigma'}
        
        where :math:`\vec{\sigma}` is the vecotr of Pauli matricies, and :math:`\vec{r}_{ij}`
        is the relative position between sites i and j.
        """
        if not rashba_lambda:
            print("no Rashba SOC alpha provided, skipping build")
            return

        if spin_symm is None:
            spin_symm = self.hamiltonian.spin_symm

        if spin_symm != model.SpinSymm.NONCOLLINEAR:
            raise ValueError("Rashba SOC is only valid for non-collinear spin symmetry")

        _key = 'tij'

        rashba_lambda = np.array(rashba_lambda)
        t = np.array(t)

        if len(t.shape) == 1:
            for n,tval in enumerate(t,start=1):
                self.rashba_soc(
                    rashba_lambda=rashba_lambda,
                    t=tval,
                    n=n,
                    spin_symm=spin_symm
                )
        elif rashba_lambda.shape == ():
            if self.hamiltonian.nbands > 1:
                warn(f"using {self.hamiltonian.nbands} bands while Rashba SOC is only tested for nbands=1")
            basis_size = self.hamiltonian.nbands * self.lattice.N_sites
            shape = (basis_size,basis_size)

            def add_pair(pair,axis=0,row=None,column=None,data=None):
                """
                add pair of indices to row,column,data,
                    accounting for the number of bands
                """
                for m in range(self.hamiltonian.nbands):
                    row.append(self._index_map(pair.i,m))
                    column.append(self._index_map(pair.j,m))
                    
                    if pair.phase != 0.0: # phase is the twist
                        prefactor = rashba_lambda*t*np.exp(-1j*pair.phase)
                    else:
                        prefactor = rashba_lambda*t

                    if axis == 0 or axis == 1:
                        data.append(prefactor*pair.r_relative[axis])
                    else:
                        raise ValueError(f"invalid axis for pairs")

            # make up-down sigma_y contribution
            row = list()
            column = list()
            data = list()  
            for pair in self.lattice.get_nth_neighbors(n=n,twist=self.hamiltonian.twist):
                add_pair(pair,axis=0,row=row,column=column,data=data)
            H_rashba_up_down_y = -1*sps.csr_array(
                (data,(row,column)),
                shape=shape
            )

            # make up-down sigma_x contribution
            row = list()
            column = list()
            data = list()  
            for pair in self.lattice.get_nth_neighbors(n=n,twist=self.hamiltonian.twist):
                add_pair(pair,axis=1,row=row,column=column,data=data)
            H_rashba_up_down_x = 1j*sps.csr_array(
                (data,(row,column)),
                shape=shape
            )
            
            H_rashba_up_down = H_rashba_up_down_x + H_rashba_up_down_y
            zero = sps.csr_array(H_rashba_up_down.shape,dtype=H_rashba_up_down.dtype)
            H_rashba = sps.bmat([
                [zero ,H_rashba_up_down],
                [H_rashba_up_down.conj().T,zero]
            ],
            format='csr')
        
            component = model.HamiltonianComponent(
                csr_array=H_rashba,
                model_type='one_body',
                spin_symm=spin_symm
            )
            self._add_term(_key,component)
        else:
            raise ValueError(
                "could not-build Rashba SOC from lambda = {rashba_lambda} "
                f"with {n}th neighbor hopping"
            )


    def afm_pinning(self,h_afm_pin,axis=0,spin_symm=None,pin_type=None):
        r"""adds anti-ferromagnetic (AFM) pinning to the Hamiltonian

        Builds an anti-ferromagnetic (AFM) pinning term of the type:
        :math:`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i \downarrow} = - v_{i \uparrow} = 1/2(-1)^(h)(i)) h_afm_pin`
        and h_afm_pin is the pinning field strength, 
        and adds it to the Hamiltonian.

        Parameters
        ----------
        h_afm_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
    
        Raises
        ------
        RuntimeError
            when the Hamiltonian as an invalid `afm_pin_type`

        Notes
        -----
        available pinning functions:
        - staggered: :math:`h(i) = i_1+i_2`    
        - fm: :math:`h(i) = i_2`    

        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.

        """
        if pin_type is None:
            pin_type = self.hamiltonian.afm_pin_type.lower()
        else:
            pin_type = pin_type.lower()
        print(f"Using afm pin type: {pin_type}")
        if pin_type in ["staggered", "afm"]:
          pinning_func = lambda coord: 0.5*(-1)**(sum(coord))
        elif pin_type in ["same", "matching", "fm"]:
          pinning_func = lambda coord: 0.5*(-1)**(coord[(axis+1)%2])
        else:
            raise RuntimeError(
                "Attempted to build AFM pinning with invalid "
                f"pin_type {pin_type}"
            )
        self.edge_pinning(pinning_func,h_afm_pin,False,axis,spin_symm)

    def fm_pinning(self,h_fm_pin,axis=0,spin_symm=None,pin_type=None):
        r"""adds ferromagnetic (FM) pinning to the Hamiltonian

        builds an FM pinning term of the type:
        :math:`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i downarrow} = - v_{i uparrow} = 1/2 h(i) h_fm_pin`
        and h_fm_pin is the pinning field strength.

        Parameters
        ----------
        h_fm_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
    
        Raises
        ------
        RuntimeError
            when the Hamiltonian as an invalid `fm_pin_type`

        Notes
        -----
        available functions:
        - staggered: :math:`h(i) = (-1)^{(i_1==0)}`
        - fm: :math:`h(i) = +0.5`

        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.

        """
        if pin_type is None:
            pin_type = self.hamiltonian.afm_pin_type.lower()
        else:
            pin_type = pin_type.lower()
        print(f"Using fm pin type: {pin_type}")
        if pin_type in ["staggered", "afm", "opposite"]:
          pinning_func = lambda coord: 0.5*(-1)**(coord[axis]==0)
        elif pin_type in ["same", "matching", "fm"]:
          pinning_func = lambda coord: 0.5
        else:
            raise RuntimeError(
                "Attempted to build AFM pinning with invalid "
                f"pin_type {pin_type}"
            )
        self.edge_pinning(pinning_func,h_fm_pin,False,axis,spin_symm)

    def charge_pinning(self,h_charge_pin,axis=0,spin_symm=None):
        r"""Adds charge pinning to the Hamiltonian

        builds a charge pinning term of the type:
        :math`\sum_{i \sigma} v_{i\sigma} \hat{n}_{i\sigma}`,
        where :math:`v_{i downarrow} = v_{i uparrow} = h_charge_pin`
        and h_charge_pin is the pinning field strength.

        Parameters
        ----------
        h_afm_pin : float
            the amplitude of the pinning field
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.
    
        Notes
        -----
        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.

        """
        pinning_func = lambda coord: 0.5
        self.edge_pinning(pinning_func,h_charge_pin,True,axis,spin_symm)

    def edge_pinning(self,pinning_func,h_pin,same_sign=False,axis=0,spin_symm=None):
        """Add general pinning, at the edge, to the Hamiltonian

        Generalized pinning function, apply pinning_func to sites at the edge
        pinning_func takes a (i,j) coordinate

        Parameters
        ----------
        pinning_func :  function(i:int, j:int) -> float
            a modulation function for the pinning field. Should compute
        h_pin : float
            the amplitude of the pinning field
        same_sign : bool
            if True, spin up and spin down sectors have the same sign.
            if Falsw, spin up and spin down sectors have opposite signs.
        axis : int, optional, default = 0
            the axis to apply pinning along
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            an override for the default spin symmetry enumerate type for the hopping matrix.
            If not given, the spin symmetry of the Hamiltonian will be used.

        Notes
        -----
        pinning is applied to an edge,
        with lattice coordinate 0 or L-1, on the given axis.

        """
        
        if spin_symm is None:
            spin_symm = self.hamiltonian.spin_symm

        _key = 'tij'

        basis_size = self.hamiltonian.nbands * self.lattice.N_sites
        shape = (basis_size,basis_size)

        row = list()
        column = list()
        data = list()

        def add_site(site):
            """
            add pair of indices to row,column,data,
                accounting for the number of bands
            """
            for m in range(self.hamiltonian.nbands):
                row.append(self._index_map(site.index,m))
                column.append(self._index_map(site.index,m))

                # apply generalized function to an edge
                factor = pinning_func(site.coord)

                data.append(factor*h_pin)
                
        # TODO: a little inefficent, i.e. could loop over just the sites 
        #           on the desired edge, not a problem for now
        for site in self.lattice.get_sites():
            r = site.coord
            #other_axis = (axis+1)%2
            if r[axis] == 0 or r[axis] == self.lattice.L[axis] - 1:
                add_site(site)

        H_pin_up = sps.csr_array(
            (data,(row,column)),
            shape=shape
        )
        H_pin_down = sps.csr_array(
            (data,(row,column)),
            shape=shape
        )
        if not same_sign:
            H_pin_down *= -1

        Hhop = self._wrap_one_body_spin_structure(H_pin_up, spin_symm, H_pin_down)

        component = model.HamiltonianComponent(
            csr_array=Hhop,
            model_type='one_body',
            spin_symm=spin_symm
        )

        self._add_term(_key,component)

    
    def _get_hst_type(self,U,is_discrete=True):
        '''
        Determine default Hubbard-Stratonovich transformation
            type based on sign of U - (any of Ui, U1ij, U2ij)

        Also applies to Jij; however, the continuous versions
            should be used.

        Parameters
        ----------
        U : float | numpy.ndarray
            the hubbard U, also applicable to J
        '''
        if is_discrete:
            prefix = "discrete_"
        else:
            prefix = "continuous_"
        if np.all(U <= 0):
            return prefix+"charge"
        elif np.all(U >= 0):
            return prefix+"spin"
        else:
            raise ValueError(
                "Unable to unambiguously choose a Hubbard-Stratonovich "
                f"transformation type for U = {U}"
            )

    def _clean_hst_type(self,hst_type:str) -> str:
        """returns a 'cleaned' Hubbard-Stratonovich transformation (HST)
        type string that the AFQMC executable can understand.

        Parameters
        ----------
        hst_type : str
            the input HST type string

        Returns
        -------
        str
            an HST string that can be directly understood in the 
            AFQMC executable.

        """
        if hst_type.lower() in ("discrete_charge","discrete charge"):
            return "discrete_charge"
        elif hst_type.lower() in ("discrete_spin","discrete spin"):
            return "discrete_spin"
        elif hst_type.lower() in ("continuous_charge","continuous charge"):
            return "continuous_charge"
        elif hst_type.lower() in ("continuous_spin","continuous spin"):
            return "continuous_spin"
        else:
            raise ValueError(
                f"Invalid hst_type '{hst_type}': valid options are: "
                "discrete_charge, discrete_spin, continuous_charge, or continuous_spin")

    def _add_interaction(self,band_U,_key,model_type=None,is_discrete=True,spin_symm=model.SpinSymm.CLOSED,hst_type=None,nth_neighbor=0):
        """
        Add an interaction term to the Hamiltonian.

        Parameters
        ----------
        band_U : numpy.ndarray
            the possibly band-dependent interaction matrix to add to the Hamiltonian
        _key : str
            the key to use for the interaction term
        model_type : str, optional
            the model type of the interaction term. If not given, the default
            model type in _add_interaction_impl is used.
        is_discrete : bool, optional
            if True, a discrete Hubbard-Stratonovich Type is used for the interaction.
        spin_symm : afqmctools.hamiltonian.model.ham_class.SpinSymm, optional
            the spin symmetry to use for the interaction term. If not given, defaults
            to model.SpinSymm.CLOSED.
        hst_type : str, optional
            the Hubbard-Stratonovich transformation type to use for the interaction term. If not
            provided, the HST type is inferred from the sign of the interaction matrix to ensure
            that constrained-path AFQMC is used instead of phaseless AFQMC.
        nth_neighbor : int, optional
            the order of neighbor to use for the interaction term. If not given, the interaction is
            assumed to be on-site.
        """

        # 1. break band_U matrix into a negative and positive part
        band_U_positive = band_U.copy()
        band_U_positive[band_U<0] = 0.0

        band_U_negative = band_U.copy()
        band_U_negative[band_U>0] = 0.0

        assert np.all(np.equal(band_U_negative.toarray() + band_U_positive.toarray(), band_U.toarray()))

        # 2. call _hubbard_impl for each non-zero term.
        if np.any(band_U_negative.toarray()):
            print(f"Building hubbard term with negative U values: {band_U_negative}")
            self._add_interaction_impl(
                band_U=band_U_negative,
                _key=_key,
                model_type=model_type,
                is_discrete=is_discrete,
                spin_symm=spin_symm,
                hst_type=hst_type,
                nth_neighbor=nth_neighbor
                )

        if np.any(band_U_positive.toarray()):
            print(f"Building hubbard term with positive U values: {band_U_positive}")
            self._add_interaction_impl(
                band_U=band_U_positive,
                _key=_key,
                model_type=model_type,
                is_discrete=is_discrete,
                spin_symm=spin_symm,
                hst_type=hst_type,
                nth_neighbor=nth_neighbor
                )


    def _add_interaction_impl(self,band_U,_key,model_type=None,is_discrete=True,spin_symm=model.SpinSymm.CLOSED,hst_type=None,nth_neighbor=0):
        r"""
        Build CSR hubbard U from the on-site interband U matrix, `band_U`.

        explicitly, build 
        
        .. math:: U_{(i,m),{j,m'}} = \delta_{ij} U^band_{m,m'}

        Notes
        -----
        this function is agnostic to the type of U term (U,U1,U2,etc.).
            it is assumed that band_U is correctly constructed.
            band_U_mm' should only inlcude m <= m'
            
        """
        if nth_neighbor == 0:
             site_matrix = sps.identity(self.lattice.N_sites)
        elif int(nth_neighbor) == nth_neighbor:
            # TODO: you are here!
            neighobrs = self.lattice.get_nth_neighbors(n=nth_neighbor)
            
            site_matrix = sps.lil_matrix((self.lattice.N_sites,self.lattice.N_sites))
            for neighbor in neighobrs:
                # convention for interactions is i < j!
                if neighbor.i < neighbor.j:
                    site_matrix[ neighbor.i, neighbor.j] += 1

        H_hubbard = sps.kron(
            A=site_matrix,
            B=band_U,
            format='csr'
        )

        print("H_U = ", H_hubbard)

        if hst_type is None:
            hst_type = self._get_hst_type(
                band_U.toarray(),
                is_discrete=is_discrete
            )
        else:
            hst_type = self._clean_hst_type(hst_type)

        component = model.HamiltonianComponent(
            csr_array=H_hubbard,
            model_type=model_type,
            spin_symm=spin_symm,
            hst_type=self._clean_hst_type(hst_type)
        )

        self._add_term(
            key=_key,
            term=component
        )

    @skip_empty_params
    def onsite_hubbard(self,U,hst_type=None):
        """adds onsite Hubbard U to the Hamiltonian with 
        possibly band-dependent amplitude U. See Notes for conventions.

        
        Parameters
        ----------
        U : float | iterable | numpy.ndarray
            the amplitude of the onsite Hubbard interaction. See Notes
            for the conventions on how U is interpreted.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to 
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U.

        Notes
        -----
        conventions:
            - a single number given for U implies the same U for all sites
            - a 1-Dimensional array with length nbands gives a different U for each
                band, but U is independent of site index.

        Advanced Notes:
            - in case of a 1-D array of U values, the U values are separated into
                positive and negative values. Up to two terms may be generated (i.e. one positive
                and one negative) to allow different Hubbard-Stratonovich Transformations to be
                used for each case.
        
        """
        U = np.array(U)

        if np.allclose(U,0.0):
            print("no Hubbard U provided, skipping build")
            return
        else:
            print(f"Building Hubbard U term with U={U}")

        U_positive = U.copy()
        U_positive[ U < 0 ] = 0.0

        U_negative = U.copy()
        U_negative[ U > 0 ] = 0.0

        if np.any(U_positive):
            print(f"Building onsite hubbard with positive U values: {U_positive}")
            self._onsite_hubbard_impl(U=U_positive,hst_type=hst_type)
        

        if np.any(U_negative):
            print(f"Building onsite hubbard with negative U values: {U_negative}")
            self._onsite_hubbard_impl(U=U_negative,hst_type=hst_type)


    def _onsite_hubbard_impl(self,U,hst_type=None):
        """
        Builds hubbard U from the on-site interband U matrix, U, and adds 
        it the the Hamiltonian.
        """
        
        if U.shape == ():
            H_U = sps.csr_array(
                U*sps.eye(self.lattice.N_sites*self.hamiltonian.nbands,format='csr')
            )
        elif len(U.shape) == 1 and U.shape[0] == self.hamiltonian.nbands:
            H_U = sps.kron(
                A=sps.identity(self.lattice.N_sites),
                B=np.diag(U),
                format='csr'
            )
        elif len(U.shape) == 1 and U.shape[0] == self.lattice.N_sites:
            H_U = sps.kron(
                A=np.diag(U),
                B=sps.identity(self.hamiltonian.nbands),
                format='csr'
            )
        else:
            raise ValueError(f"could not-build hubbard U from U = {U}")

        _key = 'Uij'
        
        if hst_type is None:
            hst_type = self._get_hst_type(U)
        else: 
            hst_type = self._clean_hst_type(hst_type)    

        component = model.HamiltonianComponent(
            csr_array=H_U,
            model_type='hubbard_u',
            hst_type=hst_type
        )

        self._add_term(
            key=_key,
            term=component
        )

    def _build_intrasite_band_matrix(self,U):
        """
        build and return a matrix, in csr format, with shape
            (nbands,nbands) which describes matrix elements
            between different bands on the same lattice site.
            All diagonal elements are 0. -> this case is handled
            elsewhere.

        Currently supports a constant value, U, used for all bands.

        TODO: this is not documented clearly.
        """
        row = list()
        column = list()
        data = list()
        
        for m,n in itertools.combinations(range(self.hamiltonian.nbands),r=2):
            data.append(U)
            row.append(m)
            column.append(n)
        
        return sps.csr_array((data,(row,column)), shape=(self.hamiltonian.nbands,self.hamiltonian.nbands))

    def _build_intersite_band_matrix(self,U):
        """
        build and return a matrix, in csr format, with shape
            (nbands,nbands) which describes matrix elements
            between different bands on the same lattice site.
            All diagonal elements are 0. -> this case is handled
            elsewhere.

        Currently supports a constant value, U, used for all bands.

        TODO: this is not documented clearly. It also looks like there is no way to
          tell it the order of neighbors to use.
        """
        row = list()
        column = list()
        data = list()
        
        for m,n in itertools.combinations_with_replacement(range(self.hamiltonian.nbands),r=2):
            data.append(U)
            row.append(m)
            column.append(n)

        return sps.csr_array((data,(row,column)), shape=(self.hamiltonian.nbands,self.hamiltonian.nbands))

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_U1_density_density(self,U1,hst_type=None,nth_neighbor=0):
        r"""
        Adds a density-density Hubbard U1 term to the Hamiltonian.

        .. math:: H_{U_1} = \sum_{ij} U^1_{ij} (n_{i\uparrow} n_{j\downarrow} + n_{i\downarrow} n_{j\uparrow})
        
        Parameters
        ----------
        U1 : float | numpy.ndarray
            the amplitude of the Hubbard U1 term. See Notes for conventions.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to 
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U1.
        nth_neighbor : int, optional
            the order of neighbor to use for the interaction term. If not given, the interaction is
            assumed to be on-site.

        Notes
        -----
        conventions:
            - a single number given for U1 implies the same U1 for all sites, and is uniform across bands

            .. TODO: fill in more details on how U1 is interpreted
        """
        if not U1:
            print("no Hubbard-Kanamori U1 provided, skipping build")
            return
        else:
            print(f"Building Hubbard-Kanamori density-density U1 term with U1={U1}")

        U1 = np.array(U1)

        if np.allclose(U1,0.0):
            print("empty U1 provided, skipping build")
            return

        # TODO: we can make a decorator for this to avoid code reproduction
        if self.hamiltonian.nbands == 1 and nth_neighbor == 0:
            raise ValueError(
                "Onsite Hubbard U1 is not supported for nbands=1" 
                " try setting nth_neighbor > 0"
            )

        if U1.shape == () and nth_neighbor == 0:
            H_U1 = self._build_intrasite_band_matrix(U1)
        elif U1.shape == () and nth_neighbor > 0:
            H_U1 = self._build_intersite_band_matrix(U1)
        elif U1.shape==(self.hamiltonian.nbands,self.hamiltonian.nbands):
            H_U1 = sps.csr_array(U1)
        else:
            raise ValueError(f"could not build hubbard U1 from U1 = {U1}: invalid shape")

        _key = 'U1ij'

        self._add_interaction(
            band_U=H_U1,
            _key=_key,
            model_type='hubbard_u',
            hst_type=hst_type,
            nth_neighbor=nth_neighbor
        )

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_U2_spin_spin(self,U2,hst_type=None,nth_neighbor=0):
        r"""   
        Adds a spin-spin Hubbard U2 term to the Hamiltonian.

        .. math:: H_{U_2} = \sum_{ij} U^2_{ij} (n_{i\uparrow} n_{j\uparrow} + n_{i\downarrow} n_{j\downarrow})
        
        Parameters
        ----------
        U2 : float | numpy.ndarray
            the amplitude of the Hubbard U2 term. See Notes for conventions.
        hst_type : str, optional
            the type of Hubbard-Stratonovich transformation (HST) to
            use in AFQMC. If not specified, an appropriate HST type
            will be inferred based on U2.
        nth_neighbor : int, optional
            the order of neighbor to use for the interaction term. If not given, the interaction is
            assumed to be on-site.
        
        Notes
        -----
        conventions:
            - a single number given for U2 implies the same U2 for all sites, and is uniform across bands

            .. TODO: fill in more details on how U2 is interpreted
        """

        # TODO: we can make a decorator for this to avoid code reproduction
        if self.hamiltonian.nbands == 1 and nth_neighbor == 0:
            raise ValueError(
                "Onsite Hubbard U2 is not supported for nbands=1" 
                " try setting nth_neighbor > 0"
            )

        U2 = np.array(U2)

        if np.allclose(U2,0.0):
            print("Empty Hubbard-Kanamori U2 provided, skipping build")
            return
        else:
            print(f"Building Hubbard-Kanamori spin-spin U2 term with U2={U2}")

        if U2.shape == () and nth_neighbor == 0:
            H_U2 = self._build_intrasite_band_matrix(U2)
        elif U2.shape == () and nth_neighbor > 0:
            H_U2 = self._build_intersite_band_matrix(U2)
        elif U2.shape==(self.hamiltonian.nbands,self.hamiltonian.nbands):
            H_U2 = sps.csr_array(U2)
        else:
            raise ValueError(f"could not build hubbard U2 from U2 = {U2}: invalid shape")

        _key = 'U2ij'

        self._add_interaction(
            band_U=H_U2,
            _key=_key,
            model_type='hubbard_u',
            spin_symm=model.SpinSymm.COLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor
        )

    @iterate_nth_order(0)
    @skip_empty_params
    def hubbard_Jij(self,J,hst_type=None,nth_neighbor=0):
        r"""
        Adds a Hund's J term to the Hamiltonian. 

        The general form of the Hund's J term is:

        .. math:: 
            \sum_{i<j} J_{ij} ( 
                \hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{j\uparrow}
                +\hat{c}^\dagger_{i\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{j\uparrow}
                +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{i\downarrow}\hat{c}_{j\downarrow}\hat{c}_{i\uparrow}
                +\hat{c}^\dagger_{j\uparrow}\hat{c}^\dagger_{j\downarrow}\hat{c}_{i\downarrow}\hat{c}_{i\uparrow} 
            ),
        
        where i,j are combined site and band indices.

        Here, we generate the $J_{ij}$ matrix which
        will be interpreted as above.
        """

        # TODO: we can make a decorator for this to avoid code reproduction
        if self.hamiltonian.nbands == 1 and nth_neighbor == 0:
            raise ValueError(
                "Onsite Hubbard J is not supported for nbands=1" 
                " try setting nth_neighbor > 0"
            )

        J = np.array(J)

        if np.allclose(J,0.0):
            print("no Hubbard-Kanamori J provided, skipping build")
            return
        else:
            print(f"Building Hubbard-Kanamori J term with J={J}")


        if J.shape == () and nth_neighbor == 0:
            H_J = self._build_intrasite_band_matrix(J)
        elif J.shape == () and nth_neighbor > 0:
            H_J = self._build_intersite_band_matrix(J)
        elif J.shape==(self.hamiltonian.nbands,self.hamiltonian.nbands):
            H_J = sps.csr_array(J)
        else:
            raise ValueError(f"could not build hubbard J from J = {J}: invalid shape")

        _key = 'Jij'

        self._add_interaction(
            band_U=H_J,
            _key=_key,
            model_type='hubbard_j',
            is_discrete=False,
            spin_symm=model.SpinSymm.COLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor
        )

    @iterate_nth_order(1)
    @skip_empty_params
    def heisenberg_J(self,J,hst_type=None,nth_neighbor=1):
        r"""
        Adds an isotropic Heisenberg spin-term.

        The isotropic Heisenberg Hamiltonian can be expressed
        in terms of the Hubbard-Kanamori terms as:

        .. math:: H_{Heisenberg}[J] = H_U1[-J/2] + H_U2[J/2] 
                          - J \sum_{i < j}( C^\dagger_{i \uparrow}C^\dagger_{j \downarrow} C_{i \downarrow}C_{j \uparrow} 
                                        + C^\dagger_{i \downarrow}C^\dagger_{j \uparrow} C_{i \uparrow}C_{j \downarrow} )

        The last two terms are two of the four terms of the Hund's J.
        """

        # TODO: we can make a decorator for this to avoid code reproduction
        if self.hamiltonian.nbands == 1 and nth_neighbor == 0:
            raise ValueError(
                "Onsite Heisenberg J is not supported for nbands=1" 
                " try setting nth_neighbor > 0"
            )

        self.hubbard_U1_density_density(-J/4,hst_type=hst_type,nth_neighbor=nth_neighbor)
        self.hubbard_U2_spin_spin(J/4,hst_type=hst_type,nth_neighbor=nth_neighbor)

        J=-J # by convention
        J = np.array(J)

        if np.allclose(J,0.0):
            print("Empty Heisenberg J provided, skipping build")
            return
        else:
            print(f"Building Heisenberg J term with J={J}")


        if J.shape == () and nth_neighbor == 0:
            H_J = self._build_intrasite_band_matrix(J)
        elif J.shape == () and nth_neighbor > 0:
            H_J = self._build_intersite_band_matrix(J)
        elif J.shape==(self.hamiltonian.nbands,self.hamiltonian.nbands):
            H_J = sps.csr_array(J)
        else:
            raise ValueError(f"could not build hubbard J from J = {J}")

        _key = 'J_heisenberg'

        self._add_interaction(
            band_U=H_J,
            _key=_key,
            model_type='heisenberg_j',
            is_discrete=False,
            spin_symm=model.SpinSymm.NONCOLLINEAR,
            hst_type=hst_type,
            nth_neighbor=nth_neighbor
        )

    @iterate_nth_order(1)
    @skip_empty_params
    def nth_order_hubbard_Vij(self,V,hst_type=None,nth_neighbor=1):
        r"""
        TODO: Be careful about the conventions on defining 'V', there could be a
           factor of 2 (or 1/2)

        Builds an extended Hubbard model V term and adds it to the Hamiltonian.
        
        The conventions used are, if i = (\mu, m) is a combined lattice and band index,
        
        .. math:: H_V = \sum_{\langle(i<j)\rangle} \sum_{\sigma \sigma'} V_{n m} ( \hat{n}_{i \sigma \sigma'} \hat{n}_{j \sigma \sigma'}  )

        which is equivalent to:
        :math:`H_V = H_{U_1}(U_1=V)` and :math:`H_{U_2}($U_2$=V)`
        """
        self.hubbard_U1_density_density(V,hst_type=hst_type,nth_neighbor=nth_neighbor)
        self.hubbard_U2_spin_spin(V,hst_type=hst_type,nth_neighbor=nth_neighbor)

    def _components_by_hst(self,components):
        """
        Sort a list of HamiltonianComponent instances (`components`) by
           their Hubbard-Stratonovich transformation type.
        """
        comps_by_hst = {}

        for comp in components:
            this_hst_type = comp.hubbard_strat_type

            if this_hst_type in comps_by_hst.keys():
                comps_by_hst[this_hst_type].append(comp)
            else:
                comps_by_hst[this_hst_type] = [comp]
    
        return comps_by_hst

    def _combine_components(self,_key='tij'):
        """
        combine all terms with key `_key` into a single
            term, if they exist.

        we kept each term separate while building the
            Hamiltonian in order to maintain a unifrom
            interface - especially for different
            types of one body terms.

        Note: for two-body terms, we want to only combine
        terms if they have the same Hubbard-Stratonovich type.
        """
    
        
        if _key in self.hamiltonian.terms.keys():
            # 1) get all terms
            terms = self.hamiltonian.terms[_key]

            # 2) sort by HST type
            terms_by_hst_type = self._components_by_hst(terms)
            
            # 3) for each HST type, create a new term by summing all of the same type
            new_terms = []
            for hst_type, new_term_by_hst in terms_by_hst_type.items():
                if len(new_term_by_hst) == 1:
                    new_terms.append(new_term_by_hst[0]) # i.e. no need to sum!
                elif len(new_term_by_hst) > 1:
                    print(f"Combining {_key} terms")
                    new_terms.append(sum(new_term_by_hst))

            # 4) overwrite the old terms *list* with the new one!
            self.hamiltonian.terms[_key] = new_terms
            self.hamiltonian.recount_components()


    #TODO:Tests: trying combining various combinations of U,U1,U2 -> ex: the case where we have only U2
    def _combine_hubbard_u(self):
        """
        For hubbard U terms, we basically have two cases:

        1. U2 is non-zero, in which case we need to build
          a matrix with dimensions 2MxM, where M is the size of
          the basis, and in which the top MxM matrix represents
          the sum of U_ij and U1_ij, and the bottom MxM matrix
          represent U2_ij
        2. U2 is zero, in which case we simply need to sum Uij
            and U1ij.

        Note: this method will preserve separation by Hubbard-Stratonovich
           transformation type.
        """

        def _get_terms_by_hst_type(local_terms:dict,hst_type:str,keys:list) -> list:
            """
            Helper function for code readability. Returns a list of all terms
               with Hubbard-Stratonovich Transformation type 'hst_type' AND 
               with type 'key'.
            """
            terms = []
            for key in keys:
                if key in local_terms.keys():
                    all_terms = local_terms[key]
                    for term in all_terms:
                        if term.hubbard_strat_type == hst_type:
                            terms.append(term)
            return terms

        print("Combining Hubbard U, U1, and U2 terms where possible")

        _intraband_den_den_key = 'Uij'
        _interband_den_den_key = 'U1ij'
        _spin_spin_key = 'U2ij'

        _all_keys = (
            _intraband_den_den_key,
            _interband_den_den_key,
            _spin_spin_key
        )
        
        hst_types = (
            "discrete_spin",
            "discrete_charge",
            "continuous_spin",
            "continuous_charge"
        )

        # alias for readability
        hamiltonian = self.hamiltonian

        local_terms = {}
        # Be careful! we're *moving* all U,U1, and U2 terms to a local dictionary
        #   where we add all of these back to hamiltoninan as we iterate through local terms!
        for _key in _all_keys:
            self._combine_components(_key)
            if _key in hamiltonian.keys():
                local_terms[_key]  = hamiltonian._pop_term(_key)
        
        for hst_type in hst_types:

            den_den_terms = []

            local_den_den = _get_terms_by_hst_type(
                    local_terms=local_terms,
                    hst_type=hst_type,
                    keys=[_intraband_den_den_key,_interband_den_den_key]
            )

            den_den_terms.extend(
                local_den_den
            )
            
            # combine density-density terms
            if len(den_den_terms) > 1:
                den_den_term = sum(den_den_terms)
            elif len(den_den_terms) == 1:
                den_den_term = den_den_terms[0]
            else:
                den_den_term = None

            # add in spin-spin interaction if present
            spin_spin_terms = _get_terms_by_hst_type(
                local_terms=local_terms,
                hst_type=hst_type,
                keys=[_spin_spin_key]
            )
    
            if spin_spin_terms:

                # this should only every have one term at this point
                if len(spin_spin_terms) > 1:
                    raise ValueError(
                        "[Error for Developers] HamiltonianBuilder. "
                        "_combine_hubbard_u has more than one spin_spin term with "
                        f"hst_type={hst_type}"
                        )
                else:
                    spin_spin_term = spin_spin_terms[0]

                shape = spin_spin_term.csr_array.shape

                if den_den_term is None:
                    den_den_term_csr = sps.csr_array(shape)
                else:
                    den_den_term_csr = den_den_term.csr_array

                hubbard_matrix = sps.vstack([
                    den_den_term_csr,
                    spin_spin_term.csr_array
                ])
                spin_symm = spin_spin_term.spin_symm
            else:

                if den_den_term is None:
                    continue
                
                spin_symm = den_den_term.spin_symm
                hubbard_matrix = den_den_term.csr_array

            self._add_term(
                key='Uij',
                term=model.HamiltonianComponent(
                    csr_array=hubbard_matrix,
                    model_type='hubbard_u',
                    spin_symm=spin_symm,
                    hst_type=hst_type
                )
            )
        
        self.hamiltonian.recount_components()


    def print_components(self,verbose=False):
        """Individually print each current Hamiltonian component
        """
        for key,components in self.hamiltonian.terms.items():
            print(f"\n === {key} terms === \n")
            for i,component in enumerate(components):
                print(f"{key} component {i} with hst_type =", component.hubbard_strat_type)
                if verbose:
                    print("csr matrix: ", component.csr_array)

    def finalize(self,verbose=False):
        """Combine Hamiltonian terms whereever possible, keeping terms
        with different `hst_type` separate.

        Parameters
        ----------
        verbose:bool
            if True, print verbose information
        """
        if verbose:
            print(" ====== Hamiltonian terms before combining components ====== ")
            self.print_components(verbose)

        print("Combining terms of the same type")
        self._combine_components('tij')
        self._combine_hubbard_u()
        self._find_max_spin_symm()

        if verbose:
            print(" ====== Hamiltonian terms after combining components ====== ")
            self.print_components(verbose)
