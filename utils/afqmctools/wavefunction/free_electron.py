# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

"""
Free-Electron Trial Wavefunction builder

Author: Kyle Eskridge
GitHub: bkesk
"""
from warnings import warn
from pathlib import Path

import numpy as np
import h5py as h5
import scipy.sparse.linalg as spsl
import scipy.linalg as spl
import scipy.sparse as sps

from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import Hamiltonian,SpinSymm,get_spin_symm_enum
from afqmctools.systems.lattice import Lattice
from afqmctools.hamiltonian.converter import read_hamiltonian
from afqmctools.systems.lattice import get_lattice
from afqmctools.utils.io import read_input_params

try:
    from autohf import lattice_hf, AutoHFHamiltonian
    HAS_AUTOHF = True
except ImportError:
    HAS_AUTOHF = False
    warn("AutoHF not found. Some functionality will be limited.")


# TODO: Validate a "good" default small irrational twist angle
# For Free Electron Trial wavefunctions - small *irrational* twist angle
THETA_X = 1/np.sqrt(592560607) # 592560607 is prime
THETA_Y = 1/np.sqrt(47603)     # 47603 is prime


def free_electron(source,nelec,twist=None,spin_symm=None,use_dense=True,lattice=None,
                  return_autohf=False,measure_spin=True,filling_strategy='aufbau',shell_tol=1e-6,
                  measure_evar=True):
    """
    Builds a free-electron trial wavefunction based on the 'source' lattice hamiltonian.

    Parameters
    ----------
    source : str | dict | Hamiltonian
        str - the name of an input file with lattice and hamiltonian blocks defined
        dict - a dict containing lattice and hamiltonian blocks
    nelec : tuple(int,int)
        number of spin-up and spin-down electrons
    twist : array-like, optional
        twist angle for the lattice (default: small irrational twist)
    spin_symm : SpinSymm or str, optional
        spin symmetry to use
    use_dense : bool
        whether to use dense matrix operations (default: True)
    lattice : Lattice, optional
        pre-constructed lattice object
    return_autohf : bool
        whether to return AutoHF results (default: False)
    measure_spin : bool
        whether to measure spin in AutoHF (default: True)
    filling_strategy : str
        strategy for filling orbitals within shells. Options:

        - 'aufbau': fill shells from lowest to highest energy (default)
        - 'balanced': for partially filled shells, select orbitals evenly 
            to balance properties like momentum
        - 'hund': apply Hund's rule (maximize spin)
        - 'alternating': fill from edges inward (0, -1, 1, -2, ...) 
            for momentum cancellation

    shell_tol : float
        tolerance for grouping eigenvalues into shells (default: 1e-6)
        Orbitals with eigenvalues within this tolerance are considered 
        to belong to the same shell.
    measure_evar : bool
        whether to measure the variational energy with respect to the interacting Hamiltonian
        using AutoHF (default: True)
    
    Returns
    -------
    wfn : list
        wavefunction as [coefficients, orbital_matrix]
    spin_symm : SpinSymm
        spin symmetry of the wavefunction
    results : dict (optional)
        AutoHF results if return_autohf=True
    """
    if twist is None:
        twist = 0.1*np.array((THETA_X,THETA_Y))
    else:
        twist = np.array(twist)

    print(f"Generating free-electron trial wavefunction with twist = {twist}")

    if isinstance(source,str|Path|dict):
        # convert Path to str
        if isinstance(source,Path):
            source = str(source)

        if isinstance(source,str) and h5.is_hdf5(source):
            hamiltonian = read_hamiltonian(source)
        
        elif isinstance(source,str|dict):
            if isinstance(source,str) and source.endswith(".toml"):
                source = read_input_params(source)

            lattice_params = source.get("lattice",dict())
            lattice_params["twist"] = twist
            lattice = get_lattice(params=lattice_params)

            hamiltonian = HamiltonianBuilder.from_input(
                source=source,
                lattice=lattice
                ).hamiltonian
    elif isinstance(source,Hamiltonian):

        hamiltonian = source

        if np.any(twist != hamiltonian.twist):
            warn(
                "Twist angle of the Hamiltonian does not match the twist angle "
                "provided to the free electron wavefunction builder. "
                "Using the twist angle from the Hamiltonian."
            )

    else:
        raise ValueError("source must be a str|Path, dict, or Hamiltonian instance")

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm

    wfn,spin_symm_wfn = _free_electron(
        hamiltonian=hamiltonian,
        nelec=nelec,
        spin_symm=spin_symm,
        use_dense=use_dense,
        filling_strategy=filling_strategy,
        shell_tol=shell_tol
    )

    # TODO: put this somewhere else, probably parse from an input file
    settings = dict(
        ansatz = 'SD',
        steps = -1, # prints only the reference energy
        verbose = True,
        nelec = nelec,
        batch_size = 1,
        measure_spin = measure_spin
    )

    if spin_symm_wfn == SpinSymm.NONCOLLINEAR:
        settings["noncollinear"] = True

    if measure_evar and not HAS_AUTOHF:
        warn("AutoHF not available. Skipping energy evaluation.")
        results = None
    elif measure_evar:
        results = lattice_hf(
            AutoHFHamiltonian(source=hamiltonian),
            settings=settings,
            initial_guess=wfn[1][0], # TODO: make a wfn container class so that we don't need to have to take awkward indices like this
            suppress_logo = True,
        )

    if return_autohf:
        return wfn,spin_symm,results
    else:
        return wfn,spin_symm_wfn
    

def _free_electron(hamiltonian:Hamiltonian,nelec,spin_symm=None,use_dense=True,filling_strategy='aufbau',shell_tol=1e-6):
    """
    Builds a free-electron trial wavefunction for the given Hamiltonian

    Parameters
    ----------        
    hamiltonian : Hamiltonian
        a Lattice Model Hamiltonian instance to build the Hamiltonian for.
    nelec : tuple(int,int)
        an iterable with the number of electrons per spin to use.
    spin_symm : SpinSymm or str, optional
        spin symmetry to use
    use_dense : bool
        whether to use dense matrix operations
    filling_strategy : str
        strategy for filling orbitals within shells ('aufbau', 'balanced', 'hund', 'alternating')
    shell_tol : float
        tolerance for grouping eigenvalues into shells

    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    spin_symm : SpinSymm
        the spin symmetry of the wave
    """

    norb = hamiltonian.nbands * hamiltonian.nsites
    T = hamiltonian.get_one_body()

    if spin_symm is None:
        spin_symm = hamiltonian.spin_symm
    else:
        spin_symm = get_spin_symm_enum(spin_symm)

    if spin_symm == SpinSymm.CLOSED:
        raise NotImplementedError("Free electron wavefunction with closed spin symmetry is not implemented")
    elif spin_symm == SpinSymm.COLLINEAR:
        wfn = _collinear_free_elec(T,nelec=nelec,Nmo=norb,use_dense=use_dense,
                                   filling_strategy=filling_strategy,shell_tol=shell_tol)
        return wfn, spin_symm
    elif spin_symm == SpinSymm.NONCOLLINEAR:
        wfn = _noncollinear_free_elec(T,nelec=nelec,Nmo=norb,use_dense=use_dense,
                                      filling_strategy=filling_strategy,shell_tol=shell_tol)
        return wfn, spin_symm
    else:
        raise ValueError("Can't make free electron wavefunction. Unknown spin_symm")


def _get_1body_eig(H1body,num_eigenvals,use_dense=True):
    """
    Get the eigenvalues and vectors of the given one-body Hamiltonian.

    This function will attempt to use the dense representation of the
      1-body Hamiltonian by default, but will use the sparse representation if requested and if possible. The sparse
      rep. is not recommended. It uses spipy.sparse.linalg.eigh which
      is known to not properly handle complex-valued matrices.
    
    Parameters
    ----------
    H1body : scipy.sparse.csr_array
        a sparse array representing the 1-body Hamiltonian
    num_eigenvals : int
        the number of eigenvalues to find
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian

    Returns
    -------
    sorted_eigenvals : np.ndarray
        the sorted eigenvalues of the 1-body Hamiltonian
    sorted_orbitals : np.ndarray
        the sorted eigenvectors of the 1-body Hamiltonian
    """

    N = H1body.shape[0]
    if num_eigenvals > N:
        print("Requested more eigenvalues than N for NxN array: finding N eigenvalues instead")
        num_eigenvals = N

    if use_dense or num_eigenvals == N:
        print("using dense representation of 1-body Hamiltonian to find eigenvalues and orbitals")
        sorted_eigenvals,sorted_orbitals = spl.eigh(
            a=H1body.toarray()
        )
    else:
        print("using sparse representation of 1-body Hamiltonian to find eigenvalues and orbitals")
        # only valid for num_eigenvales <= N-1 
        warn("Using sparse representation of 1-body Hamiltonian. This may not work for complex-valued matrices.")
        eigenvalues,orbitals = spsl.eigsh(
            A=H1body,
            k=num_eigenvals,
            which='SA' # smallest algebraic eigenvalues
            )
        # sort by eignvalue
        sorted_eigenvals,sorted_orbitals = zip(*sorted(zip(eigenvalues,orbitals.T),key=lambda x: x[0]))
        sorted_orbitals = np.array(sorted_orbitals).T

    return sorted_eigenvals,sorted_orbitals


def _group_eigenvalues_by_shell(eigenvalues, orbitals, tol=1e-6):
    """
    Group eigenvalues and their corresponding orbitals into shells based on degeneracy.
    
    Orbitals are considered to belong to the same shell if their eigenvalues 
    are within `tol` of each other.
    
    Parameters
    ----------
    eigenvalues : np.ndarray
        sorted array of eigenvalues
    orbitals : np.ndarray
        array of orbitals (columns correspond to eigenvalues)
    tol : float
        tolerance for considering eigenvalues as degenerate
    
    Returns
    -------
    shells : list of dict
        list of shells, where each shell is a dict with keys:
        - 'energy': average energy of the shell
        - 'eigenvalues': array of eigenvalues in the shell
        - 'orbitals': array of orbitals in the shell (columns)
        - 'indices': original indices of orbitals in the shell
        - 'degeneracy': number of orbitals in the shell
    """
    shells = []
    i = 0
    n_orbitals = len(eigenvalues)
    
    while i < n_orbitals:
        # Start a new shell
        shell_start = i
        shell_energy = eigenvalues[i]
        
        # Find all eigenvalues within tolerance of the first one in this shell
        j = i + 1
        while j < n_orbitals and abs(eigenvalues[j] - shell_energy) < tol:
            j += 1
        
        shell_end = j
        shell_indices = list(range(shell_start, shell_end))
        
        shell = {
            'energy': np.mean(eigenvalues[shell_start:shell_end]),
            'eigenvalues': eigenvalues[shell_start:shell_end],
            'orbitals': orbitals[:, shell_start:shell_end],
            'indices': shell_indices,
            'degeneracy': shell_end - shell_start
        }
        
        shells.append(shell)
        i = shell_end
    
    return shells


def _fill_shells(shells, n_electrons, strategy='aufbau', verbose=True):
    """
    Fill shells with electrons according to a given strategy.
    
    Parameters
    ----------
    shells : list of dict
        list of shells as returned by _group_eigenvalues_by_shell
    n_electrons : int
        number of electrons to place in orbitals
    strategy : str
        filling strategy. Options:
        - 'aufbau': fill shells from lowest to highest energy (default)
        - 'balanced': for partially filled shells, try to balance occupation
                     (useful for momentum balancing in k-space)
        - 'hund': apply Hund's rule (maximize spin, minimize pairing)
        - 'alternating': fill from edges inward (0, -1, 1, -2, 2, ...)
                        useful for momentum cancellation
    verbose : bool
        whether to print shell filling information
    
    Returns
    -------
    occupied_orbitals : np.ndarray
        array of occupied orbitals (Nmo x n_electrons)
    occupied_indices : list
        list of original orbital indices that were occupied
    """
    if n_electrons == 0:
        # Return empty arrays with proper shape
        if len(shells) > 0:
            Nmo = shells[0]['orbitals'].shape[0]
        else:
            Nmo = 0
        return np.zeros((Nmo, 0), dtype=complex), []
    
    Nmo = shells[0]['orbitals'].shape[0]
    occupied_orbitals_list = []
    occupied_indices = []
    electrons_remaining = n_electrons
    
    if verbose:
        print(f"\nFilling {n_electrons} electrons using '{strategy}' strategy:")
        print(f"Found {len(shells)} shell(s)")
    
    for shell_idx, shell in enumerate(shells):
        if electrons_remaining == 0:
            break
        
        shell_deg = shell['degeneracy']
        shell_energy = shell['energy']
        
        # How many electrons to put in this shell?
        electrons_in_shell = min(electrons_remaining, shell_deg)
        
        if verbose:
            print(f"  Shell {shell_idx}: energy={shell_energy:.6f}, "
                  f"degeneracy={shell_deg}, filling={electrons_in_shell}/{shell_deg}")
        
        # Apply filling strategy
        if strategy == 'aufbau' or electrons_in_shell == shell_deg:
            # Fill all available orbitals in order (or completely fill the shell)
            selected_orbitals = shell['orbitals'][:, :electrons_in_shell]
            selected_indices = shell['indices'][:electrons_in_shell]
            
        elif strategy == 'balanced':
            # For partially filled shells, try to balance occupation
            # This is useful for momentum balancing in k-space
            # Strategy: select orbitals evenly spaced through the shell
            if electrons_in_shell < shell_deg:
                # Select evenly spaced orbitals
                spacing = shell_deg / electrons_in_shell
                selected_idx = [int(i * spacing) for i in range(electrons_in_shell)]
                selected_orbitals = shell['orbitals'][:, selected_idx]
                selected_indices = [shell['indices'][i] for i in selected_idx]
                if verbose:
                    print(f"    Balanced filling: selected orbital indices {selected_idx} from shell")
            else:
                selected_orbitals = shell['orbitals'][:, :electrons_in_shell]
                selected_indices = shell['indices'][:electrons_in_shell]
        
        elif strategy == 'alternating':
            # For partially filled shells, fill from edges inward
            # Pattern: 0, -1, 1, -2, 2, -3, 3, ...
            # This helps cancel momentum in k-space
            if electrons_in_shell < shell_deg:
                selected_idx = []
                left = 0
                right = shell_deg - 1
                take_from_left = True
                
                for _ in range(electrons_in_shell):
                    if take_from_left:
                        selected_idx.append(left)
                        left += 1
                    else:
                        selected_idx.append(right)
                        right -= 1
                    take_from_left = not take_from_left
                
                selected_orbitals = shell['orbitals'][:, selected_idx]
                selected_indices = [shell['indices'][i] for i in selected_idx]
                if verbose:
                    print(f"    Alternating filling: selected orbital indices {selected_idx} from shell")
            else:
                selected_orbitals = shell['orbitals'][:, :electrons_in_shell]
                selected_indices = shell['indices'][:electrons_in_shell]
        
        elif strategy == 'hund':
            # Hund's rule: maximize spin (for single spin channel, just fill in order)
            # This is the same as aufbau for a single spin channel
            selected_orbitals = shell['orbitals'][:, :electrons_in_shell]
            selected_indices = shell['indices'][:electrons_in_shell]
        
        else:
            raise ValueError(f"Unknown filling strategy: {strategy}")
        
        occupied_orbitals_list.append(selected_orbitals)
        occupied_indices.extend(selected_indices)
        electrons_remaining -= electrons_in_shell
    
    if electrons_remaining > 0:
        raise ValueError(f"Not enough orbitals to accommodate {n_electrons} electrons. "
                        f"{electrons_remaining} electrons could not be placed.")
    
    # Concatenate all occupied orbitals
    occupied_orbitals = np.hstack(occupied_orbitals_list)
    
    return occupied_orbitals, occupied_indices


def _collinear_free_elec(Hfree,nelec,Nmo,use_dense=True,filling_strategy='aufbau',shell_tol=1e-6):
    """
    Given a 1-body Hamiltonian, generates a 'free-electron' wavefunction in
       a format that can be directly passed to `write_wfn()`.

    Parameters
    ----------
    Hfree : np.ndarray
        a 1-body Hamiltonian
    nelec : tuple(int,int)
        the number of electrons per spin
    Nmo : int
        the number of basis set functions
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian
    filling_strategy : str
        strategy for filling orbitals within shells. Options:
        - 'aufbau': fill shells from lowest to highest energy (default)
        - 'balanced': for partially filled shells, select orbitals evenly 
                     to balance properties like momentum
        - 'hund': apply Hund's rule (maximize spin)
        - 'alternating': fill from edges inward (0, -1, 1, -2, ...) 
                        for momentum cancellation
    shell_tol : float
        tolerance for grouping eigenvalues into shells (default: 1e-6)
    
    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    """
    Nup, Ndown = nelec
    
    # We need enough eigenvalues to fill the most occupied spin channel
    # plus some buffer to ensure we capture complete shells
    num_eigenvals = max(nelec) + 10  # Add buffer for shell structure
    num_eigenvals = min(num_eigenvals, Nmo)  # Don't exceed available orbitals
    
    Hfree_sigma = (Hfree[:Nmo,:],Hfree[Nmo:,:])

    occupied_orbitals_sigma = []

    for spin_idx, (Hfree_spin, n_elec) in enumerate(zip(Hfree_sigma, nelec)):
        spin_label = "up" if spin_idx == 0 else "down"
        print(f"\n{'='*60}")
        print(f"Processing spin-{spin_label} channel ({n_elec} electrons)")
        print(f"{'='*60}")
        
        # Get eigenvalues and eigenvectors
        eigenvals, orbitals = _get_1body_eig(Hfree_spin, num_eigenvals, use_dense=use_dense)
        print(f"\nEigenvalues of the non-interacting Hamiltonian:")
        print(eigenvals)
        
        # Group eigenvalues into shells
        shells = _group_eigenvalues_by_shell(eigenvals, orbitals, tol=shell_tol)
        
        # Fill shells according to strategy
        occupied_orbitals, occupied_indices = _fill_shells(
            shells, n_elec, strategy=filling_strategy, verbose=True
        )
        
        occupied_orbitals_sigma.append(occupied_orbitals)

    # Construct the wavefunction
    wfn = np.zeros((1, Nmo, sum(nelec)), dtype=np.complex128)
    coeffs = np.array([1.0+0j])
    wfn[0, :, :Nup] = occupied_orbitals_sigma[0]
    wfn[0, :, Nup:] = occupied_orbitals_sigma[1]

    print(f"\n{'='*60}")
    print(f"Wavefunction construction complete")
    print(f"Total electrons: {sum(nelec)} (up: {Nup}, down: {Ndown})")
    print(f"{'='*60}\n")

    return [coeffs, wfn]

def _noncollinear_free_elec(Hfree,nelec,Nmo,use_dense=True,filling_strategy='aufbau',shell_tol=1e-6):
    """
    Given a 1-body Hamiltonian, generates a 'free-electron' wavefunction in
         a format that can be directly passed to `write_wfn()`.

    Parameters
    ----------
    Hfree : np.ndarray
        a 1-body Hamiltonian
    nelec : tuple(int,int)
        the number of electrons per spin
    Nmo : int
        the number of basis set functions
    use_dense : bool
        whether to use the dense representation of the 1-body Hamiltonian
    filling_strategy : str
        strategy for filling orbitals within shells. Options:
        - 'aufbau': fill shells from lowest to highest energy (default)
        - 'balanced': for partially filled shells, select orbitals evenly 
                     to balance properties like momentum
        - 'hund': apply Hund's rule (maximize spin)
        - 'alternating': fill from edges inward (0, -1, 1, -2, ...) 
                        for momentum cancellation

    shell_tol : float
        tolerance for grouping eigenvalues into shells (default: 1e-6)

    Returns
    -------
    wfn : list
        a list containing the coefficients and Slater determinant arrays of a Wavefunction
    """
    Nup, Ndown = nelec
    n_total_electrons = Nup + Ndown
    
    # Add buffer to ensure we capture complete shells
    num_eigenvals = n_total_electrons + 10
    num_eigenvals = min(num_eigenvals, 2*Nmo)  # Don't exceed available orbitals
    
    # assumes that the input is collinear
    if Hfree.shape == (2*Nmo,Nmo):
        Hfree = sps.block_diag([Hfree[:Nmo,:],Hfree[Nmo:,:]],format="csr")
    elif Hfree.shape == (Nmo,Nmo):
        Hfree = sps.block_diag([Hfree,Hfree],format="csr")
    elif Hfree.shape != (2*Nmo,2*Nmo):
        raise ValueError(
            "Hfree has an invalid shape. "
            "shape must be any of (Nmo,Nmo) - Closed H, (2*Nmo,Nmo) - Collinear H "
            " - (2*Nmo,2*Nmo) Noncollinear H"
        )

    print(f"\n{'='*60}")
    print(f"Processing noncollinear system ({n_total_electrons} electrons)")
    print(f"{'='*60}")
    
    # Get eigenvalues and eigenvectors
    eigenvals, orbitals = _get_1body_eig(Hfree, num_eigenvals, use_dense=use_dense)
    print(f"\nEigenvalues of the non-interacting Hamiltonian:")
    print(eigenvals)
    
    # Group eigenvalues into shells
    shells = _group_eigenvalues_by_shell(eigenvals, orbitals, tol=shell_tol)
    
    # Fill shells according to strategy
    occupied_orbitals, occupied_indices = _fill_shells(
        shells, n_total_electrons, strategy=filling_strategy, verbose=True
    )
    
    # Construct the wavefunction
    npol = 2
    nelec_total = Nup + Ndown
    wfn = np.zeros((1, npol*Nmo, nelec_total), dtype=np.complex128)
    coeffs = np.array([1.0+0j])
    wfn[0, :, :] = occupied_orbitals

    print(f"\n{'='*60}")
    print(f"Wavefunction construction complete")
    print(f"Total electrons: {nelec_total} (original: up={Nup}, down={Ndown})")
    print(f"Noncollinear representation: {nelec_total} electrons, 2*{Nmo} spinor orbitals")
    print(f"{'='*60}\n")

    return [coeffs, wfn]


