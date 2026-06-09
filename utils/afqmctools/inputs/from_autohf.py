# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from autohf import AutoHFHamiltonian,lattice_hf

from afqmctools.wavefunction.model import write_wfn
from afqmctools.wavefunction.common import modified_gram_schmidt, check_orthonormality
from afqmctools.hamiltonian.model.ham_class import Hamiltonian
import numpy as np

def _autoHF_2_afqmc_wfn_noncollinear(orbs,N,nelec):
    '''
    '''
    state_afqmc = orbs[0][:,:nelec[0]+nelec[1]]

    return state_afqmc


def _autoHF_2_afqmc_wfn_collinear(orbs,N,nelec):
    '''
    '''

    state_afqmc = np.zeros((N,sum(nelec)),dtype=orbs[0].dtype)
    state_afqmc[:,:nelec[0]] = orbs[0][:,:nelec[0]]
    state_afqmc[:,nelec[0]:] = orbs[1][:,:nelec[1]]

    return state_afqmc

def autohf_to_afqmc(input_ = None,
                    output_fname = None,
                    ham = None,
                    write_hamil = False,
                    ):
  """
  convert input_ into a wavefunction in output_fname for afqmc
  Supported 3 cases:
  - string: completed autohf file
  - dictionary: settings to run autohf, convenience wrapper
  - tuple of dictionaries: passed autohf outputs, same as string
  """
  if input_ is None:
    raise ValueError("Must have either input file or autohf output to convert!")

  if output_fname is None:
    raise ValueError("Output filename is needed! Please define output_fname")


  if isinstance(input_,dict):
    # run autohf
    if ham is None:
      # TODO: allow for ham to be dictionary from toml to setup
      raise ValueError("Please supply Hamiltonian if running autohf from this function")

    # convert if handed afqmctools data, otherwise should be autohf
    if isinstance(ham,Hamiltonian):
      ham_ = AutoHFHamiltonian(source=ham)
    else:
      ham_ = ham
    data,data_f = lattice_hf(ham_,settings=input_)
  elif isinstance(input_,str):
    import pickle
    print(f"Reading from file {input_}")
    if input_[-4:] != ".pkl" and input_[-7:] != ".pickle":
      print("Assuming pkl/pickle format, other formats currently unsupported")
    with open(input_,"rb") as f:
      data = pickle.load(f)
  elif isinstance(input_,tuple):
    #TODO: better error handling here
    data = input_[0]
    if "args" not in data:
      raise ValueError("Unsupported input provided")
  else:
    raise ValueError("Unsupported input type provided")

  # by here we have by some way gotten
  # the args from autohf, the orbitals, N, and nelec
  autohf_args = data["args"]
  orbs = data["orbitals"]
  nelec = data["nelec"]
  N = orbs[0].shape[0]

  coeffs = np.array([1.0+0j])
  isnc = autohf_args.noncollinear
  if isnc:
    state_afqmc = _autoHF_2_afqmc_wfn_noncollinear(orbs,N,nelec)
    wfn = np.zeros((1,N,sum(nelec)), dtype=np.complex128)
    wfn[0,:,:] = state_afqmc[:,:]

    # Check orthonormality
    norm = check_orthonormality(wfn, sum(nelec), wfn_type='noncollinear')
    
    # Orthonormalize columns for noncollinear case if norm is not 1.0
    if abs(norm - 1.0) > 1.0e-10:
      wfn[0,:,:] = modified_gram_schmidt(wfn[0,:,:])
      # Re-check after orthonormalization
      check_orthonormality(wfn, sum(nelec), wfn_type='noncollinear')

    write_wfn(
        filename=output_fname,
        wfn=[coeffs,wfn],
        walker_type='ghf',
        nelec=(sum(nelec),0),
        norb=N//2
    )
  else:
    state_afqmc = _autoHF_2_afqmc_wfn_collinear(orbs,N,nelec)
    wfn = np.zeros((1,N,sum(nelec)), dtype=np.complex128)
    wfn[0,:,:] = state_afqmc[:,:]

    # Check orthonormality
    norm = check_orthonormality(wfn, nelec, wfn_type='collinear')
    
    if abs(norm - 1.0) > 1.0e-10:
      wfn[0,:,:nelec[0]] = modified_gram_schmidt(wfn[0,:,:nelec[0]])
      wfn[0,:,nelec[0]:] = modified_gram_schmidt(wfn[0,:,nelec[0]:])
      check_orthonormality(wfn, nelec, wfn_type='collinear')

    write_wfn(
        filename=output_fname,
        wfn=[coeffs,wfn],
        walker_type='uhf',
        nelec=nelec,
        norb=N
    )

