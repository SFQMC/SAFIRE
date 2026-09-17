# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import os
from warnings import warn

import numpy as np
import h5py


def read(fname, **kwargs):
  """ read h5 file and return a h5py File object

  Args:
    fname (str): hdf5 file
    kwargs (dict): keyword arguments to pass on to h5py.File,
      default is {'mode': 'r'}
  Return:
    h5py.File: h5py File object
  """
  if not ('mode' in kwargs):
    kwargs['mode'] = 'r'
  return h5py.File(fname, **kwargs)


def _correlation_length(edata, axis=0):
    """ Calculate the auto-correlation length of each column

    The real and the imaginary part of a column are measured from the same walker
    ensemble and share a correlation time, but each one only reveals it where it
    actually fluctuates, so take the longer of the two estimates. A part `corr`
    considers constant reports an infinite length, which says nothing about the
    correlation time and is dropped; where neither part fluctuates the length does not
    matter and 1.0 stands in, which leaves the standard error at zero.

    Args:
      edata (np.array): array of equilibrated time series data
      axis (int, optional): axis to average over, default 0 i.e. columns

    Return:
      np.array: auto-correlation length of each column
    """
    from stats.lib.stats import corr
    kappa = np.stack([np.apply_along_axis(corr, axis, edata.real),
                      np.apply_along_axis(corr, axis, edata.imag)])
    return np.where(np.isfinite(kappa), kappa, -np.inf).max(axis=0)


def me2d(edata, kappa=None, axis=0):
    """ Calculate mean and error of a table of columns

    Args:
      edata (np.array): 2D array of equilibrated time series data
      kappa (float, optional): pre-calculate auto-correlation, default is to
       re-calculate on-the-fly
      axis (int, optional): axis to average over, default 0 i.e. columns

    Return:
      (np.array, np.array): (mean, error) of each column. The error of complex data is
       complex, its real and imaginary parts being the standard errors of the real and
       imaginary parts of the data. Both use the one auto-correlation length of the
       column, so summing their variances recovers the error of the column as a whole.
    """
    # get autocorrelation
    ntrace = edata.shape[axis]
    if kappa is None:
      kappa = _correlation_length(edata, axis)

    # a column with no auto-correlation length to measure gets the neutral 1.0
    kappa = np.where(np.isfinite(kappa), kappa, 1.0)
    neffective = ntrace/kappa
    # calculate mean and error
    val_mean = edata.mean(axis=axis)
    if np.iscomplexobj(edata):
        val_std = (edata.real.std(ddof=1, axis=axis)
                   + 1j*edata.imag.std(ddof=1, axis=axis))
    else:
        val_std = edata.std(ddof=1, axis=axis)
    val_err  = val_std/np.sqrt(neffective)
    return val_mean, val_err


def afobs(fp, obs_name, nequil, kappa=None, group='BackPropagated', numer='one_rdm', iav=None):
    """ extract 1RMD output from AFQMC stat.h5 file
     assume BackPropagated (BP) 'Observables/BackPropagated'
  
    Args:
      fp (h5py.File): h5py handle of stat.h5 file
      obs_name (str): observable name, probably 'FullOneRDM'
      nequil (int): number of equilibration BP blocks to remove
      kappa (float, optional): auto-correlation, default recalculate
      numer (str, optional): numerator to extract, default 'one_rdm'
      iav (int, optional): BP level (Average_$iav), default is last level
    Return:
      tuple: (mean, error) arrays
    """
    # 1. gather meta data
    meta_paths = {
      'walker_type': 'Metadata/WalkerType',
      'nmo': 'Metadata/NMO',
      'dt': 'Metadata/Timestep',
      'free_projection': 'Metadata/FreeProjection',
    }
    meta = {}
    
    for key, path in meta_paths.items():
        meta[key] = fp[path][()]
    
    if meta['free_projection'] > 0:
        msg = 'need to consider denominator!'
        raise NotImplementedError(msg)
    nbas = int(meta['nmo'])
    itwalker = int(meta['walker_type'])
    
    if itwalker == 1:  # CLOSED
        rdm_shape = (1, nbas, nbas)
    elif itwalker == 2:  # COLLINEAR
        rdm_shape = (2, nbas, nbas)
    elif itwalker == 3:  # non-collinear
        rdm_shape = (1, 2*nbas, 2*nbas)
    else:
        msg = 'unknown walker type %d' % itwalker
        raise RuntimeError(msg)
    
    # 2. deal with back propagation (BP)
    avg_path = f'Observables/{group}/{obs_name}'

    if iav is None:  # use longest BP
        avgs = fp[avg_path].keys()
        iavgs = [int(a.replace('Average_', '')) for a in avgs]
        mav = max(iavgs)
    else:  # use user request
        mav = iav
    
    matrix_path = os.path.join(avg_path, 'Average_%d' % mav)
    # 3. get 1RDM at all equilibrated blocks
    blocks = fp[matrix_path].keys()
    rdm_blocks = [key for key in blocks if key.startswith(numer)]
    nblock = len(rdm_blocks)
    
    if nequil >= nblock:
        msg = 'cannot discard %d/%d blocks' % (nequil, nblock)
        raise RuntimeError(msg)
    
    data = []
    for block in rdm_blocks[nequil:]:
        path = os.path.join(matrix_path, block)
        rdm = fp[path][()].view(np.complex128)
        dpath = os.path.join(matrix_path, block.replace(numer, 'denominator'))
        deno = fp[dpath][()].view(np.complex128)
        if deno == 0:
            warn('In afobs: observable denominator is zero! using NaN for observable data')
            data.append(rdm*np.nan)
        else:
          data.append(rdm/deno)
    
    assert np.prod(rdm_shape) == np.prod(rdm.shape)
    
    # 4. get mean and standard error
    mat = np.array(data, dtype=np.complex128).reshape(
      -1, np.prod(rdm_shape))
    ym, ye = me2d(mat,kappa=kappa)
    dm = ym.reshape(rdm_shape)
    de = ye.reshape(rdm_shape)
    
    if itwalker == 3:  # non-collinear [up-up, dn-dn, up-dn, dn-up]
        dm = np.array([
            dm[0, :nbas, :nbas], dm[0, nbas:, nbas:],
            dm[0, :nbas, nbas:], dm[0, nbas:, :nbas],
        ])
        de = np.array([
            de[0, :nbas, :nbas], de[0, nbas:, nbas:],
            de[0, :nbas, nbas:], de[0, nbas:, :nbas],
        ])
  
    return dm, de
