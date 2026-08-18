# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

from collections.abc import Iterable
#from itertools import batched

import h5py
import numpy
from afqmctools.utils.io import from_complex
from .common import get_metadata, batched

# Map user names to internal names.
MAP = {
        'one_rdm': {
            'group': 'FullOneRDM',
            'numer': 'one_rdm'
        },
        'two_rdm': {
            'group': 'FullTwoRDM',
            'numer': 'two_rdm'
        },
        'diag_two_rdm': {
            'group': 'DiagTwoRDM',
            'numer': 'diag_two_rdm'
        },
        'spinspin': {
            'group': 'SpinSpin',
            'numer': 'spinspin'
        },
        'pair_correlation': {
            'group': 'PairCorr',
            'numer': 'pair_correlation'
        },
    }

def extract_data(filename, group, estimator, sample=None, transform=None, batch_size=None):
    r"""Extract data from HDF5 file.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    group : string
        Path to estimator.
    estimator : string
        Estimator to analyse.
    sample : int
        Sample to extract. A *single* sample of data will be extracted, with 0-based indexing starting
        from the first sample found in the HDF5 file. This is *different* than how samples are indexed 
        in the HDF5 file where "measurement block index is used". In HDF5, if equilibration is used, the
        measurement block index will not start at 0.
        Optional. Default None (return everything).
    transform : callable
        Function to transform data. Optional. See Notes for details.
    batch_size : int
        Size of batch to load data. Optional. Default None (load all data at once).
        
    Returns
    -------
    numer : :class:`numpy.ndarray`
        Numerator of estimator.
    denom : :class:`numpy.ndarray`
        Denominator of estimator.

    
    Notes
    -----

    **Transform Function Guidelines**

    - Takes a single argument (data of shape (nsamples,...)) and return transformed data
    - assume the leading dimension is nsamples

    +------------------------------+--------------------------------------+
    | Do                           | Don't                                |
    +==============================+======================================+
    | Return a new array           | Modify the data in place.            |
    +------------------------------+--------------------------------------+
    | Return the transformed data. | Return a tuple — return a single     |
    |                              | array instead.                       |
    +------------------------------+--------------------------------------+
    | Return None if data is       | Raise an exception                   |
    | invalid.                     |                                      |
    +------------------------------+--------------------------------------+

    """
    if transform is not None:
        # revisit this if issues with generators / yield arise
        if not isinstance(transform, Iterable):
            transform = [transform]
    else:
        # dummy empty transform - allows generic iteration below
        transform = []

    def apply_transform(data):
        for t in transform:
            if not callable(t):
                raise ValueError(f"Transform '{t.__name__}' is not callable. Must be a callable.")
            data = t(data)
        return data

    with h5py.File(filename, 'r') as fh5:
        dsets = list(fh5[group].keys())
        denom_id = [d for d in dsets if 'denominator' in d]
        numer_id = [d for d in dsets if estimator in d]
        if sample is not None:
            assert sample < len(numer_id)
            numer = apply_transform(from_complex(fh5[group][numer_id[sample]][:]))
            denom = from_complex(fh5[group][denom_id[sample]][:])
        else:
            # use a generator to load the data in batches
            if batch_size is None:
                numer = apply_transform(numpy.array([from_complex(fh5[group][d][:]) for d in numer_id]))
                denom = numpy.array([from_complex(fh5[group][d][:]) for d in denom_id])
            else:
                for numer_ids_batch,denom_ids_batch in zip(batched(numer_id, batch_size),batched(denom_id, batch_size)):
                    numer_batch = apply_transform(numpy.array([from_complex(fh5[group][d][:]) for d in numer_ids_batch]))
                    denom_batch = numpy.array([from_complex(fh5[group][d][:]) for d in denom_ids_batch])
                    numer = numpy.concatenate((numer, numer_batch), axis=0) if 'numer' in locals() else numer_batch
                    denom = numpy.concatenate((denom, denom_batch), axis=0) if 'denom' in locals() else denom_batch
        return numer, denom

def extract_observable(filename, estimator='back_propagated',
                       name='one_rdm', ix=None,
                       dataset_name=None, transform=None,
                       batch_size=None):
    r"""Extract observable from HDF5 file.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).
    estimator : string
        Estimator type to analyse. Options: back_propagated or mixed.
        Default: back_propagated.
    name : string
        Name of observable (see estimates.py for list).
    ix : int
        Back propagation path length to average. Optional.
        Default: None (chooses longest path).
    sample : int
        Sample to extract. Optional. Default None (return everything).
    dataset_name : string
        Name of dataset to extract. Optional. Default None (uses name).
    transform : callable
        Function to transform data. Optional. See Notes for details.
    batch_size : int
        Size of batch to load data. Optional. Default None (load all data at once).
    
    Returns
    -------
    obs : :class:`numpy.ndarray`
        Observable for a single sample or the full set of samples. Note if using
        free projection the numerator and denominator are returned separately.

    Notes
    -----

    **Transform Function Guidelines**

    - Takes a single argument (data of shape (nsamples,...)) and return transformed data
    - assume the leading dimension is nsamples

    +------------------------------+--------------------------------------+
    | Do                           | Don't                                |
    +==============================+======================================+
    | Return a new array           | Modify the data in place.            |
    +------------------------------+--------------------------------------+
    | Return the transformed data. | Return a tuple — return a single     |
    |                              | array instead.                       |
    +------------------------------+--------------------------------------+
    | Return None if data is       | Raise an exception                   |
    | invalid.                     |                                      |
    +------------------------------+--------------------------------------+

    """
    sym_md = get_metadata(filename)
    free_proj = sym_md['FreeProjection']
    if estimator == 'back_propagated':
        base = 'Observables/BackPropagated/'
        if ix is None:
            bp_md = get_metadata(filename, path=base)
            ix = bp_md['NumAverages'] - 1
        ename = MAP[name]
        base += ename['group'] + '/Average_{}'.format(ix)
    elif estimator == 'mixed':
        base = 'Observables/Mixed/'
        ename = MAP[name]
        base += ename['group'] + '/Average_0'
    else:
        print("Unknown estimator type: {} ".format(estimator))
        return None
    if dataset_name is None:
        dataset_name = ename['numer']
    numer, denom = extract_data(filename, base, dataset_name, transform=transform, batch_size=batch_size)
    if free_proj:
        return (numer, denom)
    else:
        # Use array broadcasting to divide by weights.
        #TODO: nsamples should be the tailing dimension in future versions
        return (numer.T / denom.T).T 

def get_estimator_len(filename, name='Mixed'):
    r"""Get number of samples of 1RDM from file.

    Parameters
    ----------
    filename : string
        output file containing density matrix (\*.h5 file).

    Returns
    -------
    len_rdm : int
        Number of samples of 1RDM.
    """
    with h5py.File(filename, 'r') as fh5:
        dms = fh5[name].keys()
        # Block size of RDM is not necessarily known so just be dumb and count.
        num_dm = len([n for n in dms if 'denominator' in n])
    return num_dm
