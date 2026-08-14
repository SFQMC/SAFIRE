# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import h5py as h5
import numpy
from afqmctools.utils.io import to_complex

def write_to_hdf5(f,dataset,data,dtype=None):
    if dtype is None:
        dtype = data.dtype
    if not isinstance(f,h5.File):
        raise ValueError("[Developer Error] called write_to_hdf5 on a non-h5py.File instance")
    if dataset in f:
        del f[dataset]
    f.create_dataset(dataset,data=data.astype(dtype), dtype=dtype)

# TODO: some parameter names are highly misleading. ex: "real_chol" usually refers to
#         whether the one-body terms are real valued (vs. the Cholesky vectors as the name implies)

def write_dense(
        hcore,
        chol, 
        nelec, 
        nmo, 
        enuc=0.0,
        filename='hamiltonian.h5',
        real_chol=None,
        ortho=None,
        verbose=None # unused - kept for backwards compatibility
        ):
    

    with h5.File(filename, 'a') as fh5:

        write_to_hdf5(fh5,'Hamiltonian/Energies',
                      data=numpy.array([enuc,0.]),
                      dtype=numpy.float64
        )

        if real_chol is None:
            real_chol = not numpy.any(numpy.iscomplex(chol))

        if real_chol:
            write_to_hdf5(fh5,'Hamiltonian/DenseFactorized/L',numpy.real(chol))
        else:
            write_to_hdf5(fh5,'Hamiltonian/DenseFactorized/L', data=to_complex(chol.astype(numpy.complex128)))
        
        # Test the dtype, not the values. `numpy.iscomplex` is elementwise and returns
        # False wherever the imaginary part is exactly zero, so `numpy.all(...)` only
        # holds when EVERY entry has a non-zero imaginary part. A physical hcore is
        # hermitian, so its diagonal is real, and the old predicate sent it down the
        # `numpy.real` branch -- silently discarding every off-diagonal imaginary part.
        if numpy.iscomplexobj(hcore):
            write_to_hdf5(fh5,'Hamiltonian/hcore',data=to_complex(hcore.astype(numpy.complex128)))
        else:
            write_to_hdf5(fh5,'Hamiltonian/hcore', data = numpy.real(hcore))

        write_to_hdf5(fh5,'Hamiltonian/dims', data = numpy.array([0, 0, 0, nmo,
                                               nelec[0], nelec[1], 0,
                                               chol.shape[-1]], dtype=numpy.int32))
        write_to_hdf5(fh5,'Hamiltonian/ComplexIntegrals', data=numpy.array([not int(real_chol)],
                                                          dtype=numpy.int32))
        if ortho is not None:
            write_to_hdf5(fh5,'Hamiltonian/X',data=ortho)


def to_sparse(vals, cutoff=1e-8):
    nz = numpy.where(numpy.abs(vals) > cutoff)
    ix = numpy.empty(nz[0].size+nz[1].size, dtype=numpy.int32)
    ix[0::2] = nz[0]
    ix[1::2] = nz[1]
    vals = vals[nz]
    return ix, vals
