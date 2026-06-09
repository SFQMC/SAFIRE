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
import numba as nb

jit = nb.njit

def check_integer(value, name=None):
    if not np.allclose(value, int(value)):
        error_string = f"{name} value {value} is not an integer" if name else f"Value {value} is not an integer"
        raise ValueError(error_string)
    else:
        return int(value)

def hermitize_factory(walker_type) -> callable:
    r"""
    Factory function to generate a hermitian transform function.

    The transform makes a matrix hermitian by averaging with its conjugate transpose.
    .. math::
        M_{nij} = \\frac{1}{2} (M_{nij} + M_{nji}^{*})
    
    where 'n' is the sample dimension, and 'i' and 'j' are the matrix dimensions.

    Notes
    -----
    Only works for 2D matrices, where the first dimension is the sample dimension, and the
    second dimension is the square of a the matrix dimension.
    """

    if walker_type == 'closed':
        def hermitize(data):
            M_squared = data.shape[-1]
            M = np.sqrt(M_squared)
            if M != int(M):
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not a perfect square")
            else:
                M = int(M)
            transposed_indices = np.ravel_multi_index(np.unravel_index(np.arange(M_squared), (M,M)), (M,M), order='F')
            return 0.5 * (data + data[:,transposed_indices].conj())
        return hermitize
    elif walker_type == 'collinear':
        def hermitize(data):
            nspin = 2
            M_squared = data.shape[-1]/nspin
            if not np.allclose(M_squared,int(M_squared)): 
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not divisible by number of spins {nspin}")
            else:
                M_squared = int(M_squared)

            M = np.sqrt(M_squared)
            
            if not M == int(M):
                raise ValueError(f"Flattened Matrix dimension {M_squared} is not a perfect square")
            else:
                M = int(M)
            new_data = np.zeros((data.shape[0], nspin, M_squared), dtype=data.dtype)
            for s in range(nspin):
                transposed_indices = np.ravel_multi_index(np.unravel_index(np.arange(M_squared), (M,M)), (M,M), order='F')
                # select the s-th spin channel with data[:,s*M_squared:(s+1)*M_squared]; and transpose by indexing with transposed_indices;
                new_data[:,s,:] = 0.5 * (data[:,s*M_squared:(s+1)*M_squared] + data[:,s*M_squared:(s+1)*M_squared][:,transposed_indices].conj())
            return new_data
        return hermitize
    elif walker_type == 'non_collinear':
        def hermitize(data):
            # data shape: (n_samples, 4*M²) representing flattened (2M, 2M) spinor matrices
            M_squared_times_4 = data.shape[-1]
            M_squared = M_squared_times_4 // 4
            
            if not np.allclose(M_squared_times_4 / 4, M_squared):
                raise ValueError(f"Flattened Matrix dimension {M_squared_times_4} is not divisible by 4")
            
            M = np.sqrt(M_squared)
            if M != int(M):
                raise ValueError(f"Flattened Matrix dimension {M_squared} (per spin block) is not a perfect square")
            else:
                M = int(M)
            
            M_spinor = 2 * M
            M_spinor_squared = M_spinor * M_spinor
            
            transposed_indices = np.ravel_multi_index(
                np.unravel_index(np.arange(M_spinor_squared), (M_spinor, M_spinor)), 
                (M_spinor, M_spinor), 
                order='F'
            )
            return 0.5 * (data + data[:, transposed_indices].conj())
        return hermitize
    else:
        raise ValueError(f"Unknown walker type: {walker_type}")

def upper_triangle_factory(walker_type) -> callable:
    r"""
    Factory function to generate an upper triangle transform function.

    The transform extracts the upper triangle of a matrix.

    .. math::
        M_{nij} = M_{nij} \\quad i \\leq j
    
    where 'n' is the sample dimension, and 'i' and 'j' are the matrix dimensions.

    Notes
    -----
    Only works for 2D matrices, where the first dimension is the sample dimension, and the
    second dimension is the square of a the matrix dimension.
    """

    if walker_type == 'closed':
        def upper_triangle(data):
            dm_size = data.shape[-1]
            nbasis = (1 + np.sqrt(1 + 8 * dm_size)) / 2
            nbasis = check_integer(nbasis, name="nbasis")
            
            nsample = data.shape[0]
            diag_two_rdm = np.zeros((nsample, nbasis, nbasis), dtype=data.dtype)
            ij = 0
            for i in range(nbasis):
                for j in range(i+1, nbasis):
                    diag_two_rdm[:,i,j] = data[:,ij]
                    diag_two_rdm[:,j,i] = data[:,ij].conj()
                    ij += 1
            return diag_two_rdm
        return upper_triangle
    elif walker_type == 'collinear':
        def upper_triangle(data):
            dm_size = data.shape[-1]
            nbasis = (1 + np.sqrt(1 + 8 * dm_size)) / 4
            nbasis = check_integer(nbasis, name="nbasis")
            
            nsample = data.shape[0]
            diag_two_rdm = np.zeros((nsample, 2*nbasis, 2*nbasis), dtype=data.dtype)
            ij = 0
            for i in range(2*nbasis):
                for j in range(i+1, 2*nbasis):
                    diag_two_rdm[:,i,j] = data[:,ij]
                    diag_two_rdm[:,j,i] = data[:,ij].conj()
                    ij += 1
            return diag_two_rdm
        return upper_triangle
    elif walker_type == 'non_collinear':
        # Note: this is identical to the collinear case, as the diagonal two-rdm
        #         output format is the same for both walker types.
        def upper_triangle(data):
            dm_size = data.shape[-1]
            nbasis = (1 + np.sqrt(1 + 8 * dm_size)) / 4
            nbasis = check_integer(nbasis, name="nbasis")
            
            nsample = data.shape[0]
            diag_two_rdm = np.zeros((nsample, 2*nbasis, 2*nbasis), dtype=data.dtype)
            ij = 0
            for i in range(2*nbasis):
                for j in range(i+1, 2*nbasis):
                    diag_two_rdm[:,i,j] = data[:,ij]
                    diag_two_rdm[:,j,i] = data[:,ij].conj()
                    ij += 1
            return diag_two_rdm
        return upper_triangle
    else:
        raise NotImplementedError("Upper triangle extraction only implemented for closed shell walkers.")

def eval_hubbard_from_diag_two_rdm_factory(U, walker_type):
    r"""
    Factory function to generate a Hubbard U energy transform function from diagonal two-rdm.

    The transform computes the Hubbard U energy from the diagonal of the two-body density matrix.
    
    .. math::
        E_{U} = U \sum_{i} \rho_{i,\uparrow i,\uparrow; i,\downarrow i,\downarrow}
    
    where 'i' is the number of sites.

    For closed shell, a factor of 2 is applied to the result to account for the two spin channels.
    for collinear, both spin channels are evaluated separately, and the result is summed.
    For non-collinear, the spin channels are evaluated separately, and the result is summed.

    Parameters
    ----------
    U : float
        The Hubbard U parameter.
    walker_type : str
        The type of walker ('closed', 'collinear', 'non_collinear').

    Transform Parameters
    --------------------
    diag2rdm : np.ndarray
        The input diagonal two-rdm data of shape (n_samples, M, M) - i.e. expressed as the full array versus the 
        upper-triangular form - Where M is the number of basis functions. M is the number of spin orbitals for 
        all cases except closed shell, where M is the number of spatial orbitals.
    """
    if walker_type == 'closed':
        def transform(diag2rdm):
            # Note: Closed walkers are intentially not implemented for lattice models, so no need to implement here.
            raise NotImplementedError("Hubbard U energy from diag two-rdm not implemented for closed shell walkers.")
        return transform
    elif walker_type == 'collinear':
        def transform(diag2rdm):
            nbasis = diag2rdm.shape[1] / 2
            nbasis = check_integer(nbasis, name="nbasis")
            diag2rdm_iup_idown = np.diagonal(diag2rdm[:, nbasis:, :nbasis], axis1=1, axis2=2)
            return U * np.sum(diag2rdm_iup_idown, axis=1)[:, np.newaxis]
        return transform
    elif walker_type == 'non_collinear':
        def transform(diag2rdm):
            nbasis = diag2rdm.shape[1] / 2
            nbasis = check_integer(nbasis, name="nbasis")
            diag2rdm_iup_idown = np.diagonal(diag2rdm[:, nbasis:, :nbasis], axis1=1, axis2=2)
            return U * np.sum(diag2rdm_iup_idown, axis=1)[:, np.newaxis]
        return transform
    else:
        raise ValueError(f"Unknown walker type: {walker_type}")

def eval_one_body_obs_factory(one_body_operator=None,walker_type=None):
    """
    Factory function to generate a one-body energy transform function.

    The transform computes a one body observable from the one-body operator and the density matrix.
    
    .. math::
        O_{1} = \\sum_{pq} \\rho_{pq} O_{pq}
    
    where 'p' and 'q' are the matrix dimensions, and 'h' is the one-body operator.

    For closed shell, a factor of 2 is applied to the result to account for the two spin channels.
    for collinear, both spin channels are evaluated separately, and the result is summed.
    For non-collinear, the spin channels are evaluated separately, and the result is summed.
    """

    # needed for jit below
    one_body_operator = one_body_operator.astype(np.complex128)

    if walker_type == 'closed':
        @jit(cache=True)
        def transform(data):
            return 2*(data @ one_body_operator.flatten())[:,np.newaxis]
        return transform
    elif walker_type == 'collinear':
        @jit(cache=True)
        def transform(data):
            spin_up_contribution = data[:,0] @ one_body_operator.flatten()
            spin_down_contribution = data[:,1] @ one_body_operator.flatten()
            return (spin_up_contribution + spin_down_contribution )[:,np.newaxis]
        return transform
    elif walker_type == 'noncollinear':
        @jit(cache=True)
        def transform(data):
            return (data @ one_body_operator.flatten())[:,np.newaxis]
        return transform
    else:
        raise ValueError(f"Unknown walker type: {walker_type}")


def eval_two_body_obs_factory(cholesky=None,two_body_operator=None,walker_type=None):
    """
    Factory function to generate a two-body energy transform function.

    The transform computes a two body observable from the two-body operator and the density matrix.
    
    .. math::
        O_{2} = \\frac{1}{2} \\sum_{pqrs} \\rho_{pqrs} ( pq | rs )
    
    where 'p' and 'q' are the matrix dimensions, and 'h' is the two-body operator.

    For closed shell, a factor of 2 is applied to the result to account for the two spin channels.
    for collinear, both spin channels are evaluated separately, and the result is summed.
    For non-collinear, the spin channels are evaluated separately, and the result is summed.
    """
    if two_body_operator is None and cholesky is None:
        raise ValueError("Two-body integrals not provided; please provide them in Chemist's convention via the 'eri' argument "
                         " or in Cholesky format via the 'cholesky' argument")
    # needed for jit below
    two_body_operator = two_body_operator.astype(np.complex128)

    nbasis_4 = two_body_operator.shape[-1]
    nbasis = np.power(nbasis_4, 0.25)
    if not nbasis == int(nbasis):
        raise ValueError(f"Flattened Matrix dimension {nbasis_4} is not a 4th root")
    else:
        nbasis = int(nbasis)

    if two_body_operator is not None:
        if walker_type == 'closed':
            raise NotImplementedError("Closed walker type not implemented")
        elif walker_type == 'collinear':
            num_spin_sectors = 3
            @jit(cache=True)
            def transform(data):
                data = data.reshape((-1,num_spin_sectors,nbasis_4))
                # compute the two-body energy for each spin channel
                O2_aaaa = np.dot(data[:,0], two_body_operator)
                O2_aabb = np.dot(data[:,1], two_body_operator)
                O2_bbaa = O2_aabb
                O2_bbbb = np.dot(data[:,2], two_body_operator)
                return 0.5*(O2_aaaa + O2_bbbb + O2_aabb + O2_bbaa)[:,np.newaxis]
            return transform
        elif walker_type == 'noncollinear':
            raise NotImplementedError("Non-collinear walker type not implemented")
        else:
            raise ValueError(f"Unknown walker type:{ walker_type}")
    elif cholesky is not None:
        raise NotImplementedError("Cholesky not implemented; contact the developers")

