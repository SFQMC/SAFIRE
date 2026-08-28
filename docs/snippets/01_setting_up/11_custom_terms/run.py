import scipy.sparse as sps
import numpy as np

from safiretools import Lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import HamiltonianComponent,SpinSymm
import afqmctools.utils.io as io
from afqmctools.wavefunction.free_electron import free_electron

lattice = Lattice.from_dict(
    params=dict(
        L1 = 3,
        L2 = 2,
        boundary1 = "PBC",
        boundary2 = "PBC"
    )
)
nelec = (2,2)

builder = HamiltonianBuilder(lattice=lattice)

# add some terms
builder.nth_neighbor_hopping(t=[1.0,0.5])
builder.onsite_hubbard(U=8.0)
builder.nth_order_hubbard_Vij(V=2.0)

nbasis = lattice.N_sites

# add a custom term - in this case, randon symetric noise
one_body_matrix = 0.0001*np.random.rand(nbasis,nbasis)
one_body_matrix = sps.csr_matrix(0.5*(one_body_matrix + one_body_matrix.T))
# the convention is to stack the spin-up hopping matrix on the spin-down hopping matrix
one_body_matrix = sps.vstack([one_body_matrix,one_body_matrix])

custom_one_body = HamiltonianComponent(
    csr_matrix=one_body_matrix,
    model_type='one_body',
    spin_symm=SpinSymm.COLLINEAR
)

# manually add the custom term
builder.hamiltonian["tij"] = custom_one_body

builder.finalize()

hamiltonian = builder.hamiltonian

io.write_model_hamiltonian(
    hamiltonian=hamiltonian,
    fname="afqmc.h5",
    nelec=nelec
)
free_electron(
    source=hamiltonian,
    nelec=nelec,
    output="afqmc.h5",
    lattice=lattice
)
