import numpy as np

from safiretools import HamiltonianBuilder, Lattice

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

builder = HamiltonianBuilder(lattice=lattice, nelec=nelec)

# add some terms
builder.nth_neighbor_hopping(t=[1.0,0.5])
builder.onsite_hubbard(U=8.0)
builder.nth_order_hubbard_Vij(V=2.0)

nbasis = lattice.N_sites

# add a custom term - in this case, random symmetric noise. Give it as a plain
#   (nbasis,nbasis) matrix; the builder stacks the spin sectors for you to match
#   the Hamiltonian's spin symmetry.
one_body_matrix = 0.0001*np.random.rand(nbasis,nbasis)
one_body_matrix = 0.5*(one_body_matrix + one_body_matrix.T)

builder.custom_one_body(one_body_matrix)

builder.finalize()

hamiltonian = builder.get_hamiltonian()

hamiltonian.to_hdf5("afqmc.h5")
free_electron(
    source=hamiltonian,
    nelec=nelec,
    output="afqmc.h5",
    lattice=lattice
)
