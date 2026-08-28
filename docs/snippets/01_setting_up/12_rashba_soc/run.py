from safiretools import Lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder
from afqmctools.hamiltonian.model.ham_class import SpinSymm
import afqmctools.utils.io as io
from afqmctools.wavefunction.free_electron import free_electron

lattice = Lattice.from_dict(
    params=dict(
        L1 = 3,
        L2 = 3,
        boundary1 = "PBC",
        boundary2 = "PBC"
    )
)
nelec = (5,5)

# list of hopping parameters is interpreted as follows:
#    hopping[0] is 't'
#    hopping[1] is 't^{prime}'
#    ....
#    hopping[n-1] is 't^{n}'
hopping = [1.0,0.5]

builder = HamiltonianBuilder(
    lattice=lattice,
    spin_symm=SpinSymm.NONCOLLINEAR
)
# add standard Hubbard terms
builder.nth_neighbor_hopping(t=hopping)
builder.onsite_hubbard(U=8.0)

# add Rashba SOC consistent with the hopping
builder.rashba_soc(rashba_lambda=0.3, t=hopping)
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
