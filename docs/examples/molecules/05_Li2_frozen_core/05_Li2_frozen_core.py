r"""
Example 6: Frozen core/orbital tranformation.

From:

W. Purwanto, S. Zhang, H. Krakauer. Frozen-Orbital and Downfolding Calculations with 
Auxiliary-Field Quantum Monte Carlo. J. Chem. Theory Comput. 2013, 9, 11, 4825–4833
https://pubs.acs.org/doi/10.1021/ct4006486

Computing the AFQMC ground state energy of the Li_2 molecule in the cc-pVDZ
using the frozen core approximation to freeze the Li 1s orbitals.

ncore: 2	
trial wavefunction: RHF
refrence AFQMC energy: -14.9017(1)
FCI energy:	       -14.9005
"""
from pyscf import gto,scf

from afqmctools.utils.pyscf_utils import load_from_pyscf_chk_mol
from afqmctools.hamiltonian.mol import write_hamil_mol
from afqmctools.wavefunction.mol import write_wfn_mol


# run PySCF calculation to generate RHF wavefunction
equil_bond=2.673

atoms = f"""
Li 0.0 0.0 {equil_bond/2}
Li 0.0 0.0 {-equil_bond/2}
"""

mol = gto.M(
    atom = atoms,
    basis = "ccpvdz",
    verbose = 5
)

mf = scf.RHF(mol)
mf.chkfile = 'rhf.chk'
E_HF = mf.kernel()

wfn_chk = 'rhf.chk'
chol_tol = 1e-6

fout = 'afqmc.h5'


scf_data = load_from_pyscf_chk_mol(wfn_chk)

# specify the active space similarly to CAS methods,
# `cas =(ne,no)` where `ne`/`no` is the number of active electrrons / orbitals
# use `no=-1` to include all remaining orbitlas in the CAS space
ne = 2 # [He] 2s^1 for each Li
no = -1 # use all remaining orbitals in CAS space
write_hamil_mol(
    scf_data = scf_data,
    hamil_file = fout,
    chol_cut = chol_tol,
    cas = (ne,no),
    walker_type="closed"
)

# write the frozen core trial wavefunction
wfn = write_wfn_mol(
    scf_data = scf_data,
    filename = fout,
    cas = (ne,no)
)
