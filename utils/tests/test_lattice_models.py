# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import pytest
# for Dev
import numpy as np
import scipy.sparse as sps

from afqmctools.systems.lattice import get_lattice
from afqmctools.hamiltonian.model.builder import HamiltonianBuilder, skip_empty_params
from afqmctools.hamiltonian.model.ham_class import SpinSymm


class MockBuilder:
    """Mock class to test decorators."""

    def __init__(self):
        self.called_with = []

    @skip_empty_params
    def test_method(self, params, *args, **kwargs):
        """Method decorated with skip_empty_params."""
        self.called_with.append(params)
        print(f"  MockBuilder test_method called with params={params}")


class TestLatticeClass:
    """
    use 4x4 square lattice as test case.

    Ensure that:
        - lattice that hasn't been built raises an error on .get_sites(), .get_nth_neighbors() AND no empty lists are returned
        - ensure that the ALL of correct neighbors are returned for n=1,6 (the max) AND that the list is symmetric
        - ensure that an error is raised when a site would be its own neighbor
    """

    @pytest.fixture(scope='class')
    def test_square_lattice(self):
        return get_lattice(
            params={
                'L1' : 4,
                'L2' : 4,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
                'build' : False
            }
        )

    @pytest.mark.parametrize(
        "func,arg,error",
        [
            ("get_nth_neighbors",0,RuntimeError),
            ("get_nth_neighbors",1,RuntimeError),
            ("get_nth_neighbors",None,RuntimeError),
            ("get_sites",None,RuntimeError),
            ("get_nth_image_neighbors",0,RuntimeError),
            ("get_nth_image_neighbors",1,RuntimeError),
            ("get_nth_image_neighbors",None,RuntimeError),
        ]
    )
    def test_unbuilt(self,test_square_lattice,func,arg,error):
        assert test_square_lattice._built == False
        with pytest.raises(error) as e:
            call = getattr(test_square_lattice,func)
            if arg is not None:
                call(arg)
            else:
                call()
            
    
    @pytest.fixture(scope='class')
    def test_square_lattice_built(self,test_square_lattice):
        test_square_lattice.build()
        return test_square_lattice

    @pytest.mark.parametrize(
        "n,error",
        [
            (1,None),
            (2,None),
            (3,None),
            (4,None),
            (5,None),
            #(6,ValueError) # API change, may return
        ]
    )
    def test_nearest_neighbors(self,test_square_lattice_built,n,error):
        if error is None:
            test_square_lattice_built.get_nth_neighbors(n=n)
        else:
            with pytest.raises(error) as e:
                test_square_lattice_built.get_nth_neighbors(n=n)
                assert f"attempted to build nth order neighbors for n={n} " in e

    @pytest.mark.parametrize(
        "lattice_type",
        [
            ("Square"),
            ("Triangular"),
            ("Honeycomb"),
            ("Kagome"),
        ]
    )
    def test_unique_dist(self,lattice_type):
        lattice = get_lattice(
            params={
                'L1' : 4,
                'L2' : 4,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
                'build' : True,
                'type' : lattice_type,
            }
        )
        nearest = lattice.get_nth_neighbors(1)
        dists,counts = np.unique(np.round(lattice._dist_map,8), return_counts=True)
        # ensure unique distances
        assert np.all(counts==1)

    @pytest.mark.parametrize("lattice_type,nNN", [
            ("Square",4),
            ("Triangular",6),
            ("Honeycomb",3),
            ("Kagome",4),
            ]
      )
    @pytest.mark.parametrize("axis1,axis2", [
            ("Open","Open"),
            ("Open","PBC"),
            ("PBC","PBC"),
            ]
    )
    def test_neighbor_count(self,lattice_type,nNN, axis1,axis2):
        Nx,Ny = 6,4
        lattice = get_lattice(
            params={
                'L1' : Nx,
                'L2' : Ny,
                'boundary1' : axis1,
                'boundary2' : axis2,
                'build' : True,
                'type' : lattice_type,
            }
        )
        N = lattice.N_sites
        T0 = np.zeros((N,N))
        nearests = lattice.get_nth_neighbors(1)
        for nearest in nearests:
            T0[nearest.i,nearest.j] = 1
        assert np.allclose(T0,T0.T.conj()) # check hermitian
        numNN,counts = np.unique(np.sum(T0,axis=1), return_counts=True)
        print(lattice_type,nNN,numNN,counts)
        if axis1=="PBC" and axis2=="PBC":
          assert numNN[0] == nNN
          assert counts[0] == N
          assert len(numNN)==1
        elif axis2=="Open" and axis1=="Open":
          if lattice_type=="Triangular":
            # current formulation, left corner has 3 NN, right has 2
            assert np.allclose(numNN,[2,3,4,6])
          else:
            assert np.allclose(numNN,[nNN-2,nNN-1,nNN])
            #assert np.allclose(counts,[4,(Nx-2)*2+(Ny-2)*2,N-Nx*2-Ny*2+4])
        elif axis2=="PBC" and axis1=="Open":
          if lattice_type=="Triangular":
            assert np.allclose(numNN,[nNN-2,nNN])
          elif lattice_type=="Kagome":
            assert np.allclose(numNN,[nNN-2,nNN-1,nNN])
          else:
            assert np.allclose(numNN,[nNN-1,nNN])

        else:
          assert 1==2 # not handled yet!

    @pytest.mark.parametrize("lattice_type", [
            ("Square"),
            ("Triangular"),
            ("Honeycomb"),
            ("Kagome"),
            ]
      )
    @pytest.mark.parametrize("make_cyl", [None,"XC","YC"])
    @pytest.mark.parametrize("n", range(1,6)
    )
    def test_pbc_neighbor_count(self,lattice_type,make_cyl,n):
        Nx,Ny = 6,6
        lattice = get_lattice(
            params={
                'L1' : Nx,
                'L2' : Ny,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
                'build' : True,
                'type' : lattice_type,
                'cyl_mode' : make_cyl,
            }
        )
        N = lattice.N_sites
        T0 = np.zeros((N,N))
        nearests = lattice.get_nth_neighbors(n)
        for nearest in nearests:
            T0[nearest.i,nearest.j] = 1
        assert np.allclose(T0,T0.T.conj()) # check hermitian
        numNN,counts = np.unique(np.sum(T0,axis=1), return_counts=True)
        print(lattice_type,numNN,counts)
        assert counts[0] == N
        assert len(numNN)==1 # one value everyone has

    @pytest.mark.parametrize("lattice_type", [
            ("Square"),
            ("Triangular"),
            ("Honeycomb"),
            ("Kagome"),
            ]
      )
    def test_planewaves(self,lattice_type,):
        Nx,Ny = 4,3 # rectangular to ensure more difficult case
        lattice = get_lattice(
            params={
                'L1' : Nx,
                'L2' : Ny,
                'boundary1' : "PBC",
                'boundary2' : "PBC",
                'build' : True,
                'type' : lattice_type,
            }
        )
        N = lattice.N_sites
        T0 = np.zeros((N,N))+0.j
        nearests = lattice.get_nth_neighbors(1)
        for nearest in nearests:
            T0[nearest.i,nearest.j] = -1*np.exp(1j*nearest.phase)
        assert np.allclose(T0,T0.T.conj()) # check hermitian
        # test that Uk is a proper unitary rotation and block-diagonalized T
        orig_eigs = np.linalg.eigvalsh(T0)
        Uk = lattice.get_planewaves()
        Hk = Uk@ T0@ Uk.conj().T
        new_eigs = np.linalg.eigvalsh([Hk[i*lattice.nb:(i+1)*lattice.nb,i*lattice.nb:(i+1)*lattice.nb] 
                                     for i in range(lattice.L[0]*lattice.L[1])]).flatten()
        assert np.allclose(sorted(orig_eigs),sorted(new_eigs))


class TestModelHamiltonianBuilder:
    """Test that the Model builder builds the correct Hamiltonian terms
    """

    @pytest.mark.parametrize(
        "nbands,epsilon,expected",
        [
            (
                2,
                0.5,
                0.5*np.eye(4)
            ),
            (
                2, 
                np.array(
                    [[0.0,0.5],
                     [0.5,0.0]]
                ), 
                np.array(
                    [[0.0,0.5,0.,0.],
                     [0.5,0.,0.0,0.0],
                     [0.0,0.0,0.,0.5],
                     [0.,0.0,0.5,0.0]]
                ),
            ),
            (
                2,
                np.array(
                    [[-2.0,0.0],
                    [0.0,0.5]]
                ), 
                np.array(
                    [[-2.0,0.,0.,0.],
                    [0.,0.5,0.,0.],
                    [0.,0.,-2.0,0.],
                    [0.,0.0,0.,0.5]]
                ),
            )
        ]
    )
    def test_onsite_epsilon(self,nbands,epsilon,expected):
        builder = HamiltonianBuilder(nbands=nbands,lattice=get_lattice(params={"L1":2,"L2":1,"boundary1":"PBC","boundary2":"PBC"}))
        builder.onebody_onsite(epsilon)
        builder.finalize()
        one_body = builder.hamiltonian.get_one_body()
        assert np.allclose(one_body.toarray(),expected)


    @pytest.mark.parametrize("case,expectedU1,expectedU2",[
        (
           {
            "lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC"),
            "hamiltonian" : dict(U = 4.0, V = 1.5)
            },
            np.array(
                [[4.0, 1.5, 1.5, 2*1.5, 0, 0],
                [0, 4.0, 1.5, 0, 2*1.5, 0],
                [0, 0, 4.0, 0, 0, 2*1.5],
                [0, 0, 0, 4.0, 1.5, 1.5 ],
                [0, 0, 0, 0, 4.0, 1.5 ],
                [0, 0, 0, 0, 0, 4.0]]
            ),np.array(
                [[0, 1.5, 1.5, 2*1.5, 0, 0],
                [0, 0, 1.5, 0, 2*1.5, 0],
                [0, 0, 0, 0, 0, 2*1.5],
                [0, 0, 0, 0, 1.5, 1.5 ],
                [0, 0, 0, 0, 0, 1.5 ],
                [0, 0, 0, 0, 0, 0]]
            )
        ),
        (
           {
            "lattice" : dict(L1=2,L2=3,boundary1="open",boundary2="open"),
            "hamiltonian" : dict(U = 4.0, V = 1.5)
            },
            np.array(
                [[4.0, 1.5, 0, 1.5, 0, 0],
                [0, 4.0, 1.5, 0, 1.5, 0],
                [0, 0, 4.0, 0, 0, 1.5],
                [0, 0, 0, 4.0, 1.5, 0.0 ],
                [0, 0, 0, 0, 4.0, 1.5 ],
                [0, 0, 0, 0, 0, 4.0]]
            ),np.array(
                [[0, 1.5, 0, 1.5, 0, 0],
                [0, 0, 1.5, 0, 1.5, 0],
                [0, 0, 0, 0, 0, 1.5],
                [0, 0, 0, 0, 1.5, 0.0 ],
                [0, 0, 0, 0, 0, 1.5 ],
                [0, 0, 0, 0, 0, 0]]
            )
        ),
        (
           {
            "lattice" : dict(L1=3,L2=2,boundary1="open",boundary2="PBC"),
            "hamiltonian" : dict(U = 4.0, V = 1.5)
            },
            np.array(
                [[4.0, 2*1.5, 1.5, 0, 0, 0],
                [0, 4.0, 0, 1.5, 0, 0],
                [0, 0, 4.0, 2*1.5, 1.5, 0],
                [0, 0, 0, 4.0, 0, 1.5 ],
                [0, 0, 0, 0, 4.0, 2*1.5 ],
                [0, 0, 0, 0, 0, 4.0]]
            ),np.array(
                [[0, 2*1.5, 1.5, 0, 0, 0],
                [0, 0, 0, 1.5, 0, 0],
                [0, 0, 0, 2*1.5, 1.5, 0],
                [0, 0, 0, 0, 0, 1.5 ],
                [0, 0, 0, 0, 0, 2*1.5 ],
                [0, 0, 0, 0, 0, 0]]
            )
        ),
        (
           {
            "lattice" : dict(L1=3,L2=2,boundary1="open",boundary2="PBC",type="triangular"),
            "hamiltonian" : dict(U = 4.0, V = 1.5)
            },
            np.array(
                [[4.0, 2*1.5, 1.5, 1.5, 0, 0],
                [0, 4.0, 1.5, 1.5, 0, 0],
                [0, 0, 4.0, 2*1.5, 1.5, 1.5],
                [0, 0, 0, 4.0, 1.5, 1.5 ],
                [0, 0, 0, 0, 4.0, 2*1.5 ],
                [0, 0, 0, 0, 0, 4.0]]
            ),np.array(
                [[0.0, 2*1.5, 1.5, 1.5, 0, 0],
                [0, 0.0, 1.5, 1.5, 0, 0],
                [0, 0, 0.0, 2*1.5, 1.5, 1.5],
                [0, 0, 0, 0.0, 1.5, 1.5 ],
                [0, 0, 0, 0, 0.0, 2*1.5 ],
                [0, 0, 0, 0, 0, 0.0]]
            )
        ),
        ])
    def test_hubbard_V(self,case,expectedU1,expectedU2):
        """
        Test that the Hubbard V term is probably built
        """        
        expected_Umat = np.append(expectedU1,expectedU2,axis=0)

        H = HamiltonianBuilder.from_input(source=case).hamiltonian

        interactions = H["Uij"]
        actual_Umat = sum(interactions).toarray()

        print("Expected Uij matrix:")
        print(expected_Umat)
        print("Actual Uij matrix:")
        print(actual_Umat)

        assert np.allclose(actual_Umat,expected_Umat)
        print("\nUij matricies match!\n")

    @pytest.mark.parametrize("case,expectedU1",[
        ({
            "lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC"),
            "hamiltonian" : dict(J_heisenberg = 2.0)
        },0.25*np.array(
            [[0, -2.0, -2.0, -2*2.0,      0,      0],
             [0,    0, -2.0,      0, -2*2.0,      0],
             [0,    0,    0,      0,      0, -2*2.0],
             [0,    0,    0,      0,   -2.0,  -2.0 ],
             [0,    0,    0,      0,      0,  -2.0 ],
             [0,    0,    0,      0,      0,      0]]
            )
        ),
        ({
            "lattice" : dict(L1=2,L2=3,boundary1="open",boundary2="open"),
            "hamiltonian" : dict(J_heisenberg = 2.0)
        },0.25*np.array(
            [[0, -2.0,  0.0, -2.0,      0,      0],
             [0,    0, -2.0,      0, -2.0,      0],
             [0,    0,    0,      0,    0,   -2.0],
             [0,    0,    0,      0, -2.0,    0.0],
             [0,    0,    0,      0,    0,   -2.0],
             [0,    0,    0,      0,    0,      0]]
            )
        ),
        ({
            "lattice" : dict(L1=3,L2=2,boundary1="open",boundary2="PBC"),
            "hamiltonian" : dict(J_heisenberg = 2.0)
        },0.25*np.array(
            [[0, -2.0*2.0, -2.0,        0,    0,      0],
             [0,        0,    0,     -2.0,    0,      0],
             [0,        0,    0, -2.0*2.0, -2.0,      0],
             [0,        0,    0,        0,    0,   -2.0],
             [0,        0,    0,        0,    0, -2*2.0],
             [0,        0,    0,        0,    0,      0]]
            )
        ),
        ({
            "lattice" : dict(L1=3,L2=2,boundary1="open",boundary2="PBC",type="triangular"),
            "hamiltonian" : dict(J_heisenberg = 2.0)
        },0.25*np.array(
            [[0, -2.0*2.0, -2.0,     -2.0,    0,      0],
             [0,        0, -2.0,     -2.0,  0.0,      0],
             [0,        0,    0, -2.0*2.0, -2.0,   -2.0],
             [0,        0,    0,        0,  -2.0,   -2.0],
             [0,        0,    0,        0,    0, -2*2.0],
             [0,        0,    0,        0,    0,      0]]
            )
        )
    ])
    def test_heisenberg_J(self,case,expectedU1):
        """
        Test that the Hubbard V term is probably built
        """
        print("Case: ", case)
        expected_U2mat = -1*expectedU1    
        expected_Umat = np.append(expectedU1,expected_U2mat,axis=0)
        # Note: the factor of 4 is because we are using U1 and U2 to represent part
        #         of the Hamiltonian, but for the "J" part, we explicitly include 
        #         factors of 1/2 when evaluating the energy.
        expected_Jmat = 4*expectedU1
        
        H = HamiltonianBuilder.from_input(source=case).hamiltonian
        interactions = H["Uij"]
        actual_Umat_shape = max( [interaction.csr_array.shape for interaction in interactions] )
        actual_Umat = np.zeros(actual_Umat_shape)
        for interaction in interactions:
            shape = interaction.csr_array.shape
            actual_Umat[:shape[0],:shape[1]] = interaction.csr_array.toarray()

        print("Expected Uij matrix:")
        print(expected_Umat)
        print("Actual Uij matrix:")
        print(actual_Umat)

        assert np.allclose(actual_Umat,expected_Umat)
        print("\nUij matricies match!\n")

        interactions = H["J_heisenberg"]
        actual_Jmat = sum(interactions).toarray()

        print("Expected Jij matrix:")
        print(expected_Jmat)
        print("Actual Jij matrix:")
        print(actual_Jmat)

        assert np.allclose(actual_Jmat,expected_Jmat)
        print("\nJij matricies match!\n")

    @pytest.mark.dev
    @pytest.mark.parametrize("case,expected_up_down_x,expected_up_down_y",[
        (
            {
                "lattice" : dict(L1=3,L2=3,boundary1="PBC",boundary2="PBC"),
                "hamiltonian" : dict(t=[1.0,0.5], rashba_lambda = 0.3)
            },
            -0.3*np.array([ # this is prop. to t_{<ij>}*(r_x)_{<ij>} + t'_{<<ij>>}*(r_x)_{<<ij>>}
                [     0,      0,      0, +1*1.0, +1*0.5, +1*0.5, -1*1.0, -1*0.5, -1*0.5],
                [     0,      0,      0, +1*0.5, +1*1.0, +1*0.5, -1*0.5, -1*1.0, -1*0.5],
                [     0,      0,      0, +1*0.5, +1*0.5, +1*1.0, -1*0.5, -1*0.5, -1*1.0],
                [-1*1.0, -1*0.5, -1*0.5,      0,      0,      0, +1*1.0, +1*0.5, +1*0.5],
                [-1*0.5, -1*1.0, -1*0.5,      0,      0,      0, +1*0.5, +1*1.0, +1*0.5],
                [-1*0.5, -1*0.5, -1*1.0,      0,      0,      0, +1*0.5, +1*0.5, +1*1.0],
                [+1*1.0, +1*0.5, +1*0.5, -1*1.0, -1*0.5, -1*0.5,      0,      0,      0],
                [+1*0.5, +1*1.0, +1*0.5, -1*0.5, -1*1.0, -1*0.5,      0,      0,      0],
                [+1*0.5, +1*0.5, +1*1.0, -1*0.5, -1*0.5, -1*1.0,      0,      0,      0]
            ]),
            1j*0.3*np.array([ # this is prop. to t_{<ij>}*(r_y)_{<ij>} + t'_{<<ij>>}*(r_y)_{<<ij>>}
                [     0,  1*1.0, -1*1.0,      0, +1*0.5, -1*0.5,      0, +1*0.5, -1*0.5],
                [-1*1.0,      0,  1*1.0, -1*0.5,      0, +1*0.5, -1*0.5,      0, +1*0.5],
                [ 1*1.0, -1*1.0,      0, +1*0.5, -1*0.5,      0, +1*0.5, -1*0.5,      0],
                [     0, +1*0.5, -1*0.5,      0, +1*1.0, -1*1.0,      0, +1*0.5, -1*0.5],
                [-1*0.5,      0, +1*0.5, -1*1.0,      0, +1*1.0, -1*0.5,      0, +1*0.5],
                [+1*0.5, -1*0.5,      0, +1*1.0, -1*1.0,      0, +1*0.5, -1*0.5,      0],
                [     0, +1*0.5, -1*0.5,      0, +1*0.5, -1*0.5,      0, +1*1.0, -1*1.0],
                [-1*0.5,      0, +1*0.5, -1*0.5,      0, +1*0.5, -1*1.0,      0, +1*1.0],
                [+1*0.5, -1*0.5,      0, +1*0.5, -1*0.5,      0, +1*1.0, -1*1.0,      0]
            ]),
        )
    ])
    def test_rashba_soc(self,case,expected_up_down_x,expected_up_down_y):
        print("Case:", case)
        expected_up_down = expected_up_down_x + expected_up_down_y

        exptected_rashba_matrix = np.block(
           [
              [ np.zeros_like(expected_up_down_x), expected_up_down],
              [expected_up_down.conj().T, np.zeros_like(expected_up_down_x)]
           ]
        )
        builder = HamiltonianBuilder(
           lattice=get_lattice(params=case["lattice"]),
           spin_symm=SpinSymm.NONCOLLINEAR
        )
        builder.rashba_soc(
            rashba_lambda=case["hamiltonian"]["rashba_lambda"],
            t=case["hamiltonian"].get("t",0.0)
        )
        one_body = sum(builder.hamiltonian["tij"]).csr_array.toarray()
        assert np.allclose(one_body,exptected_rashba_matrix)

        # TODO: refactor by invoking through the director

    
    @pytest.mark.dev
    @pytest.mark.parametrize("case,shape,spin_symm,error",[
        (
            {"lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC")},(6,6),SpinSymm.COLLINEAR,None
        ),
        (
            {"lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC")},(2*6,2*6),SpinSymm.NONCOLLINEAR,None
        ),
        (
            {"lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC")},(2*6,6),SpinSymm.COLLINEAR,None
        ),
        (
            {"lattice" : dict(L1=2,L2=3,boundary1="PBC",boundary2="PBC")},(3,2),SpinSymm.COLLINEAR,ValueError
        )
    ])
    def test_custom_one_body(self,case,shape,spin_symm,error):
        """
        Test that the Hubbard V term is probably built
        """
        lattice = get_lattice(params=case["lattice"])
        builder = HamiltonianBuilder(lattice=lattice)

        random_tmat = np.random.rand(*shape)
        if error is None:
            builder.custom_one_body(sps.csr_array(random_tmat),spin_symm=spin_symm)
            builder.finalize()
            hamiltonian = builder.hamiltonian
            actual_tij = sum(hamiltonian["tij"]).csr_array.toarray()
            # TODO: for collinear, and random tmat with shaoe (6,6), we need to split the matrix to compare
            nbasis = hamiltonian.nsites*hamiltonian.nbands
            if hamiltonian.spin_symm == SpinSymm.COLLINEAR and random_tmat.shape == (nbasis,nbasis):
                random_tmat = np.block(
                   [[random_tmat],
                    [random_tmat]]
                )
            assert np.allclose(actual_tij,random_tmat)
        else:
           with pytest.raises(error):
                builder.custom_one_body(random_tmat,spin_symm=spin_symm)
        
        print("\ntij matricies match!\n")

    #@pytest.mark.dev
    @pytest.mark.parametrize("params,expected",[
        ( None, False),
        ( [None, None], False),
        ( 0.0, False),
        ( [0.0, 0.0], False),
        ( [None, 0.0], False),
        ( 1.5, True),
        ( [1.5, None, 0.0], True),
        ( [1.0, 2.0], True),
        ( [[None, None], [None, None]], False),
        ( [[1.0, None], [0.0, 2.0]], True),
        ( [], False),
        ( [None], False),
        ( np.array([0.0, 0.0]), False),
        ( np.array([1.0, 2.0]), True),
    ])
    def test_skip_empty_params(self,params,expected):
        """
        Test that empty parameters are properly identified
        """
        builder = MockBuilder()
        initial_count = len(builder.called_with)
        builder.test_method(params)
        was_called = len(builder.called_with) > initial_count
        assert was_called == expected

