"""
Program to solve 2D heat transfer problems using finite element method.
Main programs:
    - mesh: mesh simple geometries like rectangular plate using quad mesh.
    - solver: using FEM to solve plane stress / 2D heat transfer problems.
    - post: visualize the result.
"""

import numpy as np

import mesh, solver, post

if __name__ == '__main__':
    v = [
        [0. , -10.],
        [30., -10.],
        [30.,  0.], 
        [0. ,  0.]
    ]
    m = mesh.MeshBox(verts=v)
    m.seed(ndx=3, ndy=2)
    m.mesh()
    m.showmesh()
    h = solver.PlaneStress(m, [[(0,0), np.array([0,3,6,9])],[(0,None),np.array([1,10,11])],[(0,-1e-5), np.array([2])], [(None,-1e-5), np.array([5])]],  
                [[(0,-1000), np.array([5])]], [], [3e7, 0.3], 1)
    h.steady_state_solver()
    