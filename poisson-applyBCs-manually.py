from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
import pygmsh
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import scipy
import numpy as np

import ufl


with pygmsh.occ.Geometry() as geom:
    rectangle = geom.add_rectangle(
        [0.0, 0.0, 0.0],
        1,   # width
        1   # height
    )

    # Set a uniform mesh size
    geom.characteristic_length_max = 0.02
    geom.characteristic_length_min = 0.01

    mesh = geom.generate_mesh()

points = mesh.points
cells = mesh.cells_dict

domain = {
    "vertices": points[:, :2].astype(float),
    "simplices": cells["triangle"].astype(int),
}
    
mesh = dune.alugrid.aluConformGrid(domain)
print("grid size:", mesh.size(0),flush=True)

fnSpace = dune.fem.space.lagrange( mesh, order=2, storage='petsc')

uTrial = ufl.TrialFunction(fnSpace)
v = ufl.TestFunction(fnSpace)
x = ufl.SpatialCoordinate(fnSpace)

u_h = fnSpace.interpolate(0, name="u_h")
u_n = u_h.copy(name="u_h_n")

f = -6

u_D = 1 + x[0]**2 + 2*x[1]**2 
u_D_fn = fnSpace.interpolate(u_D, name="u_D_fn")
test_fn = ufl.sin(x[0])

dbcBottom = dune.ufl.DirichletBC(fnSpace, test_fn, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(fnSpace, test_fn, x[1] > 1 - 1e-8 )
dbcLeft = dune.ufl.DirichletBC(fnSpace, test_fn, x[0] < 1e-8)
dbcRight = dune.ufl.DirichletBC(fnSpace, test_fn, x[0] > 1 - 1e-8)

bottomFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(x[1]) < 1e-8 else 0.0,
    name="bottomFn"
)

topFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(1 - x[1]) < 1e-8 else 0.0,
    name="topFn"
)

leftFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(x[0]) < 1e-8 else 0.0,
    name="leftFn"
)

rightFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(1 - x[0]) < 1e-8 else 0.0,
    name="rightFn"
)

bottomDoFs = np.where(bottomFn.as_numpy != 0)[0]
topDoFs = np.where(topFn.as_numpy != 0)[0]
leftDoFs = np.where(leftFn.as_numpy != 0)[0]
rightDoFs = np.where(rightFn.as_numpy != 0)[0]

a = ufl.inner( ufl.grad(uTrial), ufl.grad(v) )*ufl.dx
l = f*v*ufl.dx 

mat = dune.fem.assemble([a, dbcBottom, dbcTop, dbcLeft, dbcRight]) 
rhsVec = dune.fem.assemble(l)

u_D_vals = u_D_fn.as_numpy
A = mat.as_numpy
b = rhsVec.as_numpy 
b[bottomDoFs] = u_D_vals[bottomDoFs]
b[topDoFs] = u_D_vals[topDoFs]
b[leftDoFs] = u_D_vals[leftDoFs]
b[rightDoFs] = u_D_vals[rightDoFs]
y = u_h.as_numpy
y[:] = scipy.sparse.linalg.spsolve(A,b)

e_h = u_h - u_D
squaredErrors = dune.fem.integrate(e_h**2)
print("L^2 error:", squaredErrors)

e_h_proj = fnSpace.interpolate(e_h, name="error")


mesh.writeVTK("u_h", pointdata=[u_h])
mesh.writeVTK("err", pointdata=[e_h_proj])



