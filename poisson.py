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

fnSpace = dune.fem.space.lagrange( mesh, order=2)

uTrial = ufl.TrialFunction(fnSpace)
v = ufl.TestFunction(fnSpace)
x = ufl.SpatialCoordinate(fnSpace)

u_h = fnSpace.interpolate(0, name="u_h")
u_n = u_h.copy(name="u_h_n")

f = -6

u_D = 1 + x[0]**2 + 2*x[1]**2 

dbcBottom = dune.ufl.DirichletBC(fnSpace, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(fnSpace, u_D, x[1] > 1 - 1e-8 )
dbcLeft = dune.ufl.DirichletBC(fnSpace, u_D, x[0] < 1e-8)
dbcRight = dune.ufl.DirichletBC(fnSpace, u_D, x[0] > 1 - 1e-8)

a = ufl.inner( ufl.grad(uTrial), ufl.grad(v) )*ufl.dx
l = f*v*ufl.dx 

mat = dune.fem.assemble([a, dbcBottom, dbcTop, dbcLeft, dbcRight]) 
rhsVec = dune.fem.assemble([l, dbcBottom, dbcTop, dbcLeft, dbcRight])

A = mat.as_numpy
b = rhsVec.as_numpy 
y = u_h.as_numpy
y[:] = scipy.sparse.linalg.spsolve(A,b)

e_h = u_h - u_D
squaredErrors = dune.fem.integrate(e_h**2)
print("L^2 error:", squaredErrors)

e_h_proj = fnSpace.interpolate(e_h, name="error")


mesh.writeVTK("u_h", pointdata=[u_h])
mesh.writeVTK("err", pointdata=[e_h_proj])



