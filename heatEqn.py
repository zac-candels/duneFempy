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

alpha = 3
beta = 1.2
T = 1
numSteps = 1e2
dt = T/numSteps


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

t = dune.ufl.Constant(0.0)

f = beta - 2 - 2*alpha 
u_0 = 1 + x[0]**2 + alpha*x[1]**2

u_D = 1 + x[0]**2 + alpha*x[1]**2 + beta*t

u_nP1 = fnSpace.interpolate(u_0, name="u_h")
u_n = u_nP1.copy(name="u_n")

bilinForm = uTrial*v*ufl.dx\
    + dt*ufl.inner( ufl.grad(uTrial), ufl.grad(v) )*ufl.dx 

linForm = dt*f*v*ufl.dx + u_n*v*ufl.dx


dbcBottom = dune.ufl.DirichletBC(fnSpace, u_D, x[1] < 1e-8)
dbcTop = dune.ufl.DirichletBC(fnSpace, u_D, x[1] > 1 - 1e-8 )
dbcLeft = dune.ufl.DirichletBC(fnSpace, u_D, x[0] < 1e-8)
dbcRight = dune.ufl.DirichletBC(fnSpace, u_D, x[0] > 1 - 1e-8)

mat = dune.fem.assemble([bilinForm, dbcBottom, dbcTop,\
                         dbcLeft, dbcRight])
    
A = mat.as_numpy 
solver = scipy.sparse.linalg.factorized(A)
rhsVec = dune.fem.assemble([linForm, dbcBottom, dbcTop,\
                         dbcLeft, dbcRight])
    
vtk = mesh.sequencedVTK(
    "heat",
    pointdata=[u_n]
)

for n in range( int(numSteps)):
    t.value = t + dt
    
    rhsVec = dune.fem.assemble([linForm, dbcBottom, dbcTop,\
                             dbcLeft, dbcRight])
        
    b = rhsVec.as_numpy 
    
    u_nP1.as_numpy[:] = solver(b)

    u_n.assign(u_nP1)
    
    if n%10 == 0:
        vtk()
    

