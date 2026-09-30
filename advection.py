from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import scipy
import numpy as np

import ufl

T = 10.0
dt = 0.0005
numSteps = int(T / dt)

Lx = 2.0
Ly = 2.0

Nx = 20
Ny = 20


# Create mesh
domain = dune.grid.cartesianDomain([0.0, 0.0], [Lx, Ly], [Nx, Ny])

mesh = dune.fem.view.adaptiveLeafGridView(dune.alugrid.aluConformGrid(domain))

print("grid size:", mesh.size(0),flush=True)

fnSpace = dune.fem.space.lagrange( mesh, order=2)

uTrial = ufl.TrialFunction(fnSpace)
v = ufl.TestFunction(fnSpace)
x = ufl.SpatialCoordinate(fnSpace)

t = dune.ufl.Constant(0.0)

c = dune.ufl.Constant((1.0, 0.0))
# Initial condition
def initial_condition(x):
    
    return ufl.exp( -( (x[0]-1)**2 + (x[1]-1)**2 )/ 0.125 )

    #return 1.0 - ufl.exp(-1.0 / (8.0 * ( (x[0] - 1.0)**2 + (x[1] - 1.0)**2) ) )

u_nP1 = fnSpace.interpolate(initial_condition(x), name="u_h")
u_n = u_nP1.copy(name="u_n")

bilinForm = uTrial*v*ufl.dx\

linForm = u_n*v*ufl.dx - dt*ufl.dot(c, ufl.grad(u_n))*v*ufl.dx\
    - 0.5 * dt**2 * ufl.dot(c, ufl.grad(u_n))*ufl.dot(c, ufl.grad(v))*ufl.dx 



mat = dune.fem.assemble([bilinForm])
    
A = mat.as_numpy 
solver = scipy.sparse.linalg.factorized(A)
rhsVec = dune.fem.assemble([linForm])
    
vtk = mesh.sequencedVTK("heat", pointdata=[u_n])

for n in range( int(numSteps)):
    t.value = t + dt
    
    rhsVec = dune.fem.assemble([linForm])
        
    b = rhsVec.as_numpy 
    
    u_nP1.as_numpy[:] = solver(b)

    u_n.assign(u_nP1)
    
    if n%10 == 0:
        vtk()
    

