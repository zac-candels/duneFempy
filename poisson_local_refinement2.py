from scipy.sparse import linalg
import matplotlib.pyplot as plt
import pygmsh
import dune
import dune.fem as fem
from dune.fem.view import adaptiveLeafGridView as adaptiveGridView
from dune.ufl import Constant, DirichletBC
import scipy
import numpy as np
import ufl

# --- geometry: [-1,1] x [-1,1], centered at origin ---
with pygmsh.occ.Geometry() as geom:
    rectangle = geom.add_rectangle([-1.0, -1.0, 0.0], 2, 2)
    geom.characteristic_length_max = 0.2
    geom.characteristic_length_min = 0.1
    mesh = geom.generate_mesh()

points = mesh.points
cells = mesh.cells_dict
domain = {
    "vertices": points[:, :2].astype(float),
    "simplices": cells["triangle"].astype(int),
}

# wrap in adaptiveLeafGridView so gridAdapt/GridMarker work
mesh = adaptiveGridView(dune.alugrid.aluConformGrid(domain))
print("initial grid size:", mesh.size(0), flush=True)

fnSpace = dune.fem.space.lagrange(mesh, order=2, storage='numpy')

uTrial = ufl.TrialFunction(fnSpace)
v = ufl.TestFunction(fnSpace)
x = ufl.SpatialCoordinate(fnSpace)

u_h = fnSpace.interpolate(0, name="u_h")

# forcing: f = 3 inside disk radius 0.25, else 0
r2 = x[0]**2 + x[1]**2
f = ufl.conditional(r2 < 0.5**2, 3.0, 0.0)

a = ufl.inner(ufl.grad(uTrial), ufl.grad(v)) * ufl.dx
l = f * v * ufl.dx

# --- local refinement around E = B_0.05(0,0) ---
dist = abs(ufl.sqrt(r2) - 0.5)
indicator = ufl.conditional(dist < 0.1, 1.0, 0.0)

indicatorSpace = dune.fem.space.lagrange(mesh, order=1, storage='numpy')  # or finite volume space
indicatorFn = indicatorSpace.interpolate(indicator, name="indicator")

maxLevel = 4   # base mesh is already ~0.01-0.02; a few extra levels is plenty
marker = dune.fem.GridMarker(indicatorFn,
                    refineTolerance=0.5,
                    coarsenTolerance=0.2,
                    minLevel=0, maxLevel=maxLevel)

for i in range(maxLevel):
    # Re-interpolate the indicator on the current leaf grid at each step
    indicatorFn.interpolate(indicator)
    
    dune.fem.gridAdapt(marker, u_h)   # prolongs/restricts u_h and adapts mesh
    print(mesh.size(0), end=" ")
print()

# --- homogeneous Dirichlet BC, u = 0 on all four sides ---
u_D = 0
dbcBottom = dune.ufl.DirichletBC(fnSpace, u_D, x[1] < -1 + 1e-8)
dbcTop    = dune.ufl.DirichletBC(fnSpace, u_D, x[1] >  1 - 1e-8)
dbcLeft   = dune.ufl.DirichletBC(fnSpace, u_D, x[0] < -1 + 1e-8)
dbcRight  = dune.ufl.DirichletBC(fnSpace, u_D, x[0] >  1 - 1e-8)

# these must be built AFTER adaptation, against the final space
u_D_fn = fnSpace.interpolate(u_D, name="u_D_fn")

bottomFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(x[1] + 1) < 1e-8 else 0.0, name="bottomFn")
topFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(1 - x[1]) < 1e-8 else 0.0, name="topFn")
leftFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(x[0] + 1) < 1e-8 else 0.0, name="leftFn")
rightFn = fnSpace.interpolate(
    lambda x: 1.0 if abs(1 - x[0]) < 1e-8 else 0.0, name="rightFn")

bottomDoFs = np.where(bottomFn.as_numpy != 0)[0]
topDoFs    = np.where(topFn.as_numpy    != 0)[0]
leftDoFs   = np.where(leftFn.as_numpy   != 0)[0]
rightDoFs  = np.where(rightFn.as_numpy  != 0)[0]

mat = dune.fem.assemble([a, dbcBottom, dbcTop, dbcLeft, dbcRight])
rhsVec = dune.fem.assemble(l)

u_D_vals = u_D_fn.as_numpy
A = mat.as_numpy
b = rhsVec.as_numpy
b[bottomDoFs] = u_D_vals[bottomDoFs]
b[topDoFs]    = u_D_vals[topDoFs]
b[leftDoFs]   = u_D_vals[leftDoFs]
b[rightDoFs]  = u_D_vals[rightDoFs]

y = u_h.as_numpy
y[:] = scipy.sparse.linalg.spsolve(A, b)

mesh.writeVTK("u_h_circle", pointdata=[u_h])