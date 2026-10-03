import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

import dune
import dune.fem
import ufl


T = 10.0
dt = 0.0005
numSteps = int(T / dt)

Lx = 2.0
Ly = 2.0

Nx = 15
Ny = 15

order = 2

# Adapt the mesh every N time steps
adaptEvery = 10

# Maximum refinement level
maxLevel = 4

# Tolerance used when identifying periodic DOFs
tol = 1e-10


# parameters
t = dune.ufl.Constant(0.0)

c = dune.ufl.Constant((1.0, 0.0))


# Create mesh
domain = dune.grid.cartesianDomain(
    [0.0, 0.0],
    [Lx, Ly],
    [Nx, Ny]
)

mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain)
)

print("Initial cells:", mesh.size(0))


# Initial condition
def initial_condition(x):

    return 1.0 - ufl.exp(-1.0 / (4.0 * ( (x[0] - 1.0)**2 + (x[1] - 1.0)**2) ) )


# Fn. to build every object that depends on the mesh - ie function space,
# finite-element functions, (bi)linear forms, matrix, rhs vector
def build_system(mesh, u_n):

    # Finite element space
    fnSpace = dune.fem.space.lagrange(mesh, order=order)


    # UFL objects
    uTrial = ufl.TrialFunction(fnSpace)
    v = ufl.TestFunction(fnSpace)

    x = ufl.SpatialCoordinate(fnSpace)


    # Bilinear form
    bilinForm = (uTrial * v * ufl.dx)

    # Linear form
    linForm = u_n * v * ufl.dx - dt * ufl.dot(c, ufl.grad(u_n))* v* ufl.dx\
        - 0.5* dt**2* ufl.dot(c, ufl.grad(u_n))* ufl.dot(c, ufl.grad(v))* ufl.dx
    
    rhsOp = dune.fem.operator.galerkin(linForm - uTrial*v*ufl.dx)
    rhsFn = fnSpace.zero.copy()
    # Assemble mass matrix
    mat = dune.fem.assemble([bilinForm])

    A = scipy.sparse.csr_matrix(mat.as_numpy.copy())

    # --------------------------------------------------------
    # Get coordinates of the current DOFs

    x_dof = fnSpace.interpolate(x[0],name="x_dof").as_numpy.copy()

    y_dof = fnSpace.interpolate(x[1],name="y_dof").as_numpy.copy()

    N = len(x_dof)

    # --------------------------------------------------------
    # Apply periodic coordinate mapping
    #
    # x = Lx becomes x = 0
    # y = Ly becomes y = 0
    # --------------------------------------------------------

    x_periodic = x_dof.copy()
    y_periodic = y_dof.copy()

    x_periodic[np.abs(x_periodic - Lx) < tol] = 0.0

    y_periodic[np.abs(y_periodic - Ly) < tol] = 0.0

    # --------------------------------------------------------
    # Create a key for each periodic equivalence class
    # --------------------------------------------------------

    keys = np.column_stack((np.round(x_periodic, 10),np.round(y_periodic, 10)))

    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)

    nPeriodic = len(unique_keys)

    #print("DOFs:",N,"   periodic DOFs:",nPeriodic)

    # --------------------------------------------------------
    # Construct prolongation matrix
    #
    # u_full = P @ u_periodic
    # --------------------------------------------------------

    rows = np.arange(N)
    cols = inverse
    data = np.ones(N)

    P = scipy.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(N, nPeriodic)
    ).tocsr()

    # --------------------------------------------------------
    # Reduce matrix using periodicity
    #
    # A_periodic = P^T A P
    # --------------------------------------------------------

    A_periodic = (P.T @ A @ P).tocsc()

    # --------------------------------------------------------
    # Factorize matrix
    #
    # This factorization is valid until the mesh changes.
    # --------------------------------------------------------

    solver = scipy.sparse.linalg.factorized(A_periodic)

    return (fnSpace, uTrial, v, x, rhsOp, rhsFn, P, solver)


# ============================================================
# Initial finite element space
# ============================================================

fnSpace = dune.fem.space.lagrange(
    mesh,
    order=order
)

x = ufl.SpatialCoordinate(fnSpace)


# ============================================================
# Initial solution
# ============================================================

u_n = fnSpace.interpolate(initial_condition(x), name="u_n")


# ============================================================
# Build initial system
# ============================================================

fnSpace, uTrial, v, x, rhsOp, rhsFn, P, solver = build_system(mesh, u_n)


print("Initial DOFs:", fnSpace.size)

vtk = mesh.sequencedVTK(
    "advection",
    pointdata=[u_n]
)


for n in range(numSteps):

    # print("--------------------------------------------")
    # print("Step:", n, "/", numSteps)
    # print("Cells:", mesh.size(0))
    # print("DOFs:", fnSpace.size)


    t.value += dt

    # Assemble RHS

    rhsOp(fnSpace.zero, rhsFn)

    # --------------------------------------------------------
    # Apply periodic reduction
    #
    # b_periodic = P^T b
    # --------------------------------------------------------

    b_periodic = P.T @ rhsFn.as_numpy

    # --------------------------------------------------------
    # Solve periodic system
    #
    # A_periodic u_periodic = b_periodic
    # --------------------------------------------------------

    u_periodic = solver(
        b_periodic
    )

    # --------------------------------------------------------
    # Expand solution back to full DOF vector
    #
    # u_full = P u_periodic
    # --------------------------------------------------------

    u_full = P @ u_periodic

    # --------------------------------------------------------
    # Put solution back into DUNE function
    # --------------------------------------------------------

    u_n.as_numpy[:] = u_full

    # --------------------------------------------------------
    # Adapt mesh
    # --------------------------------------------------------

    if n % adaptEvery == 0:

        # print()
        # print("ADAPTING MESH")

        # ----------------------------------------------------
        # Create refinement indicator
        #
        # For now this uses u itself as the indicator.
        # ----------------------------------------------------

        marker = dune.fem.GridMarker(
            u_n,
            refineTolerance=0.9,
            coarsenTolerance=0.4,
            minLevel=0,
            maxLevel=maxLevel
        )

        # ----------------------------------------------------
        # Adapt the mesh.
        #
        # Passing u_n here makes DUNE transfer the solution
        # to the new mesh.
        # ----------------------------------------------------

        dune.fem.gridAdapt(
            marker,
            [u_n]
        )

        # print("New cells:", mesh.size(0))

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The mesh has changed, so all DOF-dependent objects
        # must be rebuilt.
        # ----------------------------------------------------

        (fnSpace, uTrial, v, x, rhsOp, rhsFn, P, solver) = build_system(mesh, u_n)

        # print("New DOFs:", fnSpace.size)

        # print("Finished adaptation")
        # print()

    # --------------------------------------------------------
    # VTK output
    # --------------------------------------------------------

    if n % 10 == 0:
        vtk()


print()
print("Simulation finished.")