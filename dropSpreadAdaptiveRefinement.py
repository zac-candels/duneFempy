import os
os.environ["DUNE_LOG_LEVEL"] = "warning"


from scipy.sparse import linalg
from math import sqrt
import matplotlib.pyplot as plt
import pygmsh
#from dune.alugrid import aluConformGrid as GridView
from dune.ufl import Constant, DirichletBC
import dune
import scipy
import numpy as np
import os
import time

WORKDIR = os.getcwd()
outDirName = os.path.join(WORKDIR, "drop-spread-periodic2")
os.makedirs(outDirName, exist_ok=True)

import ufl

R0 = 2
initDropDiam = 2*R0
A = 0.5
kappa = 0.02
interfaceThickness = np.sqrt(kappa/A)
M_tilde = 10
theta_deg = 30
theta = theta_deg * np.pi / 180

Q = 9
Tfinal = 1500
dt = 0.002
beta_mass_diff = 0.1*dt
numSteps = int(Tfinal/dt)
L_x = 16
L_y = 4
Nx = 40
Ny = 10


xc, yc = L_x/2, R0 - 0.6*R0

rho_init = 1.0
nu = 1/3
c_s = np.sqrt(1/3)
tau = 0.1

domain = dune.grid.cartesianDomain(
    [0.0, 0.0],
    [L_x, L_y],
    [Nx, Ny])

mesh = dune.fem.view.adaptiveLeafGridView(
    dune.alugrid.aluConformGrid(domain))

xi = [
        dune.ufl.Constant((0.0,  0.0)),
        dune.ufl.Constant((1.0,  0.0)),
        dune.ufl.Constant((0.0,  1.0)),
        dune.ufl.Constant((-1.0,  0.0)),
        dune.ufl.Constant((0.0, -1.0)),
        dune.ufl.Constant((1.0,  1.0)),
        dune.ufl.Constant((-1.0,  1.0)),
        dune.ufl.Constant((-1.0, -1.0)),
        dune.ufl.Constant((1.0, -1.0)),
    ]

# Corresponding weights
w = np.array([
    4/9,
    1/9, 1/9, 1/9, 1/9,
    1/36, 1/36, 1/36, 1/36
])

#%% Create function space and finite-element functions for the solution

V = dune.fem.space.lagrange(mesh, order=1)

V_vec3 = dune.fem.space.lagrange(mesh, order=1, dimRange=3)
vel3 = V_vec3.interpolate(dune.ufl.Constant((0, 0, 0)), name="vel")

v = ufl.TestFunction(V)

phi_trial = ufl.TrialFunction(V)
mu_trial = ufl.TrialFunction(V)
f_trial = ufl.TrialFunction(V)
f_n = []
f_star = []
f_nP1 = []

initFn = V.interpolate(0, "initFn")
for idx in range(Q):
    #f_n.append( V.interpolate(0, "f_n{}".format(idx) ) )
    f_star.append( initFn.copy(name=f"f_star{idx}") )
    f_nP1.append( initFn.copy(name=f"f_nP1{idx}") )
    
phi_nP1 = initFn.copy(name="phi_nP1")
mu_n = initFn.copy(name="mu_n")

x = ufl.SpatialCoordinate(V)

phiInitExpr = -ufl.tanh( (ufl.sqrt(pow(x[0]-xc,2) + pow(x[1]-yc,2)) - R0)\
                        / (ufl.sqrt(2)*interfaceThickness) )

phi_n = V.interpolate(phiInitExpr, name="phi_n")
mass_init = dune.fem.integrate((phi_n + 1) / 2)
mass_diff = dune.ufl.Constant(1e-9)

forceDensity = -phi_n * ufl.grad(mu_n)

t = dune.ufl.Constant(0.0)


#%% Define functions to compute density, velocity, body force
# and equilibrium distributions

def getDens(f_n):
    return sum(f_n)

def getVel(f_n, forceDensity):
    density = sum(f_n)
    momentum = f_n[0] * xi[0]
    for idx in range(1, Q):
        momentum += f_n[idx] * xi[idx]
    velTerm1 = momentum/density 
    velTerm2 = forceDensity * dt / (2 * density)
    return velTerm1 + velTerm2



def fEquilInit(vel_idx):
    
    forceDensity = dune.ufl.Constant((0.0, 0.0))
    
    vel_0 = -dune.ufl.Constant( (0.0, 0.0) )
    
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1
        + c_dot_u / c_s**2 
        + c_dot_u**2 / (2*c_s**4) 
        - ufl.inner(vel_0, vel_0) / (2*c_s**2)
        )




def f_equil(vel_idx, velocity, density, velSquared):

    # Compute ci . u for this direction
    c_dot_u = ufl.inner(velocity, xi[vel_idx])

    feq = w[vel_idx] * density * (1 + 3*c_dot_u + 4.5*c_dot_u**2 - 1.5*velSquared)

    return feq  



def body_Force(vel, vel_idx, Force_density):
    prefactor = w[vel_idx]
    inverse_cs2 = 1 / c_s**2
    inverse_cs4 = 1 / c_s**4

    xi_dot_prod_F = xi[vel_idx][0]*Force_density[0]\
        + xi[vel_idx][1]*Force_density[1]

    u_dot_prod_F = vel[0]*Force_density[0] + vel[1]*Force_density[1]

    xi_dot_u = xi[vel_idx][0]*vel[0] + xi[vel_idx][1]*vel[1]

    Force = prefactor*(inverse_cs2*(xi_dot_prod_F - u_dot_prod_F)
                       + inverse_cs4*xi_dot_u*xi_dot_prod_F)

    return Force


for idx in range(Q):
    print("idx = ", idx)
    f_n.append( V.interpolate(
        fEquilInit(idx),
        name=f"f_n{idx}") )
    
opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

    
def build_system(mesh, f_n, f_star, phi_n, mu_n):

    # =====================================================
    # Finite element spaces
    # =====================================================

    fnSpace = dune.fem.space.lagrange(
        mesh,
        order=1
    )

    V_vec3 = dune.fem.space.lagrange(
        mesh,
        order=1,
        dimRange=3
    )

    vel3 = V_vec3.interpolate(
        dune.ufl.Constant((0, 0, 0)),
        name="vel"
    )

    x = ufl.SpatialCoordinate(fnSpace)

    # =====================================================
    # Dirichlet boundary conditions
    #
    # These are retained at y=0 and y=L_y.
    # x is periodic.
    # =====================================================

    u_D = 1

    dbcBottom = dune.ufl.DirichletBC(
        fnSpace,
        u_D,
        x[1] < 1e-8
    )

    dbcTop = dune.ufl.DirichletBC(
        fnSpace,
        u_D,
        abs(L_y - x[1]) < 1e-8
    )

    bottomFn = fnSpace.interpolate(
        lambda x: 1.0 if abs(x[1]) < 1e-8 else 0.0,
        name="bottomFn"
    )

    topFn = fnSpace.interpolate(
        lambda x: 1.0 if abs(L_y - x[1]) < 1e-8 else 0.0,
        name="topFn"
    )

    bottomDoFs = np.where(
        bottomFn.as_numpy != 0
    )[0]

    topDoFs = np.where(
        topFn.as_numpy != 0
    )[0]

    # =====================================================
    # UFL objects
    # =====================================================

    trialFn = ufl.TrialFunction(fnSpace)
    v = ufl.TestFunction(fnSpace)

    # =====================================================
    # Physical quantities
    # =====================================================

    density = getDens(f_n)

    velocity_n = getVel(
        f_n,
        forceDensity
    )

    velStar_n = getVel(
        f_star,
        forceDensity
    )

    velSquared = ufl.inner(
        velocity_n,
        velocity_n
    )

    # =====================================================
    # Allen-Cahn equation
    # =====================================================

    lin_form_AC = (
        phi_n * v * ufl.dx
        - dt * v
        * ufl.dot(
            velocity_n,
            ufl.grad(phi_n)
        ) * ufl.dx
        - dt * M_tilde * v * mu_n * ufl.dx
        - (beta_mass_diff / dt)
        * mass_diff
        * ufl.sqrt(
            ufl.dot(
                ufl.grad(phi_n),
                ufl.grad(phi_n)
            )
        )
        * v * ufl.dx
        - 0.5 * dt**2
        * ufl.dot(
            velocity_n,
            ufl.grad(v)
        )
        * ufl.dot(
            velocity_n,
            ufl.grad(phi_n)
        )
        * ufl.dx
    )

    # =====================================================
    # Chemical potential
    # =====================================================

    lin_form_mu = (
        A * phi_n * (phi_n**2 - 1) * v * ufl.dx
        + kappa
        * ufl.dot(
            ufl.grad(phi_n),
            ufl.grad(v)
        ) * ufl.dx
        + kappa
        / (np.sqrt(2) * interfaceThickness)
        * np.cos(theta)
        * (phi_n**2 - 1)
        * v
        * ufl.ds(3)
    )

    # =====================================================
    # LBM forms
    # =====================================================

    bilinFormsStream = []
    bilinFormsColl = []

    linear_forms_stream = []
    linear_forms_collision = []

    for idx in range(Q):

        bilinFormsStream.append(
            trialFn * v * ufl.dx
        )

        bilinFormsColl.append(
            trialFn * v * ufl.dx
        )

        double_dot_product_term = (
            -0.5 * dt**2
            * ufl.inner(
                xi[idx],
                ufl.grad(f_star[idx])
            )
            * ufl.inner(
                xi[idx],
                ufl.grad(v)
            )
            * ufl.dx
        )

        dot_product_force_term = (
            0.5 * dt**2
            * ufl.inner(
                xi[idx],
                ufl.grad(v)
            )
            * body_Force(
                velStar_n,
                idx,
                forceDensity
            )
            * ufl.dx
        )

        lin_form_idx = (
            f_star[idx] * v * ufl.dx
            - dt
            * v
            * ufl.inner(
                xi[idx],
                ufl.grad(f_star[idx])
            )
            * ufl.dx
            + dt
            * v
            * body_Force(
                velStar_n,
                idx,
                forceDensity
            )
            * ufl.dx
            + double_dot_product_term
            + dot_product_force_term
        )

        f_eq_idx = f_equil(
            idx,
            velocity_n,
            density,
            velSquared
        )

        lin_form_coll = (
            f_n[idx]
            - dt / tau
            * (f_n[idx] - f_eq_idx)
        ) * v * ufl.dx

        linear_forms_stream.append(
            lin_form_idx
        )

        linear_forms_collision.append(
            lin_form_coll
        )

    # =====================================================
    # PERIODIC DOF MAP
    # =====================================================

    x_dof = fnSpace.interpolate(
        x[0],
        name="x_dof"
    ).as_numpy.copy()

    y_dof = fnSpace.interpolate(
        x[1],
        name="y_dof"
    ).as_numpy.copy()

    N = len(x_dof)

    # -----------------------------------------------------
    # x periodicity only
    # -----------------------------------------------------

    x_periodic = x_dof.copy()

    x_periodic[
        np.abs(x_periodic - L_x) < 1e-10
    ] = 0.0

    # -----------------------------------------------------
    # y is NOT periodic
    # -----------------------------------------------------

    keys = np.column_stack((
        np.round(x_periodic, 10),
        np.round(y_dof, 10)
    ))

    unique_keys, inverse = np.unique(
        keys,
        axis=0,
        return_inverse=True
    )

    nPeriodic = len(unique_keys)

    rows = np.arange(N)
    cols = inverse
    data = np.ones(N)

    P = scipy.sparse.coo_matrix(
        (data, (rows, cols)),
        shape=(N, nPeriodic)
    ).tocsr()

    # print(
    #     "Full DOFs:",
    #     N,
    #     "Periodic DOFs:",
    #     nPeriodic
    # )

    # =====================================================
    # Assemble matrices
    # =====================================================

    sysMatStream = []
    sysMatColl = []

    for idx in range(Q):

        # ---------------------------------------------
        # Streaming matrix
        # ---------------------------------------------

        if idx in (0, 1, 3):

            A_stream = dune.fem.assemble(
                bilinFormsStream[idx]
            )

        elif idx in (5, 2, 6):

            A_stream = dune.fem.assemble(
                [
                    bilinFormsStream[idx],
                    dbcBottom
                ]
            )

        elif idx in (4, 7, 8):

            A_stream = dune.fem.assemble(
                [
                    bilinFormsStream[idx],
                    dbcTop
                ]
            )

        # ---------------------------------------------
        # Collision matrix
        # ---------------------------------------------

        A_coll = dune.fem.assemble(
            bilinFormsColl[idx]
        )

        # ---------------------------------------------
        # PERIODIC REDUCTION
        # ---------------------------------------------

        A_stream_periodic = (
            P.T
            @ scipy.sparse.csr_matrix(
                A_stream.as_numpy
            )
            @ P
        ).tocsc()

        A_coll_periodic = (
            P.T
            @ scipy.sparse.csr_matrix(
                A_coll.as_numpy
            )
            @ P
        ).tocsc()

        sysMatStream.append(
            A_stream_periodic
        )

        sysMatColl.append(
            A_coll_periodic
        )

    # =====================================================
    # Factorizations
    # =====================================================

    # Collision matrix is identical for all directions,
    # so one factorization is sufficient.
    collSolver = scipy.sparse.linalg.factorized(
        sysMatColl[0]
    )

    streamSolvers = []

    for idx in range(Q):

        streamSolvers.append(
            scipy.sparse.linalg.factorized(
                sysMatStream[idx]
            )
        )

    # =====================================================
    # Operators
    # =====================================================

    collisionOps = []
    streamingOps = []

    for idx in range(Q):

        collForm = linear_forms_collision[idx]
        streamForm = linear_forms_stream[idx]

        collisionOps.append(
            dune.fem.operator.galerkin(
                collForm
                - trialFn * v * ufl.dx
            )
        )

        streamingOps.append(
            dune.fem.operator.galerkin(
                streamForm
                - trialFn * v * ufl.dx
            )
        )

    # =====================================================
    # AC / chemical potential operators
    # =====================================================

    phi_form_op = dune.fem.operator.galerkin(
        lin_form_AC
        - trialFn * v * ufl.dx
    )

    mu_form_op = dune.fem.operator.galerkin(
        lin_form_mu
        - trialFn * v * ufl.dx
    )

    # =====================================================
    # RHS storage
    # =====================================================

    rhsVecCollision = []

    rhsVecStreaming = []

    for idx in range(Q):

        rhsVecCollision.append(
            fnSpace.zero.copy()
        )

        rhsVecStreaming.append(
            fnSpace.zero.copy()
        )

    rhs_AC_fn = fnSpace.zero.copy()
    rhs_Mu_fn = fnSpace.zero.copy()

    # =====================================================
    # Arrays containing the full DUNE DOFs
    # =====================================================

    f_n_arrays = []
    f_star_arrays = []

    for idx in range(Q):

        f_n_arrays.append(
            f_n[idx].as_numpy
        )

        f_star_arrays.append(
            f_star[idx].as_numpy
        )

    # =====================================================
    # Return everything
    # =====================================================

    return (
        fnSpace,
        vel3,
        bottomDoFs,
        topDoFs,
        P,
        phi_form_op,
        mu_form_op,
        rhs_AC_fn,
        rhs_Mu_fn,
        collisionOps,
        streamingOps,
        collSolver,
        streamSolvers,
        rhsVecCollision,
        rhsVecStreaming,
        f_n_arrays,
        f_star_arrays
    )
    
    
    
(V, vel3, bottomDoFs, topDoFs,
 P,
 phi_form_op, mu_form_op,
 rhs_AC_fn, rhs_Mu_fn,
 collisionOps, streamingOps,
 collSolver, streamSolvers,
 rhsVecCollision, rhsVecStreaming,
 f_n_arrays, f_star_arrays) = build_system(
     mesh,
     f_n,
     f_star,
     phi_n,
     mu_n
 )
    
    
    
    
    
#%% Time loop
#rint("about to start time-stepping\n\n")
# ============================================================
# Time loop
# ============================================================

for n in range(numSteps):

    # ========================================================
    # Assemble Allen-Cahn and chemical-potential RHS
    # ========================================================

    phi_form_op(
        V.zero,
        rhs_AC_fn
    )

    mu_form_op(
        V.zero,
        rhs_Mu_fn
    )

    # ========================================================
    # COLLISION
    # ========================================================

    for idx in range(Q):

        collisionOps[idx](
            V.zero,
            rhsVecCollision[idx]
        )

        # ----------------------------------------------
        # Full RHS
        # ----------------------------------------------

        b_full = rhsVecCollision[idx].as_numpy

        # ----------------------------------------------
        # Periodic RHS
        #
        # b_p = P.T b
        # ----------------------------------------------

        b_periodic = P.T @ b_full

        # ----------------------------------------------
        # Solve reduced system
        # ----------------------------------------------

        f_star_periodic = collSolver(
            b_periodic
        )

        # ----------------------------------------------
        # Expand back to full DOFs
        #
        # f_full = P f_periodic
        # ----------------------------------------------

        f_star_arrays[idx][:] = (
            P @ f_star_periodic
        )

    # ========================================================
    # STREAMING
    # ========================================================

    for idx in range(Q):

        streamingOps[idx](
            V.zero,
            rhsVecStreaming[idx]
        )

        # ----------------------------------------------
        # Full RHS
        # ----------------------------------------------

        b_full = rhsVecStreaming[idx].as_numpy

        # ----------------------------------------------
        # Boundary condition at bottom
        # ----------------------------------------------

        if idx in (2, 5, 6):

            b_full[bottomDoFs] = (
                f_star[opp_idx[idx]]
                .as_numpy[bottomDoFs]
            )

        # ----------------------------------------------
        # Boundary condition at top
        # ----------------------------------------------

        elif idx in (4, 7, 8):

            b_full[topDoFs] = (
                f_star[opp_idx[idx]]
                .as_numpy[topDoFs]
            )

        # ----------------------------------------------
        # Periodic reduction
        # ----------------------------------------------

        b_periodic = P.T @ b_full

        # ----------------------------------------------
        # Solve
        # ----------------------------------------------

        f_n_periodic = streamSolvers[idx](
            b_periodic
        )

        # ----------------------------------------------
        # Expand solution
        # ----------------------------------------------

        f_n_arrays[idx][:] = (
            P @ f_n_periodic
        )

    # ========================================================
    # Solve Allen-Cahn equation
    # ========================================================

    b_phi_full = rhs_AC_fn.as_numpy

    b_phi_periodic = P.T @ b_phi_full

    phi_periodic = collSolver(
        b_phi_periodic
    )

    phi_nP1.as_numpy[:] = (
        P @ phi_periodic
    )

    # ========================================================
    # Solve chemical potential
    # ========================================================

    b_mu_full = rhs_Mu_fn.as_numpy

    b_mu_periodic = P.T @ b_mu_full

    mu_periodic = collSolver(
        b_mu_periodic
    )

    mu_n.as_numpy[:] = (
        P @ mu_periodic
    )

    # ========================================================
    # Update phi
    # ========================================================

    phi_n.assign(
        phi_nP1
    )

    # ========================================================
    # Mass correction
    # ========================================================

    mass_n = dune.fem.integrate(
        (phi_n + 1) / 2
    )

    mass_diff.assign(
        mass_n - mass_init
    )

    # ========================================================
    # Mesh adaptation
    # ========================================================

    if n % 20 == 0:

        marker = dune.fem.GridMarker(
            1 - phi_n**2,
            refineTolerance=0.8,
            coarsenTolerance=0.2,
            minLevel=0,
            maxLevel=3
        )

        dune.fem.gridAdapt(
            marker,
            f_n
            + f_star
            + [phi_n, mu_n]
        )

        # IMPORTANT:
        # build_system() reconstructs P and all
        # periodic matrices after mesh adaptation.

        (
            V,
            vel3,
            bottomDoFs,
            topDoFs,
            P,
            phi_form_op,
            mu_form_op,
            rhs_AC_fn,
            rhs_Mu_fn,
            collisionOps,
            streamingOps,
            collSolver,
            streamSolvers,
            rhsVecCollision,
            rhsVecStreaming,
            f_n_arrays,
            f_star_arrays
        ) = build_system(
            mesh,
            f_n,
            f_star,
            phi_n,
            mu_n
        )

    # ========================================================
    # Output
    # ========================================================

    if n % 100 == 0:

        print(
            "n = ",
            n,
            "writing to file\n"
        )

        vel_expr = getVel(
            f_n,
            forceDensity
        )

        vel3.interpolate(
            ufl.as_vector([
                vel_expr[0],
                vel_expr[1],
                0
            ])
        )

        mesh.writeVTK(
            os.path.join(
                outDirName,
                f"vel_{n:06d}"
            ),
            pointdata=[vel3]
        )

        mesh.writeVTK(
            os.path.join(
                outDirName,
                f"phi_{n:06d}"
            ),
            pointdata=[phi_n]
        )