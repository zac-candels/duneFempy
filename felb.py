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

Q = 2
Tfinal = 1500
dt = 0.01
numSteps = int(Tfinal/dt)
L_x = 32
L_y = 32
nx = 3
ny = 2
h = min(L_x/nx, L_y/ny)

forceDensity = np.array([2.6041666e-5, 0.0])
rho_init = dune.ufl.Constant(1.0)
nu = 1/3
c_s = np.sqrt(1/3)
tau = 1

u_max = forceDensity[0]*L_y**2/(8*rho_init*nu)



mesh = dune.grid.structuredGrid([0, 0], [L_x, L_y], [nx, ny],
                                periodic=[True, False])

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

v = ufl.TestFunction(V)

f_trial = ufl.TrialFunction(V)
f_n = []
f_star = []
f_nP1 = []
for idx in range(Q):
    f_n.append( V.interpolate(0, "f{}".format(idx) ) )
 #   f_star.append( V.interpolate(0, "f{}".format(idx) ) )
 #   f_nP1.append( V.interpolate(0, "f_star{}".format(idx)) )

x = ufl.SpatialCoordinate(V)

t = dune.ufl.Constant(0.0)


#%% Define functions to compute density, velocity, body force
# and equilibrium distributions

def getDens(f_n):
    return sum(f_n)

def getVel(f_n):
    density = sum(f_n)
    
    momentum = None
    for idx in range(Q):
        momentum += f_n[idx] * xi[idx]
        
    velTerm1 = momentum/density 
    
    F = dune.ufl.Constant((forceDensity[0], forceDensity[1]))
    velTerm2 = F * dt / (2 * density)
        
    return velTerm1 + velTerm2 



def fEquilInit(vel_idx, forceDensity):
    
    vel_0 = -dune.ufl.Constant( (forceDensity[0]*dt/(2*rho_init),
                               forceDensity[1]*dt/2*rho_init) )
    
    c = xi[vel_idx]
    c_dot_u = ufl.inner(c, vel_0)
    return w[vel_idx] * rho_init * (
        1
        + c_dot_u / c_s**2 
        + c_dot_u**2 / (2*c_s**4) 
        - ufl.inner(vel_0, vel_0) / (2*c_s**2)
        )



def f_equil(f_list, vel_idx):
    """
    Compute equilibrium distribution for direction idx
    Returns a NumPy array (values at all DoFs).
    """
    density = sum(f_list)
    
    # Compute velocity at each DoF
    momentum = None
    for idx in range(Q):
        momentum += f_n[idx] * xi[idx]
        
    velocity = momentum/density 

    velSquared = velocity**2

    # Compute ci . u for this direction
    c_dot_u = velocity * xi[vel_idx]

    feq = w[idx] * density * (1 + 3*c_dot_u + 4.5*c_dot_u**2 - 1.5*velSquared)

    return feq  # NumPy array



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
    f_n[idx] = V.interpolate(
        fEquilInit(idx, forceDensity),
        name=f"f{idx}"
    )
    

#%% Define linear and bilinear forms

bilinFormsStream = []
linear_forms_stream = []

bilinFormsColl = []
linear_forms_collision = []

opp_idx = {0: 0, 1: 3, 2: 4, 3: 1, 4: 2, 5: 7, 6: 8, 7: 5, 8: 6}

for idx in range(Q):

    bilinFormsStream.append(f_trial * v * ufl.dx)
    bilinFormsColl.append(f_trial*v*ufl.dx)

    double_dot_product_term = -0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(f_star[idx]))\
        * ufl.inner(xi[idx], ufl.grad(v)) * ufl.dx

    dot_product_force_term = 0.5*dt**2 * ufl.inner(xi[idx], ufl.grad(v))\
        * body_Force(getVel(f_star), idx, forceDensity) * ufl.dx


    lin_form_idx = f_star[idx]*v*ufl.dx\
        - dt*v*ufl.inner(xi[idx], ufl.inner(f_star[idx]))*ufl.dx\
        + dt*v*body_Force(getVel(f_star), idx, forceDensity)*ufl.dx\
        + double_dot_product_term\
        + dot_product_force_term
        
    lin_form_coll = (f_n[idx] - dt/(tau ) * (f_n[idx] - f_equil(f_n, idx, forceDensity)) )*v*ufl.dx

    linear_forms_stream.append(lin_form_idx)
    linear_forms_collision.append(lin_form_coll)

# Assemble matrices for first step
sysMatStream = []
sysMatColl = []
rhs_vec_streaming = [0]*Q
rhs_vec_collision = [0]*Q
for idx in range(Q):
    sysMatStream.append(dune.fem.assemble(bilinFormsStream[idx]))
    sysMatColl.append(dune.fem.assemble(bilinFormsColl[idx]))
 
rhsVecStreaming = [0]*Q
rhsVecCollision = [0]*Q
    
#%% Start time-stepping

t = 0.0
for n in range(numSteps):
    t += dt

    for idx in range(Q):
        rhsVecCollision[idx] = dune.ufl.assemble(linear_forms_collision[idx])
        



    # Assemble RHS vectors
    for idx in range(Q):
        rhsVecStreaming[idx] = (dune.fem.assemble(linear_forms_stream[idx]))


    # Update previous solutions

    for idx in range(Q):
        f_n[idx].assign(f_nP1[idx])
    
    
    
