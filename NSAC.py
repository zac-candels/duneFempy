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


Tfinal = 300
R0 = 2
initDropDiam = 2*R0
L_x = 8*R0
L_y = 4*R0
nx = 80
ny = 40
h = min(L_x/nx, L_y/ny)


mesh = dune.grid.structuredGrid([0, 0], [L_x, L_y], [nx, ny],
                                periodic=[True, False])

# Need periodic boundary conditions

Pe = 0.1275 
We = 0.01
Cn_param=  0.05
Re = 1
theta_deg = 30
dt = Cn_param*Pe*h**2
beta_mass_diff = 0.00001
num_steps = int(np.ceil(Tfinal/dt))

Cn = initDropDiam * Cn_param
xc, yc = L_x/2, R0 - 0.6*R0

theta = theta_deg * np.pi / 180

velSpace = dune.fem.space.lagrange(mesh, order=2, dimRange=2)
presSpace = dune.fem.space.lagrange(mesh, order=1, dimRange=1)
acSpace = dune.fem.space.lagrange(mesh, order=1, dimRange=1)

velTrial = ufl.TrialFunction(velSpace)
velTest = ufl.TestFunction(velSpace)

presTrial = ufl.TrialFunction(presSpace)
presTest = ufl.TestFunction(presSpace)

acTrial = ufl.TrialFunction(acSpace)
acTest = ufl.TestFunction(acSpace)

xVel = ufl.SpatialCoordinate(velSpace)


vel_nP1 = velSpace.interpolate([0,0], name="vel_nP1")
vel_n = vel_nP1.copy(name="vel_n")

vel_star = velSpace.interpolate([0,0], name="vel_star")

phi_init_expr = lambda x: -ufl.tanh( (ufl.sqrt(pow(x[0]-xc,2)
                                               + pow(x[1]-yc,2)) - R0)
                                    / (ufl.sqrt(2)*Cn) )

phi_nP1 = acSpace.interpolate(phi_init_expr, name="phi_nP1")
phi_n = phi_nP1.copy(name="phi_n")
phi_0 = acSpace.interpolate(phi_init_expr, name="phi_0")
mass_diff = dune.ufl.Constant(0.0)











