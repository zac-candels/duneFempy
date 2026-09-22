Because of the presence of UFL, the creation of (bi)linear forms is 
exactly the same as in FEniCSx. There are, however, some important differences 
in terms of how linear forms are assembled that bear mentioning. 

Suppose we have already defined a mesh and a function space and suppose 

A = V.interpolate(x[0]**2) # for instance

and 

Form = A*v*ufl.dx. 

Then, to compute \int A v dx, one must do so in a rather round-about way by defining 
Galerkin forms. 

Suppose we have a trial function u and a test function v defined on a function space V,
ie 

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

Then, we first define an operator G(u,v):= \int_{\Omega} Av - uv dx by 

op = dune.fem.operator.galerkin(Form - u*v*ufl.dx).

Now, if we want \int_{\Omega} A v dx, it should be clear that it is sufficient
to compute 

G(0, v) = \int_{\Omega} Av dx. 

To do this, define 

zeroFn = V.zero 

and 

assembledForm = V.zero.copy().

Then,


op(zeroFn, assembledForm) 

stores the result of G(0, v) in the variable assembledForm.
