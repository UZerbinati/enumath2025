from ngsolve import *
from ngstrefftz import *
from netgen.occ import *
import numpy as np
from mpi4py import MPI

order = 4
shape = WorkPlane().Rectangle(np.pi, np.pi).Face()
geo = OCCGeometry(shape, dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=.2, comm=MPI.COMM_WORLD))

def LaplaceDG(fes):
    alpha = 4
    n = specialcf.normal(mesh.dim)
    h = specialcf.mesh_size
    u = fes.TrialFunction()
    v = fes.TestFunction()

    jump = lambda u: u-u.Other()
    mean_dn = lambda u: 0.5*n * (grad(u)+grad(u).Other())

    a = BilinearForm(fes,symmetric=True)
    a += grad(u)*grad(v) * dx \
        +alpha*order**2/h*jump(u)*jump(v) * dx(skeleton=True) \
        +(-mean_dn(u)*jump(v)-mean_dn(v)*jump(u)) * dx(skeleton=True) \
        +alpha*order**2/h*u*v * ds(skeleton=True) \
        +(-n*grad(u)*v-n*grad(v)*u)* ds(skeleton=True)
    a.Assemble()
    
    m = BilinearForm(fes, symmetric=True)
    m += -1*u*v*dx
    return a,m

fes = L2(mesh, order=order,  dgjumps=True)
eps=1e-8
Lap = lambda u : sum(Trace(u.Operator('hesse')))
u = fes.TrialFunction()
trefftz_test_space = L2(mesh, order=order-2,  dgjumps=True)
v = trefftz_test_space.TestFunction()
op = Lap(u)*v*dx
with TaskManager():
    emb = TrefftzEmbedding(op)
    etfes = EmbeddedTrefftzFES(emb)

a,m = LaplaceDG(etfes)
a.Assemble()
m.Assemble()
from ngsPETSc import EigenSolver
solver = EigenSolver((m, a), etfes, 10, solverParameters={"eps_type":"lobpcg", "st_type":"precond"})
solver.solve()
for i in range(10):
   print("Normalised (by pi^2) Eigenvalue ", i, " = ", solver.eigenValue(i))
