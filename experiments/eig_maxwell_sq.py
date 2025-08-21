from ngsolve import *
from ngstrefftz import *
from netgen.occ import *
import numpy as np
from mpi4py import MPI

def rot(u):
    #J = u.Operator('Grad') 
    #return J[1,0]-J[0,1]
    return grad(u)[1,0]-grad(u)[0,1]
def MaxwellDG(fes):
    k = fes.components[0].globalorder
    u, v = fes.TrialFunction()[0], fes.TestFunction()[0]
    p, q = fes.TrialFunction()[1], fes.TestFunction()[1]

    alpha = 1e3  # interior penalty param
    n = specialcf.normal(mesh.dim)
    t = specialcf.tangential(mesh.dim)
    h = specialcf.mesh_size

    jump_u = u - u.Other()
    jump_v = v - v.Other()
    jump_p= p - p.Other()
    jump_q= q - q.Other()
    mean_rotu = 0.5*(rot(u)+rot(u.Other()))
    mean_rotv = 0.5*(rot(v)+rot(u.Other()))
    jump_ut = (u - u.Other())*t
    jump_vt = (v - v.Other())*t
    mean_dudn = 0.5 * (grad(u) + grad(u.Other())) * n
    mean_dvdn = 0.5 * (grad(v) + grad(v.Other())) * n
    mean_q = 0.5 * n * (q + q.Other())
    mean_p = 0.5 * n * (p + p.Other())

    a = BilinearForm(fes)
    #rot rot (u)
    a += InnerProduct(rot(u), rot(v)) * dx
    a += (-mean_rotu * jump_vt - mean_rotv * jump_ut) * dx(skeleton=True)
    a += alpha * k**2 / h * jump_ut * jump_vt * dx(skeleton=True)
    a += alpha * k**2 / h * (u*t) * (v*t) * ds(skeleton=True)
    a += (-rot(u) * (t * v) - rot(v) * (t * u)) * ds(skeleton=True)
    
    # div(u)*q 
    #a += (u * grad(q) + v * grad(p)) * dx
    a += (-div(u) * q - div(v) * p) * dx
    a += (mean_p * jump_v + mean_q * jump_u) * dx(skeleton=True)
    a += alpha* k**2 / h *(p *q) * ds(skeleton=True)
    a.Assemble()

    m = BilinearForm(fes)
    m += -1*u*v*dx
    return a,m

order = 2
hmax = 0.2

wp = WorkPlane()
geo = OCCGeometry(wp.Rectangle(np.pi, np.pi).Face(),dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=hmax,comm=MPI.COMM_WORLD))

V = VectorL2(mesh, order=order, dgjumps=True, type1=True)
Q = L2(mesh, order=order, dgjumps=True, dirichlet=".*")
fes = V * Q

a, m = MaxwellDG(fes)
from ngsPETSc import EigenSolver
solver = EigenSolver((m, a), fes, 10,
                     solverParameters={
                        "eps_type":"krylovschur",
                        "st_type":"sinvert",
                        "eps_target": 1,
                        "eps_monitor": "",
                        "st_pc_type": "lu",
                        "st_pc_factor_mat_solver_type": "mumps"
                    })
solver.solve()
for i in range(solver.nconv):
   print("Normalised (by pi^2) Eigenvalue ", i, " = ", solver.eigenValue(i))
