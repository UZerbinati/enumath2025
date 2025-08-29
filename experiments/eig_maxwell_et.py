from ngsolve import *
from ngstrefftz import *
from netgen.occ import *
import numpy as np
from mpi4py import MPI

def rotrot(u):
    J = u.Operator("hesse")
    return (J[0,3]-J[1,1],J[1,0]-J[0,2])
    

def TrefftzSpace(fes):
    mesh = fes.components[0].mesh
    k = fes.components[0].globalorder
    Vs = VectorL2(mesh, order=k - 2)
    Qs = L2(mesh, order=k - 1)
    test_fes = Vs * Qs

    u, p = fes.TrialFunction()
    wu, wp = test_fes.TestFunction()[0:2]

    op = (rotrot(u)[0]*wu[0]+rotrot(u)[1]*wu[1])*dx

    emb = TrefftzEmbedding(op,eps=10**-6)
    return emb, emb.GetEmbedding()

def rot(u):
    J = u.Operator("Grad")
    return J[1,0]-J[0,1]

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
    jump_pn= (p - p.Other())*n
    jump_qn= (q - q.Other())*n
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
    #a += -1*k**2/h * jump_p*jump_q*dx(skeleton=True) #Ilaria's term
    a += alpha* k**2 / (h**2) *(p *q) * ds(skeleton=True)

    m = BilinearForm(fes)
    m += u*v*dx
    return a,m

order = 8
hmax = 0.1

wp = WorkPlane()
geo = OCCGeometry(wp.Rectangle(np.pi, np.pi).Face(),dim=2)
mesh = Mesh(geo.GenerateMesh(maxh=hmax,comm=MPI.COMM_WORLD))

V = VectorL2(mesh, order=order, dgjumps=True, type1=True)
Q = L2(mesh, order=order, dgjumps=True, dirichlet=".*")
fes = V * Q
emb, P = TrefftzSpace(fes)
PT = P.CreateTranspose()
a, m = MaxwellDG(fes)
a.Assemble()
m.Assemble()
TA = PT @ a.mat @ P
TM = PT @ m.mat @ P
from ngsPETSc import Matrix
from slepc4py import SLEPc
from petsc4py import PETSc

A = Matrix(TA, parDescr=(None, None, None))
M = Matrix(TM, parDescr=(None, None, None))

eps = SLEPc.EPS().create()
solverParameters = {
        "eps_type":"krylovschur",
        "st_type":"sinvert",
        "eps_target": 1,
        "eps_monitor": "",
        "st_pc_type": "lu",
        "st_pc_factor_mat_solver_type": "mumps"
}
options_object = PETSc.Options()
if solverParameters is not None:
    for optName, optValue in solverParameters.items():
        options_object[optName] = optValue
eps.setOperators(A.mat, M.mat)
eps.setDimensions(10, SLEPc.DECIDE)
eps.setFromOptions()
eps.solve()
print("Dimension of DG Space {}, Dimension of Trefftz space: {}".format(P.shape[0],P.shape[1]))
for i in range(eps.getConverged()):
   print("Normalised  Eigenvalue ", i, " = ", eps.getEigenvalue(i))
