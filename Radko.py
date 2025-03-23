import numpy as np

import dedalus.public as d3

import logging



logger = logging.getLogger(__name__)



Lx, Lz = 24*np.pi, 12*np.pi

#Lx, Lz = 2*np.pi, 1*np.pi

Nx, Nz = 4096,2048

#Nx, Nz = 64,32

beta = 1

nv = 0.0002

dealias = 3/2

stop_sim_time = 800

#stop_sim_time = 2

timestepper = d3.RK443

max_timestep = 1e-2

dtype = np.float64



coords = d3.CartesianCoordinates('x', 'z')

dist = d3.Distributor(coords, dtype=dtype)

xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)

zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)



tau_psi=dist.Field(name='tau_psi')

u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))

psi = dist.Field(name='psi', bases=(xbasis, zbasis))

#cosi = dist.Field(name='cosi', bases=(xbasis, zbasis))

#dcosi = dist.Field(name='dcosi', bases=(xbasis, zbasis))

psi_bar = dist.Field(name='psi_bar', bases=(xbasis, zbasis))

laplacian_psi_bar = dist.Field(name='laplacian_psi_bar', bases=(xbasis, zbasis))

psi_prime = dist.Field(name='psi_prime', bases=(xbasis, zbasis))

laplacian_psi_prime = dist.Field(name='laplacian_psi_prime', bases=(xbasis, zbasis))

laplacian_laplacian_psi_prime = dist.Field(name='laplacian_laplacian_psi_prime', bases=(xbasis, zbasis))

#zeta_prime = dist.Field(name='zeta_prime', bases=(xbasis, zbasis))

#tau_p = dist.Field(name='tau_p')

#s = dist.Field(name='s', bases=(xbasis, zbasis))



x, z = dist.local_grids(xbasis, zbasis)

ex, ez = coords.unit_vector_fields(dist)

A=0.5

B=0.25

eps=2*np.pi/Lz

X = eps**2*x

Z = eps*z

#T = eps**2*t

Re = 1/nv

def jacobian(f1, f2):

    return (d3.Differentiate(f1, coords['x']) * d3.Differentiate(f2, coords['z']) -

            d3.Differentiate(f1, coords['z']) * d3.Differentiate(f2, coords['x']))





#problem = d3.IVP([laplacian_psi_prime,laplacian_laplacian_psi_prime,psi_prime,tau_psi], namespace=locals())



problem = d3.IVP([psi_prime,tau_psi],namespace=locals())

problem.namespace.update({'t':problem.time})

#problem = d3.IVP([psi_bar, psi_prime, laplacian_psi_bar, laplacian_psi_prime, laplacian_laplacian_psi_prime], namespace=locals())

#problem.add_equation(psi_bar - (A+B*np.cos(t))*np.cos(z))

#problem.add_equation(laplacian_psi_bar = eps**4 * diff(psi_bar, X, X) + diff(psi_bar, z, z) + 2 * eps * diff(psi_bar, z, Z) + eps**2 * diff(psi_bar, Z, Z) )

#problem.add_equation(laplacian_psi_prime = eps**4 * diff(psi_prime, X, X) + diff(psi_prime, z, z) + 2 * eps * diff(psi_prime, z, Z) + eps**2 * diff(psi_prime, Z, Z) )

#problem.add_equation(diff(laplacian_psi_prime, t) + jacobian(psi_prime , laplacian_psi_bar) + jacobian(psi_bar , laplacian_psi_prime) + jacobian(psi_prime , laplacian_psi_prime) + beta * diff(psi_prime, x) == nv*laplacian_laplacian_psi_prime)

problem.add_equation("d3.TimeDerivative(lap(psi_prime))- nv * lap(lap(psi_prime))+ beta * d3.Differentiate(psi_prime, coords['x'])+tau_psi=-jacobian(psi_prime, lap(psi_bar)) -jacobian(psi_bar, lap(psi_prime))-jacobian(psi_prime, lap(psi_prime))")

#problem.add_equation("laplacian_laplacian_psi_prime = d3.Laplacian(laplacian_psi_prime).evaluate()")

#problem.add_equation("laplacian_psi_prime = d3.Laplacian(psi_prime).evaluate()")

problem.add_equation("integ(psi_prime)=0")

#problem.add_equation((

#    d3.TimeDerivative(dcosi) + cosi,

#    0

#))#2

#problem.add_equation((

#    d3.TimeDerivative(cosi) - dcosi,

#    0

#))#1









#problem.add_equation(laplacian_laplacian_psi_prime = eps**4 * diff(laplacian_psi_prime, X, X) + diff(laplacian_psi_prime, z, z) + 2 * eps * diff(laplacian_psi_prime, z, Z) + eps**2 * diff(laplacian_psi_prime, Z, Z) )



solver = problem.build_solver(timestepper)

solver.stop_sim_time = stop_sim_time



#current_time = float(problem.time)

psi_bar['g'] = (A + B * np.cos(problem.time['g'])) * np.cos(z)

np.random.seed(42)

#psi_prime['g'] = eps * (np.random.rand(Nx, Nz) - 0.5)

psi_prime['g'] += eps * (np.random.rand(Nx, Nz) - 0.5)

#psi_bar['g'] = (A+B*np.cos(t))*np.cos(z)

laplacian_psi_bar = d3.Laplacian(psi_bar).evaluate()

laplacian_psi_prime = d3.Laplacian(psi_prime).evaluate()

#laplacian_laplacian_psi_prime = d3.Laplacian(laplacian_psi_prime).evaluate()

psi = psi_bar + psi_prime

# Calculate the velocity field

#u['g'][0] = d3.Differentiate(psi, coords['z']).evaluate().change_scales(1)['g']

#u['g'][1] = -d3.Differentiate(psi, coords['x']).evaluate().change_scales(1)['g']





snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=0.1, max_writes=10)

#snapshots.add_task(laplacian_psi_prime, name='laplacian_psi_prime')

snapshots.add_task(psi_prime, name='psi_prime')

snapshots.add_task(laplacian_psi_prime, name='vorticity_prime')

#snapshots.add_task(psi_prime, name='psi')



#CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,

#             max_change=1.5, min_change=0.5, max_dt=max_timestep)

#CFL.add_velocity(u)



flow = d3.GlobalFlowProperty(solver, cadence=10)

flow.add_property(psi_prime, name='psi_prime')



try:

    logger.info('Starting main loop')

    fixed_timestep = 1e-4

    while solver.proceed:

        timestep = fixed_timestep

#        timestep = CFL.compute_timestep()

        solver.step(timestep)

        if (solver.iteration-1) % 100 == 0:

            max_psi_prime = np.sqrt(flow.max('psi_prime'))

            logger.info('Iteration=%i, Time=%e, dt=%e, max(w)=%f' %(solver.iteration, solver.sim_time, timestep, max_psi_prime))

except:

    logger.error('Exception raised, triggering end of main loop.')

    raise

finally:

    solver.log_stats()