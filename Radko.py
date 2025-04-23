import numpy as np
import dedalus.public as d3
import logging

logger = logging.getLogger(__name__)

Lx, Lz = 2*np.pi, 1*np.pi
Nx, Nz = 32,16
beta = 1
nv = 0.0002
#dealias = 3/2
dealias = 3/2
stop_sim_time = 20
timestepper = d3.RK443
max_timestep = 1e-2
dtype = np.float64

coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(-Lz/2, Lz/2), dealias=dealias)

tau_psi=dist.Field(name='tau_psi')
#u = dist.VectorField(coords, name='u', bases=(xbasis, zbasis))
#psi = dist.Field(name='psi', bases=(xbasis, zbasis))
cos_z = dist.Field(name='cos_z', bases=(xbasis, zbasis))
sin_z = dist.Field(name='sin_z', bases=(xbasis, zbasis))
#velo = dist.Field(name='velo', bases=(xbasis, zbasis))
#laplacian_psi_bar = dist.Field(name='laplacian_psi_bar', bases=(xbasis, zbasis))
psi_prime = dist.Field(name='psi_prime', bases=(xbasis, zbasis))
#laplacian_psi_prime = dist.Field(name='laplacian_psi_prime', bases=(xbasis, zbasis))
#laplacian_laplacian_psi_prime = dist.Field(name='laplacian_laplacian_psi_prime', bases=(xbasis, zbasis))

x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)

A=0.5
B=0.25

#A=1
#B=0.1

eps=2*np.pi/Lz
X = eps**2*x
Z = eps*z
Re = 1/nv

cos_z['g']=np.cos(z) #*np.maximum(x/Lx,1)
sin_z['g']=np.sin(z) #*np.maximum(x/Lx,1)

def jacobian(f1, f2):
    return (d3.Differentiate(f1, coords['x']) * d3.Differentiate(f2, coords['z']) -
            d3.Differentiate(f1, coords['z']) * d3.Differentiate(f2, coords['x']))

#???

psi_bar = lambda t: (A+B * np.cos(t))*cos_z#*np.maximum(x/Lx,1)    
lap_psi_bar = lambda t: -(A+B*np.cos(t))*cos_z#*np.maximum(x/Lx,1)

#dpsi_bar_dx = lambda t: 0
dpsi_bar_dz = lambda t: -(A+B*np.cos(t))*sin_z#*np.maximum(x/Lx,1)
#dlap_psi_bar_dx = lambda t: 0
dlap_psi_bar_dz = lambda t: (A+B*np.cos(t))*sin_z#*np.maximum(x/Lx,1)

cos = lambda t: np.cos(t)
sin = lambda t: np.sin(t)
dx = lambda f: d3.Differentiate(f,coords['x'])
dz = lambda f: d3.Differentiate(f,coords['z'])

def jacobian2(f3,t):                  #for jacobian(psi_prime, lap(psi_bar))
    return (dx(f3)*dlap_psi_bar_dz(t))
def jacobian3(f4,t):                  #for jacobian(psi_bar, lap(psi_prime))
    return -dpsi_bar_dz(t)*dx(f4)

#???


problem = d3.IVP([psi_prime,tau_psi],namespace=locals())
problem.namespace.update({'t':problem.time})

#problem.add_equation("d3.TimeDerivative(lap(psi_prime))- nv * lap(lap(psi_prime))+ beta * d3.Differentiate(psi_prime, coords['x'])+tau_psi=-jacobian(psi_prime, lap(psi_bar)) -jacobian(psi_bar, lap(psi_prime))-jacobian(psi_prime, lap(psi_prime))")
problem.add_equation("d3.TimeDerivative(lap(psi_prime))- nv * lap(lap(psi_prime))+ beta * dx(psi_prime)+tau_psi=-jacobian2(psi_prime,t) -jacobian3(lap(psi_prime),t)-jacobian(psi_prime, lap(psi_prime))")

#problem.add_equation("d3.TimeDerivative(lap(psi_prime))- nv * lap(lap(psi_prime))+ beta * dx(psi_prime)+tau_psi=-dx(psi_prime)*(A+B*cos(1))*sin(z)-(-(A+B*cos(1))*sin(z))*dx(lap(psi_prime))-jacobian(psi_prime, lap(psi_prime))")

problem.add_equation("integ(psi_prime)=0")

solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

#psi_bar['g'] = (A + B * np.cos(problem.time['g'])) * np.cos(z)
#np.random.seed(42)

psi_prime.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
#psi_prime['g']=eps*(np.random.rand(Nx,Nz))
#psi_bar = lambda z, t: (A + B * np.cos(t)) * np.cos(z)
#psi_bar['g'] += eps * (np.random.rand(Nx, Nz) - 0.5)
#laplacian_psi_bar = d3.Laplacian(psi_bar).evaluate()
#laplacian_psi_prime = d3.Laplacian(psi_prime).evaluate()
#psi = psi_bar + psi_prime

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=1, max_writes=1000)
snapshots.add_task(psi_prime, name='psi_prime')
snapshots.add_task(cos_z, name='cos_z')
#snapshots.add_task(sin_z, name='sin_z')
snapshots.add_task(d3.Laplacian(psi_prime), name='vorticity_prime')
snapshots.add_task(dx(psi_prime),name='w')
snapshots.add_task(-dz(psi_prime),name='u')

velox=dpsi_bar_dz(problem.time)+dz(psi_prime)#.evaluate()['g']
veloz=dx(psi_prime)#.evaluate()['g']
velocity_expr = velox * ex + veloz * ez


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
#CFL.add_velocity((velox,veloz))
CFL.add_velocity(velocity_expr)

flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(psi_prime, name='psi_prime')

try:
    logger.info('Starting main loop')
  #  fixed_timestep = 1e-2
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        ##update
        current_t = solver.sim_time
  #      psi_bar['g'] = (A + B * np.cos(current_t)) * np.cos(z) * np.ones_like(z)
 #       psi_bar['g'] = (A + B * np.cos(current_t )) * np.cos(z)
  #      psi_bar['g'] = (A + B * np.cos(current_t))
    #    laplacian_psi_bar = d3.Laplacian(psi_bar).evaluate()
        if (solver.iteration-1) % 10 == 0:
            max_psi_prime = np.sqrt(flow.max('psi_prime'))
            logger.info('Iteration=%i, Time=%e, dt=%e, max(psi_prime)=%f' %(solver.iteration, solver.sim_time, timestep, max_psi_prime))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()
