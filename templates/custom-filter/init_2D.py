import signac
import numpy
import gsd.hoomd
import math
import numpy as np


def init(job):
    frame = gsd.hoomd.Frame()
    frame.particles.N = job.sp.N
    frame.particles.types = ["A", "B"]            #for size differences
    frame.particles.typeid = np.zeros(job.sp.N)
    
    n1 = int(((job.sp.N/2)))                     #Number of particles of type 'A'
    n2 = int(job.sp.N - n1)                      #Number of particles of type 'B'   

    L = math.sqrt((job.sp.N * 0.5 * math.pi * 0.5**2 + job.sp.N * 0.5 *
                   math.pi * (math.sqrt(2)/2)**2) / job.sp.phi)
    
    frame.configuration.box = [L, L, 0, 0, 0, 0]
    frame.particles.typeid = [0] * n1 + [1] * n2 
    frame.particles.position = np.random.uniform(
        low=-L/2, high=L/2, size=(job.sp.N, 3))
    
    for i in range(len(frame.particles.position)):
        frame.particles.position[i][2] = 0


    with gsd.hoomd.open(job.fn("initial.gsd"), "w") as traj:
        traj.append(frame)
    job.document['initialized'] = True
    job.document['totalsteps'] = int(10**6)


project = signac.init_project()
sp = {"shear": 0, "kT": 10**(-4),
"phi": 0.65, "Pe": 10.0, "N": 1000, "dt": 0.01}
        
job = project.open_job(sp).init()
init(job)
      