from flow import FlowProject
import freud
import hoomd
import math
import numpy
import random

CLUSTER_JOB_WALLTIME = 30
HOOMD_RUN_WALLTIME_LIMIT = CLUSTER_JOB_WALLTIME*3600-1800    #stop the simulation 30 mins in advance in case of delays of storing data


# This is the basic information of the simulation (those things that are not changed during the whole simulation)
def create_simulation(job):                                                            
    cpu = hoomd.device.CPU()
    sim = hoomd.Simulation(device=cpu, seed=random.randrange(0, 10000))

    # define the harmonic repulsion potential (using morse to approximate it)
    nl = hoomd.md.nlist.Cell(buffer = 0.4)
    morse = hoomd.md.pair.Morse(nlist = nl, mode = "shift")
    a1 = 1.0
    a2 = math.sqrt(2)
    a3 = (a1 + a2) / 2.0
    alpha = 0.01
    morse.params[('A', 'A')] = dict(
        D0 = 1 / (math.exp(2 * alpha * a1) - 2 * math.exp(alpha * a1) + 1), alpha = alpha, r0 = a1)
    morse.params[('B', 'B')] = dict(
        D0 = 1 / (math.exp(2 * alpha * a2) - 2 * math.exp(alpha * a2) + 1), alpha = alpha, r0 = a2)
    morse.params[('A', 'B')] = dict(
    D0 = 1 / (math.exp(2 * alpha * a3) - 2 * math.exp(alpha * a3) + 1), alpha = alpha, r0 = a3)

    morse.r_cut[('A', 'A')] = a1
    morse.r_cut[('B', 'B')] = a2
    morse.r_cut[('A', 'B')] = a3


    #defining a custom filter class
    class DensityFilter(hoomd.filter.CustomFilter):
        def __init__(self, min_den, max_den):
            self.min_den = min_den
            self.max_den = max_den

        def __hash__(self):
            return hash((self.min_den, self.max_den))

        def __eq__(self, other):
            return (
                isinstance(other, DensityFilter)
                and self.min_den == other.min_den
                and self.max_den == other.max_den
            )

        def __call__(self, state):
            with state.cpu_local_snapshot as local_snapshot:
                a = local_snapshot.particles.position
                b = local_snapshot.particles.tag
                tags = numpy.array(b, copy = True)                     #think about whether you want tags or rtags in your implementation
                positions = numpy.array(a, copy = True)
            
            ld = freud.density.LocalDensity(r_max=2, diameter=0.60)
            k = ld.compute(system=(sim.state.box, positions))  
            densities = k.density * 1.16                               #going from number density to volume

            indices = (densities > self.min_den) & (
                    densities < self.max_den)

            return (tags[indices])

    #instance of custom filter
    densityfilter = DensityFilter(0.0, 0.67)

    filter_updater = hoomd.update.FilterUpdater(
        trigger=hoomd.trigger.Periodic(100),
        filters=[densityfilter],
    )
    sim.operations += filter_updater

    #defining active forces
    active1 = hoomd.md.force.Active(filter = densityfilter)
    Dr1 = 3 * job.sp.kT / a3 ** 2                                         # this is how it is defined in the paper
    v1 = job.sp.Pe * a3 * Dr1    
    active1.active_force['A','B'] = (v1, 0, 0)                           #Active particle, swimming pressure
    active1.active_torque['A','B'] = (0, 0, 0)

    rotation_updater1 = active1.create_diffusion_updater(
        trigger = 1, rotational_diffusion = Dr1)
    sim.operations += rotation_updater1



    # define the evolving dynamics
    integrator = hoomd.md.Integrator(dt=job.sp.dt)
    integrator.forces.append(morse)
    integrator.forces.append(active1)
    bd = hoomd.md.methods.Brownian(filter=hoomd.filter.All(), kT=job.sp.kT)
    integrator.methods.append(bd)
    sim.operations.integrator = integrator
    return sim                                                                                 

class Project(FlowProject):
    pass

# This is the step to equilibrate the initial state with randomly scattered particles


@Project.pre.true('initialized')  # Pre-condition in job document
@Project.post.true('equilibrated')  # Post-condition in job document
# Workflow step
@Project.operation(directives={"memory": "64g", 'walltime': CLUSTER_JOB_WALLTIME})
def equilibrate(job):
    sim = create_simulation(job)
    sim.create_state_from_gsd(filename=job.fn('initial.gsd'))
    sim.run(1)                                                                          #modified this to minimize equilibriation
    hoomd.write.GSD.write(state=sim.state, mode='xb',
                          filename=job.fn('equilibrium.gsd'))
    job.document['equilibrated'] = True
    job.document['equil_step'] = sim.timestep

# This is the step to add the shear and only implemented after the equilibrate step has been finished  (this step might be finished within one submitted job on the cluster)


@Project.pre.true('equilibrated')
@Project.post(lambda job: job.document.get('timestep', 0) - job.document['equil_step']    # will continue until the running step is larger than the totalsteps
              >= job.document['totalsteps'])
@Project.operation(directives={"memory": "64g", 'walltime': CLUSTER_JOB_WALLTIME})
def shear(job):
    end_step = job.document['equil_step']+job.document['totalsteps']

    sim = create_simulation(job)

    if job.isfile('restart.gsd'):
        # Read the final system configuration from a previous execution.
        sim.create_state_from_gsd(filename=job.fn('restart.gsd'))
    else:
        # Or read `compressed.gsd` for the first execution of equilibrate.
        sim.create_state_from_gsd(filename=job.fn('equilibrium.gsd'))


    # # Write output for pressure-tensor and other useful information (10000 frames in total)
    # thermo_properties = hoomd.md.compute.ThermodynamicQuantities(
    #     filter=hoomd.filter.All())
    # sim.operations.computes.append(thermo_properties)
    # logger_h5 = hoomd.logging.Logger(categories=['scalar', 'sequence'])
    # logger_h5.add(sim, quantities=['timestep', 'walltime'])
    # logger_h5.add(thermo_properties, quantities=['pressure_tensor'])

    # hdf5_writer = hoomd.write.HDF5Log(trigger=hoomd.trigger.Periodic(
    #     int(job.document['totalsteps']/10000)), filename=job.fn('stresslog.h5'), mode='a', logger=logger_h5)
    # sim.operations.writers.append(hdf5_writer)

    # output all particle's trajectories   (10000 frames in total)
    logger_gsd = hoomd.logging.Logger()
    gsd_writter = hoomd.write.GSD(filename=job.fn('trajectory.gsd'), trigger=hoomd.trigger.Periodic(100),
        mode='ab', filter=hoomd.filter.All(), logger=logger_gsd)
    sim.operations.writers.append(gsd_writter)

    try:
        while sim.timestep < end_step:
            # Run the simulation in chunks of 10,000 time steps.
            sim.run(min(1e6, end_step-sim.timestep))    # 1e6 steps approximately use 10 min to run. 
            # End the workflow step early if the next run exceeds the
            # alotted walltime. Use the walltime of the current run as
            # an estimate for the next.
            if (
                sim.device.communicator.walltime + sim.walltime
                >= HOOMD_RUN_WALLTIME_LIMIT
            ):
                break
    finally:
        # Write the state of the system to `restart.gsd`.
        hoomd.write.GSD.write(
            state=sim.state, mode='wb', filename=job.fn('restart.gsd')
        )

    job.document['timestep'] = sim.timestep


if __name__ == '__main__':
    Project().main()



