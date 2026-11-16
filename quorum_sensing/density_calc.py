#File with various class definitions pertaining to density based filtering


from collections import deque
from flow import FlowProject
import freud
import hoomd
import math
import numpy
import random


"""
A custom HOOMD Action that calculates local densities once per step,
maintains a rolling history over the last n frames, {and stores the 
time-averaged densities in a 'densities' attribute for filters to read.}<- this 
needs to be done by a different function/class
"""
class DensityCalculator(hoomd.custom.Action):

    def __init__(self, ld_r_max, ld_diameter, window_size = 1):
        super().__init__()  # Call the parent class's constructor FIRST

        # Store parameters
        self.ld_r_max = ld_r_max
        self.ld_diameter = ld_diameter
        self.window_size = window_size
        
        # Rolling buffer to store density history indexed by particle tag
        self.history = deque(maxlen = self.window_size)
        self.running_sum = None
        
        # Create the freud object once
        self._freud_ld = freud.density.LocalDensity(
            r_max=self.ld_r_max, diameter = self.ld_diameter
        )
    
    def save_history(self, filepath):
        """Saves density history and running sum to disk."""
        if self.running_sum is not None:
            numpy.savez(
                filepath,
                history=numpy.array(self.history),
                running_sum=self.running_sum
            )


    def load_history(self, filepath):
        """Loads density history and running sum from disk."""
        import os

        if os.path.exists(filepath):
            data = numpy.load(filepath)

            # Restore deque
            self.history = deque(
                data["history"],
                maxlen=self.window_size
            )

            # Restore running sum directly
            self.running_sum = data["running_sum"]

    def act(self, timestep):
        """
        The method HOOMD calls at each 'trigger' step.
        """
        state = self._state
        box = state.box
        
        with state.cpu_local_snapshot as local_snapshot:
            positions = numpy.array(local_snapshot.particles.position, copy = True)
            tags = numpy.array(local_snapshot.particles.tag, copy = True)
            
        # Run the freud calculation outside the 'with' block
        self._freud_ld.compute(system=(box, positions))
        inst_densities = self._freud_ld.density
        
        # CRITICAL: HOOMD periodically reorders particles in memory for cache efficiency.
        # To correctly average a particle's density over time, we must align densities
        # by immutable particle tags rather than mutable memory indices.
        n_particles = len(tags)
        dens_by_tag = numpy.zeros(n_particles, dtype=numpy.float64)
        dens_by_tag[tags] = inst_densities


        # Initialize running sum on first frame
        if self.running_sum is None:
            self.running_sum = numpy.zeros_like(dens_by_tag)


        # Remove oldest frame if window is full
        if len(self.history) == self.window_size:
            oldest = self.history.popleft()
            self.running_sum -= oldest


        # Add newest frame
        self.running_sum += dens_by_tag

        # Store newest frame
        self.history.append(dens_by_tag)
        

"""
Custom filter that selects the top or bottom fraction of particles
based on the running sum of their local density.
"""
class DensityFilter(hoomd.filter.CustomFilter):


    def __init__(self, active_fraction, density_calculator, invert=False):
        self.active_fraction = active_fraction
        self.density_calculator = density_calculator
        self.invert = invert

    def __hash__(self):
        return hash((
            self.active_fraction,
            id(self.density_calculator),
            self.invert,
        ))

    def __eq__(self, other):
        return (
            isinstance(other, DensityFilter)
            and self.active_fraction == other.active_fraction
            and self.density_calculator is other.density_calculator
            and self.invert == other.invert
        )

    def __call__(self, state):
        running_sum = self.density_calculator.running_sum

        # Density calculator has not run yet
        if running_sum is None:
            return numpy.array([], dtype=numpy.uint32)

        num_particles = len(running_sum)
        num_active = int(num_particles * self.active_fraction)

        if num_active <= 0:
            return numpy.array([], dtype=numpy.uint32)

        if num_active >= num_particles:
            return numpy.arange(num_particles, dtype=numpy.uint32)

        if not self.invert:
            # Lowest density particles
            return numpy.argpartition(
                running_sum,
                num_active - 1
            )[:num_active].astype(numpy.uint32)

        else:
            # Highest density particles
            return numpy.argpartition(
                running_sum,
                -num_active
            )[-num_active:].astype(numpy.uint32)


"""
A custom filter that acts as a logical NOT operator for another custom filter.
Uses an O(N) boolean mask for maximum performance.
"""
class InverseCustomFilter(hoomd.filter.CustomFilter):

    def __init__(self, base_filter):
        self.base_filter = base_filter

    def __hash__(self):
        return hash((self.__class__.__name__, hash(self.base_filter)))

    def __eq__(self, other):
        return (
            isinstance(other, InverseCustomFilter)
            and self.base_filter is other.base_filter
        )

    def __call__(self, state):
        # 1. Get the tags of the particles inside the active filter
        base_tags = self.base_filter(state)

        # 2. Create an O(N) boolean mask sized to the number of particles
        mask = numpy.ones(state.N_particles, dtype=bool)
        
        # 3. Mark the active tags as False
        if len(base_tags) > 0:
            mask[base_tags] = False

        # 4. Extract the indices (tags) where the mask is still True
        return numpy.nonzero(mask)[0].astype(numpy.uint32)
