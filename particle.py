import sys
from typing import Callable

import numpy as np
import scipy as sp


class Particle:

    def __init__(self, spatial_grid: np.ndarray,
                 initial_func: Callable[[np.ndarray, ...], np.ndarray],
                 mass: float = 1,
                 **kwargs):
        """
        Initializes a particle with a given initial wave function
        :param spatial_grid: The spatial grid on which the wave function is defined
        :param initial_func: Function that returns the initial wave function \n
        Signature: initial_func(spatial_grid, **kwargs) -> np.ndarray
        :param mass: Mass of the particle
        :param kwargs: Additional arguments to be passed to initial_func
        """
        print(kwargs)
        self.spatial_grid = spatial_grid
        self.dx = spatial_grid[1] - spatial_grid[0]
        self.mass = mass
        self.psi = initial_func(spatial_grid, **kwargs)
        self.normalize()

    def normalize(self):
        """
        Normalizes the wave function Psi
        """
        self.psi /= np.sqrt(np.sum(np.square(np.abs(self.psi))) * self.dx)

    def get_kinetic_operator(self) -> np.ndarray:
        """
        Returns the kinetic operator 1/2 * d^2/dx^2
        using the finite difference method on the spatial grid
        """
        off_diag = np.ones(len(self.spatial_grid) - 1)
        diag = -2 * np.ones(len(self.spatial_grid))
        return sp.sparse.diags([off_diag, diag, off_diag],
                               [-1, 0, 1]) / self.dx ** 2

    def simulation_step(self, potential: np.ndarray, dt: float) -> None:
        """
        Performs a single simulation step using the Crank-Nicolson method
        :param potential: The potential at the current time step
        as a diagonal matrix on the spatial grid. \n
        Must be of shape (len(spatial_grid), len(spatial_grid))
        :param dt: The time step (for good results, dt should be <= 0.001)
        :return: None
        """
        if dt > 0.001:
            sys.stdout.write(
                f"Warning: dt = {dt} is larger than 0.001. "
                f"Results may be inaccurate.\n")
        if potential.shape != (len(self.spatial_grid), len(self.spatial_grid)):
            raise ValueError(
                f"Potential must be of shape {len(self.spatial_grid)} x "
                f"{len(self.spatial_grid)}")

        # Make sure that the potential is a sparse matrix
        potential = sp.sparse.csc_matrix(potential)

        # Calculate the Crank-Nicolson matrix
        u1 = sp.sparse.eye(len(self.spatial_grid)) - 0.5j * dt * (
                self.get_kinetic_operator() - potential)
        u2 = sp.sparse.eye(len(self.spatial_grid)) + 0.5j * dt * (
                self.get_kinetic_operator() - potential)

        # Solve the linear system u1 * psi_new = u2 * psi using sparse solver
        self.psi = sp.sparse.linalg.spsolve(u1, u2.dot(self.psi))

    def simulate(self, potential: np.ndarray, dt: float, steps: int) -> None:
        """
        Simulates the particle for a given number of time steps
        :param potential: The potential at the current time step
        as a diagonal matrix on the spatial grid. \n
        Must be of shape (len(spatial_grid), len(spatial_grid))
        :param dt: The time step (for good results, dt should be <= 0.001)
        :param steps: The number of time steps to simulate
        :return: None
        """
        for _ in range(steps):
            self.simulation_step(potential, dt)

    def get_probability_density(self) -> np.ndarray:
        """
        Returns the probability density of the particle
        """
        return np.square(np.abs(self.psi))
