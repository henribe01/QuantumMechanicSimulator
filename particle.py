import sys
import timeit
from typing import Callable

import numpy as np
import scipy as sp
from scipy.constants import hbar


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
        Returns the kinetic operator 1/(2m) * d^2/dx^2 as a banded matrix
        using the finite difference method on the spatial grid
        """
        off_diag = np.ones(len(self.spatial_grid) - 1)
        diag = -2 * np.ones(len(self.spatial_grid))
        return -sp.sparse.diags([off_diag, diag, off_diag],
                               [-1, 0, 1], format="csr") / (2 * self.mass * self.dx ** 2)

    def simulation_step(self, potential: sp.sparse.csr_matrix, dt: float, potential_next: sp.sparse.csr_matrix = None) -> None:
        """
        Performs a single simulation step using the Crank-Nicolson method
        :param potential: The potential at the current time step
        as a diagonal matrix on the spatial grid. \n
        Must be of shape (len(spatial_grid), len(spatial_grid)) \n
        Must be a sparse csr_matrix for good performance
        :param dt: The time step (for good results, dt should be <= 0.001)
        :param potential_next: The potential at the next time step
        as a diagonal matrix on the spatial grid. \n
        Must be of shape (len(spatial_grid), len(spatial_grid)) \n
        Must be a sparse csr_matrix for good performance
        If None, the potential at the next time step is assumed to be the same
        as the potential at the current time step
        :return: None
        """
        if dt/self.dx**2 * hbar**2/(2*self.mass) > 0.5:
            sys.stdout.write(
                f"Warning: dt/dx^2 = {dt/self.dx**2} is larger than 0.5. "
                f"Results may be inaccurate.\n")
        # if potential.shape != (len(self.spatial_grid), len(self.spatial_grid)):
        #     raise ValueError(
        #         f"Potential must be of shape {len(self.spatial_grid)} x "
        #         f"{len(self.spatial_grid)}")
        # if potential_next is not None and potential_next.shape != (
        #         len(self.spatial_grid), len(self.spatial_grid)):
        #     raise ValueError(
        #         f"Potential_next must be of shape {len(self.spatial_grid)} x "
        #         f"{len(self.spatial_grid)}")

        if potential_next is None:
            potential_next = potential

        kinetic_operator = self.get_kinetic_operator()
        # Calculate the matrix A as described in the Crank-Nicolson method
        # as a banded matrix
        off_diag = -0.5j * dt * kinetic_operator.diagonal(1)
        off_diag_lower = np.concatenate((off_diag, [0]))
        off_diag_upper = np.concatenate(([0], off_diag))
        #print(potential.diagonal(0))
        diag = np.ones(len(self.spatial_grid)) - 0.5j * dt * (kinetic_operator.diagonal(0) + potential_next.diagonal(0))
        m_next = np.array([off_diag_upper, diag, off_diag_lower])

        # Calculate right hand side of the equation
        m_current = sp.sparse.eye(len(self.spatial_grid), format='csr') + 0.5j * dt * (kinetic_operator + potential)
        b = m_current.dot(self.psi)

        # Solve the equation
        self.psi = sp.linalg.solve_banded((1, 1), m_next, b)

    def simulate(self, potential: Callable, dt: float, steps: int) -> None:
        """
        Simulates the particle for a given number of time steps
        :param potential: The potential at the current time step
        as a diagonal matrix on the spatial grid. \n
        Must be of shape (len(spatial_grid), len(spatial_grid))
        :param dt: The time step (for good results, dt should be <= 0.001)
        :param steps: The number of time steps to simulate
        :return: None
        """
        for i in range(steps):
            potential_current = potential(self.spatial_grid, i * dt) * sp.sparse.eye(len(self.spatial_grid))
            potential_next = potential(self.spatial_grid, (i + 1) * dt) * sp.sparse.eye(len(self.spatial_grid))
            print(potential_current.diagonal(0))
            self.simulation_step(potential_current, dt, potential_next)

    def get_probability_density(self) -> np.ndarray:
        """
        Returns the probability density of the particle
        """
        return np.square(np.abs(self.psi))

    def __add__(self, other):
        if not isinstance(other, Particle):
            raise TypeError(f'Cannot add Particle and {type(other)}',
                            'Only addition of two Particles is supported')
        if not np.array_equal(self.spatial_grid, other.spatial_grid):
            raise ValueError(
                'Cannot add Particles with different spatial grids',
                'Make sure that both particles have the same spatial grid')
        if self.mass != other.mass:
            raise ValueError(
                'Cannot add Particles with different masses',
                'Make sure that both particles have the same mass')

        return Particle(self.spatial_grid, lambda x: self.psi + other.psi)

    def __sub__(self, other):
        return self.__add__(-other)

    def __mul__(self, other):
        if isinstance(other, Particle):
            raise TypeError(f'Cannot multiply Particle and {type(other)}',
                            'Only multiplication of Particle with scalar is supported')
        return Particle(self.spatial_grid, lambda x: other * self.psi)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, Particle):
            raise TypeError(f'Cannot divide Particle by {type(other)}',
                            'Only division of Particle by scalar is supported')
        return Particle(self.spatial_grid, lambda x: self.psi / other)

    def __neg__(self):
        return Particle(self.spatial_grid, lambda x: -self.psi)
