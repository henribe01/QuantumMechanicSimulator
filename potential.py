import sys

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from abc import ABC


class PotentialBase(ABC):
    """
    Base class for potentials
    """

    def __init__(self, spatial_grid: np.ndarray, name: str):
        self.spatial_grid = spatial_grid
        self.dx = spatial_grid[1] - spatial_grid[0]
        self.name = name
        self.potential = np.zeros(len(spatial_grid))

    def plot(self, ax: plt.Axes, **kwargs) -> plt.Artist:
        """
        Plots the potential on the given axes
        :param ax:  on which to plot the potential
        :param kwargs: Additional arguments to be passed to ax.plot
        :return: List of artists
        """
        return ax.fill_between(self.spatial_grid, self.potential, **kwargs)

    def get_potential(self) -> sp.sparse.csr_matrix:
        """
        Returns the potential as a diagonal matrix on the spatial grid
        """
        return sp.sparse.csr_matrix(sp.sparse.diags(self.potential))

    def get_eigenstates(self, num_states: int, mass: float = 1) -> np.ndarray:
        """
        Returns the first num_states eigenstates of the potential
        :param num_states: Number of eigenstates to return
        :param mass: Mass of the particle
        :return: Array of eigenstates
        """
        raise NotImplementedError

class FreeParticle(PotentialBase):
    def __init__(self, spatial_grid: np.ndarray):
        super().__init__(spatial_grid, "Free Particle")
        self.potential = np.zeros(len(spatial_grid))


class InfiniteSquareWell(PotentialBase):
    def __init__(self, spatial_grid: np.ndarray, left_wall: float,
                 right_wall: float):
        super().__init__(spatial_grid, "Infinite Square Well")
        self.left_wall = left_wall
        self.right_wall = right_wall
        if left_wall < spatial_grid[0]:
            sys.stderr.write(
                f"Warning: left_wall = {left_wall} is smaller than the "
                f"leftmost point of the spatial grid = {spatial_grid[0]}. ")
        if right_wall > spatial_grid[-1]:
            sys.stderr.write(
                f"Warning: right_wall = {right_wall} is larger than the "
                f"rightmost point of the spatial grid = {spatial_grid[-1]}. ")
        if left_wall > right_wall:
            raise ValueError(
                f"left_wall = {left_wall} is larger than right_wall = "
                f"{right_wall}")

        # Set potential to big number outside the well
        self.potential[spatial_grid < left_wall] = 1e10
        self.potential[spatial_grid > right_wall] = 1e10


class HarmonicOscillator(PotentialBase):
    def __init__(self, spatial_grid: np.ndarray, center: float, omega: float, mass: float = 1):
        super().__init__(spatial_grid, "Harmonic Oscillator")
        self.omega = omega
        self.center = center
        self.potential = 0.5 * self.omega ** 2 * mass * np.square(
            self.spatial_grid - self.center)


class PotentialBarrier(PotentialBase):
    def __init__(self, spatial_grid: np.ndarray, center: float, width: float,
                 height: float):
        super().__init__(spatial_grid, "Potential Barrier")
        self.center = center
        self.width = width
        self.height = height
        self.potential = np.zeros(len(spatial_grid))
        self.potential[
        (spatial_grid > center - width / 2) & (spatial_grid < center + width / 2)] = height