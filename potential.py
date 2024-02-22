import sys

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from abc import ABC
# from scipy.constants import hbar
from scipy.special import hermite, factorial

hbar = 1


class PotentialBase(ABC):
    """
    Base class for potentials
    """

    def __init__(self, name: str):
        self.name = name

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        """
        Returns the potential as a function of the spatial grid and time
        :param t:  at which to evaluate the potential
        """
        raise NotImplementedError

    def plot(self, spatial_grid: np.ndarray, ax: plt.Axes, t: float = 0, **kwargs) -> list:
        """
        Plots the potential on the given axes
        :param spatial_grid: The spatial grid on which to evaluate the potential
        :param ax:  on which to plot the potential
        :param t: Time at which to evaluate the potential
        :param kwargs: Additional arguments to be passed to ax.plot
        :return: List of artists
        """
        potential = self.potential(spatial_grid, t)
        return ax.fill_between(spatial_grid, potential, animated=True, **kwargs)

    def get_csr_matrix(self, spatial_grid: np.ndarray, t: float = 0) -> sp.sparse.csr_matrix:
        """
        Returns the potential as a diagonal matrix on the spatial grid
        """
        return sp.sparse.diags(self.potential(spatial_grid, t), 0, format="csr")

    def get_analytical_eigenstates(self, spatial_grid: np.ndarray, n: int, **kwargs) -> np.ndarray:
        """
        Returns the analytical eigenstates of the potential
        """
        raise NotImplementedError

    def get_analytical_eigenvalues(self, n: int, **kwargs) -> np.ndarray:
        """
        Returns the analytical eigenvalues (energies) of the potential
        """
        raise NotImplementedError


class FreeParticle(PotentialBase):
    def __init__(self):
        super().__init__("Free Particle")

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        return np.zeros(len(spatial_grid))


class SquareWell(PotentialBase):
    def __init__(self, left_wall: float, right_wall: float, height: float = np.inf):
        """
        Initializes a square well potential
        :param left_wall: Position of the left wall
        :param right_wall: Position of the right wall
        :param height: Height of the potential outside the well (default = np.inf)
        """
        super().__init__("Infinite Square Well")
        self.left_wall = left_wall
        self.right_wall = right_wall
        self.height = height

    def check_boundaries(self, spatial_grid: np.ndarray):
        """
        Checks if the boundaries of the potential are within the spatial grid
        """
        if self.left_wall < spatial_grid[0]:
            sys.stderr.write(
                f"Warning: left_wall = {self.left_wall} is smaller than the minimum of the spatial grid = {spatial_grid[0]}. "
                f"Setting left_wall = {spatial_grid[0]}\n")
            self.left_wall = spatial_grid[0]
        if self.right_wall > spatial_grid[-1]:
            sys.stderr.write(
                f"Warning: right_wall = {self.right_wall} is larger than the maximum of the spatial grid = {spatial_grid[-1]}. "
                f"Setting right_wall = {spatial_grid[-1]}\n")
            self.right_wall = spatial_grid[-1]

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        self.check_boundaries(spatial_grid)
        potential = np.zeros(len(spatial_grid))
        potential[spatial_grid < self.left_wall] = self.height
        potential[spatial_grid > self.right_wall] = self.height
        return potential

    def get_analytical_eigenstates(self, spatial_grid: np.ndarray, n: int, **kwargs) -> np.ndarray:
        L = self.right_wall - self.left_wall
        if self.height != np.inf:
            raise NotImplementedError("Analytical eigenstates are only available for infinite square wells")
        return np.sqrt(2 / L) * np.sin(n * np.pi / L * (spatial_grid - self.left_wall))

    def get_analytical_eigenvalues(self, n: int, particle_mass: float = 1, **kwargs) -> float:
        L = self.right_wall - self.left_wall
        if self.height != np.inf:
            raise NotImplementedError("Analytical eigenvalues are only available for infinite square wells")
        return (n * np.pi * hbar) ** 2 / (2 * particle_mass * L ** 2)


class HarmonicOscillator(PotentialBase):
    def __init__(self, center: float, omega: float, mass: float):
        super().__init__("Harmonic Oscillator")
        self.center = center
        self.omega = omega
        self.mass = mass

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        return 0.5 * self.mass * np.square(self.omega * (spatial_grid - self.center))

    def get_analytical_eigenstates(self, spatial_grid: np.ndarray, n: int, **kwargs) -> np.ndarray:
        return 1 / np.sqrt(2 ** n * factorial(n)) * (self.mass * self.omega / (np.pi * hbar)) ** 0.25 * \
            np.exp(-self.mass * self.omega * np.square(spatial_grid - self.center) / (2 * hbar)) * \
            hermite(n)(np.sqrt(self.mass * self.omega / hbar) * (spatial_grid - self.center))

    def get_analytical_eigenvalues(self, n: int, **kwargs) -> float:
        return hbar * self.omega * (n + 0.5)


class PotentialBarrier(PotentialBase):
    def __init__(self, center: float, width: float, height: float):
        super().__init__("Potential Barrier")
        self.center = center
        self.width = width
        self.height = height

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        potential = np.zeros(len(spatial_grid))
        potential[np.logical_and(spatial_grid > self.center - self.width / 2,
                                 spatial_grid < self.center + self.width / 2)] = self.height
        return potential

    def get_analytical_eigenstates(self, spatial_grid: np.ndarray, n: int, **kwargs) -> np.ndarray:
        raise NotImplementedError


class SinusoidalHarmonicOscillator(PotentialBase):
    def __init__(self, center: float, omega: float, mass: float, sin_omega: float):
        super().__init__("Sinusoidal Harmonic Oscillator")
        self.center = center
        self.omega = omega
        self.mass = mass
        self.sin_omega = sin_omega

    def potential(self, spatial_grid: np.ndarray, t: float = 0) -> np.ndarray:
        omega = np.sin(self.sin_omega * t) * self.omega
        return 0.5 * self.mass * np.square(omega * (spatial_grid - self.center))
