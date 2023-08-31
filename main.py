import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
from matplotlib.animation import FuncAnimation
from particle import Particle

matplotlib.use('TkAgg')


def gaussian_wave_packet(spatial_grid: np.ndarray, x_0: float, k_0: float,
                         width: float) -> np.ndarray:
    return np.exp(
        -0.5 * np.square(spatial_grid - x_0) / np.square(width)) * np.exp(
        1j * k_0 * spatial_grid)


def infinite_potential_well_eigenfunction(spatial_grid: np.ndarray,
                                          n: int) -> np.ndarray:
    L = x_max - x_min
    return np.sqrt(2 / L) * np.sin(n * np.pi * (spatial_grid - x_min) / L)


def potential(x: np.ndarray) -> np.ndarray:
    return np.zeros(len(x))


if __name__ == '__main__':
    x_min = 0
    x_max = 10
    x_0 = x_max / 2
    k_0 = 5
    width = 0.1
    spatial_grid = np.linspace(x_min, x_max, 1000)

    particle = Particle(spatial_grid, gaussian_wave_packet, x_0=x_0, k_0=k_0, width=width)
    # particle = Particle(spatial_grid, infinite_potential_well_eigenfunction, n=5)
    potential = np.diag(potential(spatial_grid))
    dt = 0.001
    steps = 1000

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 1)
    real_line, = ax.plot([], [], label="Re")
    imag_line, = ax.plot([], [], label="Im")
    abs_line, = ax.plot([], [], label="Abs")
    prob_line, = ax.plot([], [], label="Prob")
    ax.legend()


    # Animation
    def animate(i):
        particle.simulation_step(potential, dt)
        real_line.set_data(particle.spatial_grid, np.real(particle.psi))
        imag_line.set_data(particle.spatial_grid, np.imag(particle.psi))
        abs_line.set_data(particle.spatial_grid, np.abs(particle.psi))
        prob_line.set_data(particle.spatial_grid,
                           particle.get_probability_density())
        return real_line, imag_line, abs_line, prob_line


    anim = FuncAnimation(fig, animate, frames=steps, interval=1, blit=True,
                         repeat=True)
    plt.show()
