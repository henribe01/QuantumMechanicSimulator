import math
import timeit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp
from matplotlib import animation
from matplotlib.animation import FuncAnimation
from particle import Particle
from potential import *
import math

matplotlib.use('TkAgg')


def gaussian_wave_packet(spatial_grid: np.ndarray, x_0: float, k_0: float,
                         width: float) -> np.ndarray:
    return np.exp(
        -0.5 * np.square(spatial_grid - x_0) / np.square(width)) * np.exp(-
                                                                          1j * k_0 * spatial_grid)


def infinite_potential_well_eigenfunction(spatial_grid: np.ndarray,
                                          n: int, left_wall: float,
                                          right_wall: float) -> np.ndarray:
    L = right_wall - left_wall
    inside_well = np.logical_and(spatial_grid >= left_wall,
                                 spatial_grid <= right_wall)
    eigenfunction = np.zeros(len(spatial_grid))
    eigenfunction[inside_well] = np.sqrt(2 / L) * np.sin(
        n * np.pi / L * (spatial_grid[inside_well] - left_wall))
    return eigenfunction


def ground_state_harmonic(spatial_grid: np.ndarray, x_0: float,
                          omega: float) -> np.ndarray:
    return np.power(mass * omega / (np.pi * hbar), 0.25) * np.exp(
        -mass * omega * np.square(spatial_grid - x_0) / (2 * hbar))


def time_dependent_potential(spatial_grid: np.ndarray, t: float) -> np.ndarray:
    return 0.5 * mass * omega ** 2 * np.square(spatial_grid - center) * np.cos(omega * t)


if __name__ == '__main__':
    x_min = 0
    x_max = 20
    center = (x_max - x_min) / 2
    x_0 = 3
    k_0 = 8
    mass = 1
    hbar = 1
    width = 1
    spatial_grid = np.linspace(x_min, x_max, 5000)
    dx = spatial_grid[1] - spatial_grid[0]

    omega = 5
    particle = Particle(spatial_grid, gaussian_wave_packet, x_0=center, k_0=k_0,
                       width=width)
    # particle = Particle(spatial_grid, ground_state_harmonic, x_0=center,
    #                    omega=omega)
    # particle = Particle(spatial_grid, infinite_potential_well_eigenfunction,
    #                    n=5, left_wall=x_min + 2, right_wall=x_max - 2)
    # particle = 1/math.sqrt(10)* (3 * Particle(spatial_grid,
    #                        infinite_potential_well_eigenfunction,
    #                       n=1, left_wall=x_min + 2, right_wall=x_max - 2) -
    #         Particle(spatial_grid, infinite_potential_well_eigenfunction,
    #                 n=3, left_wall=x_min + 2, right_wall=x_max - 2))
    # particle = Particle(spatial_grid, infinite_potential_well_eigenfunction,
    #                     n=1, left_wall=x_min + 2, right_wall=x_max - 2) + \
    #             Particle(spatial_grid, infinite_potential_well_eigenfunction,
    #                         n=2, left_wall=x_min + 2, right_wall=x_max - 2)

    # potential = InfiniteSquareWell(spatial_grid, x_min + 2, x_max - 2)
    # potential = HarmonicOscillator(spatial_grid, center, omega, mass)
    # potential = PotentialBarrier(spatial_grid, center, width=0.5, height=50)
    # potential = FreeParticle(spatial_grid)
    potential = time_dependent_potential
    dt = 0.001
    steps = 5000

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-1, 2)
    real_line, = ax.plot([], [], label="Re")
    imag_line, = ax.plot([], [], label="Im")
    abs_line, = ax.plot([], [], label="Abs")
    prob_line, = ax.plot([], [], label="Prob")
    ax.legend()
    #potential_line = potential.plot(ax, facecolor='gray', edgecolor='black')
    #ax.fill_between(particle.spatial_grid, -1, facecolor='gray',
    #                edgecolor='black')

    result = np.zeros((steps, len(particle.spatial_grid)), dtype=complex)
    probs = np.zeros((steps, len(particle.spatial_grid)))
    start = timeit.default_timer()
    for i in range(steps):
        potential_current = potential(particle.spatial_grid, i * dt)
        potential_current = sp.sparse.diags(potential_current, 0, format="csr")
        potential_next = potential(particle.spatial_grid, (i + 1) * dt)
        potential_next = sp.sparse.diags(potential_next, 0, format="csr")
        particle.simulation_step(potential_current, dt, potential_next)
        result[i] = particle.psi
        probs[i] = particle.get_probability_density()
    stop = timeit.default_timer()
    print('Time: ', stop - start)

    # Animation
    speed = 20


    def animate(i):
        real_line.set_data(particle.spatial_grid, np.real(result[i * speed]))
        imag_line.set_data(particle.spatial_grid, np.imag(result[i * speed]))
        abs_line.set_data(particle.spatial_grid, np.abs(result[i * speed]))
        prob_line.set_data(particle.spatial_grid, probs[i * speed])
        return real_line, imag_line, abs_line, prob_line


    anim = FuncAnimation(fig, animate, frames=steps // speed,
                         interval=1 / 60,
                         repeat=True)
    # print(particle.psi)
    plt.show()
    # anim.save('animation.gif', writer='imagemagick', fps=60)
