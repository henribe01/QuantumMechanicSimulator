import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as sp

matplotlib.use('TkAgg')


def gaussian_wave_packet(spatial_grid: np.ndarray, x_0: float, k_0: float,
                         width: float) -> np.ndarray:
    return np.exp(
        -0.5 * np.square(spatial_grid - x_0) / np.square(width)) * np.exp(
        1j * k_0 * spatial_grid)


def get_kinetic_operator(spatial_grid: np.ndarray) -> np.ndarray:
    dx = spatial_grid[1] - spatial_grid[0]
    off_diag = np.ones(len(spatial_grid) - 1)
    diag = -2 * np.ones(len(spatial_grid))
    return sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1]) / dx ** 2


def get_potential_operator(spatial_grid: np.ndarray) -> np.ndarray:
    V = np.zeros(len(spatial_grid))
    return sp.sparse.diags(V)


def get_hamiltonian(spatial_grid: np.ndarray) -> np.ndarray:
    return get_kinetic_operator(spatial_grid) + get_potential_operator(
        spatial_grid)


def crank_nicolson_step(psi: np.ndarray, hamiltonian: np.ndarray,
                        dt: float) -> np.ndarray:
    u1 = sp.sparse.eye(len(psi)) - 0.5j * dt * hamiltonian
    u2 = sp.sparse.eye(len(psi)) + 0.5j * dt * hamiltonian
    u1 = u1.tocsc()
    u2 = u2.tocsc()
    return sp.sparse.linalg.spsolve(u1, u2.dot(psi))


if __name__ == '__main__':
    fig, ax = plt.subplots()
    x_min = 0
    x_max = 10
    x_0 = x_max / 2
    k_0 = 5
    width = 0.5
    spatial_grid = np.linspace(x_min, x_max, 1000)
    n = 10
    psi = gaussian_wave_packet(spatial_grid, x_max / 2, 5, 0.5)
    #psi = np.sqrt(2 / x_max) * np.sin(n * np.pi * spatial_grid / x_max)
    real_line, = ax.plot(spatial_grid, psi.real, label='Real')
    imag_line, = ax.plot(spatial_grid, psi.imag, label='Imag')
    abs_line, = ax.plot(spatial_grid, np.abs(psi), label='Abs')

    hamiltonian = get_hamiltonian(spatial_grid)
    dt = 0.001
    for i in range(100000):
        psi = crank_nicolson_step(psi, hamiltonian, dt)
        real_line.set_ydata(psi.real)
        imag_line.set_ydata(psi.imag)
        abs_line.set_ydata(np.abs(psi))
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)