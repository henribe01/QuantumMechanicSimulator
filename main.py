import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy as sp

matplotlib.use('TkAgg')
plt.style.use('seaborn-v0_8')

hbar = 1
mass = 1


def get_time_evolution_matrix(dt: float, dx: float,
                              potential: np.ndarray) -> np.ndarray:
    """
    Returns the time evolution matrix according to the Theory in readme.md
    :param dt: The time step
    :param dx: The space step
    :param potential: The potential as a function of space (same length as spatial_grid)
    :return: The time evolution matrix
    """
    alpha = hbar * dt / (2 * mass * dx ** 2)
    off_diag = np.full(len(potential) - 1, 1j * alpha)
    diag = np.full(len(potential), 1 - 2j * alpha - 1j * dt * potential / hbar)
    return sp.sparse.diags([off_diag, diag, off_diag], [-1, 0, 1], format='csc')


def normalize(state: np.ndarray) -> np.ndarray:
    """
    Normalizes the wave function
    :param state: The wave function
    :return: The normalized wave function
    """
    return state / np.sqrt(np.sum(np.abs(state) ** 2))


def gaussian_wave_packet(spatial_grid: np.ndarray, x_0: float, k_0: float,
                         width: float) -> np.ndarray:
    """
    Creates a wave packet in form off a Gaussian wave packet
    :param spatial_grid: The discretized space
    :param x_0: The center of the wave packet
    :param k_0: The center of the momentum distribution
    :param width: The width of the wave packet
    :return: The wave packet discretized in space
    """
    return normalize(
        np.exp(-np.square(spatial_grid - x_0) / (4 * width ** 2)) * np.exp(
            -1j * k_0 * spatial_grid))


if __name__ == '__main__':
    spatial_grid = np.linspace(-10, 10, 2000)
    dt = 0.001
    dx = spatial_grid[1] - spatial_grid[0]
    x_0 = 0
    k_0 = 0
    width = 0.1
    potential = np.zeros(len(spatial_grid))
    state = gaussian_wave_packet(spatial_grid, x_0, k_0, width)

    steps = 1000
    states = np.zeros((steps, len(state)), dtype=np.complex128)
    states[0] = state
    for i in range(1, steps):
        states[i] = sp.sparse.linalg.spsolve(
            get_time_evolution_matrix(dt, dx, potential), states[i - 1])

    fig, ax = plt.subplots()
    ax.set_ylim(-0.2, 0.2)
    ax.set_xlim(-10, 10)
    real_line, = ax.plot([], [], label='Real')
    imag_line, = ax.plot([], [], label='Imag')
    abs_line, = ax.plot([], [], label='Abs')
    plt.legend(handles=[real_line, imag_line, abs_line])
    speed = 10


    def animate(i):
        real_line.set_data(spatial_grid, np.real(states[i * speed]))
        imag_line.set_data(spatial_grid, np.imag(states[i * speed]))
        abs_line.set_data(spatial_grid, np.abs(states[i * speed]))
        return real_line, imag_line, abs_line


    ani = FuncAnimation(fig, animate, frames=steps // speed, interval=1,
                        blit=True, repeat=True)
    plt.show()
