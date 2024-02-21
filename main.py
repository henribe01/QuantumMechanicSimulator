import timeit

import matplotlib
from matplotlib.animation import FuncAnimation
from particle import Particle
from potential import *

matplotlib.use('TkAgg')


def gaussian_wave_packet(spatial_grid: np.ndarray, x_0: float, k_0: float,
                         width: float) -> np.ndarray:
    return np.exp(
        -0.5 * np.square(spatial_grid - x_0) / np.square(width)) * np.exp(-
                                                                          1j * k_0 * spatial_grid)


if __name__ == '__main__':
    x_min = 0
    x_max = 20
    center = (x_max - x_min) / 2
    x_0 = 3
    k_0 = 8
    mass = 1
    width = 1
    spatial_grid = np.linspace(x_min, x_max, 5000)
    dx = spatial_grid[1] - spatial_grid[0]

    omega = 3
    particle = Particle(spatial_grid, gaussian_wave_packet, x_0=center, k_0=k_0,
                        width=width)

    potential = HarmonicOscillator(center=center, omega=omega, mass=mass)
    dt = 0.001
    steps = 5000

    # Plot
    fig, ax = plt.subplots()
    ax.set_xlim(x_min + 5, x_max - 5)
    ax.set_ylim(-1, 2)
    real_line, = ax.plot([], [], label="Re")
    imag_line, = ax.plot([], [], label="Im")
    abs_line, = ax.plot([], [], label="Abs")
    prob_line, = ax.plot([], [], label="Prob")
    ax.legend()
    potential_line = potential.plot(spatial_grid, ax, facecolor='gray', edgecolor='black')
    ax.fill_between(particle.spatial_grid, -1, facecolor='gray',
                    edgecolor='black')

    result = np.zeros((steps, len(particle.spatial_grid)), dtype=complex)
    probs = np.zeros((steps, len(particle.spatial_grid)))
    start = timeit.default_timer()
    for i in range(steps):
        potential_current = potential.get_csr_matrix(particle.spatial_grid)
        particle.simulation_step(potential_current, dt)
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
