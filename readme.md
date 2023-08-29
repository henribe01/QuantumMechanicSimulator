# Quantum Mechanic Simulator

## Description

This is a simulator for quantum mechanics.
It is still work in progress.

## Goals

- [ ] Simulate a free particle
- [ ] Simulate a particle in a potential
    - [ ] Infinite potential well
    - [ ] Finite potential well
    - [ ] Step potential
    - [ ] Tunneling
    - [ ] Harmonic oscillator
    - [ ] Hydrogen atom

## Theory

### Schrödinger's Equation

The Schrödinger's Equation is a partial differential equation that describes
how the quantum state of a physical system evolves over time.
It is defined as

````math
i \hbar \frac{\partial}{\partial t} |\psi(t)\rangle = \hat{H} |\psi(t)\rangle
````

where $\hat{H}$ is the Hamiltonian operator of the system. It is the sum of the
kinetic energy and the potential energy of the system.

````math
\hat{H} = \frac{\hat{p}^2}{2m} + V(\hat{x}) = -\frac{\hbar^2}{2m} \frac{\partial^2}{\partial x^2} + V(x)
````

### Solving the Schrödinger's Equation numerically

To solve the Schrödinger's Equation numerically, we need to discretize the
spatial and temporal dimensions. We can do this by using the finite difference
[[3](#sources)] method. For the second spatial derivative, we get

````math
\frac{\partial^2}{\partial x^2} |\psi(x)\rangle \approx \frac{|\psi(x + \Delta x)\rangle - 2|\psi(x)\rangle + |\psi(x - \Delta x)\rangle}{\Delta x^2}
````

where $\Delta x$ is the spatial step size. We can do the same for the first
temporal derivative.

````math
\frac{\partial}{\partial t} |\psi(t)\rangle \approx \frac{|\psi(t + \Delta t)\rangle - |\psi(t)\rangle}{\Delta t}
````

Substituting these approximations into the Schrödinger's Equation, we get

````math
i \hbar \frac{|\psi(t + \Delta t)\rangle} = - \frac{\hbar^2}{2m} \frac{|\psi(x + \Delta x)\rangle - 2|\psi(x)\rangle + |\psi(x - \Delta x)\rangle}{\Delta x^2} + V(x)|\psi(x)\rangle}{\Delta t}
````

Solving for $|\psi(t + \Delta t)\rangle$, we get

````math
|\psi(x, t + \Delta t)\rangle = |\psi(x, t)\rangle + \frac{i \hbar \Delta t}{2m \Delta x^2} \left(|\psi(x + \Delta x, t)\rangle - 2|\psi(x, t)\rangle + |\psi(x - \Delta x, t)\rangle\right) - \frac{i \Delta t}{\hbar} V(x)|\psi(x, t)\rangle
````

If we define $\alpha = \frac{i \hbar \Delta t}{2m \Delta x^2}$ and rewrite the
equation in matrix form, we get

````math
\begin{pmatrix}
\vdots \\
|\psi(x - \Delta x, t + \Delta t)\rangle \\
|\psi(x, t + \Delta t)\rangle \\
|\psi(x + \Delta x, t + \Delta t)\rangle \\
\vdots
\end{pmatrix}
=
\begin{pmatrix}
    \ddots & \ddots & \ddots & \ddots & \ddots \\
    \ddots & 1 + 2 i \alpha - \frac{i \Delta t}{\hbar} V(x - \Delta x) & i \alpha & 0 & 0 \\
    \ddots & i \alpha & 1 - 2 i \alpha - \frac{i \Delta t}{\hbar} V(x) & i \alpha & 0 \\
    \ddots & 0 & i \alpha & 1 + 2 i \alpha - \frac{i \Delta t}{\hbar} V(x + \Delta x) & \ddots \\
    \ddots & 0 & 0 & \ddots & \ddots
\end{pmatrix}
\begin{pmatrix}
\vdots \\
|\psi(x - \Delta x, t)\rangle \\
|\psi(x, t)\rangle \\
|\psi(x + \Delta x, t)\rangle \\
\vdots
\end{pmatrix} = \hat{A}_t |\psi_t\rangle
````

where $\hat{A}_t$ is the matrix that describes the system at time $t$ and
$|\psi_t\rangle$ is the state of the system at time $t$.

We can solve this equation by using the Crank-Nicolson method [[4](#sources)].
We get

````math
|\psi_{t + \Delta t}\rangle = \frac{1}{2}\left(\hat{A}_t |\psi_t\rangle + \hat{A}_{t + \Delta t} |\psi_{t + \Delta t}\rangle\right)
````

Solving for $|\psi_{t + \Delta t}\rangle$, we get

````math
|\psi_{t + \Delta t}\rangle = \left(\hat{1} - \frac{1}{2}\hat{A}_{t + \Delta t}\right)^{-1} \hat{A}_t |\psi_t\rangle
````

where $\hat{1}$ is the identity matrix. We can use this equation to solve the
Schrödinger's Equation numerically.

## Technologies used

- Python

## Challenges and Solutions

- Finding a starting point for the project:
    - I found two blog posts [[1](#sources)], [[2](#sources)] where the Authors
      implemented something similar. I used this as a starting point for my
      project.

## Installation

## How to use

## Sources

- [1] [Schrödinger's Equation with Python and Numpy](https://maxtyler.net/blog/one-dim-quantum-mechanics)
- [2] [Simulating quantum mechanics with Python](https://ben.land/post/2022/03/09/quantum-mechanics-simulation/)
- [3] [Finite difference](https://en.wikipedia.org/wiki/Finite_difference)
- [4] [Crank-Nicolson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method)