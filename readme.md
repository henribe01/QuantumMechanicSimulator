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
i \hbar \frac{|\psi(t + \Delta t)\rangle = - \frac{\hbar^2}{2m} \frac{|\psi(x + \Delta x)\rangle - 2|\psi(x)\rangle + |\psi(x - \Delta x)\rangle}{\Delta x^2} + V(x)|\psi(x)\rangle}{\Delta t}
````

Solving for $|\psi(t + \Delta t)\rangle$, we get

````math
|\psi(t + \Delta t)\rangle = \left(1 + \frac{i \hbar \Delta t}{2m \Delta x^2}\right)^{-1} \left(2|\psi(x)\rangle - \left(1 - \frac{i \hbar \Delta t}{2m \Delta x^2}\right) \left(|\psi(x + \Delta x)\rangle + |\psi(x - \Delta x)\rangle\right)\right)
````

This equation can then be solved using the Crank-Nicolson
method [[4](#sources)].

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