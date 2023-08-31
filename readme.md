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
- [ ] Calculate the eigenvalues and eigenfunctions of a given potential

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