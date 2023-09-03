# Quantum Mechanic Simulator

## Description

This is a simulator for quantum mechanics.
It is still work in progress.

## Goals

- [x] Simulate a free particle
- [ ] Simulate a particle in a potential
    - [x] Infinite potential well
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
- Numpy
- Matplotlib (for plotting/animate the results)
- Scipy (for efficient matrix operations and solving the linear system)

## Challenges and Solutions

- Finding a starting point for the project:
    - I found two blog posts [[1](#sources)], [[2](#sources)] where the Authors
      implemented something similar. I used this as a starting point for my
      project.
- The runtime to calculate the results was too long (30s for 1000 time steps):
    1. First Problem was, that I transformed a Numpy array into a Scipy Sparse
       CSC matrix in every iteration. This was the biggest bottleneck. I
       solved this by transforming the Numpy array into a Scipy Sparse CSC
       matrix once and then using this matrix in every iteration.
       This reduced the runtime from 30s to 2.5s. (Improvement-factor: 12)
    2. The second problem was, that I used the 'sp.sparse.linalg.solve' function
       which is efficient for sparse matrices, but there is a more efficient
       algorithm for tridiagonal matrices. I transformed the matrix into a
       band matrix and solved the linear system with the '
       sp.linalg.solve_banded'.
       This reduced the runtime from 2.5s to 1s. (Improvement-factor: 2.5)

## Installation

## How to use

## Sources

- [1] [Schrödinger's Equation with Python and Numpy](https://maxtyler.net/blog/one-dim-quantum-mechanics)
- [2] [Simulating quantum mechanics with Python](https://ben.land/post/2022/03/09/quantum-mechanics-simulation/)
- [3] [Finite difference](https://en.wikipedia.org/wiki/Finite_difference)
- [4] [Crank-Nicolson method](https://en.wikipedia.org/wiki/Crank%E2%80%93Nicolson_method)
- [5] [Numerical Solutions to the Time-dependent Schrödinger Equation](http://staff.ustc.edu.cn/~zqj/posts/Numerical_TDSE/)
- 