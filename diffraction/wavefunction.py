#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class WaveFunctionCN:
    def __init__(self, x, y, psi_0, V, dt, hbar=1.0, m=1.0, t0=0.0):
        self.x = np.array(x)
        self.y = np.array(y)
        self.psi = np.array(psi_0, dtype=np.complex128)
        self.V = np.array(V, dtype=np.complex128)
        self.dt = dt
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.hbar = hbar
        self.m = m
        self.t = t0

        if not np.isclose(self.dx, self.dy):
            raise ValueError("WaveFunctionCN assumes dx = dy")

        alpha = dt / (4 * self.dx**2)
        self.alpha = alpha
        self.size_x = len(x)
        self.size_y = len(y)
        dimension = self.size_x * self.size_y

        # Build matrix A 
        N = (self.size_x - 1) * (self.size_y - 1)
        size = 5 * N + 2 * self.size_x + 2 * (self.size_y - 2)
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                idx = i + j * self.size_y

                if i == 0 or i == (self.size_y - 1) or j == 0 or j == (self.size_x - 1):
                    I[k] = idx
                    J[k] = idx
                    K[k] = 1
                    k += 1
                    continue

                Vp = self.V[idx]

                # Center
                I[k] = idx; J[k] = idx; K[k] = 1.0j - 4 * alpha - Vp * dt / 2; k += 1
                # Neighbors
                I[k] = idx; J[k] = idx - 1; K[k] = alpha; k += 1
                I[k] = idx; J[k] = idx + 1; K[k] = alpha; k += 1
                I[k] = idx; J[k] = idx - self.size_y; K[k] = alpha; k += 1
                I[k] = idx; J[k] = idx + self.size_y; K[k] = alpha; k += 1

        self.Mat1 = sparse.coo_matrix((K, (I, J)), shape=(dimension, dimension)).tocsc()

        # Build matrix B 
        I = np.zeros(size)
        J = np.zeros(size)
        K = np.zeros(size, dtype=np.complex128)

        k = 0
        for i in range(0, self.size_y):
            for j in range(0, self.size_x):
                idx = i + j * self.size_y

                if i == 0 or i == (self.size_y - 1) or j == 0 or j == (self.size_x - 1):
                    I[k] = idx; J[k] = idx; K[k] = 0; k += 1
                    continue

                Vp = self.V[idx]

                I[k] = idx; J[k] = idx; K[k] = 1.0j + 4 * alpha + Vp * dt / 2; k += 1
                I[k] = idx; J[k] = idx - 1; K[k] = -alpha; k += 1
                I[k] = idx; J[k] = idx + 1; K[k] = -alpha; k += 1
                I[k] = idx; J[k] = idx - self.size_y; K[k] = -alpha; k += 1
                I[k] = idx; J[k] = idx + self.size_y; K[k] = -alpha; k += 1

        self.Mat2 = sparse.coo_matrix((K, (I, J)), shape=(dimension, dimension)).tocsc()

    def get_prob(self):
        return np.abs(self.psi) ** 2

    def compute_norm(self):
        prob = self.get_prob().reshape(self.size_y, self.size_x)
        return np.trapezoid(np.trapezoid(prob, self.x, axis=1), self.y).real

    def step(self):
        self.psi = spsolve(self.Mat1, self.Mat2.dot(self.psi))
        self.t += self.dt
