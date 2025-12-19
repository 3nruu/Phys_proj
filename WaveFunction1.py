#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu


class WaveFunction:
    """
    2D free-particle Schrödinger equation
    solved with Crank–Nicolson scheme.
    
    Physical assumptions:
    - ℏ = 1, m = 1 by default
    - dx = dy (required!)
    - Dirichlet BC: ψ = 0 at boundaries
    """

    def __init__(self, x, y, psi_0, V, dt, hbar=1.0, m=1.0, t0=0.0):

        # --- grids ---
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.Nx = len(x)
        self.Ny = len(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]

        if not np.isclose(self.dx, self.dy):
            raise ValueError("This implementation assumes dx = dy")

        self.dt = dt
        self.hbar = hbar
        self.m = m
        self.t = t0

        # --- wavefunction ---
        self.psi = np.asarray(psi_0, dtype=np.complex128)
        self.V = np.asarray(V, dtype=np.complex128)

        N = self.Nx * self.Ny

        # --- Crank–Nicolson coefficient ---
        dx2 = self.dx * self.dx
        alpha = self.dt / (4.0 * dx2)
        self.alpha = alpha

        # --- sparse matrix assembly ---
        rows = []
        cols = []
        data_A = []
        data_B = []

        def idx(i, j):
            return i + j * self.Ny

        for j in range(self.Nx):
            for i in range(self.Ny):

                p = idx(i, j)

                # --- Dirichlet boundaries ---
                if i == 0 or i == self.Ny - 1 or j == 0 or j == self.Nx - 1:
                    rows.append(p)
                    cols.append(p)
                    data_A.append(1.0)
                    data_B.append(0.0)
                    continue

                Vp = self.V[p]

                # --- central point ---
                rows.append(p)
                cols.append(p)
                data_A.append(1.0j - 4 * alpha - Vp * self.dt / 2)
                data_B.append(1.0j + 4 * alpha + Vp * self.dt / 2)

                # --- neighbors ---
                for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                    q = idx(i + di, j + dj)
                    rows.append(p)
                    cols.append(q)
                    data_A.append(alpha)
                    data_B.append(-alpha)

        self.Mat1 = sparse.coo_matrix(
            (data_A, (rows, cols)), shape=(N, N)
        ).tocsc()

        self.Mat2 = sparse.coo_matrix(
            (data_B, (rows, cols)), shape=(N, N)
        ).tocsr()

        # --- LU factorization (KEY SPEEDUP) ---
        self.lu = splu(self.Mat1)

    # --------------------------------------------------

    def step(self):
        """Perform one Crank–Nicolson time step"""
        rhs = self.Mat2 @ self.psi
        self.psi = self.lu.solve(rhs)
        self.t += self.dt

    # --------------------------------------------------

    def get_prob(self):
        """Return |ψ|²"""
        return np.abs(self.psi) ** 2

    # --------------------------------------------------

    def compute_norm(self):
        """Compute ∫|ψ|² dx dy"""
        prob = self.get_prob().reshape(self.Ny, self.Nx)
        return np.trapz(
            np.trapz(prob, self.x, axis=1),
            self.y
        ).real
