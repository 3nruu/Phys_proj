#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import pickle
from Wave_Function import WaveFunction
from FresnelDiffraction import FresnelDiffraction

# -----------------------------
# Parameters
# -----------------------------
dt = 0.005
hbar = 1.0
m = 1.0
kx0 = 30.0
ky0 = 0.0
wavelength = 2 * np.pi / kx0

# Spatial grid
x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08

x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

xx, yy = np.meshgrid(x, y)

# -----------------------------
# Double slit
# -----------------------------
mask = np.zeros_like(xx, dtype=bool)
x_slit, slit_width = 0.0, 0.2
y1, y2, h = -2.0, 2.0, 1.5

mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y1 - h/2) & (yy <= y1 + h/2)] = True
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y2 - h/2) & (yy <= y2 + h/2)] = True

# -----------------------------
# Initial QM wavefunction
# -----------------------------
psi0 = np.zeros((Nx, Ny), dtype=complex)
for i in range(Nx):
    for j in range(Ny):
        if mask[j, i]:
            psi0[i, j] = np.exp(1j * (kx0 * x[i] + ky0 * y[j]))

psi0_flat = psi0.flatten()
psi0_flat /= np.sqrt(np.sum(np.abs(psi0_flat)**2) * dx * dy)

V = np.zeros(Nx * Ny)

S = WaveFunction(x=x, y=y, psi_0=psi0_flat, V=V,
                 dt=dt, hbar=hbar, m=m)

# -----------------------------
# Fresnel–Kirchhoff
# -----------------------------
U0 = np.zeros((Ny, Nx), dtype=complex)
U0[mask] = np.exp(1j * kx0 * xx[mask])

FD = FresnelDiffraction(x=x, y=y, U0=U0,
                        wavelength=wavelength)

# -----------------------------
# Animation setup
# -----------------------------
nb_frame = 200
z0 = 5.0   # минимальное расстояние применимости Френеля

fig, ax = plt.subplots(figsize=(7, 4))
line_qm, = ax.plot([], [], lw=2, label="QM")
line_f,  = ax.plot([], [], "--", lw=2, label="Fresnel")

ax.set_xlim(y_min, y_max)
ax.set_ylim(0, 0.15)
ax.set_xlabel(r"$y$")
ax.set_ylabel("Intensity")
ax.legend()
ax.grid(alpha=0.3)

# -----------------------------
# Storage
# -----------------------------
I_qm_store = np.zeros((nb_frame, Ny))
I_f_store = np.zeros((nb_frame, Ny))

# -----------------------------
# Animation function
# -----------------------------
def animate(i):
    S.step()
    t = S.t

    # связь времени и расстояния (ħ = m = 1)
    z = z0 + kx0 * t

    # ---- Quantum mechanics ----
    prob = S.get_prob().reshape(Nx, Ny)


    # ---- Fresnel diffraction ----
    U = FD.propagate(z)
  

    # QM
    I_qm = np.trapezoid(prob, x, axis=0)
    I_qm /= np.trapezoid(I_qm, y)

# Fresnel
    U = FD.propagate(z)
    I_f = np.trapezoid(np.abs(U)**2, x, axis=1)
    I_f /= np.trapezoid(I_f, y)

    # store
    I_qm_store[i] = I_qm
    I_f_store[i] = I_f

    # update plot
    line_qm.set_data(y, I_qm)
    line_f.set_data(y, I_f)

    ax.set_title(rf"$t = {t:.2f},\quad z = {z:.2f}$")

    return line_qm, line_f

# -----------------------------
# Run animation
# -----------------------------
anim = animation.FuncAnimation(
    fig, animate, frames=nb_frame, interval=100, blit=True
)

anim.save("screen_intensity_comparison.gif",
          writer="pillow", fps=10)

# -----------------------------
# Save data
# -----------------------------
with open("screen_intensity_data.pkl", "wb") as f:
    pickle.dump({
        "y": y,
        "t": np.arange(nb_frame) * dt,
        "I_qm": I_qm_store,
        "I_fresnel": I_f_store
    }, f)

plt.show()
