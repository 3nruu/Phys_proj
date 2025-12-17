#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.gridspec as gridspec
from Wave_Function import WaveFunction
from FresnelDiffraction import FresnelDiffraction
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle


# -------------------------------------------------
# Основной код
# -------------------------------------------------

plt.rcParams.update({'font.size': 7})

#####################################
#       1) Create the system        #
#####################################

# Parameters
dt = 0.005
hbar = 1.0
m = 1.0
kx0 = 30.0
ky0 = 0.0
wavelength = 2 * np.pi / kx0  # длина волны де Бройля

# Spatial grid
x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08
x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

# Plotting grid
ni = 150
xi = np.linspace(x.min(), x.max(), ni)
yi = np.linspace(y.min(), y.max(), ni)
xig, yig = np.meshgrid(xi, yi)

xx, yy = np.meshgrid(x, y)  # (Ny, Nx)

# === Initial condition: double slit ===
mask = np.zeros_like(xx, dtype=bool)
x_slit, slit_width = 0.0, 0.2
y1, y2, h = -2.0, 2.0, 1.5
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y1 - h/2) & (yy <= y1 + h/2)] = True
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y2 - h/2) & (yy <= y2 + h/2)] = True

# Initial wave function (Nx, Ny)
psi_0 = np.zeros((Nx, Ny), dtype=complex)
for i in range(Nx):
    for j in range(Ny):
        if mask[j, i]:
            psi_0[i, j] = np.exp(1j * (kx0 * x[i] + ky0 * y[j]))

# Flatten for WaveFunction (i + j * Ny)
psi_0_flat = psi_0.flatten()
V_xy = np.zeros(Nx * Ny)
norm = np.sqrt(np.sum(np.abs(psi_0_flat)**2) * dx * dy)
psi_0_flat /= norm

# Initial field for Fresnel (Ny, Nx)
U0 = np.zeros((Ny, Nx), dtype=complex)
U0[mask] = np.exp(1j * kx0 * xx[mask])  # xx is (Ny, Nx)

# Initialize solvers
S = WaveFunction(x=x, y=y, psi_0=psi_0_flat, V=V_xy, dt=dt, hbar=hbar, m=m)
FD = FresnelDiffraction(x=x, y=y, U0=U0, wavelength=wavelength)


######################################
#       2) Setting up the plot       #
######################################

nb_frame = 200
nbr_level = 150
x_screen = 7.0
kx = np.argmin(np.abs(x - x_screen))

fig = plt.figure(figsize=(11, 8))
gs = gridspec.GridSpec(3, 3, width_ratios=[2, 1, 1], height_ratios=[1, 0.05, 1])
ax1 = plt.subplot(gs[:, 0])          # 2D QM
ax2 = plt.subplot(gs[0, 1], projection='3d')  # 3D QM
ax3 = plt.subplot(gs[2, 1])          # 1D QM slice
ax4 = plt.subplot(gs[2, 2])          # 1D Fresnel slice

div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '2%', '2%')

# Aesthetics
ax1.set_aspect('equal')
ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
ax1.set_xlabel(r"$x\ (a_0)$", fontsize=12)
ax1.set_ylabel(r"$y\ (a_0)$", fontsize=12)

ax2.view_init(elev=30., azim=-40.)
ax2.set_box_aspect([1, 1, 0.3])
ax2.set_xlabel(r"$x$", fontsize=8)
ax2.set_ylabel(r"$y$", fontsize=8)
ax2.set_zticks([])

for ax in [ax3, ax4]:
    ax.set_xlim([y_min, y_max])
    ax.set_xlabel(r"$y\ (a_0)$", fontsize=9)
    ax.set_ylabel(r"$|\psi|^2$", fontsize=9)

ax3.set_title("Квантовая модель", fontsize=10)
ax4.set_title("Теория Френеля", fontsize=10)

# Initial state
prob = S.get_prob().reshape(Nx, Ny)
z_plot = prob.T
level = np.linspace(0, z_plot.max(), nbr_level)
cset = ax1.contourf(xx, yy, z_plot, levels=level, cmap='jet')

zi = griddata((xx.ravel(), yy.ravel()), z_plot.ravel(), (xig, yig), method='cubic')
ax2.plot_surface(xig, yig, zi, cmap='jet', rcount=ni//2, ccount=ni//2, alpha=0.9)

# Initial slices
ax3.plot(y, prob[kx, :], 'b', label='QM')
y_fres, I_fres = FD.get_slice(z=kx0 * S.t, x_obs=x_screen)
ax4.plot(y_fres, I_fres, 'r', label='Френель')

ax1.axvline(x[kx], color='orange', linestyle='--', linewidth=1)

cbar1 = fig.colorbar(cset, cax=cax1)
plt.tight_layout()

# Storage
slice_data_qm = np.zeros((nb_frame, Ny))
slice_data_fres = np.zeros((nb_frame, Ny))

######################################
#       3) Animation function        #
######################################

def animate(i):
    S.step()
    t = S.t
    z0 = 5.0              # минимальное расстояние применимости Френеля
    z = z0 + kx0 * t

    # Quantum
    prob = S.get_prob().reshape(Nx, Ny)
    z_plot = prob.T
    slice_data_qm[i, :] = prob[kx, :]

    # Fresnel
    z_fixed = x_screen - x_slit  # например, 8.0 - 0.0 = 8.0
    y_fres, I_fres = FD.get_slice(z=z, x_obs=x_screen)
    slice_data_fres[i, :] = I_fres

    # Clear
    ax1.clear()
    ax2.clear()
    ax3.clear()
    ax4.clear()

    # 2D QM
    level = np.linspace(0, z_plot.max(), nbr_level)
    cset = ax1.contourf(xx, yy, z_plot, levels=level, cmap='jet')
    ax1.set_aspect('equal')
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlabel(r"$x\ (a_0)$", fontsize=12)
    ax1.set_ylabel(r"$y\ (a_0)$", fontsize=12)
    ax1.axvline(x[kx], color='orange', linestyle='--', linewidth=1)
    ax1.text(0.02, 0.92, f"$t = {t:.2f}$", color='white', transform=ax1.transAxes, fontsize=10)

    # 3D QM
    zi = griddata((xx.ravel(), yy.ravel()), z_plot.ravel(), (xig, yig), method='cubic')
    ax2.plot_surface(xig, yig, zi, cmap='jet', rcount=ni//2, ccount=ni//2, alpha=0.9)
    ax2.set_box_aspect([1, 1, 0.3])
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([0, zi.max() if np.any(zi) else 1])
    ax2.set_xlabel(r"$x$", fontsize=8)
    ax2.set_ylabel(r"$y$", fontsize=8)
    ax2.set_zticks([])

    # Slices
    ax3.plot(y, prob[kx, :], 'b')
    ax4.plot(y_fres, I_fres, 'r')

    for ax in [ax3, ax4]:
        ax.set_xlim([y_min, y_max])
        ax.set_ylim([0, 0.15])
        ax.set_xlabel(r"$y\ (a_0)$", fontsize=9)
        ax.set_ylabel(r"$|\psi|^2$", fontsize=9)

    ax3.set_title("Квантовая модель", fontsize=10)
    ax4.set_title("Теория Френеля", fontsize=10)

    print(f"Frame {i+1}/{nb_frame}")

######################################
#       4) Run and save              #
######################################

anim = animation.FuncAnimation(fig, animate, frames=nb_frame, interval=100, blit=False)

# Save
anim.save('diffraction_2slit_comparison.gif', writer='pillow', fps=10)

with open("diffraction_comparison.pkl", 'wb') as f:
    pickle.dump({
        'y': y,
        'x_screen': x_screen,
        'qm': slice_data_qm,
        'fresnel': slice_data_fres,
        't_vec': np.arange(nb_frame) * dt
    }, f)

plt.show()