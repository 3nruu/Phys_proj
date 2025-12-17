#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.gridspec as gridspec
from Wave_Function import WaveFunction
from scipy.interpolate import griddata
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

plt.rcParams.update({'font.size': 7})

#####################################
#       1) Create the system        #
#####################################

# Parameters
dt = 0.005
hbar = 1.0
m = 1.0

# Spatial grid
x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08
x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

# Interpolation grid for smoother 3D surface
ni = 150
xi = np.linspace(x.min(), x.max(), ni)
yi = np.linspace(y.min(), y.max(), ni)
xig, yig = np.meshgrid(xi, yi)

# Meshgrid for plotting (standard 'xy' indexing)
xx, yy = np.meshgrid(x, y)  # shape (Ny, Nx)

# === Initial wave function just after the double slit ===
# Create mask on the full grid (same shape as xx, yy)
mask = np.zeros_like(xx, dtype=bool)

# Slit parameters
x_slit = 0.0
slit_width = 0.2
y_center1 = -2.0
y_center2 = 2.0
slit_height = 1.5

# Define slits in the (x, y) plane
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y_center1 - slit_height/2) & (yy <= y_center1 + slit_height/2)] = True
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y_center2 - slit_height/2) & (yy <= y_center2 + slit_height/2)] = True

# Initial wave: plane wave e^{i kx x}
kx0 = 30.0
ky0 = 0.0
psi_0 = np.zeros((Nx, Ny), dtype=complex)
# Fill psi_0 in (i, j) = (x-index, y-index)
for i in range(Nx):
    for j in range(Ny):
        if mask[j, i]:  # note: mask[j,i] because xx,yy are (Ny,Nx)
            psi_0[i, j] = np.exp(1j * (kx0 * x[i] + ky0 * y[j]))

# Flatten in the order expected by WaveFunction: psi[i + j * Ny]
psi_0_flat = psi_0.flatten()
V_xy = np.zeros(Nx * Ny)

# Normalize
norm = np.sqrt(np.sum(np.abs(psi_0_flat)**2) * dx * dy)
psi_0_flat /= norm

# Initialize solver
S = WaveFunction(x=x, y=y, psi_0=psi_0_flat, V=V_xy, dt=dt, hbar=hbar, m=m)

######################################
#       2) Setting up the plot       #
######################################

nb_frame = 250
nbr_level = 150

fig = plt.figure(figsize=(10, 7))
gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1], height_ratios=[1, 1])
ax1 = plt.subplot(gs[:, 0])        # 2D density
ax2 = plt.subplot(gs[0, 1], projection='3d')
ax3 = plt.subplot(gs[1, 1])

div1 = make_axes_locatable(ax1)
cax1 = div1.append_axes('right', '3%', '3%')

ax1.set_aspect('equal')
ax1.set_xlim([x_min, x_max])
ax1.set_ylim([y_min, y_max])
ax1.set_xlabel(r"$x\ (a_0)$", fontsize=12)
ax1.set_ylabel(r"$y\ (a_0)$", fontsize=12)

ax2.view_init(elev=30., azim=-40.)
ax2.set_box_aspect([1, 1, 0.3])
ax2.set_xlabel(r"$x$", fontsize=9)
ax2.set_ylabel(r"$y$", fontsize=9)
ax2.set_zticks([])

ax3.set_xlim([y_min, y_max])
ax3.set_xlabel(r"$y\ (a_0)$", fontsize=9)
ax3.set_ylabel(r"$|\psi|^2$", fontsize=9)

# Initial state
prob = S.get_prob().reshape(Nx, Ny)  # (Nx, Ny)
z_plot = prob.T                      # (Ny, Nx) → matches xx, yy

level = np.linspace(0, z_plot.max(), nbr_level)
cset = ax1.contourf(xx, yy, z_plot, levels=level, cmap='jet')

zi = griddata((xx.ravel(), yy.ravel()), z_plot.ravel(), (xig, yig), method='cubic')
ax2.plot_surface(xig, yig, zi, cmap='jet', rcount=ni//2, ccount=ni//2, alpha=0.9)

x_screen = 8.0
kx = np.argmin(np.abs(x - x_screen))
ax3.plot(y, prob[kx, :], 'b')
ax1.axvline(x[kx], color='orange', linestyle='--', linewidth=1)

cbar1 = fig.colorbar(cset, cax=cax1)

# Storage
t_vec = np.arange(nb_frame) * dt
slice_data = np.zeros((nb_frame, Ny))

######################################
#       3) Animation function        #
######################################

def animate(i):
    S.step()
    prob = S.get_prob().reshape(Nx, Ny)  # (Nx, Ny)
    z_plot = prob.T                      # (Ny, Nx)
    slice_data[i, :] = prob[kx, :]

    ax1.clear()
    ax2.clear()
    ax3.clear()

    level = np.linspace(0, z_plot.max(), nbr_level)
    cset = ax1.contourf(xx, yy, z_plot, levels=level, cmap='jet')
    ax1.set_aspect('equal')
    ax1.set_xlim([x_min, x_max])
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlabel(r"$x\ (a_0)$", fontsize=12)
    ax1.set_ylabel(r"$y\ (a_0)$", fontsize=12)
    ax1.axvline(x[kx], color='orange', linestyle='--', linewidth=1)
    ax1.text(0.02, 0.92, f"$t = {S.t:.2f}$", color='white', transform=ax1.transAxes, fontsize=10)

    zi = griddata((xx.ravel(), yy.ravel()), z_plot.ravel(), (xig, yig), method='cubic')
    ax2.plot_surface(xig, yig, zi, cmap='jet', rcount=ni//2, ccount=ni//2, alpha=0.9)
    ax2.set_box_aspect([1, 1, 0.3])
    ax2.set_xlim([x_min, x_max])
    ax2.set_ylim([y_min, y_max])
    ax2.set_zlim([0, zi.max() if np.any(zi) else 1])
    ax2.set_xlabel(r"$x$", fontsize=9)
    ax2.set_ylabel(r"$y$", fontsize=9)
    ax2.set_zticks([])

    ax3.plot(y, prob[kx, :], 'b')
    ax3.set_xlim([y_min, y_max])
    ax3.set_ylim([0, 0.15])
    ax3.set_xlabel(r"$y\ (a_0)$", fontsize=9)
    ax3.set_ylabel(r"$|\psi|^2$", fontsize=9)

    print(f"Frame {i+1}/{nb_frame}")
    # НЕ возвращаем ничего при blit=False

######################################
#       4) Run and save              #
######################################

anim = animation.FuncAnimation(fig, animate, frames=nb_frame, interval=100, blit=False)

# Save as GIF
anim.save('diffraction_2slit.gif', writer='pillow', fps=10)

# Save slice data
with open("diffraction_slice.pkl", 'wb') as f:
    pickle.dump(slice_data, f)

plt.show()