#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize a single parameter case with:
1. Comparison maps (QM, Fresnel, Difference)
2. Animated screen intensity comparison
"""

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import matplotlib.gridspec as gridspec
from diffraction import WaveFunctionCN, Fresnel1D
from diffraction.visualization_utils import compute_smart_color_limits, normalize_for_display
import pickle

plt.rcParams.update({'font.size': 8})

# -------------------------------------------------
# Parameters - EDIT THESE
# -------------------------------------------------
kx0 = 25.0
d = 1.5  # slit separation
h = 1.0  # hole width

# Fixed parameters
dt = 0.005
hbar = 1.0
m = 1.0

x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08

x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

# Double slit parameters
x_slit, slit_width = 0.0, 0.2

# -------------------------------------------------
# Calculate derived parameters
# -------------------------------------------------
wavelength = 2 * np.pi / kx0
x_screen_fixed = (d**2) / wavelength  # Screen at d²/λ

y1 = -d / 2
y2 = d / 2

print(f"Parameters:")
print(f"  kx0 = {kx0}, d = {d}, h = {h}")
print(f"  wavelength = {wavelength:.3f}")
print(f"  screen at x = {x_screen_fixed:.2f} (d²/λ)")
print(f"  Fresnel number F = 1")
print("="*60)

# -------------------------------------------------
# Setup simulation
# -------------------------------------------------
xx, yy = np.meshgrid(x, y, indexing='ij')

# Create mask for double slit
mask = np.zeros((Nx, Ny), dtype=bool)
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y1 - h/2) & (yy <= y1 + h/2)] = True
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y2 - h/2) & (yy <= y2 + h/2)] = True

# Initial wave function (QM)
psi0 = np.zeros((Nx, Ny), dtype=complex)
psi0[mask] = np.exp(1j * kx0 * xx[mask])

# Normalize
psi0_flat = psi0.flatten()
psi0_flat /= np.sqrt(np.sum(np.abs(psi0_flat)**2) * dx * dy)

# Create WaveFunctionCN object
S = WaveFunctionCN(x=x, y=y, psi_0=psi0_flat, V=np.zeros(Nx*Ny), 
                   dt=dt, hbar=hbar, m=m)

# Fresnel initial field
U0 = np.zeros_like(y, dtype=complex)
U0[(y >= y1 - h/2) & (y <= y1 + h/2)] = 1.0
U0[(y >= y2 - h/2) & (y <= y2 + h/2)] = 1.0
U0 /= np.sqrt(np.trapezoid(np.abs(U0)**2, y))

FD = Fresnel1D(y=y, U0=U0, wavelength=wavelength)

# Screen index
ix_screen = np.argmin(np.abs(x - x_screen_fixed))

# -------------------------------------------------
# Evolve to screen
# -------------------------------------------------
print(f"Evolving to screen at x={x_screen_fixed:.2f}...")
t_target = x_screen_fixed / kx0
n_steps = int(t_target / dt)

for _ in range(n_steps):
    S.step()

print(f"Evolved {n_steps} steps to t={S.t:.3f}")

# -------------------------------------------------
# Compute full spatial maps
# -------------------------------------------------
print("Computing spatial maps...")

prob_final = S.get_prob().reshape(Nx, Ny)

qm_full_map = np.zeros((Nx, Ny))
fresnel_full_map = np.zeros((Nx, Ny))

for ix, x_val in enumerate(x):
    if x_val <= x_slit:
        continue
    
    # QM map
    I_qm = prob_final[ix, :]
    norm_qm = np.trapezoid(I_qm, y)
    if norm_qm > 0:
        I_qm /= norm_qm
    qm_full_map[ix, :] = I_qm
    
    # Fresnel map
    z = x_val - x_slit
    if z > 0:
        I_fr = FD.intensity(z)
        norm_fr = np.trapezoid(I_fr, y)
        if norm_fr > 0:
            I_fr /= norm_fr
        fresnel_full_map[ix, :] = I_fr

# Difference map
diff_map = np.abs(qm_full_map - fresnel_full_map)

# -------------------------------------------------
# Plot 1: Comparison Maps
# -------------------------------------------------
print("Creating comparison maps...")

fig1, axs = plt.subplots(1, 3, figsize=(14, 4))

im0 = axs[0].imshow(
    qm_full_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='jet'
)
axs[0].axvline(x_screen_fixed, color='white', ls='--', lw=2, label=f'screen')
axs[0].set_title(f"QM map (k={kx0}, d={d}, h={h})")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(
    fresnel_full_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='inferno'
)
axs[1].axvline(x_screen_fixed, color='white', ls='--', lw=2, label=f'screen')
axs[1].set_title("Fresnel map (normalized)")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(
    diff_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='magma'
)
axs[2].axvline(x_screen_fixed, color='white', ls='--', lw=2, label=f'screen')
axs[2].set_title("|QM − Fresnel|")
axs[2].set_xlabel("x")
axs[2].set_ylabel("y")
plt.colorbar(im2, ax=axs[2])

plt.tight_layout()
plt.savefig(f'comparison_maps_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.png', dpi=150, bbox_inches='tight')
print(f"Saved: comparison_maps_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.png")

# -------------------------------------------------
# Plot 2: Screen intensity at fixed position
# -------------------------------------------------
print("Creating screen intensity plot...")

qm_slice = qm_full_map[ix_screen, :]
fresnel_slice = fresnel_full_map[ix_screen, :]

# Get unnormalized for actual intensity comparison
qm_slice_raw = prob_final[ix_screen, :]
fresnel_slice_raw = FD.intensity(x_screen_fixed - x_slit)

fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Normalized comparison
ax1.plot(y, qm_slice, 'b-', lw=2, label='QM', alpha=0.8)
ax1.plot(y, fresnel_slice, 'r--', lw=2, label='Fresnel', alpha=0.8)
ax1.set_xlabel('y')
ax1.set_ylabel('Normalized Intensity')
ax1.set_title(f'Screen at x={x_screen_fixed:.2f} (normalized)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(y_min, y_max)

# Raw intensity comparison
ax2.plot(y, qm_slice_raw, 'b-', lw=2, label='QM', alpha=0.8)
ax2.plot(y, fresnel_slice_raw, 'r--', lw=2, label='Fresnel', alpha=0.8)
ax2.set_xlabel('y')
ax2.set_ylabel('Intensity')
ax2.set_title(f'Screen at x={x_screen_fixed:.2f} (raw)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim(y_min, y_max)

plt.tight_layout()
plt.savefig(f'screen_intensity_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.png', dpi=150, bbox_inches='tight')
print(f"Saved: screen_intensity_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.png")

# -------------------------------------------------
# Animation: Evolving screen comparison
# -------------------------------------------------
print("Creating animated screen comparison...")

# Reset simulation
psi0_flat = psi0.flatten()
psi0_flat /= np.sqrt(np.sum(np.abs(psi0_flat)**2) * dx * dy)
S = WaveFunctionCN(x=x, y=y, psi_0=psi0_flat, V=np.zeros(Nx*Ny), 
                   dt=dt, hbar=hbar, m=m)

nb_frames = min(200, n_steps)  # Limit animation frames
frame_skip = max(1, n_steps // nb_frames)

fig3, ax = plt.subplots(figsize=(8, 5))

# Initial plot
prob0 = S.get_prob().reshape(Nx, Ny)
qm_line, = ax.plot(y, prob0[ix_screen, :], 'b-', lw=2, label='QM', alpha=0.8)
fresnel_line, = ax.plot(y, fresnel_slice_raw, 'r--', lw=2, label='Fresnel', alpha=0.8)

ax.set_xlabel('y', fontsize=12)
ax.set_ylabel('Intensity', fontsize=12)
ax.set_title(f'Screen at x={x_screen_fixed:.2f}, t=0.00', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(y_min, y_max)
ax.set_ylim(0, max(prob0[ix_screen, :].max(), fresnel_slice_raw.max()) * 1.2)

def animate_screen(i):
    # Step simulation
    for _ in range(frame_skip):
        S.step()
    
    t = S.t
    prob = S.get_prob().reshape(Nx, Ny)
    qm_intensity = prob[ix_screen, :]
    
    # Update lines
    qm_line.set_ydata(qm_intensity)
    
    # Update title
    ax.set_title(f'Screen at x={x_screen_fixed:.2f}, t={t:.2f}', fontsize=12)
    
    # Dynamic y-limit
    max_val = max(qm_intensity.max(), fresnel_slice_raw.max())
    ax.set_ylim(0, max_val * 1.2)
    
    if (i+1) % 20 == 0:
        print(f"  Frame {i+1}/{nb_frames}")
    
    return qm_line,

print(f"Animating {nb_frames} frames...")
anim = animation.FuncAnimation(fig3, animate_screen, frames=nb_frames, 
                               interval=50, blit=True)
anim.save(f'screen_animation_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.gif', 
          writer='pillow', fps=20)
print(f"Saved: screen_animation_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.gif")

# -------------------------------------------------
# Calculate errors
# -------------------------------------------------
error_L2 = np.sqrt(np.trapezoid((qm_slice - fresnel_slice)**2, y))
print("\n" + "="*60)
print(f"L2 error at screen: {error_L2:.6f}")
print("="*60)

# -------------------------------------------------
# Plot 3: Spacetime comparison GIF
# -------------------------------------------------
print("Creating spacetime comparison GIF...")

# Reset simulation
psi0_flat = psi0.flatten()
psi0_flat /= np.sqrt(np.sum(np.abs(psi0_flat)**2) * dx * dy)
S = WaveFunctionCN(x=x, y=y, psi_0=psi0_flat, V=np.zeros(Nx*Ny), 
                   dt=dt, hbar=hbar, m=m)

nb_frames_gif = min(200, n_steps)
frame_skip = max(1, n_steps // nb_frames_gif)

# Use the already computed normalized QM map from comparison maps section
# (This is the static normalized map used in the comparison)
qm_normalized_static = qm_full_map.copy()  # Already computed earlier

# Setup figure with 3 panels
fig_gif = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1])
ax_qm = plt.subplot(gs[0])
ax_fr = plt.subplot(gs[1])
ax_scr = plt.subplot(gs[2])

# Initial QM state
prob0 = S.get_prob().reshape(Nx, Ny)
im_qm = ax_qm.contourf(xx, yy, prob0, levels=100, cmap='jet')
ax_qm.axvline(x_screen_fixed, color='white', ls='--', lw=1)
ax_qm.set_title('Schrödinger |ψ|², t=0.00')
ax_qm.set_xlabel('x')
ax_qm.set_ylabel('y')
ax_qm.set_xlim(x_min, x_max)
ax_qm.set_ylim(y_min, y_max)

# Draw double slit barriers
ax_qm.vlines([x_slit, x_slit + slit_width], y1 - h/2, y1 + h/2, colors='white', linewidth=2)
ax_qm.vlines([x_slit, x_slit + slit_width], y2 - h/2, y2 + h/2, colors='white', linewidth=2)
ax_qm.hlines([y1 - h/2, y1 + h/2], x_slit, x_slit + slit_width, colors='white', linewidth=2)
ax_qm.hlines([y2 - h/2, y2 + h/2], x_slit, x_slit + slit_width, colors='white', linewidth=2)




# Static normalized QM map
ax_fr.imshow(
    fresnel_full_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='inferno'
)
ax_fr.axvline(x_screen_fixed, color='white', ls='--', lw=1)
ax_fr.set_title('Fresnel map (normalized)')
ax_fr.set_xlabel('x')
ax_fr.set_ylabel('y')




# Screen comparison
qm_slice_0 = prob0[ix_screen, :]
fresnel_slice_0 = fresnel_slice_raw
line_qm, = ax_scr.plot(y, qm_slice_0, 'b-', lw=2, label='Schrödinger', alpha=0.8)
line_fr, = ax_scr.plot(y, fresnel_slice_0, 'r--', lw=2, label='Fresnel', alpha=0.8)
ax_scr.set_title(f'Screen at x={x_screen_fixed:.2f}')
ax_scr.set_xlabel('y')
ax_scr.set_ylabel('Intensity')
ax_scr.set_xlim(y_min, y_max)
ax_scr.set_ylim(0, max(qm_slice_0.max(), fresnel_slice_0.max()) * 1.2)
ax_scr.legend()
ax_scr.grid(True, alpha=0.3)

plt.tight_layout()

def animate_spacetime(i):
    # Step simulation
    for _ in range(frame_skip):
        S.step()
    
    t = S.t
    prob = S.get_prob().reshape(Nx, Ny)
    
    # QM p Updateanel
    ax_qm.clear()
    ax_qm.contourf(xx, yy, prob, levels=100, cmap='jet')
    ax_qm.axvline(x_screen_fixed, color='white', ls='--', lw=1)
    ax_qm.set_title(f'Schrödinger |ψ|², t={t:.2f}')
    ax_qm.set_xlabel('x')
    ax_qm.set_ylabel('y')
    ax_qm.set_xlim(x_min, x_max)
    ax_qm.set_ylim(y_min, y_max)
    
    # Redraw barriers
    ax_qm.vlines([x_slit, x_slit + slit_width], y1 - h/2, y1 + h/2, colors='white', linewidth=2)
    ax_qm.vlines([x_slit, x_slit + slit_width], y2 - h/2, y2 + h/2, colors='white', linewidth=2)
    ax_qm.hlines([y1 - h/2, y1 + h/2], x_slit, x_slit + slit_width, colors='white', linewidth=2)
    ax_qm.hlines([y2 - h/2, y2 + h/2], x_slit, x_slit + slit_width, colors='white', linewidth=2)
    
    # Update screen comparison
    qm_intensity = prob[ix_screen, :]
    line_qm.set_ydata(qm_intensity)
    
    max_val = max(qm_intensity.max(), fresnel_slice_raw.max())
    ax_scr.set_ylim(0, max_val * 1.2)
    ax_scr.set_title(f'Screen at x={x_screen_fixed:.2f}, t={t:.2f}')
    
    if (i+1) % 20 == 0:
        print(f"  Frame {i+1}/{nb_frames_gif}")
    
    return line_qm,

print(f"Animating {nb_frames_gif} frames...")
anim_spacetime = animation.FuncAnimation(fig_gif, animate_spacetime, frames=nb_frames_gif, 
                                        interval=50, blit=False)
anim_spacetime.save(f'spacetime_comparison_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.gif', 
                    writer='pillow', fps=20)
print(f"Saved: spacetime_comparison_k{kx0:.0f}_d{d:.1f}_h{h:.1f}.gif")

plt.show()
print("\nVisualization complete!")
