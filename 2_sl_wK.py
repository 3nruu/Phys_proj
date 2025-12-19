#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import matplotlib.gridspec as gridspec
from Wave_Function import WaveFunction
from FresnelDiffraction_1D import Fresnel1D
import pickle

plt.rcParams.update({'font.size': 8})

# -------------------------------------------------
# Parameters
# -------------------------------------------------
dt = 0.005
hbar = 1.0
m = 1.0
kx0 = 20.0
wavelength = 2 * np.pi / kx0

x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08

x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)
xx, yy = np.meshgrid(x, y, indexing='ij')  # (Nx, Ny)

# -------------------------------------------------
# Double slit
# -------------------------------------------------
x_slit, slit_width = 0.0, 0.2
y1, y2, h = -2.0, 2.0, 1.5

mask = np.zeros((Nx, Ny), dtype=bool)
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y1 - h/2) & (yy <= y1 + h/2)] = True
mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
     (yy >= y2 - h/2) & (yy <= y2 + h/2)] = True

# -------------------------------------------------
# Initial wave function (QM)
# -------------------------------------------------
psi0 = np.zeros((Nx, Ny), dtype=complex)
psi0[mask] = np.exp(1j * kx0 * xx[mask])

psi0_flat = psi0.flatten()
psi0_flat /= np.sqrt(np.sum(np.abs(psi0_flat)**2) * dx * dy)

S = WaveFunction(x=x, y=y, psi_0=psi0_flat, V=np.zeros(Nx*Ny), dt=dt, hbar=hbar, m=m)

# -------------------------------------------------
# Fresnel initial field
# -------------------------------------------------
U0 = np.zeros_like(y, dtype=complex)
U0[(y >= y1 - h/2) & (y <= y1 + h/2)] = 1.0
U0[(y >= y2 - h/2) & (y <= y2 + h/2)] = 1.0
U0 /= np.sqrt(np.trapezoid(np.abs(U0)**2, y))

FD = Fresnel1D(y=y, U0=U0, wavelength=wavelength)

# -------------------------------------------------
# Screen
# -------------------------------------------------
x_screen = 7.0
ix = np.argmin(np.abs(x - x_screen))

# -------------------------------------------------
# Precompute z for Fresnel spacetime
# -------------------------------------------------
nb_frame = 200
z0 = 0
t_vals = np.arange(nb_frame) * dt
z_vals = z0 + kx0 * t_vals
I_fres_spacetime = np.zeros((nb_frame, Ny))

# -------------------------------------------------
# Figure: 2 maps + 1 slice
# -------------------------------------------------
fig = plt.figure(figsize=(12, 4))
gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1])
ax_qm = plt.subplot(gs[0])   # Quantum map (x, y)
ax_fr = plt.subplot(gs[1])   # Fresnel spacetime (z, y)
ax_scr = plt.subplot(gs[2])  # Screen intensity (y)

# === Initial state ===
prob0 = S.get_prob().reshape(Nx, Ny)
I0 = FD.intensity(z0)
I_fres_spacetime[0, :] = I0

# QM map
ax_qm.contourf(x, y, prob0.T, levels=100, cmap='jet')
ax_qm.axvline(x_screen, color='white', ls='--', lw=1)
ax_qm.set_title(r"QM $|\psi|^2$, $t=0.00$")
ax_qm.set_xlabel("x")
ax_qm.set_ylabel("y")
ax_qm.set_xlim(x_min, x_max)
ax_qm.set_ylim(y_min, y_max)

# Fresnel spacetime (start with first column)
Z0, Y0 = np.meshgrid([z0], y, indexing='ij')
ax_fr.pcolormesh(Z0, Y0, I0[None, :], cmap='inferno', shading='auto')
ax_fr.set_title(r"Fresnel $|U(y,z)|^2$")
ax_fr.set_xlabel("z")
ax_fr.set_ylabel("y")
ax_fr.set_xlim(z0, z0 + 0.1)
ax_fr.set_ylim(y_min, y_max)

# Screen
ax_scr.plot(y, prob0[ix, :], 'b', lw=2, label="QM")
ax_scr.plot(y, I0, 'r--', lw=2, label="Fresnel")
ax_scr.set_title("Screen intensity")
ax_scr.set_xlabel("y")
ax_scr.set_ylabel("Intensity")
ax_scr.set_xlim(y_min, y_max)
ax_scr.set_ylim(bottom=0)
ax_scr.legend()

plt.tight_layout()
fig.canvas.draw()  # ← ← ← КЛЮЧЕВОЙ ВЫЗОВ ДЛЯ WINDOWS
qm_slices = np.zeros((nb_frame, Ny))
Px_t = np.zeros((nb_frame, Nx))
x_screen_fixed = 5.0
ix_fixed = np.argmin(np.abs(x - x_screen_fixed))

# Хранилище для срезов на фиксированном экране
qm_slices_fixed = np.zeros((nb_frame, Ny))    # QM на x=8.0
fresnel_slices = np.zeros((nb_frame, Ny))     # Френель на z = 8.0

# Список для ошибок (опционально, можно вычислить потом)
errors = []
# -------------------------------------------------
# Animation function
# -------------------------------------------------
def animate(i):
    S.step()
    t = S.t

    # Расстояние от щелей для Френеля
    z = kx0 * t
    prob = S.get_prob().reshape(Nx, Ny)

    # --- QM map (остаётся как есть) ---
    ax_qm.clear()
    ax_qm.contourf(x, y, prob.T, levels=100, cmap='jet')
    ax_qm.axvline(x_screen_fixed, color='white', ls='--', lw=1)
    ax_qm.set_title(rf"QM $|\psi|^2$, $t={t:.2f}$")
    ax_qm.set_xlim(x_min, x_max)
    ax_qm.set_ylim(y_min, y_max)
    ax_qm.set_xlabel("x")
    ax_qm.set_ylabel("y")

    # --- Fresnel spacetime (остаётся) ---
    I_f = FD.intensity(z) if z > 0 else np.zeros_like(y)
    I_fres_spacetime[i, :] = I_f

    ax_fr.clear()
    z_current = kx0 * t_vals[:i+1]
    C = I_fres_spacetime[:i+1].T
    ax_fr.pcolormesh(z_current, y, C, cmap='inferno', shading='auto')
    ax_fr.set_title(r"Fresnel $|U(y,z)|^2$")
    ax_fr.set_xlabel("z")
    ax_fr.set_ylabel("y")
    ax_fr.set_xlim(0, max(z_current[-1], 0.1))
    ax_fr.set_ylim(y_min, y_max)

    # --- Срезы на ФИКСИРОВАННОМ экране (для сравнения!) ---
    qm_slice = prob[ix_fixed, :]  # QM на x = x_screen_fixed
    fresnel_slice = FD.intensity(x_screen_fixed)  # Френель на z = x_screen_fixed (т.к. x_slit=0)

    qm_slices_fixed[i, :] = qm_slice
    fresnel_slices[i, :] = fresnel_slice

    # --- Screen plot (теперь с фиксированным экраном) ---
    ax_scr.clear()
    ax_scr.plot(y, qm_slice, 'b', lw=2, label="QM")
    ax_scr.plot(y, fresnel_slice, 'r--', lw=2, label="Fresnel")
    ax_scr.set_xlim(y_min, y_max)
    ax_scr.set_ylim(bottom=0)
    ax_scr.set_xlabel("y")
    ax_scr.set_ylabel("Intensity")
    ax_scr.legend()
    ax_scr.set_title(f"Screen at x = {x_screen_fixed}")

    print(f"Frame {i+1}/{nb_frame}")




# -------------------------------------------------
# Save animation
# -------------------------------------------------
anim = animation.FuncAnimation(fig, animate, frames=nb_frame, interval=100, blit=False)

anim.save("QM_vs_Fresnel_Simple.gif", writer="pillow", fps=10)

# -------------------------------------------------
# Полное пространственное сравнение
# -------------------------------------------------

fresnel_full_map = np.zeros((Nx, Ny))

for ix, x_val in enumerate(x):
    z = x_val - x_slit   # расстояние от плоскости щели

    if z <= 0:
        fresnel_full_map[ix, :] = 0.0
        continue

    I = FD.intensity(z)

    # НОРМИРОВКА ПО y (сравниваем форму!)
    norm = np.trapezoid(I, y)
    if norm > 0:
        I /= norm

    fresnel_full_map[ix, :] = I


# Получим последнюю (или лучшую) QM карту
# Вариант A: последний кадр
prob_final = S.get_prob().reshape(Nx, Ny)

qm_full_map = np.zeros_like(prob_final)

for ix, x_val in enumerate(x):
    if x_val <= x_slit:
        qm_full_map[ix, :] = 0.0
        continue

    I = prob_final[ix, :]

    # НОРМИРОВКА ПО y
    norm = np.trapezoid(I, y)
    if norm > 0:
        I /= norm

    qm_full_map[ix, :] = I


# Вариант B: кадр, когда пакет в центре области (лучше)
# Но для простоты — возьмём последний

# Нормируем обе карты на максимум для сравнения формы
diff_map = np.abs(qm_full_map - fresnel_full_map)


# Метрики по всей области
error_L1_full = np.trapezoid(
    np.trapezoid(diff_map, y, axis=1),
    x
)

error_L2_full = np.sqrt(
    np.trapezoid(
        np.trapezoid(diff_map**2, y, axis=1),
        x
    )
)


# Save data
# Параметры эксперимента (обязательно!)
experiment_params = {
    'kx0': kx0,
    'slit_width': slit_width,
    'slit_separation': y2 - y1,  # d = 4.0 у вас
    'slit_height': h,            # высота щели
    'x_screen_fixed': x_screen_fixed,
    'dx': dx,
    'dt': dt
}

with open("QM_vs_Fresnel_Comparison.pkl", "wb") as f:
    pickle.dump({
        # Параметры
        'params': experiment_params,
        
        # Сетки
        'x': x,
        'y': y,
        't_vals': t_vals,

        'qm_full_map': qm_full_map,
        'fresnel_full_map': fresnel_full_map,
        'diff_map': diff_map,
        'error_L1_full': error_L1_full,
        'error_L2_full': error_L2_full
    }, f)
plt.show()