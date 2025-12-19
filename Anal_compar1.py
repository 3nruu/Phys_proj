#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Загрузка данных
# -------------------------------------------------
filename = "QM_vs_Fresnel_Comparison.pkl"

with open(filename, "rb") as f:
    data = pickle.load(f)

# Извлекаем необходимые данные
x = data['x']
y = data['y']
qm_full_map = data['qm_full_map']          # (Nx, Ny)
fresnel_full_map = data['fresnel_full_map']  # (Nx, Ny)
diff_map = data['diff_map']

x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0

fig, axs = plt.subplots(1, 3, figsize=(14, 4))

im0 = axs[0].imshow(
    qm_full_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='jet'
)
axs[0].set_title("QM карта (норм. по y)")
plt.colorbar(im0, ax=axs[0])

im1 = axs[1].imshow(
    fresnel_full_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='inferno'
)
axs[1].set_title("Fresnel карта (норм. по y)")
plt.colorbar(im1, ax=axs[1])

im2 = axs[2].imshow(
    diff_map.T,
    extent=[x_min, x_max, y_min, y_max],
    origin='lower',
    aspect='auto',
    cmap='magma'
)
axs[2].set_title("|QM − Fresnel|")
plt.colorbar(im2, ax=axs[2])

for ax in axs:
    ax.set_xlabel("x")
    ax.set_ylabel("y")

plt.tight_layout()
plt.show()


x_error = []
error_L2_vs_x = []

for ix, x_val in enumerate(x):

    # только после щели
    if x_val <= 0:
        continue

    I_qm = qm_full_map[ix, :]
    I_fr = fresnel_full_map[ix, :]

    diff = I_qm - I_fr

    # L2-норма по y
    err = np.sqrt(np.trapezoid(diff**2, y))

    x_error.append(x_val)
    error_L2_vs_x.append(err)

x_error = np.array(x_error)
error_L2_vs_x = np.array(error_L2_vs_x)

def running_mean(a, window=7):
    return np.convolve(a, np.ones(window)/window, mode='same')

error_smooth = running_mean(error_L2_vs_x, window=9)


plt.figure(figsize=(6,4))

plt.plot(x_error, error_L2_vs_x, alpha=0.4, label="raw")
plt.plot(x_error, error_smooth, lw=2, label="smoothed")

plt.xlabel("x")
plt.ylabel(r"$\|QM - Fresnel\|_{L^2(y)}$")
plt.title("Ошибка приближения Френеля vs x")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()



