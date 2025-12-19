#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# 1. Load data
# -----------------------------
with open("screen_intensity_data.pkl", "rb") as f:
    data = pickle.load(f)

y = data["y"]
t_vec = data["t"]
I_qm = data["I_qm"]      # shape (nb_frame, Ny)
I_f = data["I_fresnel"]  # shape (nb_frame, Ny)

nb_frame = len(t_vec)
Ny = len(y)

# -----------------------------
# 2. Compute discrepancy
# -----------------------------
# Среднеквадратичное отклонение (RMS)
discrepancy_rms = np.sqrt(np.mean((I_qm - I_f)**2, axis=1))

# Альтернатива: L1-норма (среднее абсолютное отклонение)
discrepancy_l1 = np.mean(np.abs(I_qm - I_f), axis=1)

# Или корреляция (1 - correlation ≈ discrepancy)
correlation = np.array([
    np.corrcoef(I_qm[i], I_f[i])[0, 1] for i in range(nb_frame)
])
discrepancy_corr = 1 - correlation

# -----------------------------
# 3. Plot discrepancy vs time
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(t_vec, discrepancy_rms, 'b-', lw=2, label='RMS отклонение')
plt.plot(t_vec, discrepancy_l1,  'g--', lw=2, label='L1 отклонение')
plt.plot(t_vec, discrepancy_corr, 'r-.', lw=2, label='1 - корреляция')

plt.xlabel("Время $t$", fontsize=12)
plt.ylabel("Мера расхождения", fontsize=12)
plt.title("Расхождение между QM и Френелем во времени", fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("discrepancy_vs_time.png", dpi=150)
plt.show()

# -----------------------------
# 4. Optional: Print min discrepancy time
# -----------------------------
min_idx = np.argmin(discrepancy_rms)
print(f"Минимальное RMS-расхождение: {discrepancy_rms[min_idx]:.4f} при t = {t_vec[min_idx]:.3f}")