import numpy as np
import matplotlib.pyplot as plt
import pickle

# -----------------------------
# Load saved data
# -----------------------------
with open("diffraction_comparison.pkl", "rb") as f:
    data = pickle.load(f)

y = data["y"]
t_vec = data["t_vec"]
I_qm = data["qm"]          # shape (Nt, Ny)
I_fres = data["fresnel"]  # shape (Nt, Ny)

dy = y[1] - y[0]
Nt = len(t_vec)

# -----------------------------
# Compute errors
# -----------------------------

# Pointwise absolute error
error_point = np.abs(I_qm - I_fres)

# Relative error (normalized by max QM intensity at each time)
error_rel = error_point / np.max(I_qm, axis=1)[:, None]

# Integrated (L1) error
error_int = np.zeros(Nt)
for i in range(Nt):
    num = np.trapz(np.abs(I_qm[i] - I_fres[i]), y)
    den = np.trapz(I_qm[i], y)
    error_int[i] = num / den

# -----------------------------
# Plot 1: intensity comparison (final time)
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(y, I_qm[-1], label="QM", lw=2)
plt.plot(y, I_fres[-1], "--", label="Fresnel", lw=2)
plt.xlabel(r"$y$")
plt.ylabel("Intensity")
plt.title("Intensity comparison at final time")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 2: pointwise error
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(y, error_point[-1])
plt.xlabel(r"$y$")
plt.ylabel(r"$|I_{QM} - I_F|$")
plt.title("Pointwise error (final time)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 3: relative error
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(y, error_rel[-1])
plt.xlabel(r"$y$")
plt.ylabel("Relative error")
plt.title("Relative error (final time)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# -----------------------------
# Plot 4: integrated error vs time
# -----------------------------
plt.figure(figsize=(6, 4))
plt.plot(t_vec, error_int)
plt.xlabel(r"$t$")
plt.ylabel(r"$E(t)$")
plt.title("Integrated error vs time")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
