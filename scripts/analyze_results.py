#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# Ensure project root (src) is on path before importing diffraction package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pickle
import matplotlib.pyplot as plt

from diffraction.simulation import normalize_1d

BASE_DIR = ROOT
RESULTS_DIR = BASE_DIR / "results"


def load_data(filename="Schrodinger_vs_Fresnel_Comparison.pkl"):
    with open(RESULTS_DIR / filename, "rb") as f:
        return pickle.load(f)

def plot_spatial_maps(data):
    x = data["x"]
    y = data["y"]
    
    # Используем масштабированные версии для визуализации
    qm_full_map = data.get("qm_plane_map_scaled", data.get("qm_plane_map"))
    fresnel_full_map = data.get("fresnel_full_map_scaled", data.get("fresnel_full_map"))
    
    if qm_full_map is None or fresnel_full_map is None:
        raise ValueError("Missing required fields in data")
    
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Получаем сохраненные пределы цветовой шкалы
    color_limits = data.get("color_limits", {})
    
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    
    # Карта Шрёдингера
    # В plot_spatial_maps, в imshow добавьте:
    im0 = axs[0].imshow(
        qm_full_map.T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=color_limits.get("qm_vmin", None),
        vmax=color_limits.get("qm_vmax", None),
        interpolation='bilinear',  # или 'spline16' для более гладкого
    )
    axs[0].set_title("Schrödinger: |ψ|² at plane times (scaled)")
    plt.colorbar(im0, ax=axs[0])
    
    # Карта Френеля
    im1 = axs[1].imshow(
        fresnel_full_map.T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=color_limits.get("fresnel_vmin", None),
        vmax=color_limits.get("fresnel_vmax", None),
    )
    axs[1].set_title("Fresnel: |U|² (scaled)")
    plt.colorbar(im1, ax=axs[1])
    
    # Разность (используем магму с фиксированным диапазоном)
    diff_map = np.abs(qm_full_map - fresnel_full_map)
    im2 = axs[2].imshow(
        diff_map.T,
        extent=[x_min, x_max, y_min, y_max],
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0,
        vmax=1.0,  # фиксированный максимум для разности
    )
    axs[2].set_title("|Schrödinger − Fresnel|")
    plt.colorbar(im2, ax=axs[2])
    
    # Общие настройки
    for ax in axs:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    
    plt.tight_layout()
    out = RESULTS_DIR / "comparison_maps.png"
    plt.savefig(out, dpi=300, bbox_inches='tight')  # увеличим DPI и уберем лишние поля
    plt.close(fig)
    print(f"Saved spatial maps -> {out}")

def plot_error_vs_x(data):
    x = data["x"]
    y = data["y"]
    qm_full_map = data.get("qm_plane_map")
    fresnel_full_map = data.get("fresnel_full_map")
    if qm_full_map is None or fresnel_full_map is None:
        raise ValueError("Missing qm_plane_map or fresnel_full_map")

    x_error = []
    error_L2_vs_x = []

    for ix, x_val in enumerate(x):
        if x_val <= 0:
            continue
        diff = qm_full_map[ix, :] - fresnel_full_map[ix, :]
        err = np.sqrt(np.trapezoid(diff**2, y))
        x_error.append(x_val)
        error_L2_vs_x.append(err)

    x_error = np.array(x_error)
    error_L2_vs_x = np.array(error_L2_vs_x)

    def running_mean(a, window=9):
        return np.convolve(a, np.ones(window) / window, mode="same")

    error_smooth = running_mean(error_L2_vs_x)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_error, error_L2_vs_x, alpha=0.4, label="raw")
    ax.plot(x_error, error_smooth, lw=2, label="smoothed")
    ax.set_xlabel("x (distance from origin)")
    ax.set_ylabel(r"$||Schrödinger - Fresnel||_{L^2(y)}$")
    ax.set_title("Fresnel approximation error vs x")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    out = RESULTS_DIR / "error_vs_x.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved error_vs_x -> {out}")


def plot_screen_peak(data):
    y = data["y"]
    qm_slices = data.get("qm_slices_screen")
    frame_idx = data.get("frame_screen_idx")
    t_screen = data.get("t_screen")
    if qm_slices is None or frame_idx is None:
        print("screen peak data missing; skipping screen_peak plot")
        return
    qm_raw = qm_slices[frame_idx]
    fres_raw = data["fresnel_slices_screen"][frame_idx]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y, qm_raw, "b", lw=2, label="Schrödinger (raw)")
    ax.plot(y, fres_raw, "r--", lw=2, label="Fresnel (raw)")
    ax.set_xlabel("y")
    ax.set_ylabel("Intensity (raw)")
    if t_screen is not None:
        ax.set_title(f"Screen at x={data['params']['x_screen']}, t={t_screen:.2f}")
    else:
        ax.set_title(f"Screen at x={data['params']['x_screen']}")
    ax.set_xlim(y.min(), y.max())
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = RESULTS_DIR / "screen_peak_intensity.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved screen_peak_intensity -> {out}")


def main():
    data = load_data()
    plot_spatial_maps(data)
    plot_error_vs_x(data)
    plot_screen_peak(data)

    params = data["params"]
    print("\n=== COMPARISON SUMMARY ===")
    print(f"Wave number kx0: {params['kx0']}")
    print(f"Wavelength: {params['wavelength']:.4f}")
    print(f"Slit opening: {params['slit_opening']}")
    print(f"Screen position: {params['x_screen']}")
    print(f"Global L1 error: {data['error_L1_full']:.6f}")
    print(f"Global L2 error: {data['error_L2_full']:.6f}")


if __name__ == "__main__":
    main()
