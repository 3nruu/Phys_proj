#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import sys

# Ensure project root (src) is on path before importing diffraction package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from diffraction.visualization_utils import compute_smart_color_limits, normalize_for_display

# Ensure project root (src) is on path before importing diffraction package
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib.gridspec as gridspec
import pickle

from diffraction import (
    Fresnel1D,
    WaveFunctionCN,
    DoubleSlitConfig,
    build_double_slit_mask,
    mask_to_potential,
)
from diffraction.simulation import (
    SimulationConfig,
    build_grids,
    make_initial_gaussian,
    normalize_1d,
    results_path,
    NORM_EPS,
)

plt.rcParams.update({"font.size": 8})

BASE_DIR = ROOT
RESULTS_DIR = BASE_DIR / "results"


def build_fresnel_U0(y, slit_cfg: DoubleSlitConfig):
    U0 = np.zeros_like(y, dtype=complex)
    open_min = slit_cfg.y_lower_max
    open_max = slit_cfg.y_upper_min
    U0[(y >= open_min) & (y <= open_max)] = 1.0
    # Normalize amplitude
    norm = np.trapezoid(np.abs(U0) ** 2, y)
    if norm > NORM_EPS:
        U0 /= np.sqrt(norm)
    else:
        U0 = np.zeros_like(U0)
    return U0


def main():
    cfg = SimulationConfig()
    slit_cfg = DoubleSlitConfig(
        x_slit=0.0,
        slit_width=0.3,
        y_lower_min=cfg.y_min,
        y_lower_max=-0.5,
        y_upper_min=0.5,
        y_upper_max=cfg.y_max,
        V0=400.0,
    )

    x, y, xx, yy = build_grids(cfg)
    Nx, Ny = len(x), len(y)

    # Potential
    mask = build_double_slit_mask(x, y, slit_cfg)
    V = mask_to_potential(mask, slit_cfg.V0)

    # Initial wavefunction (Schrödinger)
    psi0_grid = make_initial_gaussian(xx, yy, cfg).swapaxes(0, 1)  # (Ny, Nx)
    psi0 = psi0_grid.flatten(order="F")  # y-major: idx = i_y + j_x*Ny

    S = WaveFunctionCN(x=x, y=y, psi_0=psi0, V=V, dt=cfg.dt, hbar=cfg.hbar, m=cfg.m)
    S.psi = S.psi / S.compute_norm()

    # Fresnel setup
    wavelength = 2 * np.pi / cfg.kx0
    U0 = build_fresnel_U0(y, slit_cfg)
    FD = Fresnel1D(y=y, U0=U0, wavelength=wavelength)

    # Screen index
    ix_screen = np.argmin(np.abs(x - cfg.screen_x))

    # Group velocity and target times for each x
    v = cfg.hbar * cfg.kx0 / cfg.m
    t_targets = np.maximum((x - cfg.x0) / v, 0.0)  # time for wavepacket center to reach plane x

    # Choose nb_frame so that max target time is covered
    t_max = np.nanmax(t_targets[np.isfinite(t_targets)])
    nb_frame = max(cfg.nb_frame, int(np.ceil(t_max / cfg.dt)) + 20)
    t_vals = np.arange(nb_frame) * cfg.dt

    # Storage
    qm_slices_screen = np.zeros((nb_frame, Ny))
    fresnel_slices_screen = np.zeros((nb_frame, Ny))
    I_fres_spacetime = np.zeros((nb_frame, Ny))
    qm_plane_map = np.zeros((Nx, Ny))  # Schrödinger at plane times
    captured = np.zeros(Nx, dtype=bool)
    # arrival time at screen and corresponding frame index
    #t_screen = (cfg.screen_x - cfg.x0) / v
    t_screen = 1.60
    frame_screen_idx = int(np.argmin(np.abs(t_vals - t_screen)))

    # --- Figure setup ---
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.3, 1.3, 1])
    ax_qm = plt.subplot(gs[0])
    ax_fr = plt.subplot(gs[1])
    ax_scr = plt.subplot(gs[2])

    # Initial state
    prob0 = S.get_prob().reshape(Ny, Nx, order="F")
    I_fres_0 = FD.intensity(cfg.screen_x - slit_cfg.x_slit)
    I_fres_spacetime[0, :] = FD.intensity(0)

    # QM map
    xx_plot, yy_plot = xx, yy
    ax_qm.contourf(xx_plot, yy_plot, prob0.T, levels=100, cmap="jet")
    ax_qm.axvline(cfg.screen_x, color="white", ls="--", lw=1)
    ax_qm.set_title("Schrodinger |psi|^2, t=0.00")
    ax_qm.set_xlabel("x")
    ax_qm.set_ylabel("y")
    ax_qm.set_xlim(cfg.x_min, cfg.x_max)
    ax_qm.set_ylim(cfg.y_min, cfg.y_max)

    # Draw potential
    ax_qm.vlines([slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width], slit_cfg.y_lower_min, slit_cfg.y_lower_max, colors="white", linewidth=2)
    ax_qm.vlines([slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width], slit_cfg.y_upper_min, slit_cfg.y_upper_max, colors="white", linewidth=2)
    ax_qm.hlines([slit_cfg.y_lower_max, slit_cfg.y_upper_min], slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width, colors="white", linewidth=2)

    # Fresnel static map over x (no animation)
    fresnel_plot = np.zeros((Nx, Ny))
    for ix_val, x_val in enumerate(x):
        z = x_val - slit_cfg.x_slit
        if z <= 0:
            continue
        fresnel_plot[ix_val, :] = FD.intensity(z)
    # Вычисляем разумные пределы для Френеля
    fresnel_vmin, fresnel_vmax = compute_smart_color_limits(fresnel_plot[fresnel_plot > 0])
    
    # Статичная карта Френеля в GIF
    ax_fr.imshow(
        fresnel_plot.T,
        extent=[x.min(), x.max(), y.min(), y.max()],
        origin="lower",
        aspect="auto",
        cmap="inferno",
        vmin=fresnel_vmin,
        vmax=fresnel_vmax,
    )
    ax_fr.set_title(r"Fresnel $|U(y,z)|^2$ (static)")
    ax_fr.set_xlabel("x (distance from slit)")
    ax_fr.set_ylabel("y")
    
    # Screen plot (raw)
    qm_slice_0 = prob0[:, ix_screen]
    I_fres_0_plot = I_fres_0
    ax_scr.plot(y, qm_slice_0, "b", lw=2, label="Schrödinger")
    ax_scr.plot(y, I_fres_0_plot, "r--", lw=2, label="Fresnel")
    ax_scr.set_title(f"Screen at x = {cfg.screen_x}")
    ax_scr.set_xlabel("y")
    ax_scr.set_ylabel("Intensity (raw)")
    ax_scr.set_xlim(cfg.y_min, cfg.y_max)
    ax_scr.set_ylim(bottom=0)
    ax_scr.legend()

    # Pre-fill plane map for targets at t=0
    for ix_plane in range(Nx):
        if t_targets[ix_plane] <= 0:
            qm_plane_map[ix_plane, :] = normalize_1d(prob0[:, ix_plane], y)
            captured[ix_plane] = True

    plt.tight_layout()

    # Animation function
    def animate(i):
        S.step()
        t = S.t

        prob = S.get_prob().reshape(Ny, Nx, order="F")

        # QM map
        ax_qm.clear()
        ax_qm.contourf(xx_plot, yy_plot, prob.T, levels=100, cmap="jet")
        ax_qm.axvline(cfg.screen_x, color="white", ls="--", lw=1)
        ax_qm.set_title(f"Schrodinger |psi|^2, t={t:.2f}")
        ax_qm.set_xlim(cfg.x_min, cfg.x_max)
        ax_qm.set_ylim(cfg.y_min, cfg.y_max)
        ax_qm.set_xlabel("x")
        ax_qm.set_ylabel("y")
        ax_qm.vlines([slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width], slit_cfg.y_lower_min, slit_cfg.y_lower_max, colors="white", linewidth=2)
        ax_qm.vlines([slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width], slit_cfg.y_upper_min, slit_cfg.y_upper_max, colors="white", linewidth=2)
        ax_qm.hlines([slit_cfg.y_lower_max, slit_cfg.y_upper_min], slit_cfg.x_slit, slit_cfg.x_slit + slit_cfg.slit_width, colors="white", linewidth=2)


        # Screen comparison (raw)
        qm_slice = prob[:, ix_screen]
        fresnel_slice = FD.intensity(cfg.screen_x - slit_cfg.x_slit)

        qm_slices_screen[i, :] = qm_slice
        fresnel_slices_screen[i, :] = fresnel_slice

        ax_scr.clear()
        ax_scr.plot(y, qm_slice, "b", lw=2, label="Schrödinger")
        ax_scr.plot(y, fresnel_slice, "r--", lw=2, label="Fresnel")
        ax_scr.set_xlim(cfg.y_min, cfg.y_max)
        ax_scr.set_ylim(bottom=0, top=max(qm_slice.max(), fresnel_slice.max(), 1e-9) * 1.1)
        ax_scr.set_xlabel("y")
        ax_scr.set_ylabel("Intensity (raw)")
        ax_scr.legend()
        ax_scr.set_title(f"Screen at x = {cfg.screen_x}")

        print(f"Frame {i + 1}/{nb_frame}")

        # Capture plane-crossing slices when center reaches each x
        # Note: loop modest Nx, acceptable
        for ix_plane in range(Nx):
            if captured[ix_plane]:
                continue
            if t >= t_targets[ix_plane]:
                qm_plane_map[ix_plane, :] = normalize_1d(prob[:, ix_plane], y)
                captured[ix_plane] = True


    print("Creating animation...")
    anim = animation.FuncAnimation(fig, animate, frames=nb_frame, interval=100, blit=False)
    gif_path = results_path(RESULTS_DIR, "Schrodinger_vs_Fresnel_comparison.gif")
    anim.save(gif_path, writer="pillow", fps=10)
    print(f"Animation saved to {gif_path}")

    # Full spatial comparison (using final state)
    print("Computing full spatial comparison...")
    prob_final = S.get_prob().reshape(Ny, Nx, order="F")

    fresnel_full_map = np.zeros((Nx, Ny))
    qm_full_map = qm_plane_map.copy()  # already normalized per plane

    for ix_val, x_val in enumerate(x):
        z = x_val - slit_cfg.x_slit
        if z <= 0:
            continue
        I_fres = FD.intensity(z)
        fresnel_full_map[ix_val, :] = normalize_1d(I_fres, y)

    diff_map = np.abs(qm_full_map - fresnel_full_map)

    error_L1_full = np.trapezoid(np.trapezoid(diff_map, y, axis=1), x)
    error_L2_full = np.sqrt(np.trapezoid(np.trapezoid(diff_map**2, y, axis=1), x))

    print(f"L1 error: {error_L1_full:.4f}")
    print(f"L2 error: {error_L2_full:.4f}")

    # Save data
    experiment_params = {
        "kx0": cfg.kx0,
        "wavelength": 2 * np.pi / cfg.kx0,
        "slit_opening": slit_cfg.y_upper_min - slit_cfg.y_lower_max,
        "x_screen": cfg.screen_x,
        "dx": cfg.dx,
        "dy": cfg.dy,
        "dt": cfg.dt,
        "V0": slit_cfg.V0,
    }

    # После создания qm_plane_map добавьте масштабирование:
    qm_full_map_scaled, qm_vmin, qm_vmax = normalize_for_display(qm_full_map)
    fresnel_full_map_scaled, _, _ = normalize_for_display(fresnel_full_map, 
                                                         vmin=fresnel_vmin, 
                                                         vmax=fresnel_vmax)
    
    # Сохраняйте scaled версии для визуализации
    pkl_path = results_path(RESULTS_DIR, "Schrodinger_vs_Fresnel_Comparison.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "params": experiment_params,
                "x": x,
                "y": y,
                "t_vals": t_vals,
                "qm_plane_map": qm_full_map,          # исходные данные
                "qm_plane_map_scaled": qm_full_map_scaled,  # для визуализации
                "fresnel_full_map": fresnel_full_map,
                "fresnel_full_map_scaled": fresnel_full_map_scaled,
                "diff_map": diff_map,
                "error_L1_full": error_L1_full,
                "error_L2_full": error_L2_full,
                "qm_slices_screen": qm_slices_screen,
                "fresnel_slices_screen": fresnel_slices_screen,
                "t_screen": t_screen,
                "frame_screen_idx": frame_screen_idx,
                "color_limits": {  # сохраняем пределы для согласованности
                    "qm_vmin": qm_vmin,
                    "qm_vmax": qm_vmax,
                    "fresnel_vmin": fresnel_vmin,
                    "fresnel_vmax": fresnel_vmax,
                }
            },
            f,
        )

    print(f"Data saved to {pkl_path}")


if __name__ == "__main__":
    main()
