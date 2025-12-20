#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from WaveFunction1 import WaveFunction
from FresnelDiffraction_1D import Fresnel1D
import pickle
from multiprocessing import Pool
import time

plt.rcParams.update({'font.size': 8})

# -------------------------------------------------
# Parameter sweep configuration
# -------------------------------------------------
# Define parameter ranges to sweep
kx0_values = [10.0, 15.0, 20.0]  # wave numbers
h_values = [1.0, 1.5, 2.0]  # hole widths
x_screen_values = [5.0, 7.0, 9.0]  # screen distances

# Fixed parameters
dt = 0.005
hbar = 1.0
m = 1.0

x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08  # Original resolution for accuracy

x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

# Double slit fixed parameters
x_slit, slit_width = 0.0, 0.2
y1, y2 = -2.0, 2.0

# -------------------------------------------------
# Optimized simulation function
# -------------------------------------------------
def run_simulation(params):
    """
    Run single simulation with given parameters
    Returns comparison data for plotting
    """
    kx0, h, x_screen_fixed = params
    
    print(f"Running simulation: kx0={kx0}, h={h}, x_screen={x_screen_fixed}")
    start_time = time.time()
    
    wavelength = 2 * np.pi / kx0
    
    # Create meshgrid
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
    
    # Create WaveFunction object
    S = WaveFunction(x=x, y=y, psi_0=psi0_flat, V=np.zeros(Nx*Ny), 
                     dt=dt, hbar=hbar, m=m)
    
    # Fresnel initial field
    U0 = np.zeros_like(y, dtype=complex)
    U0[(y >= y1 - h/2) & (y <= y1 + h/2)] = 1.0
    U0[(y >= y2 - h/2) & (y <= y2 + h/2)] = 1.0
    U0 /= np.sqrt(np.trapezoid(np.abs(U0)**2, y))
    
    FD = Fresnel1D(y=y, U0=U0, wavelength=wavelength)
    
    # Screen index
    ix_fixed = np.argmin(np.abs(x - x_screen_fixed))
    
    # Calculate time to reach screen
    # z = kx0 * t, so t = z / kx0
    t_target = x_screen_fixed / kx0
    n_steps = int(t_target / dt)
    
    # Evolve QM system to target time
    for _ in range(n_steps):
        S.step()
    
    # Get final probability distribution
    prob_final = S.get_prob().reshape(Nx, Ny)
    
    # Extract slice at screen
    qm_slice = prob_final[ix_fixed, :]
    
    # Get Fresnel intensity at same location
    fresnel_slice = FD.intensity(x_screen_fixed)
    
    # Normalize both for comparison
    qm_slice_norm = qm_slice / np.trapezoid(qm_slice, y)
    fresnel_slice_norm = fresnel_slice / np.trapezoid(fresnel_slice, y)
    
    # Calculate error at screen
    error_L2 = np.sqrt(np.trapezoid((qm_slice_norm - fresnel_slice_norm)**2, y))
    
    # Compute full spatial maps for error vs x analysis
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
    
    # Calculate error vs x
    x_error = []
    error_L2_vs_x = []
    
    for ix, x_val in enumerate(x):
        if x_val <= 0:
            continue
        
        I_qm = qm_full_map[ix, :]
        I_fr = fresnel_full_map[ix, :]
        diff = I_qm - I_fr
        err = np.sqrt(np.trapezoid(diff**2, y))
        
        x_error.append(x_val)
        error_L2_vs_x.append(err)
    
    x_error = np.array(x_error)
    error_L2_vs_x = np.array(error_L2_vs_x)
    
    elapsed = time.time() - start_time
    print(f"  Completed in {elapsed:.2f}s, L2 error: {error_L2:.6f}")
    
    return {
        'kx0': kx0,
        'h': h,
        'x_screen': x_screen_fixed,
        'wavelength': wavelength,
        'y': y,
        'qm_slice': qm_slice_norm,
        'fresnel_slice': fresnel_slice_norm,
        'error_L2': error_L2,
        'time': elapsed,
        'x_error': x_error,
        'error_L2_vs_x': error_L2_vs_x,
        'qm_full_map': qm_full_map,
        'fresnel_full_map': fresnel_full_map
    }

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == '__main__':
    # Generate all parameter combinations
    param_combinations = []
    for kx0 in kx0_values:
        for h in h_values:
            for x_screen in x_screen_values:
                param_combinations.append((kx0, h, x_screen))
    
    print(f"Total simulations to run: {len(param_combinations)}")
    print("=" * 60)
    
    # Run simulations (sequentially for now, can parallelize if needed)
    results = []
    total_start = time.time()
    
    for params in param_combinations:
        result = run_simulation(params)
        results.append(result)
    
    total_time = time.time() - total_start
    print("=" * 60)
    print(f"All simulations completed in {total_time:.2f}s")
    
    # Save results
    with open("parameter_sweep_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to parameter_sweep_results.pkl")
    
    # -------------------------------------------------
    # Plotting all comparisons
    # -------------------------------------------------
    n_results = len(results)
    n_cols = len(kx0_values)
    n_rows = len(h_values) * len(x_screen_values)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, result in enumerate(results):
        kx0 = result['kx0']
        h = result['h']
        x_screen = result['x_screen']
        y = result['y']
        qm = result['qm_slice']
        fresnel = result['fresnel_slice']
        error = result['error_L2']
        wavelength = result['wavelength']
        
        # Determine subplot position
        kx0_idx = kx0_values.index(kx0)
        h_idx = h_values.index(h)
        x_screen_idx = x_screen_values.index(x_screen)
        
        row = h_idx * len(x_screen_values) + x_screen_idx
        col = kx0_idx
        
        ax = axes[row, col]
        
        ax.plot(y, qm, 'b-', lw=2, label='QM', alpha=0.7)
        ax.plot(y, fresnel, 'r--', lw=2, label='Fresnel', alpha=0.7)
        
        ax.set_xlabel('y')
        ax.set_ylabel('Normalized Intensity')
        ax.set_title(f'k={kx0:.1f}, λ={wavelength:.3f}\\nh={h:.1f}, x={x_screen:.1f}\\nL2 error={error:.4f}', 
                     fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_comparison.png', dpi=150, bbox_inches='tight')
    print("Comparison plot saved to parameter_sweep_comparison.png")
    
    # -------------------------------------------------
    # Error summary plot
    # -------------------------------------------------
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
    
    # Error vs wave number
    for h in h_values:
        for x_screen in x_screen_values:
            errors = []
            kx0_plot = []
            for result in results:
                if result['h'] == h and result['x_screen'] == x_screen:
                    errors.append(result['error_L2'])
                    kx0_plot.append(result['kx0'])
            ax1.plot(kx0_plot, errors, 'o-', label=f'h={h}, x={x_screen}')
    
    ax1.set_xlabel('Wave number kx0')
    ax1.set_ylabel('L2 Error')
    ax1.set_title('Error vs Wave Number')
    ax1.legend(fontsize=7)
    ax1.grid(True)
    
    # Error vs hole width
    for kx0 in kx0_values:
        for x_screen in x_screen_values:
            errors = []
            h_plot = []
            for result in results:
                if result['kx0'] == kx0 and result['x_screen'] == x_screen:
                    errors.append(result['error_L2'])
                    h_plot.append(result['h'])
            ax2.plot(h_plot, errors, 'o-', label=f'k={kx0}, x={x_screen}')
    
    ax2.set_xlabel('Hole width h')
    ax2.set_ylabel('L2 Error')
    ax2.set_title('Error vs Hole Width')
    ax2.legend(fontsize=7)
    ax2.grid(True)
    
    # Error vs screen distance
    for kx0 in kx0_values:
        for h in h_values:
            errors = []
            x_plot = []
            for result in results:
                if result['kx0'] == kx0 and result['h'] == h:
                    errors.append(result['error_L2'])
                    x_plot.append(result['x_screen'])
            ax3.plot(x_plot, errors, 'o-', label=f'k={kx0}, h={h}')
    
    ax3.set_xlabel('Screen distance x')
    ax3.set_ylabel('L2 Error')
    ax3.set_title('Error vs Screen Distance')
    ax3.legend(fontsize=7)
    ax3.grid(True)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_error_analysis.png', dpi=150, bbox_inches='tight')
    print("Error analysis plot saved to parameter_sweep_error_analysis.png")
    
    # -------------------------------------------------
    # Error vs x distance plots for each parameter combination
    # -------------------------------------------------
    print("Creating error vs x plots...")
    
    def running_mean(a, window=7):
        return np.convolve(a, np.ones(window)/window, mode='same')
    
    n_results = len(results)
    n_cols = len(kx0_values)
    n_rows = len(h_values) * len(x_screen_values)
    
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes3 = np.array([[axes3]])
    elif n_rows == 1:
        axes3 = axes3.reshape(1, -1)
    elif n_cols == 1:
        axes3 = axes3.reshape(-1, 1)
    
    for idx, result in enumerate(results):
        kx0 = result['kx0']
        h = result['h']
        x_screen = result['x_screen']
        x_error = result['x_error']
        error_L2_vs_x = result['error_L2_vs_x']
        wavelength = result['wavelength']
        
        # Smooth the error curve
        error_smooth = running_mean(error_L2_vs_x, window=9)
        
        # Determine subplot position
        kx0_idx = kx0_values.index(kx0)
        h_idx = h_values.index(h)
        x_screen_idx = x_screen_values.index(x_screen)
        
        row = h_idx * len(x_screen_values) + x_screen_idx
        col = kx0_idx
        
        ax = axes3[row, col]
        
        ax.plot(x_error, error_L2_vs_x, alpha=0.4, label='raw', color='gray')
        ax.plot(x_error, error_smooth, lw=2, label='smoothed', color='blue')
        
        # Mark the screen position
        ax.axvline(x_screen, color='red', ls='--', lw=1.5, alpha=0.7, label=f'screen at x={x_screen}')
        
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\|QM - Fresnel\|_{L^2(y)}$')
        ax.set_title(f'k={kx0:.1f}, λ={wavelength:.3f}, h={h:.1f}', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_max)
    
    plt.tight_layout()
    plt.savefig('parameter_sweep_error_vs_x.png', dpi=150, bbox_inches='tight')
    print("Error vs x plot saved to parameter_sweep_error_vs_x.png")
    
    plt.show()
