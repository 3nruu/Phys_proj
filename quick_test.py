#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Quick test script with reduced parameters for fast testing
Perfect for verifying everything works before running full parameter sweep
"""

import matplotlib.pyplot as plt
import numpy as np
from WaveFunction1 import WaveFunction
from FresnelDiffraction_1D import Fresnel1D
import pickle
import time

plt.rcParams.update({'font.size': 8})

# -------------------------------------------------
# Quick test configuration
# -------------------------------------------------
kx0_values = [20.0, 25.0]  # 2 wave numbers
d_values = [1.5]  # 1 slit separation (small enough to keep screen < 12)
h_values = [1.5]  # 1 hole width
# Screen will be placed at d²/λ
# For k=20, d=1.5: x = 1.5²/(2π/20) = 7.2 ✓
# For k=25, d=1.5: x = 1.5²/(2π/25) = 9.0 ✓

# Fixed parameters
dt = 0.005  # Original time step for accuracy
hbar = 1.0
m = 1.0

x_min, x_max = -2.0, 12.0
y_min, y_max = -6.0, 6.0
dx = dy = 0.08  # Original grid resolution

x = np.arange(x_min, x_max + dx, dx)
y = np.arange(y_min, y_max + dy, dy)
Nx, Ny = len(x), len(y)

# Double slit parameters
x_slit, slit_width = 0.0, 0.2

print(f"Grid size: {Nx} x {Ny} = {Nx*Ny} points")
print("=" * 60)

# -------------------------------------------------
# Simulation function
# -------------------------------------------------
def run_simulation(params):
    kx0, d, h = params
    
    wavelength = 2 * np.pi / kx0
    
    # Calculate screen position: d²/λ
    x_screen_fixed = (d**2) / wavelength
    
    # Set slit positions
    y1 = -d / 2
    y2 = d / 2
    
    print(f"Running: kx0={kx0:.1f}, d={d:.1f}, h={h:.1f}")
    print(f"  Screen at x={x_screen_fixed:.2f} (d²/λ)")
    start_time = time.time()
    
    # Create meshgrid
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # Create mask for double slit
    mask = np.zeros((Nx, Ny), dtype=bool)
    mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
         (yy >= y1 - h/2) & (yy <= y1 + h/2)] = True
    mask[(xx >= x_slit) & (xx <= x_slit + slit_width) &
         (yy >= y2 - h/2) & (yy <= y2 + h/2)] = True
    
    # Initial wave function
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
    
    # Evolve to screen
    t_target = x_screen_fixed / kx0
    n_steps = int(t_target / dt)
    
    for _ in range(n_steps):
        S.step()
    
    # Get results
    prob_final = S.get_prob().reshape(Nx, Ny)
    qm_slice = prob_final[ix_fixed, :]
    fresnel_slice = FD.intensity(x_screen_fixed)
    
    # Normalize
    qm_slice_norm = qm_slice / np.trapezoid(qm_slice, y)
    fresnel_slice_norm = fresnel_slice / np.trapezoid(fresnel_slice, y)
    
    # Error
    error_L2 = np.sqrt(np.trapezoid((qm_slice_norm - fresnel_slice_norm)**2, y))
    
    elapsed = time.time() - start_time
    print(f"  ✓ Completed in {elapsed:.2f}s, L2 error: {error_L2:.6f}")
    
    return {
        'kx0': kx0,
        'd': d,
        'h': h,
        'x_screen': x_screen_fixed,
        'wavelength': wavelength,
        'y': y,
        'qm_slice': qm_slice_norm,
        'fresnel_slice': fresnel_slice_norm,
        'error_L2': error_L2,
        'time': elapsed
    }

# -------------------------------------------------
# Main execution
# -------------------------------------------------
if __name__ == '__main__':
    # Generate parameter combinations
    param_combinations = []
    for kx0 in kx0_values:
        for d in d_values:
            for h in h_values:
                param_combinations.append((kx0, d, h))
    
    print(f"Total simulations: {len(param_combinations)}")
    print("=" * 60)
    
    # Run simulations
    results = []
    total_start = time.time()
    
    for params in param_combinations:
        result = run_simulation(params)
        results.append(result)
    
    total_time = time.time() - total_start
    print("=" * 60)
    print(f"✓ All simulations completed in {total_time:.2f}s")
    print(f"  Average: {total_time/len(results):.2f}s per simulation")
    
    # Save results
    with open("quick_test_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Results saved to quick_test_results.pkl")
    
    # -------------------------------------------------
    # Plotting
    # -------------------------------------------------
    n_results = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    for idx, result in enumerate(results):
        ax = axes[idx]
        
        kx0 = result['kx0']
        d = result['d']
        h = result['h']
        x_screen = result['x_screen']
        y = result['y']
        qm = result['qm_slice']
        fresnel = result['fresnel_slice']
        error = result['error_L2']
        wavelength = result['wavelength']
        
        ax.plot(y, qm, 'b-', lw=2.5, label='QM', alpha=0.8)
        ax.plot(y, fresnel, 'r--', lw=2.5, label='Fresnel', alpha=0.8)
        
        ax.set_xlabel('y', fontsize=10)
        ax.set_ylabel('Normalized Intensity', fontsize=10)
        ax.set_title(f'k={kx0:.1f}, d={d:.1f}, h={h:.1f}\nx={x_screen:.1f} (d²/λ)\nL2 error={error:.4f}', 
                     fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('quick_test_comparison.png', dpi=150, bbox_inches='tight')
    print("Plot saved to quick_test_comparison.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for result in results:
        print(f"kx0={result['kx0']:5.1f}, d={result['d']:4.1f}, h={result['h']:4.1f}, "
              f"x={result['x_screen']:5.2f} → L2 error={result['error_L2']:.6f} "
              f"({result['time']:.2f}s)")
    print("=" * 60)
    
    plt.show()
    print("\nQuick test complete! ✓")
