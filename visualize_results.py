#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualize saved parameter sweep results
Useful for re-plotting without re-running simulations
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

def running_mean(a, window=7):
    return np.convolve(a, np.ones(window)/window, mode='same')

def load_and_plot(filename='parameter_sweep_results.pkl'):
    """Load results and create all plots"""
    
    print(f"Loading results from {filename}...")
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    
    print(f"Loaded {len(results)} simulation results")
    
    # Extract parameter values
    kx0_values = sorted(list(set(r['kx0'] for r in results)))
    h_values = sorted(list(set(r['h'] for r in results)))
    x_screen_values = sorted(list(set(r['x_screen'] for r in results)))
    
    print(f"Parameters:")
    print(f"  kx0: {kx0_values}")
    print(f"  h: {h_values}")
    print(f"  x_screen: {x_screen_values}")
    
    x_min, x_max = -2.0, 12.0
    y_min, y_max = -6.0, 6.0
    
    # -------------------------------------------------
    # Plot 1: Comparison at screen
    # -------------------------------------------------
    print("\nCreating comparison plots...")
    n_cols = len(kx0_values)
    n_rows = len(h_values) * len(x_screen_values)
    
    fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes1 = np.array([[axes1]])
    elif n_rows == 1:
        axes1 = axes1.reshape(1, -1)
    elif n_cols == 1:
        axes1 = axes1.reshape(-1, 1)
    
    for idx, result in enumerate(results):
        kx0 = result['kx0']
        h = result['h']
        x_screen = result['x_screen']
        y = result['y']
        qm = result['qm_slice']
        fresnel = result['fresnel_slice']
        error = result['error_L2']
        wavelength = result['wavelength']
        
        kx0_idx = kx0_values.index(kx0)
        h_idx = h_values.index(h)
        x_screen_idx = x_screen_values.index(x_screen)
        
        row = h_idx * len(x_screen_values) + x_screen_idx
        col = kx0_idx
        
        ax = axes1[row, col]
        
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
    plt.savefig('replot_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved: replot_comparison.png")
    
    # -------------------------------------------------
    # Plot 2: Error vs x distance
    # -------------------------------------------------
    print("Creating error vs x plots...")
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes2 = np.array([[axes2]])
    elif n_rows == 1:
        axes2 = axes2.reshape(1, -1)
    elif n_cols == 1:
        axes2 = axes2.reshape(-1, 1)
    
    for idx, result in enumerate(results):
        kx0 = result['kx0']
        h = result['h']
        x_screen = result['x_screen']
        x_error = result['x_error']
        error_L2_vs_x = result['error_L2_vs_x']
        wavelength = result['wavelength']
        
        error_smooth = running_mean(error_L2_vs_x, window=9)
        
        kx0_idx = kx0_values.index(kx0)
        h_idx = h_values.index(h)
        x_screen_idx = x_screen_values.index(x_screen)
        
        row = h_idx * len(x_screen_values) + x_screen_idx
        col = kx0_idx
        
        ax = axes2[row, col]
        
        ax.plot(x_error, error_L2_vs_x, alpha=0.4, label='raw', color='gray')
        ax.plot(x_error, error_smooth, lw=2, label='smoothed', color='blue')
        ax.axvline(x_screen, color='red', ls='--', lw=1.5, alpha=0.7, label=f'screen at x={x_screen}')
        
        ax.set_xlabel('x')
        ax.set_ylabel(r'$\|QM - Fresnel\|_{L^2(y)}$')
        ax.set_title(f'k={kx0:.1f}, λ={wavelength:.3f}, h={h:.1f}', fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, x_max)
    
    plt.tight_layout()
    plt.savefig('replot_error_vs_x.png', dpi=150, bbox_inches='tight')
    print("Saved: replot_error_vs_x.png")
    
    # -------------------------------------------------
    # Plot 3: Full maps comparison for a selected case
    # -------------------------------------------------
    print("Creating full map comparison for selected case...")
    
    # Choose middle parameter case
    mid_result = results[len(results)//2]
    
    qm_full_map = mid_result['qm_full_map']
    fresnel_full_map = mid_result['fresnel_full_map']
    diff_map = np.abs(qm_full_map - fresnel_full_map)
    
    x = np.linspace(x_min, x_max, qm_full_map.shape[0])
    y = mid_result['y']
    
    fig3, axs = plt.subplots(1, 3, figsize=(16, 4))
    
    im0 = axs[0].imshow(
        qm_full_map.T,
        extent=[x_min, x_max, y_min, y_max],
        origin='lower',
        aspect='auto',
        cmap='jet'
    )
    axs[0].set_title(f"QM (k={mid_result['kx0']}, h={mid_result['h']})")
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
    axs[1].set_title("Fresnel")
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
    axs[2].set_title("|QM − Fresnel|")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    plt.colorbar(im2, ax=axs[2])
    
    plt.tight_layout()
    plt.savefig('replot_full_maps.png', dpi=150, bbox_inches='tight')
    print("Saved: replot_full_maps.png")
    
    # -------------------------------------------------
    # Summary statistics
    # -------------------------------------------------
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for result in results:
        print(f"kx0={result['kx0']:5.1f}, h={result['h']:4.1f}, "
              f"x={result['x_screen']:4.1f} → "
              f"L2 error={result['error_L2']:.6f}, "
              f"time={result['time']:.1f}s")
    
    print("="*60)
    print(f"\nTotal runtime: {sum(r['time'] for r in results):.1f}s")
    print(f"Average per simulation: {np.mean([r['time'] for r in results]):.1f}s")
    
    # Find best/worst cases
    best = min(results, key=lambda r: r['error_L2'])
    worst = max(results, key=lambda r: r['error_L2'])
    
    print(f"\nBest agreement: kx0={best['kx0']}, h={best['h']}, "
          f"x={best['x_screen']}, error={best['error_L2']:.6f}")
    print(f"Worst agreement: kx0={worst['kx0']}, h={worst['h']}, "
          f"x={worst['x_screen']}, error={worst['error_L2']:.6f}")
    
    plt.show()
    print("\nVisualization complete!")

if __name__ == '__main__':
    import sys
    
    filename = 'parameter_sweep_results.pkl'
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    
    try:
        load_and_plot(filename)
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        print("Please run parameter_sweep.py first to generate results.")
        print("\nUsage: python visualize_results.py [filename.pkl]")
