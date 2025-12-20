# Electron Diffraction — Fresnel vs Schrödinger

## Structure
```
src/
  diffraction/        # reusable modules
    fresnel.py        # Fresnel1D propagator (FFT-based)
    wavefunction.py   # 2D Schrödinger CN solver
    geometry.py       # double-slit config + potential masks
    simulation.py     # grids, initial packet, normalization helpers
  scripts/
    run_animation.py  # runs simulation + saves gif/pkl into results/
    analyze_results.py# generates plots from saved pkl
  results/            # generated gif/png/pkl (gitignored)
  .gitignore
  venv/ (gitignored)
```

## How to run
From `src/`:
```bash
python scripts/run_animation.py
python scripts/analyze_results.py
```
Outputs are written to `src/results/`:
- `Schrodinger_vs_Fresnel_comparison.gif`
- `Schrodinger_vs_Fresnel_Comparison.pkl`
- `comparison_maps.png`, `error_vs_x.png`, `time_evolution_screen.png`, `time_evolution_screen_raw.png`, `error_vs_time.png`

## Key defaults
- kx0 = 20 (λ ≈ 0.3142)
- dt = 0.01, nb_frame = 250
- Grid: x ∈ [-8, 13], y ∈ [-12, 12], dx = dy = 0.1
- Screen at x = 11.0
- Double slit: barriers x ∈ [0, 0.3], openings y ∈ [-0.5, 0.5]

## Notes on normalization
- Screen plots are normalized (∫I dy = 1) to compare fringe shapes.
- Raw data are still saved in the pkl and plotted in `time_evolution_screen_raw.png`.
- Fresnel colormap uses dynamic vmax for visibility; normalization guards avoid division by ~0.
