# Electron Diffraction — Fresnel vs Schrödinger

## Structure

```t
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

```bash
python scripts/run_animation.py
python scripts/analyze_results.py
```
