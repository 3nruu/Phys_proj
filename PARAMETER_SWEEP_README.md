# Parameter Sweep and Optimization Scripts

This document describes the new scripts for parameter iteration and optimization of the QM vs Fresnel comparison.

## Files

### 1. `parameter_sweep.py`
A comprehensive parameter sweep script that:
- Iterates over different wave numbers (kx0), hole widths (h), and screen distances
- Runs QM simulations and compares with Fresnel diffraction
- Generates comparison plots for all parameter combinations
- Produces error analysis plots

**Key Features:**
- Configurable parameter ranges at the top of the script
- Saves all results to `parameter_sweep_results.pkl`
- Creates two output plots:
  - `parameter_sweep_comparison.png`: Grid of all QM vs Fresnel comparisons
  - `parameter_sweep_error_analysis.png`: Error trends vs each parameter
  - `parameter_sweep_error_vs_x.png`: Error as a function of distance from slits for each parameter combination

**Usage:**
```bash
python parameter_sweep.py
```

**Configuration:**
Edit these variables in the script to change parameter ranges:
```python
kx0_values = [20.0, 25.0, 30.0]  # wave numbers
d_values = [1.4, 1.5, 1.55]      # slit separations
h_values = [1.0, 1.5, 2.0]       # hole widths
# Screen is automatically calculated as d²/λ
```

**Screen Placement:**
The screen is automatically placed at **x = d²/λ** where:
- d = slit separation (varied parameter)
- λ = de Broglie wavelength = 2π/kx0

This corresponds to **Fresnel number F = 1** for all simulations, the characteristic distance for Fresnel diffraction.

**Screen Distances** (all within x_max = 12):
- kx0=20, d=1.40 → **x = 6.24** ✓
- kx0=20, d=1.50 → **x = 7.16** ✓
- kx0=25, d=1.55 → **x = 9.56** ✓
- kx0=30, d=1.55 → **x = 11.47** ✓ (maximum)

All 27 combinations (3×3×3) produce screens within the simulation domain!

### 2. `2_sl_wK_optimized.py`
Optimized version of `2_sl_wK.py` with performance improvements:

**Optimizations:**
1. **Fewer animation frames**: 100 frames (vs 200) → 50% faster animation
2. **Pre-allocated arrays**: All arrays created at initialization
3. **Progress indicators**: Print updates every 10 frames
4. **Same grid resolution**: dx = dy = 0.08 (maintains accuracy)

**Expected speedup**: ~2x faster than original (mainly from fewer frames)

**Usage:**
```bash
python 2_sl_wK_optimized.py
```

**Output:**
- `QM_vs_Fresnel_Optimized.gif`: Animation (smaller file size)
- `QM_vs_Fresnel_Optimized.pkl`: Comparison data

## Parameter Descriptions

### Wave Number (kx0)
- Physical meaning: Momentum of the particle (p = ħk)
- Related to wavelength: λ = 2π/kx0
- Higher kx0 → shorter wavelength → less diffraction
- Range: 15-30 (limited to keep screen in view)

### Slit Separation (d)
- Physical meaning: Distance between the centers of the two slits
- Determines the screen position (x = d²/λ)
- Affects fringe spacing
- Range: 1.4-1.55 (carefully chosen to keep screen < 12)

### Hole Width (h)
- Physical meaning: Height of each slit opening
- Affects the amount of diffraction
- Larger h → more light passes through
- Range: 1.0-2.0

### Screen Distance (x_screen)
- **Automatically calculated** as x = d²/λ
- Physical meaning: Characteristic distance for Fresnel diffraction
- Corresponds to Fresnel number **F = 1** for all cases
- Time relationship: t = x_screen / kx0
- Varies from 6.2 to 11.5 (all within simulation domain)

## Understanding the Results

### L2 Error Metric
The L2 error quantifies the difference between QM and Fresnel predictions:
```
L2_error = sqrt(∫(I_QM - I_Fresnel)² dy)
```

**Expected trends:**
- Error **decreases** with increasing kx0 (shorter wavelength)
- Error increases with distance (more quantum effects accumulate)
- Error depends on slit geometry

### When Fresnel Approximation Works Best
The Fresnel-Kirchhoff approximation is valid when:
1. **High momentum**: kx0 >> 1 (short wavelength limit)
2. **Far field**: x >> h² / λ (Fraunhofer regime)
3. **Paraxial**: Small angles from optical axis

## Performance Notes

### Original vs Optimized
- Grid: ~175 × 151 = 26,425 grid points (same as original)
- Animation frames: 50% reduction (100 vs 200)
- Overall speedup: ~2x (mainly from fewer frames)

### Parameter Sweep Performance
For default configuration (3×3×3 = 27 simulations):
- Estimated time: ~10-20 minutes (depends on hardware)
- Screen distances range from 6.2 to 11.5, keeping evolution times reasonable
- Memory usage: ~2-3 GB
- Can be parallelized for faster execution (see comments in code)

## Customization

### Adding More Parameters
To add more parameters to sweep (e.g., slit separation or screen distance):
```python
d_values = [3.0, 4.0, 5.0]  # Slit separations
x_screen_values = [5.0, 7.0, 9.0]  # Different screen distances

# Update param_combinations:
for kx0 in kx0_values:
    for h in h_values:
        for d in d_values:
            for x_screen in x_screen_values:
                param_combinations.append((kx0, h, d, x_screen))

# In run_simulation:
kx0, h, d, x_screen_fixed = params
y1 = -d / 2
y2 = d / 2
```

### Enabling Parallel Processing
Uncomment the parallel processing section in `parameter_sweep.py`:
```python
# from multiprocessing import Pool
# with Pool(processes=4) as pool:
#     results = pool.map(run_simulation, param_combinations)
```

## Troubleshooting

### Out of Memory
- Reduce grid resolution (increase dx/dy to 0.10 or 0.12)
- Reduce number of frames
- Run fewer parameter combinations at once

### Slow Execution
- Use the optimized version (fewer frames)
- Enable parallel processing
- Reduce parameter ranges
- Use coarser grid (dx=dy=0.10 provides ~2.5x speedup with minor quality loss)

### Poor Accuracy
- Decrease time step dt (currently 0.005)
- Increase grid resolution (decrease dx/dy to 0.06)
- Check that grid boundaries are far enough from the wave

## References

See main README.md for physics background and equations.
