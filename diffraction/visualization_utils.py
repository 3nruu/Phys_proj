import numpy as np

def compute_smart_color_limits(data, percentile_high=99.5, percentile_low=1):
    data_flat = data.flatten()
    data_positive = data_flat[data_flat > 0]
    
    if len(data_positive) == 0:
        return 0, 1
    
    vmin = np.percentile(data_positive, percentile_low)
    vmax = np.percentile(data_positive, percentile_high)
    
    if vmax - vmin < 0.01 * np.max(data_positive):
        vmin = 0
        vmax = np.max(data_positive)
    
    return float(vmin), float(vmax)

def normalize_for_display(data, vmin=None, vmax=None):
    if vmin is None or vmax is None:
        vmin, vmax = compute_smart_color_limits(data)
    
    scaled = (data - vmin) / (vmax - vmin + 1e-12)
    scaled = np.clip(scaled, 0, 1)
    return scaled, vmin, vmax