from dataclasses import dataclass
import numpy as np

@dataclass
class DoubleSlitConfig:
    x_slit: float = 0.0
    slit_width: float = 0.3
    y_lower_min: float = -12.0
    y_lower_max: float = -0.5
    y_upper_min: float = 0.5
    y_upper_max: float = 12.0
    V0: float = 400.0  # barrier height

def build_double_slit_mask(x, y, cfg: DoubleSlitConfig):
    x = np.asarray(x)
    y = np.asarray(y)
    Nx, Ny = len(x), len(y)
    mask = np.zeros((Nx, Ny), dtype=bool)

    xx, yy = np.meshgrid(x, y, indexing="ij")

    # Lower barrier
    lower = (xx >= cfg.x_slit) & (xx <= cfg.x_slit + cfg.slit_width) & (yy >= cfg.y_lower_min) & (yy <= cfg.y_lower_max)
    # Upper barrier
    upper = (xx >= cfg.x_slit) & (xx <= cfg.x_slit + cfg.slit_width) & (yy >= cfg.y_upper_min) & (yy <= cfg.y_upper_max)

    mask[lower] = True
    mask[upper] = True
    return mask

def mask_to_potential(mask, V0):
    mask_yx = mask.T 
    V = np.zeros(mask.size)
    V[mask_yx.flatten(order="F")] = V0
    return V
