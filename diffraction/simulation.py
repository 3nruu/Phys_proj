from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Small epsilon to avoid divide-by-zero in normalization
NORM_EPS = 1e-12

@dataclass
class SimulationConfig:
    dt: float = 0.01
    hbar: float = 1.0
    m: float = 1.0
    kx0: float = 20.0
    ky0: float = 0.0
    delta_x: float = 1.0
    delta_y: float = 1.0
    x0: float = -2.0
    y0: float = 0.0
    x_min: float = -6.0
    x_max: float = 10.0
    y_min: float = -12.0
    y_max: float = 12.0
    dx: float = 0.1
    dy: float = 0.1
    screen_x: float = 5.0
    nb_frame: int = 250

def build_grids(cfg: SimulationConfig):
    x = np.arange(cfg.x_min, cfg.x_max + cfg.dx, cfg.dx)
    y = np.arange(cfg.y_min, cfg.y_max + cfg.dy, cfg.dy)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return x, y, xx, yy

def make_initial_gaussian(xx, yy, cfg: SimulationConfig):
    return (1/(2*cfg.delta_x**2*np.pi)**(1/4) *
            1/(2*cfg.delta_y**2*np.pi)**(1/4) *
            np.exp(-((xx-cfg.x0)/(2*cfg.delta_x)) ** 2) *
            np.exp(-((yy-cfg.y0)/(2*cfg.delta_y)) ** 2) *
            np.exp(1j * (cfg.kx0*xx + cfg.ky0*yy)))

def normalize_1d(arr, y):
    norm = np.trapezoid(arr, y)
    if norm > NORM_EPS:
        return arr / norm
    return np.zeros_like(arr)

def results_path(base_dir: Path, filename: str) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir / filename
