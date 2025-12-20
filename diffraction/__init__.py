# Core diffraction modules
from .fresnel import Fresnel1D
from .wavefunction import WaveFunctionCN
from .geometry import DoubleSlitConfig, build_double_slit_mask, mask_to_potential
from .simulation import SimulationConfig, build_grids, make_initial_gaussian
