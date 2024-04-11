import os

# Fix for Intel OpenMP runtime error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from smartscan.smartscan import run 
from smartscan.io import load_smartscan, load_h5, save_h5
from smartscan.plot import vornoi_plot

__all__ = ["run"]
__version__ = "1.0.1"
