import os
# Fix for Intel OpenMP runtime error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .asyncscanner import AsyncScanManager
from .virtualSGM4 import VirtualSGM4

__version__ = "0.5.0"
__all__ = [AsyncScanManager, VirtualSGM4]
