
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from .asyncscanner import AsyncScanManager
from .virtualSGM4 import VirtualSGM4

__version__ = "0.4.1"
__all__ = [AsyncScanManager, VirtualSGM4]
