import os
# Fix for Intel OpenMP runtime error on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 

from smartscan.smartscan import run # noqa: F402

__all__ = ["run"]
__version__ = "0.5.0"
