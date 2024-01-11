# Purpose: Core elements of the SmartScan program
from hmac import new
import logging
from math import e
import time
from functools import partial

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from fvgp import fvGPOptimizer

from smartscan.gp import cost_functions, aquisition_functions
from smarstcan.utilities import pretty_print_time
from .gp import GPManager
from .tcp import TCPManager

class SmartScanManager(QtCore.QObject):

    new_raw_data = QtCore.pyqtSignal(np.ndarray)
    new_processed_data = QtCore.pyqtSignal(np.ndarray)
    new_hyperparameters = QtCore.pyqtSignal(np.ndarray)
    new_points = QtCore.pyqtSignal(np.ndarray)

    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, settings: dict=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.ThreadManager")
        self.logger.debug("init ThreadManager")
        self.threadpool = QtCore.QThreadPool()
        self.logger.debug(
            f"Multithreading with maximum {self.threadpool.maxThreadCount()} threads"
        )
        self.gp_manager = GPManager(parent=self)
        self.TCP_manager = TCPManager(parent=self)

    def start_gp(self) -> None:
        """Start the GP loop."""
        self.logger.debug("Starting GP loop.")
        self.gp_manager.moveToThread(self.threadpool)
        self.gp_manager.finished.connect(self.on_gp_finished)
        self.gp_manager.status.connect(self.on_gp_status)
        self.gp_manager.error.connect(self.on_gp_error)
        self.gp_manager.new_points.connect(self.on_new_points)
        self.gp_manager.new_hyperparameters.connect(self.on_new_hyperparameters)
        self.threadpool.start(self.gp_manager)

    def stop_gp(self) -> None:
        """Stop the GP loop."""
        self.logger.debug("Stopping GP loop.")
        self.gp_manager.stop()
    
    def pause_gp(self) -> None:
        """Pause the GP loop."""
        self.logger.debug("Pausing GP loop.")
        self.gp_manager.pause()

    def resume_gp(self) -> None:
        """Resume the GP loop."""
        self.logger.debug("Resuming GP loop.")
        self.gp_manager.resume()

    def on_gp_finished(self) -> None:
        """Handle the finished signal from the GP loop."""
        self.logger.debug("GP loop finished.")

        self.gp_manager.deleteLater()

    def on_gp_status(self, status: str) -> None:
        """Handle the status signal from the GP loop."""
        self.logger.debug(f"GP loop status: {status}")
        self.status.emit(status)
    
    def on_gp_error(self, error: str) -> None:
        """Handle the error signal from the GP loop."""
        self.logger.error(error)
        self.error.emit(error)

    def on_new_points(self, points: np.ndarray) -> None:
        """Handle the new_points signal from the GP loop."""
        self.logger.debug(f"New points: {points}")
        self.new_points.emit(points)
        raise NotImplementedError("on_new_points not implemented")
    
    def on_new_hyperparameters(self, hyperparameters: np.ndarray) -> None:
        """Handle the new_hyperparameters signal from the GP loop."""
        self.logger.debug(f"New hyperparameters: {hyperparameters}")
        self.new_hyperparameters.emit(hyperparameters)
        raise NotImplementedError("on_new_hyperparameters not implemented")
    