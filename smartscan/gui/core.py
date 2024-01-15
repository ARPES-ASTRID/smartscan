# Purpose: Core elements of the SmartScan program
from __future__ import annotations
import logging
from pathlib import Path
import multiprocessing as mp
import sys
from typing import Any

import numpy as np
import yaml
from PyQt5 import QtCore
import traceback

from smartscan import tasks, preprocessing

from .gp import GPManager
from .sgm4 import DataFetcher

class SmartScanManager(QtCore.QObject):

    new_raw_data = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    new_reduced_data = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    new_hyperparameters = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    new_points = QtCore.pyqtSignal(np.ndarray)

    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.ThreadManager")
        self.logger.debug("init ThreadManager")

        self.settings = settings

        self.gp_manager = None
        self.gp_manager_thread = None
        self.data_fetcher = None
        self.data_fetcher_thread = None

        self.raw_data_queue = mp.Queue()
        self.raw_data_history = []
        self.reduced_data_queue = mp.Queue()
        self.reduced_data_history = []
        self.hyperparameters_queue = mp.Queue()
        self.hyperparameters_history = []
        self.points_queue = mp.Queue()
        self.points_history = []

        self.pool = QtCore.QThreadPool()
        self.pool.setMaxThreadCount(self.settings['core'].get('n_threads',4))
        self.logger.debug(f"Max thread count: {self.pool.maxThreadCount()}")
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.settings['core'].get('master_clock',50)) # in ms
        self.timer.timeout.connect(self.update)
        self.logger.debug(f"Master clock: {self.timer.interval()} ms")

        self._should_stop = False

    @QtCore.pyqtSlot()
    def start(self) -> None:
        """Start the scan."""
        self.start_gp()
        self.start_data_fetcher()
        self.timer.start()

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        """Stop the scan."""
        self.logger.info('Stopping all scan loops')
        # self._should_stop = True
        self.stop_gp()
        self.stop_data_fetcher()
        try:
            self.timer.stop()
        except RuntimeError:
            pass
        self.logger.info('All scan loops stopped')
        self.finished.emit()

    @QtCore.pyqtSlot()
    def update(self) -> None:
        """ run timed tasks """
        if not self.timer.isActive():
            return
        self.data_fetcher.fetch_data()
        if not self.should_stop:
            self.check_queues()
        else:
            self.stop()
    
    @property
    def should_stop(self) -> bool:
        """ check if the scan should stop """
        return self._should_stop
    def check_queues(self) -> None:
        """ check the queues for new data"""
        if not self.raw_data_queue.empty():
            pos, data = self.raw_data_queue.get()
            self.reduce_data(pos,data)
        if not self.reduced_data_queue.empty():
            pos, data = self.reduced_data_queue.get()
            self.gp_manager.update_data_and_positions(pos, data)
        if not self.hyperparameters_queue.empty():
            # self.gp_manager.update_hyperparameters(self.hyperparameters_queue.get())
            pass
        if not self.points_queue.empty():
            pass

    # @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def reduce_data(self, pos:np.ndarray, data: np.ndarray) -> None:
        """ reduce data """
        self.logger.debug(f"Reducing data: pos {pos} {data.shape}")
        self.raw_data_history.append((pos,data))
        runnable = Runnable(
            reduce,
            pos,
            data,
            settings=self.settings,
            logger=self.logger,
            )
        self.pool.start(runnable)
        runnable.signals.result.connect(self.on_reduced_data)
        runnable.signals.error.connect(self.on_thread_error)
        # runnable.finished.connect(self.on_thread_finished)

    def start_gp(self) -> None:
        """Start the GP loop."""
        self.logger.debug("Creating GP manager")
        self.gp_manager = GPManager(settings=self.settings)
        self.gp_manager_thread = QtCore.QThread()
        self.logger.debug("Moving GP manager to thread")
        self.gp_manager.moveToThread(self.gp_manager_thread)

        self.gp_manager.finished.connect(self.on_gp_finished)
        self.gp_manager.status.connect(self.on_gp_status)
        self.gp_manager.error.connect(self.on_gp_error)
        self.gp_manager.new_points.connect(self.on_new_points)
        self.gp_manager.new_hyperparameters.connect(self.on_new_hyperparameters)
        self.gp_manager_thread.started.connect(self.gp_manager.run)

        self.logger.debug("Starting GP loop.")
        self.gp_manager_thread.start()

    def stop_gp(self) -> None:
        """Stop the GP loop."""
        self.logger.debug("Stopping GP loop.")
        if self.gp_manager is not None:
            try:
                self.gp_manager.stop()
            except AttributeError:
                self.logger.error("GP manager not running.")

    def start_data_fetcher(self) -> None:
        """ Start the data fetcher """
        self.logger.debug("Creating data fetcher")
        self.data_fetcher = DataFetcher(settings=self.settings)
        self.data_fetcher_thread = QtCore.QThread()
        self.logger.debug("Moving data fetcher to thread")
        self.data_fetcher.moveToThread(self.data_fetcher_thread)

        self.data_fetcher.finished.connect(self.on_data_fetcher_finished)
        self.data_fetcher.status.connect(self.on_data_fetcher_status)
        self.data_fetcher.error.connect(self.on_data_fetcher_error)
        self.data_fetcher.new_data.connect(self.on_new_data)
        self.data_fetcher_thread.started.connect(self.data_fetcher.run)

        self.logger.debug("Starting data fetcher.")
        self.data_fetcher_thread.start()

    def stop_data_fetcher(self) -> None:
        """Stop the data fetcher."""
        self.logger.debug("Stopping data fetcher.")
        if self.data_fetcher is not None:
            try:
                self.data_fetcher.stop()
            except AttributeError:
                self.logger.error("Data fetcher not running.")

    @QtCore.pyqtSlot(object)
    def on_reduced_data(self, result:object) -> None:
        """Handle the reduced data from the thread."""
        try:
            pos, data = result
            self.logger.debug(f"Reduced data: pos{pos} shape {data.shape}")
            self.reduced_data_history.append((pos,data))
            self.new_reduced_data.emit(pos,data)
            self.reduced_data_queue.put((pos,data))
        except Exception as e:
            self.logger.error(f"{type(e)} while handling reduced data: {e} \n {traceback.format_exc()}")
            self.error.emit(str(e))

    @QtCore.pyqtSlot(tuple)
    def on_thread_error(self, error: str) -> None:
        """Handle the error signal from the thread."""
        exctype, value, traceback_ = error
        self.logger.error(value)
        self.error.emit(value)

    @QtCore.pyqtSlot()
    def on_gp_finished(self) -> None:
        """Handle the finished signal from the GP loop."""
        self.logger.debug("GP loop finished.")
        self.gp_manager.deleteLater()
    
    @QtCore.pyqtSlot(str)
    def on_gp_status(self, status: str) -> None:
        """Handle the status signal from the GP loop."""
        self.logger.debug(f"GP loop status: {status}")
        self.status.emit(status)
    
    @QtCore.pyqtSlot(str)
    def on_gp_error(self, error: str) -> None:
        """Handle the error signal from the GP loop."""
        self.logger.error(error)
        self.error.emit(error)

    @QtCore.pyqtSlot(np.ndarray)
    def on_new_points(self, points: np.ndarray) -> None:
        """Handle the new_points signal from the GP loop."""
        self.logger.debug(f"New points: {points}")
        self.new_points.emit(points)
        self.points_queue.put(points)
        raise NotImplementedError("on_new_points not implemented")
    
    @QtCore.pyqtSlot(np.ndarray)
    def on_new_hyperparameters(self, hyperparameters: np.ndarray) -> None:
        """Handle the new_hyperparameters signal from the GP loop."""
        self.logger.debug(f"New hyperparameters: {hyperparameters}")
        self.new_hyperparameters.emit(hyperparameters)
        self.hyperparameters_history.append(hyperparameters)
    
    @QtCore.pyqtSlot()
    def on_data_fetcher_finished(self) -> None:
        """Handle the finished signal from the data fetcher."""
        self.logger.debug("Data fetcher finished.")
        self.data_fetcher.deleteLater()

    @QtCore.pyqtSlot(str)
    def on_data_fetcher_status(self, status: str) -> None:
        """Handle the status signal from the data fetcher."""
        self.logger.debug(f"Data fetcher status: {status}")

    @QtCore.pyqtSlot(str)
    def on_data_fetcher_error(self, error: str) -> None:
        """Handle the error signal from the data fetcher."""
        self.logger.error(error)
        self.error.emit(error)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def on_new_data(self, pos:np.ndarray, data: np.ndarray) -> None:
        """Handle the new_data signal from the data fetcher."""
        self.logger.debug(f"New data: {data}")
        self.new_raw_data.emit(pos,data)
        self.raw_data_queue.put((pos,data))

    def __del__(self) -> None:
        """Delete the object."""
        self.logger.debug("Deleting ThreadManager")
        try:
            self.stop()
        except Exception as e:
            pass
            # self.logger.error(f"{str(type(e))} stopping the scan: {e}")

class RunnableSignals(QtCore.QObject):
    """Signals for the Runnable object."""

    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)

class Runnable(QtCore.QRunnable):
    """A runnable object for multithreading."""

    def __init__(self, function: callable, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = RunnableSignals()

    def run(self) -> None:
        """Run the function."""
        try:
            result = self.function(*self.args, **self.kwargs)
        except Exception:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


class Settings:

    def __init__(self, settings_file: str | Path) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.Settings")
        self._settings_file = Path(settings_file)
        self.logger.debug(f"Settings file: {self._settings_file}")
        assert self._settings_file.exists(), f"Settings file {self._settings_file} does not exist."
        self.update()
        self.logger = logging.getLogger(f"{__name__}.Settings")
        self.logger.debug("init Settings")

    def update(self) -> None:
        """ read the settings file"""
        self.logger.debug(f"Reading settings file: {self._settings_file}")
        with open(self._settings_file) as f:
            self._settings = yaml.load(f, Loader=yaml.FullLoader)

    def __getitem__(self, key: str) -> Any:
        self.logger.debug(f"Getting settings key: {key}")
        return self._settings[key]
    
    def save(self, path: str | Path) -> None:
        """ save the settings file """
        self.logger.debug(f"Saving settings file: {path}")
        with open(path, 'w') as f:
            yaml.dump(self._settings, f)

def reduce(pos: np.ndarray, data: np.ndarray, settings: dict, logger: logging.Logger) -> np.ndarray:
    """ reduce data """
    logger.debug(f"Reducing data: {data.shape}")
    # t0 = time.time()
    pp = data.copy()
    for _, d in settings["preprocessing"].items():
        func = getattr(preprocessing, d["function"])
        kwargs = d.get("params", {})
        if kwargs is None:
            pp = func(pp)
        else:
            pp = func(pp, **kwargs)
    # last_spectrum = pp
    # t1 = time.time()
    # self.logger.debug(
    #     f"Preprocessing {pos} | shape {pp.shape} | mean : {pp.mean():.3f} ± {pp.std():.3f} | time: {t1-t0:.3f} s"
    # )

    # # reduce data
    reduced = []
    for _, d in settings["tasks"].items():
        func = getattr(tasks, d["function"])
        kwargs = d.get("params", {})
        if kwargs is None:
            reduced.append(func(pp))
        else:
            reduced.append(func(pp, **kwargs))
    reduced = np.asarray(reduced, dtype=float).flatten()
    # if len(reduced) != len(self.task_labels):
    #     raise RuntimeError(
    #         f"Length mismatch between tasks {len(reduced)}"
    #         f"and task labels {len(self.task_labels)}."
    #     )
    # logger.debug(f"Reduction {pos} | time: {time.time()-t1:.3f} s")
    return pos, reduced