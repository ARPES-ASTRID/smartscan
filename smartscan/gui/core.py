# Purpose: Core elements of the SmartScan program
import logging
import multiprocessing as mp
import sys

import numpy as np
from PyQt5 import QtCore
import traceback

from smartscan import tasks, preprocessing

from .gp import GPManager
from .sgm4 import DataFetcher

class SmartScanManager(QtCore.QObject):

    new_raw_data = QtCore.pyqtSignal(np.ndarray)
    new_reduced_data = QtCore.pyqtSignal(np.ndarray)
    new_hyperparameters = QtCore.pyqtSignal(np.ndarray)
    new_points = QtCore.pyqtSignal(np.ndarray)

    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def __init__(self, parent=None, settings: dict=None):
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.ThreadManager")
        self.logger.debug("init ThreadManager")
        
        self.p = parent
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
        self.pool.setMaxThreadCount(self.n_processors)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.settings['timers'].get('master',50)) # in ms
        self.timer.timeout.connect(self.update)
        self.timer.start()

    def update(self) -> None:
        """ run timed tasks """
        if not self.timer.isActive():
            return
        if not self.data_fetcher_thread.isRunning():
            self.start_data_fetcher()
        if not self.gp_manager_thread.isRunning():
            self.start_gp()
        if not self.should_stop:
            self.check_queues()
        else:
            self.stop()
        
    def check_queues(self) -> None:
        """ check the queues for new data"""
        if not self.raw_data_queue.empty():
            self.reduce_data(self.raw_data_queue.get())
        if not self.reduced_data_queue.empty():
            self.gp_manager.update_data_and_positions(self.processed_data_queue.get())
        if not self.hyperparameters_queue.empty():
            # self.gp_manager.update_hyperparameters(self.hyperparameters_queue.get())
            pass
        if not self.points_queue.empty():
            pass

    def reduce_data(self, data: np.ndarray) -> None:
        """ reduce data """
        self.logger.debug(f"Reducing data: {data.shape}")
        self.raw_data_history.append(data)
        runnable = Runnable(
            reduce, 
            data,
            settings=self.settings,
            logger=self.logger,
            )
        self.pool.start(runnable)
        runnable.result.connect(self.on_reduced_data)
        runnable.error.connect(self.on_thread_error)
        # runnable.finished.connect(self.on_thread_finished)

    def on_reduced_data(self, data: tuple[np.ndarray]) -> None:
        """Handle the reduced data from the thread."""
        self.logger.debug(f"Reduced data: pos{data[0]} shape {data.shape}")
        self.reduced_data_history.append(data)
        self.new_reduced_data.emit(data)
        self.reduced_data_queue.put(data)

    def on_thread_error(self, error: str) -> None:
        """Handle the error signal from the thread."""
        self.logger.error(error)
        self.error.emit(error)

    def start_gp(self) -> None:
        """Start the GP loop."""
        self.logger.debug("Creating GP manager")
        self.gp_manager = GPManager(parent=self)
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
        self.points_queue.put(points)
        raise NotImplementedError("on_new_points not implemented")
    
    def on_new_hyperparameters(self, hyperparameters: np.ndarray) -> None:
        """Handle the new_hyperparameters signal from the GP loop."""
        self.logger.debug(f"New hyperparameters: {hyperparameters}")
        self.new_hyperparameters.emit(hyperparameters)
        self.hyperparameters_history.append(hyperparameters)
    
    def start_data_fetcher(self) -> None:
        """ Start the data fetcher """
        self.logger.debug("Creating data fetcher")
        self.data_fetcher = DataFetcher(parent=self)
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
        self.data_fetcher.stop()

    def pause_data_fetcher(self) -> None:
        """Pause the data fetcher."""
        self.logger.debug("Pausing data fetcher.")
        self.data_fetcher.pause()

    def resume_data_fetcher(self) -> None:
        """Resume the data fetcher."""
        self.logger.debug("Resuming data fetcher.")
        self.data_fetcher.resume()

    def on_data_fetcher_finished(self) -> None:
        """Handle the finished signal from the data fetcher."""
        self.logger.debug("Data fetcher finished.")
        self.data_fetcher.deleteLater()

    def on_data_fetcher_status(self, status: str) -> None:
        """Handle the status signal from the data fetcher."""
        self.logger.debug(f"Data fetcher status: {status}")

    def on_data_fetcher_error(self, error: str) -> None:
        """Handle the error signal from the data fetcher."""
        self.logger.error(error)
        self.error.emit(error)

    def on_new_data(self, data: np.ndarray) -> None:
        """Handle the new_data signal from the data fetcher."""
        self.logger.debug(f"New data: {data}")
        self.new_raw_data.emit(data)
        self.raw_data_queue.put(data)

    def start(self) -> None:
        """Start the scan."""
        self.start_gp()
        self.start_TCP()

    def stop(self) -> None:
        """Stop the scan."""
        self.stop_gp()
        self.stop_TCP()

    def pause(self) -> None:
        """Pause the scan."""
        self.pause_gp()

    def resume(self) -> None:
        """Resume the scan."""
        self.resume_gp()

    def END(self) -> None:
        """Stop the scan."""
        self.stop()

    def __del__(self) -> None:
        """Delete the object."""
        self.logger.debug("Deleting ThreadManager")
        self.stop()


class Runnable(QtCore.QRunnable):
    """A runnable object for multithreading."""

    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(tuple)
    result = QtCore.pyqtSignal(object)

    def __init__(self, function: callable, *args, **kwargs):
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def run(self) -> None:
        """Run the function."""
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done



def reduce(data: np.ndarray, settings: dict, logger: logging.Logger) -> np.ndarray:
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
    return reduced