import logging
import time
from functools import partial

from PyQt5 import QtCore
import numpy as np
from gpcam.gp_optimizer import fvGPOptimizer

from smartscan.utils import pretty_print_time
import smartscan.gp.aquisition_functions as aquisition_functions
import smartscan.gp.cost_functions as cost_functions


class GPManager(QtCore.QObject):
    """ Manager for the graphic interface of the SmartScan program

    Args:
        parent (QtWidgets.QWidget, optional): Parent widget. Defaults to None.
        settings (dict, optional): Settings dictionary. Defaults to None.

    Signals:
        new_hyperparameters (QtCore.pyqtSignal): Signal emitted when new hyperparameters are available.
        new_points (QtCore.pyqtSignal): Signal emitted when new points are available.
        finished (QtCore.pyqtSignal): Signal emitted when the scan is finished.
        status (QtCore.pyqtSignal): Signal emitted to update the status bar.
            Can be used to update the status bar of the graphic interface.
            Possible values are:
                - "Running": the GP loop is running
                - "Stopped": the GP loop is stopped
                - "Telling GP": the GP is being told about new data
                - "Training GP": the GP is being trained
                - "Asking GP": the GP is being asked for the next point
                - "Finished": the GP loop is finished
        error (QtCore.pyqtSignal): Signal emitted when an error occurs.
   
    """
    new_hyperparameters = QtCore.pyqtSignal(np.ndarray)
    new_points = QtCore.pyqtSignal(np.ndarray)
    finished = QtCore.pyqtSignal()
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, settings) -> None:
        super().__init__()
        self.logger = logging.getLogger(f"{__name__}.GPManager")
        self.logger.debug("init GPManager")

        self.settings = settings

        self.gp = None

        self._should_stop = False
        self._running = False
        self.iter_counter = 0
        self.hyperparameter_history = {}

        self._values = None
        self._positions = None

        self.task_labels = self.settings["scanning"]["tasks"]
        self.task_normalization_weights = None

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.settings["core"].get("gp_loop_clock", 50))
        self.timer.timeout.connect(self.run)
        

    @property
    def positions(self) -> np.ndarray:
        """Get the list of positions.

        Returns:
            list: List of positions.
        """
        return self._positions

    @property
    def values(self) -> np.ndarray:
        """Get the list of values.

        Returns:
            list: List of values.
        """
        return self._values

    @property
    def should_train(self) -> bool:
        """Check if the GP should be trained.

        Returns:
            bool: True if the GP should be trained, False otherwise.
        """
        return self.iter_counter % self.settings["gp"]["training"]["train_every"] == 0

    @property
    def running(self) -> bool:
        """Check if the GP loop is running.

        Returns:
            bool: True if the GP loop is running, False otherwise.
        """
        return self._running

    @running.setter
    def running(self, value: bool) -> None:
        """Set the running status of the GP loop.

        Args:
            value (bool): True if the GP loop is running, False otherwise.
        """
        self.status.emit("Running" if value else "Stopped")
        self._running = value

    @QtCore.pyqtSlot()
    def stop(self) -> None:
        """Stop the GP loop."""
        self.logger.debug("Stopping GP loop.")
        self._should_stop = True

    @QtCore.pyqtSlot()
    def pause(self) -> None:
        """Pause the GP loop."""
        self.logger.debug("Pausing GP loop.")
        self.running = False

    @QtCore.pyqtSlot()
    def resume(self) -> None:
        """Resume the GP loop."""
        self.logger.debug("Resuming GP loop.")
        self.running = True

    @QtCore.pyqtSlot()
    def update_data_and_positions(self, pos:np.ndarray, vals:np.ndarray) -> None:
        """Update the data and positions lists."""
        self.logger.debug("Updating data and positions.")
        self._positions = pos
        self._values = vals
        
    def get_taks_normalization_weights(self, update: bool = False) -> np.ndarray:
        """Get the normalization weights for the tasks.

        Args:
            update (bool, optional): If True, update the normalization weights. Defaults to False.

        Returns:
            np.ndarray: Normalization weights.
        """
        if update:
            if self.settings["scanning"]["normalize_values"] == "init":
                self.task_normalization_weights = np.mean(self.values, axis=0)
            elif self.settings["scanning"]["normalize_values"] == "fixed":
                self.task_normalization_weights = self.settings["scanning"]["fixed_normalization"]
            elif self.settings["scanning"]["normalize_values"] == "dynamic":
                self.task_normalization_weights = np.mean(self.values, axis=0)
        return self.task_normalization_weights

    def start(self) -> None:
        """ init graphic interface """
        """Initialize the GP.

        This method is called at the first iteration of the GP loop.
        """
        self.logger.debug("Starting GP initialization.")
        isd = len(self.remote.axes)
        osd = 1
        on = len(self.task_labels)
        self.gp = fvGPOptimizer(
            input_space_bounds=self.remote.limits,
            input_space_dimension=isd,
            output_space_dimension=osd,
            output_number=on,
        )
        self.logger.debug(
            f"GP object created. Input Space Dimension () = {isd} | "
            f"Output Space Dimension () = {osd} | "
            f"Output Number () = {on}"
        )
        self.tell_gp()
        self.logger.info(f"Initialized GP with {len(self.positions)} samples.")
        fvgp_pars = self.settings["gp"]["fvgp"].copy()
        init_hyperparameters = np.array(
            [float(n) for n in fvgp_pars.pop("init_hyperparameters")]
        )
        self.logger.debug("Initializing GP:")
        self.logger.debug(f"\tinit_hyperparameters: {init_hyperparameters}")
        for k,v in fvgp_pars.items():
            self.logger.debug(f"\t{k} = {v}")
        self.gp.init_fvgp(init_hyperparameters=init_hyperparameters, **fvgp_pars)

        self.train_gp()

        cost_function_dict = self.settings.get("cost_function", None)
        if cost_function_dict is not None:
            self.logger.debug("Initializing cost function: cost_function_dict['function']")
            
            cost_func_callable = getattr(cost_functions, cost_function_dict["function"])
            cost_func_params = cost_function_dict.get("params", {})
            for k,v in cost_func_params.items():
                self.logger.debug(f"\t{k} = {v}")
            self.gp.init_cost(
                cost_func_callable,
                cost_function_parameters=cost_func_params,
            )
    
    def tell(self, update_normalization: bool = False) -> None:
        """Tell the GP about the current available data.

        This method is called every time new data is added to the data queue.
        If scanning/normalize_values is not false, values are normalized.
            - 'init': values are normalized by the mean of the first batch of data
            - 'fixed': values are normalized by the value of scanning/fixed_normalization
            - 'dynamic': values are normalized by the mean of all the current data
        """
        self.logger.debug("Telling GP about new data.")
        self.status.emit("Telling GP")
        if self.gp is not None:
            pos = np.asarray(self.positions)
            vals = np.asarray(self.values)
            if self.settings["scanning"]["normalize_values"] == "always":
                vals = vals * self.get_taks_normalization_weights(update=True)
            elif self.settings["scanning"]["normalize_values"] != "never":
                vals = vals * self.get_taks_normalization_weights(update=update_normalization)
            self.gp.tell(pos, vals)

    def train(self) -> None:
        """Train the GP."""
        self.status.emit("Training GP")
        hps_old = self.gp.hyperparameters.copy()
        train_pars = self.settings["gp"]["training"].copy()
        hps_bounds = np.asarray(train_pars.pop("hyperparameter_bounds"))
        self.logger.info("Training GP:")
        self.logger.info(f"\titeration {self.iter_counter} | {len(self.positions)} samples")
        self.logger.debug(f"\thyperparameter_bounds: {hps_bounds}")
        for k,v in train_pars.items():
            self.logger.debug(f"\t{k} = {v}")
        self.tell_gp(update_normalization=True)
        t = time.time()
        hps_new = self.gp.train_gp(hyperparameter_bounds=hps_bounds, **train_pars)
        self.logger.info(f"Training complete in {pretty_print_time(time.time()-t)} s")
        self.logger.debug("Hyperparameters: ")
        for old, new, bounds in zip(hps_old, hps_new, hps_bounds):
            change = (new - old) / old
            self.logger.debug(f"\t{old:.2f} -> {new:.2f} ({change:.2%}) | {bounds}")
        self.hyperparameter_history[self.iter_counter] = hps_new

    def ask(self) -> None:
        """Ask the GP for the next point to measure."""
        self.logger.debug("Asking GP for next point.")
        self.status.emit("Asking GP")
        if self.gp is not None:
            acq_func_callable = getattr(
                aquisition_functions,
                self.settings["acquisition_function"]["function"],
            )
            acq_func_params = self.settings["acquisition_function"]["params"]
            aqf = partial(acq_func_callable, **acq_func_params)

            ask_pars = self.settings["gp"]["ask"]
            all_positions = self.remote.all_positions
            visited_positions = np.unique(self.positions)
            missing_positions = np.setdiff1d(all_positions,visited_positions)
            self.logger.debug(
                f"Positions evaluated: {len(visited_positions)}/{len(all_positions)} | "
                f"{len(visited_positions)/len(all_positions):.2%}"
    )
            if len(missing_positions) == 0:
                self.logger.info(
                    "No more positions to evaluate. Ending scan."
                )
                self._should_stop = True
                self.finished.emit()
            
            answer = self.gp.ask(
                acquisition_function=aqf,
                x0 = missing_positions,
                **ask_pars
            )
            next_pos = answer["x"]
            self.new_points.emit(next_pos)

    def run(self) -> None:
        """Run the GP loop."""
        self.logger.debug("Starting GP loop.")
        self._should_stop = False
        self.iter_counter = 0
        self.hyperparameter_history = {}
        self._running = True
        while self._running:
            if self._should_stop:
                self.logger.debug("Stopping GP loop.")
                self._running = False
                break
            elif self.has_new_data():
                self.logger.debug(f"GP loop iteration {self.iter_counter}")
                self.status.emit(f"GP loop iteration {self.iter_counter}")
                if self.gp is None:
                    self.start()
                self.tell()
                if self.should_train():
                    self.train()
                self.ask()
                self.iter_counter += 1
        self.logger.debug("GP loop finished.")
        self.finished.emit()
        self.status.emit("Finished")


