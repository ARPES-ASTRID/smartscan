from typing import List, Any
from numpy.typing import NDArray
from pathlib import Path
import logging
from functools import partial
import time
import shutil
import traceback

import yaml
import asyncio
import numpy as np
from .TCP import send_tcp_message
from .gp import aquisition_functions, cost_functions, plot
from .sgm4commands import SGM4Commands
from .utils import closest_point_on_grid, pretty_print_time
from . import preprocessing
from . import tasks
import matplotlib.pyplot as plt

from gpcam.gp_optimizer import fvGPOptimizer


class AsyncScanManager:
    """ AsyncScanManager class.
    
    This class is responsible for managing the scan.
    It connects to the SGM4, fetches data, reduces it, trains the GP and asks for the next position.
    
    Args:
        settings (dict | str | Path): A dictionary with the settings or a path to a yaml file with the settings.
        logger (logging.Logger, optional): A logger object. Defaults to None.
        
    Attributes:
        settings (dict): A dictionary with the settings.
        logger (logging.Logger): A logger object.
        remote (SGM4Commands): An object to communicate with the SGM4.
        gp (fvGPOptimizer): The GP object.
        positions (List): A list of positions.
        values (List): A list of values.
        hyperparameter_history (dict): A dictionary with the hyperparameters history.
        last_spectrum (NDArray[Any]): The last spectrum.

        _raw_data_queue (asyncio.Queue): A queue to store the raw data.
        _reduced_data_queue (asyncio.Queue): A queue to store the reduced data.
        _should_stop (bool): A flag to stop the scan.
        _replot (bool): A flag to replot the data.
        _task_weights (NDArray[Any]): An array with the task weights.
        
    TODOs:

        - [ ] Add a method to save the data.
        - [x] Add a method to save the settings.
        - [ ] Add a method to save the hyperparameters history.
        - [ ] Check why ask is returning points out of the grid
        - [ ] improve the plotting.
        - [ ] add interactive control of the scan.
        - [ ] move initialization points to the settings file.
        - [ ] add aks for multiple points.
        

    """
    def __init__(
        self,
        settings: str | Path | dict = None,
    ) -> None:
        """
        Initialize the AsyncScanManager object.

        Args:
            settings (dict | str | Path): A dictionary with the settings or a path to a yaml file with
            the settings.
        """
        # load settings
        if isinstance(settings, str | Path):
            self.settings_file = settings
            with open(settings) as f:
                self.settings = yaml.load(f, Loader=yaml.FullLoader)
        elif isinstance(settings, dict):
            self.settings = settings.copy()
            self.settings_file = None
        else:
            raise ValueError("Settings must be a path to a yaml file or a dict.")
        
        self.logger =  logging.getLogger("AsyncScanManager")
        self.logger.info("Initialized AsyncScanManager.")

        self.task_labels: List[str] = list(self.settings["tasks"].keys())

        # connect to SGM4
        self.remote = SGM4Commands(
            self.settings["TCP"]["host"],
            self.settings["TCP"]["port"],
            buffer_size=self.settings["TCP"]["buffer_size"],
        )

        # init queues
        self._raw_data_queue: asyncio.Queue = asyncio.Queue()
        self._reduced_data_queue: asyncio.Queue = asyncio.Queue()

        # init GP
        self.gp = None
        self._should_stop: bool = False
        self._ready_for_gp: bool = False

        # init data
        self.positions: List = []
        self.unique_positions: List = []
        self.values: List = []
        self.hyperparameter_history = {}
        self.task_weights = None  # for fixed task weights
        self.last_asked_position = None
        # init plotting
        self.replot: bool = False
        self.last_spectrum = None

        # scan initialization points
        self.relative_inital_points = [ # currently only the border
            [0, 0],
            [0, 0.5],
            [0, 1],
            [0.5, 1],
            [1, 1],
            [1, 0.5],
            [1, 0],
            [0.5, 0],
        ]

    @property
    def val_array(self) -> NDArray[Any]:
        """Get the values as an array."""
        return np.asarray(self.values, dtype=float)

    @property
    def pos_array(self) -> NDArray[Any]:
        """Get the positions as an array."""
        return np.asarray(self.positions, dtype=float)

    def get_taks_normalization_weights(self, update=False) -> NDArray[Any]:
        """Get the weights to normalize the tasks.

        Returns:
            NDArray[float]: An array with the weights.
        """
        if self.settings["scanning"]["normalize_values"] == "fixed":
            self.task_weights = 1 / np.array(self.settings["scanning"]["fixed_normalization"])
            self.logger.debug(f"Fixed Task weights: {self.task_weights}")
        elif self.task_weights is None or update:
            self.task_weights = 1 / self.val_array.mean(axis=0)
            self.logger.debug(f"Updated Task weights: {self.task_weights}")
        return self.task_weights

    @property
    def filename(self) -> Path:
        """Get the file stem."""
        filename: Path = Path(self.remote.filename)
        folder: Path = filename.parent
        if not filename.exists():
            if not folder.exists():
                raise FileNotFoundError(f"Folder {folder} does not exist.")
        self.logger.debug(f"Saving settings to {folder}.")
        return filename

    # get data from SGM4
    async def fetch_data(
        self,
    ) -> tuple[str, None] | tuple[NDArray[Any], NDArray[Any]] | None:
        """Get data from SGM4.

        Returns:
            tuple[str, None] | tuple[NDArray[Any], NDArray[Any]] | None:
                - tuple[str, None] if there was an error
                - tuple[NDArray[Any], NDArray[Any]] if there was no error
                - None if no data was received
        """
        self.logger.debug("Fetching data...")
        t0 = time.time()
        message: str = send_tcp_message(
            host=self.settings["TCP"]["host"],
            port=self.settings["TCP"]["port"],
            msg="MEASURE",
            buffer_size=self.settings["TCP"]["buffer_size"],
            logger=self.logger,
        )
        msg: list[str] = message.strip("\r\n").split(" ")
        msg_code: str = msg[0]
        vals: list[str] = [v for v in msg[1:] if len(v) > 0]
        self.logger.debug(f"MEASURE answer: {msg_code}: {len(message)/1024:,.1f} kB")
        match msg_code:
            case "ERROR":
                self.logger.error(message)
                return message, None
            case "NO_DATA":
                self.logger.debug(f"No data received: {message}")
                return message, None
            case "MEASURE":
                try:
                    n_pos = int(vals[0])
                    pos: NDArray[Any] = np.asarray(vals[1 : n_pos + 1], dtype=float)
                    data: NDArray[Any] = np.asarray(vals[n_pos + 1 :], dtype=float)
                    data = data.reshape(self.remote.spectrum_shape)
                    return pos, data
                except ValueError as e:
                    self.logger.critical(f'Failed interpreting received data with shape {data.shape} ')
            case _:
                self.logger.warning(f"Unknown message code: {msg_code}")
                return message, None

    async def fetch_data_loop(self) -> None:
        """Loop to fetch data from SGM4 and put it in the raw data queue."""
        self.logger.info("Starting fetch data loop.")
        await asyncio.sleep(1)  # wait a bit before starting
        while not self._should_stop:
            self.logger.debug("Fetch data looping...")
            t0 = time.time()
            pos, data = await self.fetch_data()
            if data is not None:
                self._raw_data_queue.put_nowait((pos, data))
                self.logger.info(
                    f"+ RAW queue     | pos {pos} | shape {data.shape} | "
                    f"queue size: {self._raw_data_queue.qsize()} | "
                    f"time: {time.time()-t0:.3f} s"
                )
            else:
                self.logger.debug("No data received.")
                await asyncio.sleep(0.2)
            await asyncio.sleep(0.1)  # wait a bit before trying again

    def reduce(self, pos: NDArray[Any], data: NDArray[Any]) -> NDArray[Any]:
        """ preprocess and reduce data.

        Args:
            data (NDArray[Any]): The data to reduce.

        Returns:
            NDArray[Any]: The reduced data.
        """
        t0 = time.time()
        pp = data.copy()
        if self.settings["preprocessing"] is not None:
            for _, d in self.settings["preprocessing"].items():
                func = getattr(preprocessing, d["function"])
                kwargs = d.get("params", {})
                if kwargs is None:
                    pp = func(pp)
                else:
                    pp = func(pp, **kwargs)
            self.last_spectrum = pp
            t1 = time.time()
            self.logger.debug(
                f"Preprocessing {pos} | shape {pp.shape} | mean : {pp.mean():.3f} ± {pp.std():.3f} | time: {t1-t0:.3f} s"
            )
        else:
            t1 = time.time()

        # reduce data
        reduced = []
        for _, d in self.settings["tasks"].items():
            func = getattr(tasks, d["function"])
            kwargs = d.get("params", {})
            if kwargs is None:
                reduced.append(func(pp))
            else:
                reduced.append(func(pp, **kwargs))
        reduced = np.asarray(reduced, dtype=float).flatten()
        if len(reduced) != len(self.task_labels):
            raise RuntimeError(
                f"Length mismatch between tasks {len(reduced)}"
                f"and task labels {len(self.task_labels)}."
            )
        self.logger.debug(f"Reduction {pos} | time: {time.time()-t1:.3f} s")
        return reduced

    # reduce data and update GP
    async def reduction_loop(self) -> None:
        """Reduce raw spectra to an array of N tasks and put it in the processed queue"""
        self.logger.info("Starting reduction loop.")
        while not self._should_stop:
            self.logger.debug("Running reduction step...")
            if self._raw_data_queue.qsize() > 0:
                pos, data = await self._raw_data_queue.get()
                # preprocess data
                t0 = time.time()
                reduced = self.reduce(pos,data)
                dt = time.time() - t0
                self._reduced_data_queue.put_nowait((pos, reduced))
                self.logger.info(
                    f"+ REDUCED queue | pos: {pos}, tasks: {reduced} | "
                    f"queue size: {self._reduced_data_queue.qsize()} | "
                    f"time: {dt:.3f} s"
                )
            else:
                self.logger.debug("No data in raw data queue.")
                await asyncio.sleep(0.2)  # wait a bit before trying again

    def update_data_and_positions(self) -> bool:
        """Update data and positions from the processed queue.

        Return True if new data was added, False if not.
        """
        if self._reduced_data_queue.qsize() > 0:
            self.logger.debug(
                f"Updating data and positions. Queue size: {self._reduced_data_queue.qsize()}"
            )
            n_new = 0
            while True:
                try:
                    pos, data = self._reduced_data_queue.get_nowait()
                    self.positions.append(pos)
                    self.values.append(data)
                    n_new += 1
                except asyncio.QueueEmpty:
                    break
            # self.tell_gp()
            self.logger.debug(
                f"Updated data with {n_new} new points. Total: {len(self.positions)} last Pos {self.positions[-1]} {self.values[-1]}."
            )
            return True
        else:
            self.logger.debug("Reduced queue is empty. No data to update.")
            return False

    def tell_gp(self, update_normalization: bool = False) -> None:
        """Tell the GP about the current available data.

        This method is called every time new data is added to the data queue.
        If scanning/normalize_values is not false, values are normalized.
            - 'init': values are normalized by the mean of the first batch of data
            - 'fixed': values are normalized by the value of scanning/fixed_normalization
            - 'dynamic': values are normalized by the mean of all the current data
        """
        self.logger.debug("Telling GP about new data.")
        if self.gp is not None:
            pos = np.asarray(self.positions)
            vals = np.asarray(self.values)
            if self.settings["scanning"]["normalize_values"] == "always":
                vals = vals * self.get_taks_normalization_weights(update=True)
            elif self.settings["scanning"]["normalize_values"] != "never":
                vals = vals * self.get_taks_normalization_weights(update=update_normalization)
            self.logger.info(f"TELL GP | pos: {pos[-1]} | tasks: {vals[-1]}")
            self.gp.tell(pos, vals)

    # GP loop
    def init_gp(self) -> None:
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

    def should_train(self) -> bool:
        if self.iter_counter in self.settings["scanning"]["train_at"]:
            return True
        if self.settings["scanning"]["train_every"] > 0:
            if self.iter_counter % self.settings["scanning"]["train_every"] == 0:
                return True
        return False

    def was_already_measured(self, pos:np.ndarray) -> bool:
        """ check if the given position is in the positions list."""
        return np.any(np.all(np.isclose(self.pos_array, pos), axis=1))

    def train_gp(self) -> None:
        """Train the GP."""
        hps_old = self.gp.hyperparameters.copy()
        train_pars = self.settings["gp"]["training"].copy()
        hps_bounds = np.asarray(train_pars.pop("hyperparameter_bounds"))
        if "bounds" not in self.hyperparameter_history.keys():
            self.hyperparameter_history["bounds"] = hps_bounds.tolist() 
        self.logger.info("Training GP:")
        self.logger.info(f"\titeration {self.iter_counter} | {len(self.positions)} samples")
        self.logger.debug(f"\thyperparameter_bounds: {hps_bounds}")
        for k,v in train_pars.items():
            self.logger.debug(f"\t{k} = {v}")
        self.tell_gp(update_normalization=True)
        t = time.time()
        hps_new = self.gp.train_gp(hyperparameter_bounds=hps_bounds, **train_pars)
        if not all(hps_new == self.gp.hyperparameters):
            self.logger.warning('Something wrong with training, hyperparameters not updated?')
        self.logger.info(f"Training complete in {pretty_print_time(time.time()-t)} s")
        self.logger.info("Hyperparameters: ")
        for old, new, bounds in zip(hps_old, hps_new, hps_bounds):
            change = (new - old) / old
            self.logger.info(f"\t{old:.2f} -> {new:.2f} ({change:.2%}) | {bounds}")
        self.hyperparameter_history[f"training{self.iter_counter:04.0f}"] = {
            "hyperparameters": [float(f) for f in self.gp.hyperparameters],
            "time": time.time() - t,
            "iteration": self.iter_counter,
            "samples": len(self.positions),
        }
        self.save_hyperparameters()
        
    def ask_gp(self) -> None:
        """Ask the GP for the next position."""
        acq_func_callable = getattr(
            aquisition_functions,
            self.settings["acquisition_function"]["function"],
        )
        acq_func_params = self.settings["acquisition_function"]["params"]
        aqf = partial(acq_func_callable, **acq_func_params)

        ask_pars = self.settings["gp"]["ask"]

        if self.gp.cost_function_parameters is not None:
            self.gp.cost_function_parameters.update({'prev_points': self.gp.x_data})

        if self.last_asked_position is None:
            self.last_asked_position = self.positions[-1]
        
        self.logger.debug(f"ASK: Last asked position: {self.last_asked_position}")
        next_pos = self.gp.ask(
            position=np.array(self.last_asked_position),
            acquisition_function=aqf,
            **ask_pars
        )["x"]
        for point in next_pos:
            rounded_point = closest_point_on_grid(point, axes=self.remote.axes)
            self.logger.info(
                f"ASK GP          | Adding {rounded_point} to scan. rounded from {point}"
            )
            if any([all(rounded_point == prev) for prev in self.pos_array]):
                self.logger.warning(
                f"ASK GP          | Point {rounded_point} already evaluated!"
                )
            self.remote.ADD_POINT(*rounded_point)
            self.last_asked_position = rounded_point
        return True

    async def gp_loop(self) -> None:
        """GP loop.

        This loop is responsible for training the GP and asking for the next position.
        """
        self.iter_counter = 0
        self.logger.info("Starting GP loop.")
        await asyncio.sleep(1)  # wait a bit before starting
        while not self._ready_for_gp and self.relative_inital_points is not None:
            has_new_data = self.update_data_and_positions()
            if len(self.positions) > len(self.relative_inital_points):
                self._ready_for_gp = True
            else:
                self.logger.debug(f"Waiting for data to be ready for GP. {len(self.positions)}/{len(self.relative_inital_points)} ")
                await asyncio.sleep(0.2)
        self.logger.info("Data ready for GP. Starting GP loop.")
        while not self._should_stop:
            self.logger.debug("GP looping...")
            if self.iter_counter >= self.settings["scanning"]["max_points"]:
                self.logger.warning(
                    f"Max number of iterations of {self.settings['scanning']['max_points']}"
                    " reached. Ending scan."
                )
                self._should_stop = True
                break
            has_new_data = self.update_data_and_positions()
            if has_new_data:
                self.iter_counter += 1
                self.logger.info(
                    f"GP iter: {self.iter_counter:3.0f}/{self.settings['scanning']['max_points']:4.0f} | {len(self.positions)} samples"
                )
                if self.gp is None:
                    self.init_gp()  # initialize GP at first iteration
                else:
                    self.tell_gp()
                    if self.should_train():
                        self.train_gp()
                    success = self.ask_gp()
                    if not success:
                        break
                    self.replot = True
            else:
                self.logger.debug("No data to update.")
                await asyncio.sleep(0.2)

        self.remote.END()

    # plotting loop
    async def plotting_loop(self) -> None:
        """Plotting loop.

        This loop is responsible for plotting the data.
        """
        self.logger.info("Starting plotting loop.")
        await asyncio.sleep(1)  # wait a bit before starting
        self.logger.info("starting plotting tool loop")
        self.fig = None
        aqf = None
        while not self._should_stop:
            if self.replot:
                self.replot = False
                self.logger.debug("Plotting...")
                fig, aqf = plot.plot_acqui_f(
                    gp=self.gp,
                    fig=self.fig,
                    pos=np.asarray(self.positions),
                    val=np.asarray(self.values),
                    old_aqf=aqf,
                    last_spectrum=self.last_spectrum,
                    settings=self.settings,
                )
                self.fig = fig
                plt.pause(0.01)
            else:
                await asyncio.sleep(0.2)
            # if fig is not None and self.iter_counter % self.settings['plot']['save_every'] == 0:
            #     fig.savefig(f'../results/{self.remote.filename.with_suffix("pdf").name}')

    async def killer_loop(self, duration=None) -> None:
        self.logger.info(
            f"Starting killer loop. Will kill process after {duration} seconds."
        )
        if duration is None:
            duration = self.settings["scanning"]["duration"]
        if duration is not None:
            time_left = duration
            while not self._should_stop:
                time_left -= 1
                if time_left <= 0:
                    break
                await asyncio.sleep(1)
            self.logger.info(
                f"Killer loop strikes! Scan interrupted after {duration} seconds."
            )
            self.stop()

    # all loops and initialization
    async def all_loops(self) -> None:
        """
        Start all the loops.
        """
        self.logger.info("Starting all loops.")
        tasks = (
                self.killer_loop(),
                self.fetch_data_loop(),
                self.reduction_loop(),
                self.gp_loop(),
                self.plotting_loop(),
            )
        try:
            await asyncio.gather(*tasks)
                # self.training_loop(),
        except KeyboardInterrupt:
            self.logger.warning("KeyboardInterrupt. Stopping all loops.")
            for t in tasks:
                try:
                    t.cancel()
                except:
                    pass
        except Exception as e:
            self.logger.critical(f"{type(e).__name__} stopping all loops. ")
            error_traceback = ''
            for line in traceback.format_tb(e.__traceback__):
                error_traceback += line
            self.logger.error(f"Traceback: {error_traceback}")
            self.logger.error(f"{type(e).__name__}: {e}")
            
        finally:
            self.logger.info("Finalizing loop gathering: All loops finished.")

            self.stop()

    async def start(self) -> None:
        """Initialize scan and start all loops."""
        self.init_scan()

        self._ready_for_gp = False
        self._should_stop = False
        
        await self.all_loops()

    def connect(self) -> None:
        self.remote.connect()
        if len(self.remote.axes[0]) == 0:
            raise ValueError("failed initializing axes!!")
        else:
            self.logger.info(
                f"Axes: {[a.shape for a in self.remote.axes]} | Limits: {self.remote.limits} | Step size: {self.remote.step_size} "
            )
    
    def init_scan(self) -> None:
        """Initialize the scan."""
        self.logger.info(f"Initializing scan.")
        # TODO: add this to settings and give more options
        self.remote.START()
        while True:
            try:
                s = self.remote.STATUS()
                break
            except Exception as e:
                self.logger.error(f"Error setting up scan: {e}")
                self.logger.info("Trying again in 1 second...")
                time.sleep(1)
        self.logger.info(f"Scan initialized. Status: {s}")
        if status := self.remote.STATUS() != "READY":
            raise RuntimeError(f"Scan not ready. Status: {status}")
        self.connect()
        self.save_log_to_file()
        self.save_settings()

        if self.relative_inital_points is not None:
            self.logger.info(f"Adding {len(self.relative_inital_points)} points to scan.")
            for p in self.relative_inital_points:
                x = p[0] * self.remote.limits[0][1] + (1 - p[0]) * self.remote.limits[0][0]
                y = p[1] * self.remote.limits[1][1] + (1 - p[1]) * self.remote.limits[1][0]
                self.remote.ADD_POINT(x, y)
                self.last_asked_position = (x,y)
                self.logger.debug(f"Added point {p} to scan.")

    def stop(self) -> None:
        self.logger.info("Stopping all loops.")
        self.kill()

    async def interrupt(self) -> None:
        """Interrupt the scan."""
        self.logger.info("Interrupting scan.")
        self.kill()
        await asyncio.sleep(1)

    def kill(self) -> None:
        self.logger.info("Killing all loops.")
        self._should_stop = True

    def finalize(self) -> None:
        self.logger.info("Finalizing scan.")
        self.logger.info("Saving figure...")
        self.save_figure()
        self.logger.info("Saving hyperparameters...")
        self.save_hyperparameters()
        self.logger.info("Scan finalized.")

    def save_figure(self) -> None:
        try:
            self.fig.savefig(self.filename.with_suffix(".pdf"))
            self.logger.info(f"Saved figure to {self.filename.with_suffix('.pdf')}")
            plt.close(self.fig)
        except Exception as e:
            self.logger.error(f"{type(e)} saving figure: {e}")
        
    def save_hyperparameters(self) -> None:
        try:
            target = self.filename.parent / (self.filename.stem + "_hps.yaml")
            with open(target, "w") as f:
                yaml.dump(self.hyperparameter_history, f)
            self.logger.info(f"Saved hyperparameters to {target}")
        except Exception as e:
            self.logger.error(f"{type(e)} saving hyperparameters: {e}")
    
    def save_settings(
        self,
    ) -> Path:
        """Save the settings in the data directory next to the acquired data.

        Args:
            filename (Path | str, optional): _description_. Defaults to "settings.json".
            folder (Path | str, optional): _description_. Defaults to "./".

        Raises:
            NotImplementedError: _description_

        Returns:
            Path: _description_
        """
        target = self.filename.parent / (self.filename.stem + "_settings.yaml")
        # i = 0
        if not target.exists():
            if self.settings_file is None:
                with open(target, "w") as f:
                    yaml.dump(self.settings, f)
                self.logger.info(f"Settings saved to {target}")
            else:
                shutil.copy(self.settings_file, target)
                self.logger.info(f"Settings copied to {target}")
        else:
            self.logger.critical(f"FAILED TO SAVE SETTINGS TO {target}. File exists!!")
    
    def save_log_to_file(self) -> logging.Logger:
        """Setup the logger."""
        logging_filename = self.filename.with_suffix(".log")
        self.logger.info(f"Saving INFO log to {logging_filename}")
        fh = logging.FileHandler(logging_filename)
        fh.setLevel("DEBUG")
        formatter = logging.Formatter(self.settings['logging']['formatter'])
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.info(f"Logging to file: {logging_filename}.")
    
    def __del__(self) -> None:
        self.logger.critical("Deleted instance. scan stopping")
        try:
            self.finalize()
        except Exception as e:
            self.logger.warn(f"{type(e).__name__} while deleting asyncscanner instance: {e}")


if __name__ == "__main__":
    pass
