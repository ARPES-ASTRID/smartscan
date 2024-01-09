from typing import Callable, Sequence, Tuple, Union, List, Any
from numpy.typing import NDArray, ArrayLike
from pathlib import Path
import logging
from functools import partial
import time
import shutil
from sympy import bool_map

import yaml
import asyncio
import numpy as np
from .TCP import send_tcp_message
from .gp import aquisition_functions, cost_functions, plot
from .sgm4commands import SGM4Commands
from .utils import closest_point_on_grid
from . import preprocessing
from . import tasks
import matplotlib.pyplot as plt

from gpcam.gp_optimizer import fvGPOptimizer

# import matplotlib.pyplot as plt

PathLike: object = Union[str, Path]


class Logger:
    def __init__(self) -> None:
        self.info: Callable = print
        self.debug: Callable = print
        self.error: Callable = print
        self.warning: Callable = print
        self.critical: Callable = print


LoggerLike: object = Union[Logger, logging.Logger]


class AsyncScanManager:
    def __init__(
        self,
        settings: PathLike | dict = None,
        logger: LoggerLike = None,
    ) -> None:
        """
        Initialize the AsyncScanManager object.

        Args:
            settings (dict | PathLike): A dictionary with the settings or a path to a yaml file with
            the settings.
        """
        # load settings
        if isinstance(settings, PathLike):
            self.settings_file = settings
            with open(settings) as f:
                self.settings = yaml.load(f, Loader=yaml.FullLoader)
        else:
            raise ValueError("Settings must be a path to a yaml file.")

        # set logger
        if logger is not None:
            self.logger: Logger = logger
        else:
            self.logger = Logger()
        self.logger.info("Initialized AsyncScanManager.")

        self.task_labels: List[str] = list(self.settings["tasks"].keys())

        # connect to SGM4
        self.remote = SGM4Commands(
            self.settings["TCP"]["host"],
            self.settings["TCP"]["port"],
            buffer_size=self.settings["TCP"]["buffer_size"],
        )
        self.remote.connect()
        if len(self.remote.axes[0]) == 0:
            raise ValueError("failed initializing axes!!")
        else:
            print(
                f"Axes: {[a.shape for a in self.remote.axes]} | Limits: {self.remote.limits} | Step size: {self.remote.step_size} "
            )

        # init queues
        self._raw_data_queue: asyncio.Queue = asyncio.Queue()
        self._reduced_data_queue: asyncio.Queue = asyncio.Queue()

        # init GP
        self.gp = None
        self._should_stop: bool = False

        # init data
        self.positions: List = []
        self.values: List = []
        self.hyperparameter_history = {}
        self._task_weights = None  # for fixed task weights

        # init plotting
        self.replot: bool = False
        self.last_spectrum = None

        self.save_settings()

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
            NDArray[Any]: An array with the weights.
        """
        if self.settings["scanning"]["normalize_values"] == "fixed":
            return 1 / np.array(self.settings["scanning"]["fixed_normalization"])
        if self._task_weights is None or update:
            self._task_weights = 1 / self.val_array.mean(axis=0)
        return self._task_weights

    def init_scan(self) -> None:
        """Initialize the scan."""
        self.logger.info("Initializing scan.")
        # TODO: add this to settings and give more options
        relative_points = [
            [0, 0],
            [0, 0.5],
            [0, 1],
            [0.5, 1],
            [1, 1],
            [1, 0.5],
            [1, 0],
            [0.5, 0],
            [0.5, 0.5],
        ]
        for p in relative_points:
            x = p[0] * self.remote.limits[0][1] + (1 - p[0]) * self.remote.limits[0][0]
            y = p[1] * self.remote.limits[1][1] + (1 - p[1]) * self.remote.limits[1][0]
            self.remote.ADD_POINT(x, y)
        # time.sleep(len(relative_points) * 1) # TODO: get waiting time from sgm4 or settings
        self.logger.info("Scan initialized.")

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
        filename: Path = Path(self.remote.filename)
        folder: Path = filename.parent
        if not filename.exists():
            if not folder.exists():
                raise FileNotFoundError(f"Folder {folder} does not exist.")

        target = filename.with_suffix(".yaml")
        i = 0
        while target.exists():
            i += 1
            # target = folder / filename.append("_{i:03.0f}").with_suffix(".yaml")
            raise NotImplementedError("multiple runs for the same measurement run is not working yet...")
        shutil.copy(self.settings_file, target)
        self.logger.info(f"Settings saved to {target}")

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
        # message = self.async_fetch_data()
        message: str = send_tcp_message(
            host=self.settings["TCP"]["host"],
            port=self.settings["TCP"]["port"],
            msg="MEASURE",
            buffer_size=self.settings["TCP"]["buffer_size"],
        )
        self.logger.info(f"Received data with length {len(message)/1024/1024:.2f} MB")
        msg: list[str] = message.strip("\r\n").split(" ")
        msg_code: str = msg[0]
        vals: list[str] = [v for v in msg[1:] if len(v) > 0]
        match msg_code:
            case "ERROR":
                self.logger.warning(message)
                return message, None
            case "NO_DATA":
                self.logger.warning(message)
                return message, None
            case "MEASURE":
                n_pos = int(vals[0])
                pos: NDArray[Any] = np.asarray(vals[1 : n_pos + 1], dtype=float)
                data: NDArray[Any] = np.asarray(vals[n_pos + 1 :], dtype=float)
                # shape = self.settings['data']['shape'] # TODO: request from SGM4
                data = data.reshape(self.remote.spectrum_shape)

                return pos, data
            case _:
                self.logger.warning(f"Unknown message code: {msg_code}")
                return message, None

    async def fetch_data_loop(self) -> None:
        """Loop to fetch data from SGM4 and put it in the raw data queue."""
        self.logger.info("Starting fetch data loop.")
        await asyncio.sleep(1)  # wait a bit before starting
        while not self._should_stop:
            self.logger.info("Fetch data looping...")
            pos, data = await self.fetch_data()
            await asyncio.sleep(0.1)
            if data is not None:
                self._raw_data_queue.put_nowait((pos, data))
                self.logger.debug(
                    f"Added raw data to queue: pos: {pos}, data shape: {data.shape} | queue size: {self._raw_data_queue.qsize()}"
                )
            else:
                self.logger.debug("No data received.")
                await asyncio.sleep(0.2)  # wait a bit before trying again

    # reduce data and update GP
    async def reduction_loop(self) -> None:
        """Reduce raw spectra to an array of N tasks and put it in the processed queue"""
        self.logger.info("Starting reduction loop.")
        while not self._should_stop:
            self.logger.debug("Running reduction step...")
            if self._raw_data_queue.qsize() > 0:
                pos, data = await self._raw_data_queue.get()
                self.logger.debug(f"Reduction for pos: {pos} | with shape {data.shape}")
                # preprocess data
                pp = data.copy()
                for _, d in self.settings["preprocessing"].items():
                    func = getattr(preprocessing, d["function"])
                    kwargs = d.get("params", {})
                    if kwargs is None:
                        pp = func(pp)
                    else:
                        pp = func(pp, **kwargs)
                self.last_spectrum = pp
                self.logger.debug(
                    f"Reduction for pos: {pos} | preprocessed shape {pp.shape} | pp data : {pp.mean():.3f} ± {pp.std():.3f}"
                )

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
                self._reduced_data_queue.put_nowait((pos, reduced))
                self.logger.info(
                    f"Added reduced data to queue || pos: {pos}, tasks: {reduced} | queue size: {self._reduced_data_queue.qsize()}"
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
            self.tell_gp()
            self.logger.info(
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
        if self.gp is not None:
            pos = np.asarray(self.positions)
            vals = np.asarray(self.values)
            if self.settings["scanning"]["normalize_values"] == "always":
                vals = vals * self.get_taks_normalization_weights(update=True)
            elif self.settings["scanning"]["normalize_values"] != "never":
                vals = vals * self.get_taks_normalization_weights(update=update_normalization)
            self.gp.tell(pos, vals)

    # GP loop
    def init_gp(self) -> None:
        """Initialize the GP.

        This method is called at the first iteration of the GP loop.
        """
        self.logger.debug("Starting GP initialization.")
        # TODO: make automatic detection of optimizer parameters
        self.gp = fvGPOptimizer(
            input_space_bounds=self.remote.limits,
            input_space_dimension=int(
                self.settings["gp"]["optimizer"]["input_space_dimension"]
            ),
            output_space_dimension=int(
                self.settings["gp"]["optimizer"]["output_space_dimension"]
            ),
            output_number=int(self.settings["gp"]["optimizer"]["output_number"]),
        )
        self.tell_gp()
        self.logger.info(f"Initialized GP with {len(self.positions)} samples.")
        fvgp_pars = self.settings["gp"]["fvgp"].copy()
        init_hyperparameters = np.array(
            [float(n) for n in fvgp_pars.pop("init_hyperparameters")]
        )
        self.logger.debug(
            f"Initializing GP with parameters: {fvgp_pars} and init_hyperparameters: {init_hyperparameters}"
        )
        self.gp.init_fvgp(init_hyperparameters=init_hyperparameters, **fvgp_pars)
        self.logger.info("Initialized GP. First Training...")
        self.train_gp()
        self.logger.info("GP trained.")
        cost_function_dict = self.settings.get("cost_function", None)

        if cost_function_dict is not None:
            self.logger.info("Initializing cost function.")
            cost_func_callable = getattr(cost_functions, cost_function_dict["function"])
            self.gp.init_cost(
                cost_func_callable,
                cost_function_parameters=cost_function_dict.get("params", {}),
            )

    def train_gp(self) -> None:
        """Train the GP."""
        # print('############################################################\n\n\n')
        self.logger.info(
            f"Training GP at iteration {self.iter_counter}, with {len(self.positions)} samples."
        )
        old_params = self.gp.hyperparameters.copy()
        t = time.time()
        self.tell_gp(update_normalization=True)
        train_pars = self.settings["gp"]["training"].copy()
        hps_bounds = np.asarray(train_pars.pop("hyperparameter_bounds"))
        self.logger.debug(f"Hyperparameter bounds: {hps_bounds}")
        self.gp.train_gp(hyperparameter_bounds=hps_bounds, **train_pars)

        dt = time.time() - t
        new_params = self.gp.hyperparameters.copy()
        s = "hyperparams: "
        changes = []
        for new, old in zip(new_params, old_params):
            changes.append((new - old) / old)
            s += f"{new:,.2f} ({changes[-1]:.2%}) |"

        self.logger.debug(s)
        self.logger.info(
            f"Trained GP in {dt:.2f} s. Avg change: {np.mean(changes):.2%}"
        )
        self.hyperparameter_history[self.iter_counter] = new_params
        # print('\n\n\n############################################################')

    async def gp_loop(self) -> None:
        """GP loop.

        This loop is responsible for training the GP and asking for the next position.
        """
        self.iter_counter = 0
        self.logger.info("Starting GP loop.")
        await asyncio.sleep(1)  # wait a bit before starting
        while not self._should_stop:
            self.logger.debug("GP looping...")
            if self.iter_counter > self.settings["scanning"]["max_points"]:
                self.logger.info(
                    f"Max number of iterations of {self.settings['scanning']['max_points']}"
                    " reached. Ending scan."
                )
                self._should_stop = True
            has_new_data = self.update_data_and_positions()
            if has_new_data:
                self.iter_counter += 1
                self.logger.debug(
                    f"GP iteration {self.iter_counter} | {len(self.positions)} samples"
                )
                if self.gp is None:
                    self.init_gp()  # initialize GP at first iteration
                else:
                    retrain = False
                    if self.iter_counter in self.settings["scanning"]["train_at"]:
                        retrain = True
                    elif (
                        self.settings["scanning"]["train_every"] > 0
                        and self.iter_counter % self.settings["scanning"]["train_every"]
                        == 0
                    ):
                        retrain = True
                    # f len(self.positions) in self.train_at:
                    if retrain:
                        self.train_gp()

                    acq_func_callable = getattr(
                        aquisition_functions,
                        self.settings["acquisition_function"]["function"],
                    )
                    acq_func_params = self.settings["acquisition_function"]["params"]
                    aqf = partial(acq_func_callable, **acq_func_params)

                    ask_pars = self.settings["gp"]["ask"]
                    # TODO: add points to evaluate based on the map settings.
                    answer = self.gp.ask(**ask_pars, acquisition_function=aqf)
                    next_pos = answer["x"]
                    # Remove once the correct points are passed to the ask function.
                    # |<--
                    try:
                        pos_on_grid: Tuple[int] = closest_point_on_grid(
                            next_pos, axes=self.remote.axes
                        )
                        self.logger.info(
                            f"Next suggestesd position: {next_pos} rounded to {pos_on_grid}"
                        )
                        self.remote.ADD_POINT(*pos_on_grid)
                    except ValueError:
                        self.logger.error(
                            f"Error comparing {next_pos} to previous positions to set it on grid. with axes {self.remote.axes}"
                        )

                    # -->|
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

        pass
        self.logger.info("starting plotting tool loop")
        fig = None
        aqf = None

        iteration = 0
        while not self._should_stop:
            iteration += 1
            if self.replot:
                self.replot = False
                fig, aqf = plot.plot_acqui_f(
                    gp=self.gp,
                    fig=fig,
                    pos=np.asarray(self.positions),
                    val=np.asarray(self.values),
                    old_aqf=aqf,
                    last_spectrum=self.last_spectrum,
                )
                plt.pause(0.01)
            else:
                await asyncio.sleep(0.2)
            # if fig is not None and iteration %100 == 0:
            #     fig.savefig(f'../results/{self.remote.filename.with_suffix("pdf").name}')

    # all loops and initialization
    async def all_loops(self) -> None:
        """
        Start all the loops.
        """
        # loop_methods = [getattr(self, method)() for method in dir(self) if method.endswith('_loop')]
        # await asyncio.gather(*loop_methods)
        await asyncio.gather(
            # self.training_loop(),
            self.killer_loop(),
            self.fetch_data_loop(),
            self.reduction_loop(),
            self.gp_loop(),
            self.plotting_loop(),
        )
        self.logger.info("All loops finished.")
        self.remote.END()

    async def start(self) -> None:
        """Initialize scan and start all loops."""
        self.init_scan()

        self.logger.info("Starting all loops.")
        self._should_stop = False

        await self.all_loops()

    # stop and kill
    def stop(self) -> None:
        self.logger.info("Stopping all loops.")
        self.kill()

    def kill(self) -> None:
        self.logger.info("Killing all loops.")
        self._should_stop = True

    async def killer_loop(self, duration=None) -> None:
        self.logger.info(
            f"Starting killer loop. Will kill process after {duration} seconds."
        )
        if duration is None:
            duration = self.settings["scanning"]["duration"]
        if duration is not None:
            await asyncio.sleep(duration)
            self.logger.info(
                f"Killer loop strikes! Scan interrupted after {duration} seconds."
            )
            self.stop()


if __name__ == "__main__":
    pass
