from typing import Callable, Sequence, Tuple, Union, List, Any
from numpy.typing import NDArray, ArrayLike
from pathlib import Path
from datetime import datetime
import logging

import json
import asyncio
import numpy as np
from smartscan.TCP import send_tcp_message
from smartscan.gp import fvGPOptimizer, ndim_aqfunc, compute_costs, plot_acqui_f
from smartscan.sgm4commands import SGM4Commands
from smartscan.utils import closest_point_on_grid
from smartscan.reductions import compose, sharpness, mean_std, select_roi
import matplotlib.pyplot as plt 

PathLike: object = Union[str, Path]

optimizer_pars: dict[str, int] = {
    'input_space_dimension': 2,
    'output_space_dimension': 1,
    'output_number': 2,
}
hyperparameter_bounds: ArrayLike = np.array(
    [[0.001,1e9],[1,1000],[1,1000],[1,1000],[1,1000]]
)
init_hyperparameters: ArrayLike = np.array(
        [4.71907062e+06, 4.07439017e+02, 3.59068120e+02, 4e2, 4e2]
)
train_pars: dict[str, Any] = {
    'hyperparameter_bounds': hyperparameter_bounds,
    'pop_size': 20,
    'tolerance': 1e-6,
    'max_iter': 2,
}
fvgp_pars: dict[str, Any] = {
    'init_hyperparameters' : init_hyperparameters,
    'compute_device': 'cpu',
    'gp_kernel_function': None,
    'gp_mean_function': None,
    'use_inv':False,
    'ram_economy': True,
}
ask_pars: dict[str, Any] = {
    'n': 1, 
    'bounds': None,
    'method': 'global', 
    'pop_size': 20, 
    'max_iter': 10, 
    'tol': 10e-6, 
    'x0': None, 
    'dask_client': None,
}
cost_func_params: dict[str, Any] = {
    'speed':300,
    'dwell_time':1,
    'dead_time':0.6,
    # 'point_to_um':15,
    'weight':.1,
    'min_distance':.99
}


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
            host: str = 'localhost', 
            port: int = 54333,
            buffer_size: int =  1024*1024*8,
            train_at: Sequence[int]= [20,40,80,160,320,640,1280,2560,5120,10240],
            train_every: int = 0,
            duration: float=None,
            max_iterations: int=1000,
            use_cost_function: bool = True,
            batch_normalize: bool = True,
            logger: LoggerLike =None
        ) -> None:
        self.host: str = host
        self.port: int = port
        self.buffer_size: int = buffer_size
        self.duration: float = duration
        self.max_iterations: int = max_iterations
        self.remote = SGM4Commands(host, port, buffer_size=buffer_size)
        self.remote.connect()
        if len(self.remote.axes[0]) == 0:
            raise ValueError('failed initializing axes!!')
        else:
            print(f' Axes is {self.remote.axes}')

        self._raw_data_queue: asyncio.Queue = asyncio.Queue()
        self._reduced_data_queue: asyncio.Queue = asyncio.Queue()
        self.gp = None
        self.positions: List = []
        self.values: List = []
        self.train_at: Sequence[int] = train_at
        self.train_every: int = train_every
        self.use_cost_function: bool = use_cost_function
        self.replot: bool = False
        self.batch_normalize: bool = batch_normalize
        self.last_spectrum = None

        if logger is not None:

            self.logger: Logger = logger
            # self.logger = logging.getLogger(__name__)
            # self.logger.setLevel(logging.DEBUG)
        else:
            logger = Logger()
        self.logger.info('Initialized AsyncScanManager.')

    def save_settings(self, filename: Path | str ='settings.json', folder: Path | str ='./') -> Path:
        # save to file
        folder = Path(folder) if folder is not None else Path.cwd()
        filename = Path(filename)
        if (folder/filename).exists():
            filename = Path(f'{filename.stem}_{datetime.now().strftime("%Y%m%d_%H%M%S")}{filename.suffix}')
        filepath = folder/filename
        all_dict = {
            'host': self.host,
            'port': self.port,
            'buffer_size': self.buffer_size,
            'duration': self.duration,
            'max_iterations': self.max_iterations,
            'train_at': self.train_at,
        }
        for dict_ in [optimizer_pars, train_pars, fvgp_pars, ask_pars, cost_func_params]:
            # make lists of any numpy array
            out = {}
            for k,v in dict_.items():
                if isinstance(v, np.ndarray):
                    out[k] = v.tolist()
                else:
                    out[k] = v
            all_dict.update(out)

        with open(filepath, 'w') as fp:
            json.dump(all_dict, fp)
        return filepath

    async def fetch_data(self) -> tuple[str, None] | tuple[NDArray[Any], NDArray[Any]] | None:
        # message = await self.async_fetch_data()
        message: str = send_tcp_message(
            host = self.host, 
            port = self.port,
            msg = 'MEASURE',
            buffer_size= self.buffer_size
        )
        self.logger.info(f'Received data with length {len(message)/1024/1024:.2f} MB')
        msg: list[str] = message.strip('\r\n').split(' ')
        msg_code: str = msg[0]
        vals: list[str] = [v for v in msg[1:] if len(v) >0]
        match msg_code:
            case 'ERROR':
                self.logger.warning(message)
                return message, None
            case 'NO_DATA':
                self.logger.warning(message)
                return message, None
            case 'MEASURE':
                n_pos = int(vals[0])
                pos: NDArray[Any] = np.asarray(vals[1:n_pos+1], dtype=float)
                data: NDArray[Any] = np.asarray(vals[n_pos+1:], dtype=float)
                # data = data.reshape(640,400)
                return pos, data
            case _:
                self.logger.warning(f'Unknown message code: {msg_code}')
                return message, None
        
    def update_data_and_positions(self) -> bool:
        """ Get data from SGM4"""
        if self._reduced_data_queue.qsize() > 0:
            self.logger.debug('Updating data and positions.')
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
            self.logger.info(f'Updated data with {n_new} new points. Total: {len(self.positions)} last Pos {self.positions[-1]} {self.values[-1]}.')
            return True
        else:
            self.logger.debug('No data in processed queue.')
            return False

    async def fetch_data_loop(self) -> None:
        """ Get data from SGM4"""
        self.logger.info('Starting fetch data loop.')
        while not self._should_stop:
            self.logger.info('Fetch data looping...')
            pos, data  = await self.fetch_data()
            if data is not None:
                self._raw_data_queue.put_nowait((pos, data))
                self.logger.debug(f'Put raw data in queue: {pos}, {data}')
            else:
                self.logger.debug('No data received.')
                await asyncio.sleep(.5)

    async def reduction_loop(self) -> None:
        self.logger.info('Starting reduction loop.')
        while not self._should_stop:
            self.logger.info('reduction looping...')
            if self._raw_data_queue.qsize() > 0:
                pos, data = await self._raw_data_queue.get()
                self.logger.debug(f'reducing data with shape {data.shape}')
                data = select_roi(
                    data.reshape((640,400)),
                    x_lim = (330,500),
                    y_lim = (20,380),
                )
                self.last_spectrum = data.copy()
                reduced = compose(
                    data, 
                    np.mean, 
                    sharpness, 
                    func_b_kwargs = {'sigma':5, 'r':1, 'reduce':np.mean}
                )
                # reduced = reduced * 1000
                # self.logger.info(f'adding {(pos,reduced)} to processed queue')
                self._reduced_data_queue.put_nowait((pos,reduced))
                self.logger.info(f'added {(pos,reduced)} to processed queue, currently {self._reduced_data_queue.qsize()}')
            else:
                self.logger.debug('No data in raw data queue.')
                await asyncio.sleep(.2)

    async def gp_loop(self) -> None:
        iter_counter = 0
        self.logger.info('Starting GP loop.')

        while not self._should_stop:
            self.logger.debug('GP looping...')
            if iter_counter > self.max_iterations:
                self.logger.info(f"Max number of iterations of {self.max_iterations} reached. Ending scan.")
                self._should_stop = True
            has_new_data = self.update_data_and_positions()
            if has_new_data:
                iter_counter += 1
                if self.gp is None:
                    # filename = self.save_settings()
                    # self.logger.info(f'saved settings to {filename}')
                    self.logger.debug('Starting GP initialization.')
                    self.gp = fvGPOptimizer(
                        input_space_bounds = self.remote.limits,
                        input_space_dimension= 2,
                        output_space_dimension = 1,
                        output_number = 2,  
                    )
                    self.tell_gp()
                    self.logger.info(f'Initialized GP with {len(self.positions)} samples.')
                    self.gp.init_fvgp(**fvgp_pars)
                    self.logger.info('Initialized GP. Training...')
                    self.gp.train_gp(**train_pars)
                    if self.use_cost_function:
                        cost_func_params.update({
                                        'prev_points': self.gp.x_data, 
                                        'point_to_um': self.remote.step_size[0],
                        })
                        self.gp.init_cost(
                            compute_costs, 
                            cost_function_parameters = cost_func_params,
                        )
                else:
                    self.logger.info('gp looping...')
                    retrain = False
                    if iter_counter in self.train_at:
                        retrain = True
                    elif self.train_every > 0 and iter_counter % self.train_every == 0:
                        retrain = True
                    # f len(self.positions) in self.train_at:
                    if retrain:
                        print('############################################################\n\n\n')
                        self.logger.info(f'Training GP at iteration {iter_counter}, with {len(self.positions)} samples.')
                        
                        old_params = self.gp.hyperparameters.copy()
                        self.gp.train_gp(**train_pars)
                        new_params =  self.gp.hyperparameters.copy()
                        s = 'hyperparams: '
                        for new,old in zip(new_params,old_params):
                            s += f"{new:,.2f} ({(new-old)/old:.2%}) |"
                        self.logger.debug(s)
                        print('\n\n\n############################################################')
                    answer = self.gp.ask(**ask_pars, acquisition_function=ndim_aqfunc)
                    next_pos = answer['x']
                    try:
                        pos_on_grid: Tuple[int] = closest_point_on_grid(next_pos, axes=self.remote.axes)
                        self.logger.info(f'Next suggestesd position: {next_pos} rounded to {pos_on_grid}')
                        self.remote.ADD_POINT(*pos_on_grid)
                    except ValueError:
                        self.logger.error(f'Error comparing {next_pos} to previous positions to set it on grid. with axes {self.remote.axes}')
                    self.replot = True
            else:
                self.logger.debug('No data to update.')
                await asyncio.sleep(.2)
    
        self.remote.END()
    
    def tell_gp(self) -> None:
        if self.gp is not None:
            pos = np.asarray(self.positions)
            vals = np.asarray(self.values)
            weights = np.asarray([100,10_000])
            if self.batch_normalize:
                vals = weights * vals / np.mean(vals)
            self.gp.tell(pos,vals)

    async def plotting_loop(self) -> None:
        self.logger.info('starting plotting tool loop')
        fig = None
        aqf = None

        iteration = 0
        while not self._should_stop:
            iteration += 1
            if self.replot:
                self.replot = False
                fig, aqf = plot_acqui_f(
                    gp=self.gp,
                    fig=fig,
                    pos=np.asarray(self.positions),
                    val=np.asarray(self.values),
                    old_aqf = aqf,
                    last_spectrum=self.last_spectrum,
                )
                plt.pause(0.01)
            else:
                await asyncio.sleep(.2)
            # if fig is not None and iteration %100 == 0:
            #     fig.savefig(f'../results/{self.remote.filename.with_suffix("pdf").name}')

    async def all_loops(self) -> None:
        """
        Start all the loops.
        """
        # loop_methods = [getattr(self, method)() for method in dir(self) if method.endswith('_loop')]
        # await asyncio.gather(*loop_methods)
        await asyncio.gather(
            # self.training_loop(),
            self.killer_loop(),
            self.reduction_loop(),
            self.fetch_data_loop(),
            self.gp_loop(),
            self.plotting_loop(),
        )
        self.logger.info('All loops finished.')
        self.remote.END()

    async def start(self) -> None:
        self.logger.info('Starting all loops.')
        self._should_stop = False
        await self.all_loops()

    def stop(self) -> None:
        self.logger.info('Stopping all loops.')
        self.kill()

    def kill(self) -> None:
        self.logger.info('Killing all loops.')
        self._should_stop = True

    async def killer_loop(self,duration=None) -> None:
        self.logger.info(f'Starting killer loop. Will kill process after {duration} seconds.')
        if duration is None:
            duration = self.duration
        if duration is not None:
            await asyncio.sleep(duration)
            self.logger.info('Killer loop strikes!.')
            self.stop()


if __name__ == '__main__':
    print('Running asyncscanner...')
    import os
    # an unsafe, unsupported, undocumented workaround :
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    import argparse
    parser = argparse.ArgumentParser(description='AsyncScanManager')
    parser.add_argument('--host', type=str, default='localhost', help='SGM4 host')
    parser.add_argument('--port', type=int, default=54333, help='SGM4 port')
    parser.add_argument('--loglevel', type=str, default='DEBUG', help='Log level')
    parser.add_argument('--logdir', type=str, default='logs', help='Log directory')
    parser.add_argument('--duration', type=int, default=None, help='Duration of the scan in seconds')
    parser.add_argument('--train_at', type=int, nargs='+', default=[10,20,30,40,50,60,70,80,90,100,200,400,800], help='Train GP at these number of samples')
    args = parser.parse_args()



    # init logger
    logger = logging.getLogger('async_scan_manager')
    logger.setLevel(args.loglevel)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel(args.loglevel)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # fh = logging.FileHandler(os.path.join(args.logdir, args.logfile))
    # fh.setLevel(args.loglevel)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    print('init asyncscanner object')
    # init scan manager
    scan_manager = AsyncScanManager(
        host=args.host,
        port=args.port,
        logger=logger,
        train_at=args.train_at,
        train_every=10,
        duration = args.duration,
        use_cost_function = True,
        batch_normalize = True,
    )

    # start scan manager
    loop = asyncio.get_event_loop()
    loop.run_until_complete(scan_manager.start())
    loop.close()
    logger.info('Scan manager stopped.')

