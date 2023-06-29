from typing import List,Tuple,Sequence,Union
import time
import asyncio
from pathlib import Path
import logging

import xarray as xr
# import dataloader as dl
import h5py
import numpy as np

from smartscan.TCP import TCPServer
from smartscan.file import SGM4FileManager


class VirtualSGM4(TCPServer):

    MOTOR_SPEED = 300 # um/s

    def __init__(
            self,
            ip: str,
            port: int,
            ndim: int = 2,
            source_file: Union[str, Path] = None,
            limits: List[Tuple[float]] = None,
            step_size: Sequence[float] = None,
            verbose: bool = True,
            buffer_size:int = 1024*1024*8,
            dwell_time: float = 1,
            logger=None,
    ) -> None:
        super().__init__(ip, port)
        self.logger = logger or logging.getLogger(__name__)
        self.logger.info('init VirtualSGM4 object')
        self.queue = []
        self.status = 'IDLE' # TODO: implement status
        if source_file is not None:
            self.init_scan_from_file(source_file)
        elif limits is not None and step_size is not None:
            self.init_scan(ndim, limits, step_size, verbose, dwell_time)
        self.file = None

    def init_scan(
            self,
            *args,
            scan_name: str = 'test',
            dirname: str = '.',
            verbose: bool = True,
            dwell_time: float = 0.5,
        ) -> None:
        """ Initialize the scan.
        
        Args:
            ndim: Number of dimensions of the scan.
            filename: Name of the file to save the scan.
            limits: List of tuples (min, max) for each axis.
            step_size: List of step sizes for each axis.
            verbose: If True, print messages.
            dwell_time: Dwell time at each point.
        """
        # file management
        filename = Path(scan_name).with_suffix('.h5')
        if filename.exists():
            raise FileExistsError(f'File {self.source_file} already exists.')
        else:
            self.target_file_name = Path(dirname) / filename
        self.source_file = None # we use no source file, data will be random generated

        # scan parameters
        self.dims = []
        self.starts = []
        self.stops = []
        self.steps = []
        self.coords = {}
        for arg in args:
            dim, start, stop, step = arg
            self.dims.append(dim)
            self.starts.append(start)
            self.stops.append(stop)
            self.steps.append(step)
            self.lengths.append(int((stop-start)/step))
            axis = np.arange(start, stop, step)
            self.axes.append(axis)
            self.coords[dim] = axis
        self.ndim = len(self.dims)
        self.limits = [(f,t) for f,t in zip(self.starts, self.stops)]
        self.map_shape = [len(c) for c in self.coords.values()]
        self.signal_shape = [500,500]
        self.dwell_time = dwell_time + 0.6
        self.verbose = verbose
        self.current_pos = [c[l//2] for c, l in zip(self.coords.values(), self.map_shape)]
        self.positions = list(zip(*[c.ravel() for c in np.meshgrid(*self.coords.values())]))

    def init_scan_from_file(self, filename: Union[str, Path], scan_name: str = None,) -> None:
        """ Initialize the scan from a file.

        Args:
            filename: Name of the file to read.
        """
        self.source_file = SGM4FileManager(filename)
        if scan_name is None:
            scan_name = Path(filename).stem + '_virtual'
        self.target_file_name = Path(scan_name).with_suffix('.h5')

        with self.source_file as f:
            self.dims = f.dims
            self.ndim = f.ndim
            self.coords = f.coords
            self.axes = f.axes
            self.map_shape = f.map_shape
            self.signal_shape = f.file['Entry/Data/TransformedData'].shape[1:]
            self.verbose = False
            self.dwell_time = f.dwell_time
            self.starts = f.starts
            self.stops = f.stops
            self.steps = f.steps
            self.lengths = f.lengths
            self.positions = f.positions
            self.limits = f.limits
            self.current_pos = [c[l//2] for c, l in zip(self.coords.values(), self.map_shape)]
        self.queue.append(self.current_pos)

    def nearest_position_on_grid(self, position: Sequence[float]) -> Tuple[int]:
        """Find nearest position in the grid.

        Args:
            position: position to find
        
        Returns:
            index: index of nearest position
        """
        assert len(position) == self.ndim, 'length of position does not match dimensionality'
        nearest = []
        for pos, axis in zip(position, self.axes):
            assert axis.min() <= pos <= axis.max(), 'position is outside of limits {} {}'.format(pos, axis)
            nearest.append(axis[np.argmin(np.abs(axis - pos))])
        return tuple(nearest)

    def read_data(self, position:Tuple[float]) -> np.ndarray:
        """ Read the data from the file.
        
        Args:
            position: Position to read.
        
        Returns:
            data: Data at the given position.
        """
        assert len(position) == self.ndim, f'Position {position} has wrong dimension.'
        pos = self.nearest_position_on_grid(position)
        if self.source_file is not None:
            with self.source_file as f:
                poslist = [tuple(p) for p in self.positions]
                index = poslist.index(pos)
                return pos, f.file['Entry/Data/TransformedData'][index]
        else:
            return pos, np.random.rand(*self.signal_shape)

    def create_file(
            self,
            filename:str|Path=None,
            filedir:str|Path='../data',
            mode="x",
        ) -> None:
        """ Create the file for the scan.
        
        Args:
            mode: Mode of the file. See h5py.File for details.
            
        Raises:
            FileExistsError: If the file already exists.
            """
        
        if filename is not None:
            self.target_file_name = Path(filename).with_suffix('.h5')
        if filedir is not None:
            self.target_file_name = Path(filedir) / self.target_file_name
        if self.target_file_name.exists():
            raise FileExistsError(f'File {self.target_file_name} already exists.')
        else:
            self.target_file_name.parent.mkdir(parents=True, exist_ok=True)
        self.logger.info(f'Creating file {self.target_file_name}')
        self.file = h5py.File(self.target_file_name, mode=mode, libver='latest')
        
        self.file.create_dataset(
            name = "Entry/Data/TransformedData", 
            shape = (0, *self.signal_shape), 
            maxshape = (None, *self.signal_shape),
            chunks = (1,*self.signal_shape), 
            dtype='f4'
        )
        self.file.create_dataset(
            name = 'Entry/Data/ScanDetails/SetPositions', 
            shape=(0, self.ndim), 
            maxshape=(None, self.ndim), 
            chunks=(1, self.ndim),
            dtype='f4'
        )
        self.file.create_dataset(name='Entry/Data/ScanDetails/SlowAxis_length',data=self.lengths, dtype='i4')
        self.file.create_dataset(name='Entry/Data/ScanDetails/SlowAxis_start',data=self.starts, dtype='f4')
        self.file.create_dataset(name='Entry/Data/ScanDetails/SlowAxis_step',data=self.steps, dtype='f4')
        self.file.create_dataset(name='Entry/Data/ScanDetails/SlowAxis_names',data=self.dims, dtype='S10')
        self.file.create_dataset(name='Entry/Data/ScanDetails/Dimensions',data=self.ndim, dtype='i4')
        self.file.swmr_mode = True

    def close_file(self) -> None:
        """ Close the file.
        """
        if self.file is not None:        
            self.file.close()

    def position_is_allowed(self, axis: int, target: float) -> bool:
        """ Check if the target position is allowed for the specified axis.
        """
        return self.limits[axis*self.ndim] <= target <= self.limits[axis*self.ndim+1]

    def measure(self, changed:Sequence[bool]) -> float:
        """ Fake measuring the current position."""

        self.logger.debug(f'Measuring position {self.current_pos}, changed {changed}')
        pos, data = self.read_data(self.current_pos)
        data_ds = self.file["Entry/Data/TransformedData"]
        data_ds.resize((data_ds.shape[0] + 1),axis=0)
        data_ds[-1,...] = data
        data_ds.flush()
        pos_ds = self.file["Entry/Data/ScanDetails/SetPositions"]
        pos_ds.resize((pos_ds.shape[0] + 1), axis=0)
        pos = list(pos)
        pos[not changed] = -99999999. 
        pos_ds[-1,...] = pos
        pos_ds.flush()
        return pos, data

    async def go_to_position(self, position: Sequence[float]) -> None:
        """ Move to the specified position.

        Args:
            attrs: List of coordinates of each axis to move to.
        """
        assert len(position) == self.ndim, f'Invalid number of attributes {len(position)}'
        old_pos = self.current_pos.copy()
        t0 = time.time()
        for i in range(self.ndim):
            await self.move_axis(i, position[i])
        self.logging.debug('moving from {} to {} took {:.3f} seconds'.format(old_pos, position, time.time()-t0))

    async def move_axis(self, axis: int, target: float) -> None:
        """ Move the specified axis to the specified target position.

        Args:
            axis: The axis to move.
            target: The target position.
        """
        assert axis in range(self.ndim), f'Invalid axis {axis}'
        if not self.position_is_allowed(axis, target):
            self.logger.warning(f'Invalid target {target} for axis {axis}')
        delay = max(0.05,abs(target - self.current_pos[axis]) / self.MOTOR_SPEED)
        await asyncio.sleep(delay)
        self.current_pos[axis] = target

    async def scan_loop(self) -> None:
        """
        Start the scan.

        Start a loop that moves the scanner to the next position in the queue
        and waits for the dwell time.
        Once the queue is empty, if self.wait_at_queue_empty is False, the loop stops.
        Otherwise, it waits for new points to be added to the queue.
        """
        self.logger.info('Starting scan...')
        self.wait_at_queue_empty = True
        self.current_pos = [np.mean(l) for l in self.limits]
        while True:
            if len(self.queue) == 0:
                if not self.wait_at_queue_empty:
                    self.logger.info('queue is empty, stopping scan')
                    break
                # self.log('queue is empty, waiting is {}...'.format(self.wait_at_queue_empty), end='\r')
                # await asyncio.sleep(self.dwell_time)
                changed = [False] * self.ndim
            else:
                next_pos = self.queue.pop(0)
                # manhattan distance:
                distance = sum([abs(next_pos[i] - self.current_pos[i]) for i in range(self.ndim)])
                self.logger.debug(f'Moving to {next_pos}. takes {distance/self.MOTOR_SPEED:.2f} seconds')
                delay = max(0.05, distance / self.MOTOR_SPEED)
                await asyncio.sleep(delay)
                changed = [new != old for new, old in zip(next_pos, self.current_pos)]
                self.current_pos = next_pos
                # await self.go_to_position(next_pos)
            _ = self.measure(changed)
            await asyncio.sleep(self.dwell_time)

        self.logger.info('Scan finished')
        self.source_file.close()

    def parse_message(self, message: str) -> str:
        """ Parse a message received from the client.

        these are the possible commands and expected responses:

        # ORDERS:

        ADD_POINT xx yy zz - add position xxyyzz to the queue -> ADD_POINT xx yy zz queue_length
        CLEAR - reset the queue -> CLEAR
        SCAN - start the scan -> SCAN
        END - stop waiting at queue empty -> END queue_length
        ABORT - stop the scan -> ABORT
        PAUSE - pause the scan or resumes it -> PAUSE

        # REQUESTS:

        LIMITS - returns the limits of the scanner -> LIMITS xmin,xmax ymin,ymax zmin zmax
        NDIM - returns the number of dimensions -> NDIM n
        CURRENT_POS - returns the current position -> CURRENT_POS xx yy zz
        QUEUE - returns all the points in the queue -> QUEUE xx,yy,zz, xx,yy,zz xx yy zz
        STATUS - returns the status of the scanner -> STATUS status
        FILENAME - returns the filename of the current scan -> FILENAME filename

        ERROR errortype parameters

        
        Args:
            message (str): The message to parse.

        Returns:
            str: The response to send to the client.    
        """
        f'Received message "{message}"\n'
        msg = message.strip('\r\n').split(' ')
        try:
            attr = getattr(self, msg[0])
            if len(msg) > 1:
                answer = attr(*msg[1:])

            else:
                answer = attr()
        # except AttributeError:
        #     answer = f'INVALID_COMMAND {msg[0]}'
        except Exception as e:
            answer = f'ERROR {type(e).__name__} {e}'
            raise e
        finally:
            truncated_message = message[:50] + '...' if len(message) > 50 else message
            truncated_answer = answer[:50] + '...' if len(answer) > 50 else answer
            self.logger.info(f'Received message "{truncated_message}", answer "{truncated_answer}"')
            return answer

    def ADD_POINT(self, *args) -> str:
        assert len(args) == self.ndim, f'expected {self.ndim} arguments, got {len(args)}'
        points = [float(x) for x in args]
        self.queue.append(points)
        pts = ' '.join([str(x) for x in args])
        return f'ADD_POINT {pts}'# {len(self.queue)}

    def CLEAR(self) -> str:
        self.queue = []
        return f'CLEAR'

    def SCAN(self) -> str:
        self.status = 'SCANNING'
        return f'SCAN'    
    
    def END(self) -> str:
        self.wait_at_queue_empty = False
        return f'END {len(self.queue)}'

    def ABORT(self) -> str:
        self.status = 'ABORTED'
        # TODO: stop the measurement loop
        return f'ABORT'

    def PAUSE(self) -> str:
        self.status = 'PAUSED' if self.status == 'SCANNING' else 'SCANNING'
        return f'PAUSE {self.status}'

    def LIMITS(self) -> str:
        # assert len(args) == 2*self.ndim, f'expected {2*self.ndim} arguments, got {len(args)}'
        limits = ' '.join([f'{x},{y}' for x,y in  zip(self.limits[::2], self.limits[1::2])])
        return f'LIMITS {limits}'
    
    def NDIM(self) -> str:
        assert self.ndim in [1,2,3], f'invalid ndim {self.ndim}'
        return f'NDIM {self.ndim}'

    def CURRENT_POS(self) -> str:
        pos_str = ' '.join([ f'{d}:{x}' for d,x in zip(self.dims, self.current_pos)])
        return f'CURRENT_POS {pos_str}'
    
    def QUEUE(self) -> str:
        return f'QUEUE {len(self.queue)}'
    
    def STATUS(self) -> str:
        return f'STATUS {self.status}'
    
    def FILENAME(self) -> Path:
        return f'FILENAME {self.target_file_name}'

    def ERROR(self, error: str) -> str:
        return f'ERROR {error}'
    
    def MEASURE(self) -> str:
        # pos, data = self.measure()
        pos_str =  ' '.join([str(v) for v in self.current_pos])
        data_str = ' '.join([str(np.round(v,4).astype(np.float32)) for v in np.random.rand(640,400).ravel()])
        time.sleep(np.random.rand(1)[0])
        print()
        return f'MEASURE {len(self.current_pos)} {pos_str} {data_str}'

    def __del__(self) -> None:
        self.close_file()

class FileSGM4(SGM4FileManager, VirtualSGM4):

    def __init__(
            self, 
            ip: str, 
            port: int, 
            filename: str | Path, 
            verbose: bool = True, 
            dwell_time: float = 0.1
        ) -> None:
        """ A virtual SGM4 that reads a file containing a list of points to scan.

        Args:
            ip (str): The IP address of the server.
            port (int): The port of the server.
            filename (str | Path): The file containing the points to scan.
            verbose (bool, optional): Whether to print messages. Defaults to True.
            dwell_time (float, optional): The time to wait at each point. Defaults to 0.1.
        """
        if filename is not None:
            self.filename = Path(filename)
            self.open()
        else:
            raise ValueError('filename cannot be None')

        super().__init__(
            ip = ip,
            port = port,
            ndim = self.ndim,
            filename = self.filename,
            limits = self.limits,
            verbose = verbose,
            dwell_time = dwell_time,
        )
        self.measured = []

    async def measure(self, position: Sequence[float]) -> xr.DataArray:
        """ Measure the specified point.
        
        Args:
            position: The coordinates of the point to measure.
            
        Returns:
            the spectrum measured at the specified point.
        """
        assert len(position) == self.ndim, f'Invalid number of attributes {len(position)}'
        value = self.xdata.sel({self.dims[i]: position[i] for i in range(self.ndim)}).values
        self.logger.info(f'Waiting for {self.dwell_time} seconds')
        await asyncio.sleep(self.dwell_time)
        self.measured.loc[{self.dims[i]: position[i] for i in range(self.ndim)}] = value
        return value
    

if __name__ == '__main__':

    # source_file = r"D:\data\SGM4 - 2022 - CrSBr\data\Kiss05_15_1.h5"
    source_file =  Path(r"D:\data\SGM4\SmartScan\Z006_46.h5")
    source_file =  Path(r"D:\data\SGM4\SmartScan\Z006_35_0.h5")

    name = Path(source_file).stem

        # init logger
    logger = logging.getLogger('virtualSGM4')
    logger.setLevel('DEBUG')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setLevel('DEBUG')
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # fh = logging.FileHandler(os.path.join(args.logdir, args.logfile))
    # fh.setLevel(args.loglevel)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)
    logger.info('Intialized Logger')
    # init scan manager

    vm = VirtualSGM4(
        'localhost', 
        54333, 
        verbose=True,
        logger=logger,
    )
    vm.init_scan_from_file(filename=source_file)
    filedir = Path(
        r"C:\Users\stein\OneDrive\Documents\_Work\_code\ARPES-ASTRID\smartscan\data"
        )
    i=0
    while True:
        filename = filedir / f'{name}_virtual_{i:03.0f}.h5'
        if not filename.exists():
            break
        else:
            i += 1
    
    vm.create_file(
        mode='x',
        filename=filename
    )

    vm.current_pos = np.mean(vm.limits[:2]), np.mean(vm.limits[2:])
    print('set current pos to', vm.current_pos)

    vm.run()
    print('All done. Quitting...')
