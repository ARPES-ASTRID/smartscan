from typing import Any, Dict, Callable, Tuple, Union, List, Sequence
import multiprocessing as mp
import time
import itertools
import asyncio
from pathlib import Path
import logging

import numpy as np
import xarray as xr
from tqdm.auto import trange, tqdm

import h5py

from smartscan.TCP import TCPServer
import smartscan.tasks as processing


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
            buffer_size:int = 1024*1024*8,
            dwell_time: float = None,
            simulate_times: bool = False,
            save_to_file: bool = True,
            logger=None,
    ) -> None:
        super().__init__(ip, port)
        self.logger = logger or logging.getLogger(f"{__name__}.VirtualSGM4")
        self.logger.info('init VirtualSGM4 object')
        self.output_queue = mp.Queue()
        self.input_queue = mp.Queue()
        self.status = 'IDLE' # TODO: implement status
        if source_file is not None:
            self.init_scan_from_file(source_file)
        elif limits is not None and step_size is not None:
            self.init_scan(ndim, limits, step_size, dwell_time)
        self.file = None
        self.simulate_times = simulate_times
        self.save_to_file = save_to_file
        self.dwell_time = dwell_time

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
            self.dwell_time = self.dwell_time or f.dwell_time
            self.starts = f.starts
            self.stops = f.stops
            self.steps = f.steps
            self.step_sizes = []
            self.lengths = f.lengths
            self.positions = f.positions
            self.limits = f.limits
            self.current_pos = [c[l//2] for c, l in zip(self.coords.values(), self.map_shape)]
        self.input_queue.put(self.current_pos)

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
        t0 = time.time()
        assert len(position) == self.ndim, f'Position {position} has wrong dimension.'
        pos = self.nearest_position_on_grid(position)
        if self.source_file is not None:
            with self.source_file as f:
                poslist = [tuple(p) for p in self.positions]
                index = poslist.index(pos)
                data = f.file['Entry/Data/TransformedData'][index]
                self.logger.debug('reading data from file took {:.3f} seconds'.format(time.time()-t0))
        else:
            data = np.random.rand(*self.signal_shape)
            self.logger.debug('Generating fake data took {:.3f} seconds'.format(time.time()-t0))
        return pos, data

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

    def acquire_data(self, changed:Sequence[bool]) -> float:
        """ Fake measuring the current position."""

        self.logger.debug(f'Measuring position {self.current_pos}, changed {changed}')
        pos, data = self.read_data(self.current_pos)
        if self.save_to_file:
            self.write_data(pos, data)
        self.output_queue.put_nowait((pos, data))
        self.logger.info(f'Acquired data at position {pos} | output queue size: {self.output_queue.qsize()}')
        return pos, data

    def write_data(self, position: Tuple[float], data: np.ndarray) -> None:
        """ Write data to the file.

        Args:
            position (Tuple[float]): _description_
            data (np.ndarray): _description_
        """
        t0 = time.time()
        data_ds = self.file["Entry/Data/TransformedData"]
        data_ds.resize((data_ds.shape[0] + 1),axis=0)
        data_ds[-1,...] = data
        data_ds.flush()
        pos_ds = self.file["Entry/Data/ScanDetails/SetPositions"]
        pos_ds.resize((pos_ds.shape[0] + 1), axis=0)
        pos = list(position)
        # pos[not changed] = -99999999. 
        pos_ds[-1,...] = pos
        pos_ds.flush()
        self.logger.debug('writing data to file took {:.3f} seconds'.format(time.time()-t0))

    async def scan_loop(self) -> None:
        """
        Start the scan.

        Start a loop that moves the scanner to the next position in the queue
        and waits for the dwell time.
        Once the queue is empty, if self.wait_at_queue_empty is False, the loop stops.
        Otherwise, it waits for new points to be added to the queue.
        """
        self.logger.info('Starting scan...')
        self.logger.info('Scan initialized, waiting for start command...')
        while True:
            if self.status != 'IDLE':
                self.logger.info('Scan started!')
                break
            await asyncio.sleep(0.1)   
        self.logger.debug('Freezing server for ~2 sec...')
        time.sleep(2) # yes, this should freeze the server for 2 seconds
        self.logger.info('Server Ready!')
        self.status = 'ready'
        
        self.wait_at_queue_empty = True
        # self.current_pos = [np.mean(l) for l in self.limits]
             

        while True:
            if self.input_queue.empty():
                if not self.wait_at_queue_empty:
                    self.logger.info('queue is empty, stopping scan')
                    break
                # self.log('queue is empty, waiting is {}...'.format(self.wait_at_queue_empty), end='\r')
                # await asyncio.sleep(self.dwell_time)
                changed = [False] * self.ndim
            else: # move
                next_pos = self.input_queue.get_nowait()
                # manhattan distance:
                distance = sum([abs(next_pos[i] - self.current_pos[i]) for i in range(self.ndim)])
                travel_time = distance / self.MOTOR_SPEED
                self.logger.info(f'Moving to {next_pos} | {travel_time:.2f} seconds | {distance:.2f} um | input queue size: {self.input_queue.qsize()}')
                if self.simulate_times:
                    await asyncio.sleep(travel_time)
                else:
                    await asyncio.sleep(0.05)
                changed = [new != old for new, old in zip(next_pos, self.current_pos)]
                self.current_pos = next_pos
                # await self.go_to_position(next_pos)
            if self.output_queue.qsize() < 100:
                _ = self.acquire_data(changed)
            else:
                self.logger.warning(f'Queue is full, not acquiring data. Queue size: {self.output_queue.qsize()}'
                                    f'Input queue size: {self.input_queue.qsize()}')
                await asyncio.sleep(1)
            if self.simulate_times:
                self.logger.debug('waiting for {:.2f} seconds'.format(self.dwell_time))
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
        # f'Received message "{message}"\n'
        msg = message.strip('\r\n').split(' ')
        try:
            attr = getattr(self, msg[0])
            if len(msg) > 1:
                answer = attr(*msg[1:])

            else:
                answer = attr()
        except Exception as e:
            answer = f'ERROR {type(e).__name__} {e}'
            raise e
        finally:
            truncated_message = message[:50] + '...' if len(message) > 50 else message
            truncated_answer = answer[:50] + '...' if len(answer) > 50 else answer
            # self.logger.debug(f'Received message "{truncated_message.strip('\n')}", answer "{truncated_answer.strip('\n')}"')
            return answer

    def ADD_POINT(self, *args) -> str:
        assert len(args) == self.ndim, f'expected {self.ndim} arguments, got {len(args)}'
        points = [float(x) for x in args]
        self.input_queue.put_nowait(points)
        pts = ' '.join([str(x) for x in args])
        return f'ADD_POINT {pts}'# {len(self.queue)}

    def CLEAR(self) -> str:
        self.input_queue.clear()
        return f'CLEAR'

    def SCAN(self) -> str:
        self.status = 'SCANNING'
        return f'SCAN'    
    
    def START(self) -> str:
        self.status = 'READY'
        return f'START'

    def END(self) -> str:
        self.wait_at_queue_empty = False
        return f'END {self.input_queue.qsize()}'

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
        return f'QUEUE {self.input_queue.qsize()}'
    
    def STATUS(self) -> str:
        return f'STATUS {self.status}'
    
    def FILENAME(self) -> Path:
        return f'FILENAME {self.target_file_name}'

    def ERROR(self, error: str) -> str:
        return f'ERROR {error}'
    
    def MEASURE(self) -> str:

        # pos, data = self.acquire_data(changed = [False,False])
        if self.output_queue.empty():
            return f'NO_DATA'
        pos, data = self.output_queue.get_nowait()
        pos_str =  ' '.join([str(v) for v in pos])
        data_str = ' '.join([str(np.round(v,4).astype(np.float32)) for v in data.ravel()])
        self.logger.info(f'MEASURE {len(pos)} {pos_str} {data_str[:30]}...')
        # pos_str =  ' '.join([str(v) for v in self.current_pos])
        # data_str = ' '.join([str(np.round(v,4).astype(np.float32)) for v in np.random.rand(640,400).ravel()])
        # time.sleep(np.random.rand(1)[0])
        # print()
        return f'MEASURE {len(self.current_pos)} {pos_str} {data_str}'

    def SHAPE(self) -> str:
        return f'SHAPE {self.signal_shape[0]} {self.signal_shape[1]}'

    def STEP_SIZE(self) -> str:

        return f'STEP_SIZE {" ".join([str(s) for s in self.steps])}'



    def __del__(self) -> None:
        self.close_file()


class SGM4FileManager:
    """Reads SMG4 files."""
    INVALID_POS = -999999999.0

    def __init__(self, filename: str, swmr: bool=True) -> None:
        """Initialize reader.

        Args:
            filename: path to file to read
        """
        self.filename = filename
        self.file = None
        self.swmr = swmr
        self._stack = None
        self._data = None
        self._positions = None

    def __enter__(self) -> Any:
        """Open file."""
        self.open()
        return self
    
    def __exit__(self, *args) -> None:
        """Close file."""
        self.close()

    def __del__(self) -> None:
        """Close file."""
        self.close()

    def open(self) -> None:
        """Open file."""
        Warning('File is opened, this might mess up saving data!')
        self.file = h5py.File(self.filename, 'r', swmr=self.swmr)
    
    def close(self) -> None:
        """Close file."""
        if self.file is not None:
            self.file.close()

    def touch(self) -> bool:
        """Touch file to check if all is ok.
        
        TODO: add initialization checks
        """
        try:
            with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
                pass
        except FileNotFoundError:
            return False
        return True
    
    def __len__(self) -> int:
        """Get number of spectra in file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/TransformedData'].shape[0]

    def get_data(self,index: int | slice) -> np.ndarray:
        """Get data from file.

        Args:
            index: index of spectrum to get. If slice, get slice of data
                if int, get single spectrum

        Returns:
            data: data from file
        """
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            ds = f['Entry/Data/TransformedData']
            assert ds is not None, 'File does not contain data'
            # cache data
            if self._data is None:
                self._data = np.zeros(ds.shape)
            elif self._data.shape != ds.shape: 
                # resize if shape is different
                old = self._data
                self._data = np.zeros(ds.shape)
                self._data[:old.shape[0],:old.shape[1]] = old
            if isinstance(index, int):
                index = slice(index,None,None)
            # read data from file if not cached
            if self._data[index].sum() == 0:
                self._data[index] = ds[index]
            # print(f'sending data with shape {ds[index,...].shape}')
            return self._data[index]
    
    def get_positions(self, index: int | slice) -> np.ndarray:
        """Get positions from file.

        Args:
            index: index of position to get. If slice, get slice of data

        Returns:
            positions: positions from file
        """
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            # ds = f['Entry/Data/ScanDetails/SetPositions']
            ds = f['Entry/Data/ScanDetails/TruePositions']
            assert ds is not None, 'File does not contain positions'
            # cache positions
            if self._positions is None:
                self._positions = np.zeros(ds.shape)
            elif self._positions.shape != ds.shape: 
                # resize if shape is different
                old = self._positions
                self._positions = np.zeros(ds.shape)
                self._positions[:old.shape[0],:old.shape[1]] = old
            if isinstance(index, int):
                index = slice(index,None,None)
            # read positions from file if not cached
            if self._positions[index].sum() == 0:
                self._positions[index] = ds[index]
        return self._positions[index]
        
    def get_new_data(self,len_old_data: int) -> Tuple[np.ndarray]:
        """Get new data from file.

        args:
            len_old_data: length of old data

        Returns:
            positions,data: positions and data from file
        """
        if not self.has_new_data(len_old_data):
            return None,None
        else:
            new = len(self) - len_old_data
            self.logger.debug(f'found {new} spectra')
            positions = self.get_positions(slice(-new,None,None))
            data = self.get_last_n_spectra(new)#get_data(slice(-new,None,None))
            self.logger.debug(f'data shape {data.shape}')
            return positions, data
        
    def get_last_n_spectra(self,n):
        out = np.zeros((n,640,400))
        for i in range(n):
            with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
                ds = f['Entry/Data/TransformedData']
                out[-i,:,:] = ds[-i,:,:]
        return out

    def get_merged_data(self, index: int, func:str | Callable='mean') -> dict:
        """Get data from file. compute mean of data with the same position
        
        combine positions and reduced data to a list of 1D np.array
        
        Args:
            index: index of spectrum to get. If slice, get slice of data
                if int, get single spectrum
                
        Returns:
            data: data from file
        """
        data = self.get_data(index)
        positions = self.get_positions(index)
        merged = {}
        counts = {}
        # combine data with the same position
        for p,d in tqdm(zip(positions,data),total=len(data),desc='Merging data'):
            if p in merged:
                merged[p] += d
                counts[p] += 1
            else:
                merged[p] = d
                counts[p] = 1
        # get the mean of data with the same position
        merged = {k:v/counts[k] for k,v in merged.items()}
        return merged
    
    def has_new_data(self, len_old_data:int) -> bool: 
        """Check if file has new data.

        Args:
            len_old_data: length of old data

        Returns:
            has_new_data: True if file has new data
        """
        return len(self) > len_old_data
    
    def get_reduced_data(self, index: int | slice, pipe: Sequence[str | Callable]=["sum"]) -> np.ndarray:
        """ Apply dimensionality reduction to data from file.

        Args:
            index: index of spectrum to get. If slice, get slice of data
                if int, get single spectrum
            func: function to reduce data with
        
        Returns:
            reduced: reduced data
        """
        data = self.get_combined_data(index)
        reduced = np.zeros((len(data),len(data[0])))
        for i,d in tqdm(enumerate(data),total=len(data),desc='Reducing data'):
            reduced[i] = self.process(d,pipe)
        return reduced

    def process(self, data:np.ndarray, pipe: Sequence[str | Callable]) -> np.ndarray:
        """Apply a sequence of functions to data.
        
        Args:
            data: data to process
            pipe: sequence of functions to apply to data
        
        Returns:
            processed: processed data
        """
        processed = data
        for f in pipe:
            if isinstance(f,str):
                processed = getattr(processing,f)(processed)
            else:
                processed = f(processed)
        return processed
    
    @property
    def stack(self) -> np.ndarray:
        """Get stack from file."""
        if self._stack is None:
            raise ValueError('Stack has not been loaded yet. Run reduce() first.')
        else:
            return self._stack

    def reduce(self, func: Union[callable, str]) -> np.ndarray:
        """Reduce each spectrum to a smaller feature space.

        Args:
            func: function to reduce stack with

        Returns:
            reduced: reduced stack
        """
        if hasattr(np, func): # check if function is a numpy function
            func = getattr(np, func)
        elif callable(func):
            pass
        elif hasattr(processing, func): # check if function is a function in the reduce module
            func = getattr(processing, func)
        else:
            raise ValueError("Function is not a numpy function nor a callable nor a "/
                             "function in the reduce module")
        for i in trange(len(self), desc='Reducing spectra'):
            reduced = func(self.get_data(i))
            if not isinstance(reduced, np.ndarray):
                reduced = np.array(reduced)
            if reduced.ndim == 0:
                reduced = reduced.reshape((1,))
            elif reduced.ndim >= 2:
                reduced = reduced.flatten()
            # cache reduced stack
            if self._stack is None:
                self._stack = np.empty((len(self),len(reduced)+self.ndim))
            if self._stack.shape[1] != len(reduced)+self.ndim:  
                # resize if shape is different
                old = self._stack
                self._stack = np.empty((len(self),len(reduced)+self.ndim))
                self._stack[:old.shape[0],:old.shape[1]] = old

            pos = self.get_positions(i)
            if pos.ndim == 0:
                pos = pos.reshape((1,))
            elif pos.ndim >= 2:
                pos = pos.flatten()
    
            self._stack[i,...] = np.concatenate([pos,reduced])

        return self._stack
    
    @property
    def raw_data(self) -> np.ndarray:
        """Get stack from file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f["Entry/Data/TransformedData"][()]
    
    @property
    def raw_dims(self) -> Tuple[int]:
        """Get shape of spectra."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/ScanDetails/SlowAxis_names'][()]

    @property
    def raw_shape(self) -> Tuple[int]:
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/ScanDetails/SlowAxis_names'][()]

    @property
    def positions(self) -> Tuple[np.ndarray]:
        """Get positions from file.

        Also substitutes invalid positions with the previous valid position.

        Returns:
            positions: array of positions
        """
        try:
            with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
                pos_array = f['Entry/Data/ScanDetails/SetPositions'][()]
                corrected = []
                previous = None
                for line in pos_array:
                    assert len(line) == self.ndim, 'length of position does not match dimensionality'
                    corr_line = []
                    for i, p in enumerate(line):
                        if p == self.INVALID_POS:
                            if previous is None:
                                raise ValueError('First position is invalid')
                            else:
                                p = previous[i]
                        corr_line.append(p)
                    corrected.append(corr_line)
                    previous = corr_line
        except KeyError:
            Warning('File does not contain positions. Probably loading old data. Using axes instead.')
            corrected =  list(itertools.product(*self.axes))[:len(self)]
        return np.array(corrected)
    
    @property
    def ndim(self) -> int:
        """Get number of dimensions from file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return int(f['Entry/Data/ScanDetails/Dimensions'][()])
    
    @property
    def limits(self) -> List[Tuple[float]]:
        """Get limits from file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:

            lengths = f['Entry/Data/ScanDetails/SlowAxis_length'][()]
            starts = f['Entry/Data/ScanDetails/SlowAxis_start'][()]
            steps = f['Entry/Data/ScanDetails/SlowAxis_step'][()]
            assert len(lengths) == len(starts) == len(steps) == self.ndim, \
                'lengths of limits and dimensionality do not match'
            limits = [(start, start + step * (length - 1)) for start, step, length in zip(starts, steps, lengths)]
            limits = [item for sublist in limits for item in sublist]
            return limits
    
    @property
    def starts(self) -> List[float]:
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/ScanDetails/SlowAxis_start'][()]
    
    @property
    def steps(self) -> List[float]:
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/ScanDetails/SlowAxis_step'][()]
    
    @property
    def stops(self) -> List[float]:
        return [start + step * (length - 1) for start, step, length in zip(self.starts, self.steps, self.lengths)]
    
    @property
    def lengths(self) -> List[int]:
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return f['Entry/Data/ScanDetails/SlowAxis_length'][()]
        
    @property
    def map_shape(self):
        """Get shape of map."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            return tuple(f['Entry/Data/ScanDetails/SlowAxis_length'][()])
    
    @property
    def map_size(self):
        """Get size of map."""
        return np.prod(self.map_shape)

    @property
    def axes(self) -> List[np.ndarray]:
        """Get coordinate axes from file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            lengths = f['Entry/Data/ScanDetails/SlowAxis_length'][()]
            starts = f['Entry/Data/ScanDetails/SlowAxis_start'][()]
            steps = f['Entry/Data/ScanDetails/SlowAxis_step'][()]
            axes = [np.linspace(start, start + step * (length -1), length) for start, step, length in zip(starts, steps, lengths)]
            return axes
    
    @property
    def coords(self) -> Dict[str, np.ndarray]:
        """Get coordinates from file."""
        coords = {dim:axis for dim, axis in zip(self.dims, self.axes)}
        return coords
    
    @property
    def dims(self) -> np.ndarray:
        """Get dimensions from file."""
        with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
            dims = f['Entry/Data/ScanDetails/SlowAxis_names'][()]
        dims = [d.decode('utf-8') for d in dims]
        return dims
    
    @property
    def dwell_time(self) -> float:
        """Get dwell time from file."""
        return 1.1 # TODO: alfred will fix
    
    @property
    def beam_current(self) -> float:
        """ get beam current from file

        Returns:
            _description_
        """
        try:
            with h5py.File(self.filename, 'r', swmr=self.swmr) as f:
                bc = f['Entry/Instrument/Monochromator/Beam Current'][()]
                if bc[0] == bc[-1] == -1:
                    bc = np.ones(len(self))
                    Warning('Beam current not recorded. Using 1.0 instead.')
        except KeyError:
            bc = np.ones(len(self))
            Warning('File does not contain beam current. Using 1.0 instead.')
        
        return bc

    def normalize_to_beam_current(self, img) -> np.ndarray:
        """Normalize image to beam current.

        Args:
            img: image to normalize
        
        Returns:
            img: normalized image
        """
        return img /(self.beam_current[:, None, None] * self.dwell_time)

    def to_xarray(self) -> np.ndarray:
        """Unravel stack into an nD array."""
        raise NotImplementedError
    
    def nearest(self, position: Sequence[float]) -> Tuple[int]:
        """Find nearest position in the grid.

        Args:
            position: position to find
        
        Returns:
            index: index of nearest position
        """
        assert len(position) == self.ndim, 'length of position does not match dimensionality'
        nearest = []
        for pos, axis in zip(position, self.axes):
            assert axis.min() <= pos <= axis.max(), 'position is outside of limits'
            nearest.append(axis[np.argmin(np.abs(axis - pos))])
        return tuple(nearest)
   

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
    pass