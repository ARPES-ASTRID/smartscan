from typing import Dict, Tuple, List, Callable
import time
import itertools
import random
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm, trange

from .controller import SGM4Controller, Fake_SGM4Controller
from .file import SGM4FileManager
from .virtualSGM4 import VirtualSGM4


class SmartScan:

    def __init__(
            self, 
            host: str = None, 
            port: int = None,
            **kwargs,
        ) -> None:
        self._sgm4 = None
        if host is not None and port is not None:
            self.connect(host, port, **kwargs)
        self._file = None
        self._raw_data = None
        self._raw_positions = None
        self._data_dict = None
        self._counts_dict = None
        self._reduced_data = None
        self._limits = None

    @property
    def limits(self) -> List[Tuple[float]]:
        if self._limits is None:
            file_lims = self.file.limits
            sgm4_lims = self.sgm4.limits
            if file_lims != sgm4_lims:
                Warning('limits from file and from sgm4 do not match!')
            else:
                self._limits = file_lims
        return self._limits

    @property
    def sgm4(self) -> SGM4Controller:
        if self._sgm4 is None:
            raise ValueError("SGM4 not connected")
        return self._sgm4

    def connect(
            self, 
            host: str, 
            port: int, 
            checksum: bool = False, 
            verbose: bool = True, 
            timeout: float = 1, 
            buffer_size: int = 1024,
            **kwargs,
        ) -> None:
        """ Connect to the device """
        if self._sgm4 is not None:
            raise ValueError("Already connected")
        if isinstance(host,Path):
            Ctrl = Fake_SGM4Controller
        # if ldr is an IP:
        elif host == 'localhost': #TODO: impolement real IP address
            Ctrl = SGM4Controller
        self._sgm4 = Ctrl(
            host=host, 
            port=port, 
            checksum=checksum, 
            verbose=verbose, 
            timeout=timeout, 
            buffer_size=buffer_size, 
            **kwargs,
        )

    def disconnect(self) -> None:
        """ Disconnect from the device """
        if self._sgm4 is None:
            raise ValueError("Not connected")
        self._sgm4.disconnect()
        self._sgm4 = None

    @property
    def file(self) -> SGM4FileManager:
        if self._file is None:
            raise ValueError("No file open")
        return self._file

    def open(self) -> None:
        """ Open a file """
        filename = self.sgm4.FILENAME()
        if self._file is not None:
            raise ValueError("Already open")
        self._file = SGM4FileManager(filename)
        if not self._file.touch():
            raise ValueError("Could not read file")

    def close(self) -> None:
        """ Close the file """
        if self._file is None:
            raise ValueError("No file open")
        self._file.close()
        self._file = None

    def __del__(self) -> None:
        """ Close the file and disconnect from the device """
        if self._file is not None:
            self.close()
        if self._sgm4 is not None:
            self.disconnect()

    @property
    def raw_data(self):
        if self._raw_data is None:
            raise ValueError("No data")
        return self._raw_data
    
    @property
    def values(self):
        if self._data_dict is None:
            raise ValueError("No data")
        vals = np.array(tuple(self._data_dict.values()))
        weights = self.weights
        assert len(vals) == len(weights), "data values weights dont have the same shape!"
        return vals * weights[:,None,None]
    
    @property
    def positions(self):
        if self._data_dict is None:
            raise ValueError("No data")
        return np.array(tuple(self._data_dict.keys()))
    
    @property
    def counts(self):
        if self._counts_dict is None:
            raise ValueError("No data")
        return np.array(tuple(self._counts_dict.values()))
    
    @property
    def weights(self):
        # TODO: implement beam current normalization here
        return self.counts
    
    @property
    def reduced_data(self):
        if self._reduced_data is None:
            raise ValueError("No data")
        elif len(self._reduced_data) != len(self._data_dict):
            raise ValueError("Data not reduced")
        return np.array(tuple(self._reduced_data.values()))

    @property
    def reduced_values(self):
        if self._reduced_data is None:
            raise ValueError("No data")
        elif len(self._reduced_data) != len(self._data_dict):
            raise ValueError("Data not reduced")
        return np.array(tuple(self._reduced_data.values()))
    
    def update_data(self) -> None:
        """ Look if there is new data available and update the data attribute """
        len_data_so_far = len(self._raw_data) if self._raw_data is not None else 0
        new_positions, new_data = self.file.get_new_data(len_data_so_far)
        data_dict, counts_dict = self.combine_data_by_position(new_positions, new_data)
        if new_data is not None:
            if self._raw_data is None:
                self._raw_data = new_data
                self._raw_positions = new_positions
                self._data_dict = data_dict
                self._counts_dict = counts_dict
            else:
                self._raw_data = np.concatenate([self._raw_data, new_data], axis=0)
                self._raw_positions = np.concatenate([self._raw_positions, new_positions], axis=0)
                self._data_dict = self._data_dict.update(data_dict)
                self._counts_dict = self._counts_dict.update(counts_dict)
        
    @staticmethod
    def combine_data_by_position(positions, data, pbar=True) -> Dict[Tuple[float], np.ndarray]:
        """ Combine data by position

        Args:
            positions: positions of the data
            data: data

        Returns:
            data: combined data
        """
        unique_positions = np.unique(positions, axis=0)
        combined_data = {}# = np.zeros((len(unique_positions), *data.shape[1:]))
        counts = {}
        # for pos in tqdm(unique_positions, disable=not pbar, desc="Combining data"):
        #     combined_data[pos] = np.mean(data[positions == pos], axis=0)
        for pos, d in tqdm(zip(positions,data),disable=not pbar, desc='Combining same position data'):
            pos_tuple = tuple(pos)
            if pos_tuple in combined_data.keys():
                combined_data[pos_tuple] += d
                counts[pos_tuple] += 1
            else:
                combined_data[pos_tuple] = d
                counts[pos_tuple] = 1
        return combined_data, counts

    def reduce_data(
            self,  
            transform:Callable,
            *args,
            pbar: bool = True,
            **kwargs
        ) -> None:
        """ reduce the data to a single value

        Args:
            transform: transformation to apply to the data. Must be a callable which 
                takes the data as first argument, args and kwargs, and returns n values 
                equal to the number of input parameters of the gaussian process.
            args: args to pass to the transform
            kwargs: kwargs to pass to the transform  
            pbar: show progress bar
        """
        n_new = len(self.data) - len(self.reduced_data)
        new_data = self.data[-n_new:]
        new_pos = self.positions[-n_new:]
        for pos, data in tqdm(zip(new_pos,new_data), desc="Reducing data", total=n_new, disable=not pbar):
            self._reduced_data[pos] = transform(data, *args, **kwargs)

    def init_random(self, n_init:int = 10, ) -> None:
        """ Initialize the measurement loop with random points """
        remaining_idx = itertools.product(*self.file.axes)
        # shuffle the remaining_idx
        remaining_idx = list(remaining_idx)
        random.shuffle(remaining_idx)
        for next in remaining_idx[:n]:
            self.sgm4.ADD_POINT(*next)
        
    def measurement_loop(self, max_iter:int = None, ) -> None:
        """ Start the measurement loop """
        if self._sgm4 is None:
            raise ValueError("Not connected")
        if self._file is None:
            raise ValueError("No file open")
        print('\n\n Starting the measurement loop \n\n')
        if max_iter is None:
            max_iter = self.file.map_size

        try:
            for i in range(max_iter-self.n_init):
                t0 = time.time()
                # read the data from the hdf5 file
                self.update_data()
                self.reduce_data(pbar=False)
                # evaluate the next position
                next = self.evaluation()
                # send a move command to the controller
                self.sgm4.ADD_POINT(*next)
                if time.time() - t0 < self.sleep_time:
                    time.sleep(self.sleep_time - (time.time() - t0))
            # end the scan
        finally:
            self.END()
            
    def evaluation(self) -> Tuple[float]:
        """ Evaluate the data and return the next position """
        raise NotImplementedError


class SmartScanRandom(SGM4Controller):

    def evaluation(self) -> Tuple[float]:
        """ Evaluate the data and return the next position """
        # get the data from the stack
        all_idx = itertools.product(*self.file.axes)
        remaining_idx = [idx for idx in all_idx if idx not in self.positions.keys()]
        random.shuffle(remaining_idx)
        return remaining_idx[0]


class RandomCommander(SGM4Controller):

    def __init__(
            self, 
            host: str, 
            port: int, 
            checksum: bool = False, 
            verbose: bool = True, 
            timeout: float = 1, 
            buffer_size: int = 1024,
            sleep_time=0.1,
    ) -> None:
        super().__init__(host, port, checksum, verbose, timeout, buffer_size)
        self.name = 'RandomController'
        self.status = 'unpaused'
       
        self.stack = []
        self.data_by_position = {}
        self.sleep_time = sleep_time
        
    def start_scan(self, n_init:int=10, max_iter:int=None) -> None:
        """ Start a smart scan
         using asyncio:
        1. initialize the scan with n random points
        2. start the measurement loop in which:
            1. read the data from the hdf5 file
            2. launch the evaluation of the data
            3. send a move command to the controller
            4. wait for the dwell time, if points 2 to 4 are faster than the dwell time
            5. repeat until all points are measured
        3. end the scan

        Args:
            n_init: number of initial random points

        Returns:
            None
        """
        if max_iter is None:
            max_iter = np.product(self.map_shape)
        # initialize the scan
        self.stack = []
        self.data_by_position = {}
        self.init_random(n_init)
        # wait for the intialization scan to finish
        time.sleep(self.sleep_time*(n_init+1))
        # start the measurement loop
        print('\n\n Starting the measurement loop \n\n')
        try:
            for i in range(max_iter-n_init):
                t0 = time.time()
                # read the data from the hdf5 file
                has_new = self.update_stack()
                # launch the evaluation of the data
                if has_new:
                    next = self.evaluation()
                    # send a move command to the controller
                    self.ADD_POINT(*next)
                if time.time() - t0 < self.sleep_time:
                    time.sleep(self.sleep_time - (time.time() - t0))
            # end the scan
        finally:
            self.END()

    def update_stack(self) -> dict:
        """ Update the stack with the data from the hdf5 file

        Returns:
            dict: {position: reduced data}
        """
        with SGM4FileManager(self._filename) as file:
            if len(file) > len(self.stack):
                n = len(file) - len(self.stack)
                data = list(file.get_data(slice(-n,None,None)))
                positions = list(file.positions[-n:])
                for i in trange(n,desc='Updating stack'):
                # for d,p in tqdm(zip(self.process(data),positions),total=n,desc='Processing data'):
                    d = self.process(data[i])
                    p = tuple(positions[i])
                    self.stack.append([*p,d])
                    if p not in self.data_by_position:
                        self.data_by_position[p] = [d]
                    else:
                        self.data_by_position[p].append(d)
                print(f"Added {n} points to the stack")
                return True
        return False

    def process(self,data:List[np.ndarray]) -> np.ndarray:
        """ Process the data

        Args:
            data: list of data

        Returns:
            list: processed data
        """
        return np.array([np.sum(data),np.mean(data),np.std(data)])

    @property
    def samples(self) -> np.ndarray:
        """ Return the samples from the stack

        Returns:
            np.ndarray: samples
        """
        out = []
        for k,v in self.data_by_position.items(): 
            out.append([*k,*np.mean(v,axis=0)])
        return np.array(out)
    
    def init_random(self,n) -> None:
        """ Initialize the scan with n random points

        Args:
            n: number of random points
        
        Returns:
            None
        """
        with SGM4FileManager(self._filename) as file:
            remaining_idx = itertools.product(*file.axes)
        # shuffle the remaining_idx
        remaining_idx = list(remaining_idx)
        random.shuffle(remaining_idx)
        for next in remaining_idx[:n]:
            print(f"Measuring {next}")
            self.ADD_POINT(*next)
        
    def evaluation(self) -> List[tuple]:
        """ Evaluate the data and return the next point to measure

        Returns:
            List[tuple]: list of next points to measure
        """
        with SGM4FileManager(self._filename) as file:
            all_idx = itertools.product(*file.axes)
        remaining_idx = [idx for idx in all_idx if idx not in self.data_by_position.keys()]
        random.shuffle(remaining_idx)
        return remaining_idx[0]

    def start_random_scan(self) -> None:
        """ Start a random scan

        Returns:
            None
        """
        with SGM4FileManager(self._filename) as file:
            remaining_idx = itertools.product(*file.axes)
        # shuffle the remaining_idx
        remaining_idx = list(remaining_idx)
        random.shuffle(remaining_idx)
        for next in remaining_idx:
            print(f"Measuring {next}")
            self.ADD_POINT(*next)
        self.END()
            # self.CURRENT_POS()
        print("Scan complete!")

