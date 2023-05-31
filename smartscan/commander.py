import socket
import itertools
import random
import asyncio
import time

from pathlib import Path
from typing import List, Tuple, Sequence, Union

import numpy as np
from tqdm.auto import tqdm, trange

from .TCP import send_tcp_message
from .file import SGM4FileReader

class SGM4Commander:
    """ Controller for the SGM4

    Args:
        host: host to connect to
        port: port to connect to
        checksum: add a checksum to the message
        verbose: print out extra information
        timeout: timeout for the connection
        buffer_size: size of the buffer to use
    """
    INVALID_NUMBER = -999999999.

    def __init__(
            self, 
            host: str, 
            port: int,
            checksum:bool=False,
            verbose:bool=True,
            timeout:float=1.0,
            buffer_size:int=1024,
            ) -> None:
        # TCP
        self.host = host
        self.port = port
        self.checksum = checksum
        self.verbose = verbose
        self.timeout = timeout
        self.buffer_size = buffer_size

        # SGM4
        self.filename = None
        self.ndim = None
        self.limits = None
        self.current_pos = None

    def send_command(self, command, *args) -> None:
        """ send a command to SGM4 and wait for a response
        
        Args:
            command: command to send
            args: arguments to send with the command
            
        Returns:
            response: response from SGM4
        """
        message = command.upper()
        for arg in args:
            message += f' {arg}'
        response = send_tcp_message(
            host=self.host,
            port=self.port,
            msg=message,
            checksum=self.checksum,
            verbose=self.verbose,
            timeout=self.timeout,
            buffer_size=self.buffer_size,
            CLRF=True,
        )
        if "INVALID" in response:
            raise RuntimeError(f"Invalid command: {command}")
        return response
    
    def get_scan_info(self) -> None:
        """ ask SGM4 for the scan info 
        
        command order:
        NDIM - get number of dimensions
        LIMITS - after NDIM, as you need to know how many to expect
        FILENAME - get the filename of the scan
        CURRENT_POS - where are we starting from?
        
        """
        self.ndim = self.NDIM()
        self.limits = self.LIMITS()
        assert len(self.limits) == self.ndim, f"Expected {self.ndim} limits, got {len(self.limits)}"
        try:
            self.current_pos = self.CURRENT_POS()
            assert len(self.current_pos) == self.ndim, f"Expected {self.ndim} current positions, got {len(self.current_pos)}"
        except IndexError:
            pass
        self.filename = Path(self.FILENAME())
        if self.filename.is_file():
            self.parse_file()            
        else:
            Warning(f"Expected {self.filename} to be a file")        

    def parse_file(self,filename:Path | str = None) -> None:
        if filename is None:
            filename = self.filename
        else:
            filename = Path(filename)
        assert filename.is_file(), f"Expected {filename} to be a file"
        self.filename = filename

        with SGM4FileReader(filename) as file:
            assert self.ndim == file.ndim, f"Expected {self.ndim} dimensions, got {file.ndim}"
            sgm4_limits = [sorted(l) for l in self.limits],
            file_limits = [sorted(l) for l in file.limits],
            assert sgm4_limits == file_limits, f"Expected {sgm4_limits} limits from SGM4, got {file_limits} from file"
            self.map_shape = file.map_shape

    def parse_h5_file(self, filename:str | Path) -> None:
        """Read the h5 file and get the scan info

        assert the info found in the file matches the info found in the SGM4


        Args:
            filename: path to the h5 file

        Returns:
            None
        """
        raise NotImplementedError
        
    def ADD_POINT(self, *args) -> None:
        """ add a point to the scan queue 
        
        Args:
            args: position to add to the scan queue

        Returns:
            flag indicating success
        """
        assert len(args) == self.ndim, f"Expected {self.ndim} args, got {len(args)}"
        assert all(a != self.INVALID_NUMBER for a in args), f"DO NOT move to {self.INVALID_NUMBER}"
        response = self.send_command('ADD_POINT', *args)
        split = response.split(' ')
        cmd = split.pop(0)
        vals = [float(x) for x in split]
        assert cmd == 'ADD_POINT', f"Expected ADD_POINT, got {cmd}"
        assert len(vals) == self.ndim, f"Expected {self.ndim} args, got {len(vals)}"
        if any(a == self.INVALID_NUMBER for a in vals):
            Warning(f"The position provided on axis {vals.index(self.INVALID_NUMBER)}"\
                    f" is invalid. "
            )
        return True
     
    def CLEAR(self) -> None:
        """ clear the scan queue 
        
        Returns:
            flag indicating success
        """
        response = self.send_command('CLEAR')
        split = response.split(' ')
        assert split[0] == 'CLEAR', f"Expected CLEAR, got {split[0]}"
        return True

    def LIMITS(self) -> List[Tuple[float]]:
        """ get the limits of the scan and store them in self.limits
        
        Returns:
            limits: list of tuples of floats
        """
        response = self.send_command('LIMITS')
        split = response.split(' ')
        assert split[0] == 'LIMITS', f"Expected LIMITS, got {split[0]}"
        limits = [tuple([float(l) for l in lim.split(',')]) for lim in split[1:]]
        assert len(limits) == self.ndim, f"Expected {self.ndim} limits, got {len(limits)}"
        self.limits = limits
        return limits
    
    def QUEUE(self) -> List[Tuple[float]]:
        """ get the queue of the scan and store it in self.queue

        Returns:
            queue: list of tuples of floats
        """
        response = self.send_command('QUEUE')
        split = response.split(' ')
        assert split[0] == 'QUEUE', f"Expected QUEUE, got {split[0]}"
        queue_size = int(split[1])
        return queue_size

    def NDIM(self) -> int:
        """ get the number of dimensions and store it in self.ndim 
        
        Returns:
            ndim: number of dimensions
        """
        response = self.send_command('NDIM')
        split = response.split(' ')
        assert split[0] == 'NDIM', f"Expected NDIM, got {split[0]}"
        self.ndim = int(split[1])
        return self.ndim
    
    def FILENAME(self) -> str:
        """ get the filename of the scan and store it in self.filename
        
        Returns:
            filename: filename of the scan
        """
        response = self.send_command('FILENAME')
        split = response.split(' ')
        assert split[0] == 'FILENAME', f"Expected FILENAME, got {split[0]}"
        self.filename = split[1]
        return self.filename
    
    def CURRENT_POS(self) -> List[float]:
        """ get the current position 


        Returns:
            dict: {axis: position}
        """
        response = self.send_command('CURRENT_POS')
        split = response.split(' ')
        assert split[0] == 'CURRENT_POS', f"Expected CURRENT_POS, got {split[0]}"
        current_pos = {str(p[0]):float(p[1]) for p in [x.split(',') for x in split[1:]]}
        assert len(current_pos) == self.ndim, f"Expected {self.ndim} positions, got {len(current_pos)}"
        self.current_pos = current_pos
        return current_pos 

    def END(self):
        """ End the scan after completeing the current queue

        Returns:
            ack: the size of the remaining queue
        """
        response = self.send_command('END')
        split = response.split(' ')
        assert split[0] == 'END', f"Expected END, got {split[0]}"
        assert len(split) == 2, f"Expected 2 args, got {len(split)}"
        return split[1]

    def ABORT(self):
        """ Abort the scan

        Returns:
            ack: ABORT
        """
        response = self.send_command('ABORT')
        assert response == 'ABORT', f"Expected ABORT, got {response}"
        # TODO: stop the measurement loop
        return True

    def PAUSE(self):
        """ Pause the scan
        
        Returns:
            paused or unpaused status
        """
        response = self.send_command('PAUSE')
        split = response.split(' ')
        assert split[0] == 'PAUSE', f"Expected PAUSE, got {response}"
        self.status = split[1]
        return str(split[1])


class RandomController(SGM4Commander):

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
                self.update_stack()
                # launch the evaluation of the data
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
        with SGM4FileReader(self.filename) as file:
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
        with SGM4FileReader(self.filename) as file:
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
        with SGM4FileReader(self.filename) as file:
            all_idx = itertools.product(*file.axes)
        remaining_idx = [idx for idx in all_idx if idx not in self.data_by_position.keys()]
        random.shuffle(remaining_idx)
        return remaining_idx[0]

    def start_random_scan(self) -> None:
        """ Start a random scan

        Returns:
            None
        """
        with SGM4FileReader(self.filename) as file:
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

