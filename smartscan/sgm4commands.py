from typing import Callable, Sequence, Tuple, Union, List
from tqdm.auto import tqdm
import numpy as np
import time
from pathlib import Path

from .utils import manhattan_distance, closest_point_on_int_grid
from .tcp import send_tcp_message


class SGM4Commands:
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

        self._limits = None
        self._step_size = None
        self._map_shape = None
        self._axes = None
        self._ndim = None



    @property
    def ndim(self) -> int:
        """ number of dimensions of the scan """
        if self._ndim is None:
            self._ndim = self.NDIM()
        return self._ndim

    @property
    def limits(self) -> List[List[float]]:
        """ limits of the scan """
        if self._limits is None:
            self._limits = self.LIMITS()
        return self._limits
    
    @property
    def step_size(self) -> List[float]:
        """ step size of the scan """
        if self._step_size is None:
            self._step_size = self.STEP_SIZE()
        return self._step_size

    @property
    def map_shape(self) -> Tuple[int]:
        """ shape of the map """
        if self._map_shape is None:
            self._map_shape = []
            for (start,stop), step in zip(self.limits, self.step_size):
                self._map_shape.append(int(np.ceil((stop-start)/step)))
        return tuple(self._map_shape)

    @property
    def axes(self) -> List[List[float]]:
        """ axes of the scan """
        if self._axes is None:
            self._axes = []
            for (start,stop), step in zip(self.limits, self.step_size):
                self._axes.append(np.arange(start, stop, step))
        return self._axes
    
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
    
    def connect(self) -> None:
        """ ask SGM4 for the scan info 
        
        TODOs
        - check if it is a new connection or not (i.e. if we have a filename)
        - add status information on both sides

        command order:
        NDIM - get number of dimensions
        LIMITS - after NDIM, as you need to know how many to expect
        FILENAME - get the filename of the scan
        CURRENT_POS - where are we starting from?
        
        """
        self._ndim = self.NDIM()
        self._limits = self.LIMITS()
        assert len(self._limits) == self._ndim, f"Expected {self._ndim} limits, got {len(self._limits)}"
        try:
            self._current_pos = self.CURRENT_POS()
            assert len(self._current_pos) == self._ndim, f"Expected {self._ndim} current positions, got {len(self._current_pos)}"
        except IndexError:
            pass
        self._filename = Path(self.FILENAME())
        if self._filename.is_file():
            self.parse_file()
        else:
            Warning(f"Expected {self._filename} to be a file")        

    def disconnect(self) -> None:
        """ disconnect from SGM4 """
        Warning("Disconnecting from SGM4 is not yet implemented")

    def ADD_POINT(self, *args) -> None:
        """ add a point to the scan queue 
        
        Args:
            args: position to add to the scan queue

        Returns:
            flag indicating success
        """
        assert len(args) == self._ndim, f"Expected {self._ndim} args, got {len(args)}"
        assert all(a != self.INVALID_NUMBER for a in args), f"DO NOT move to {self.INVALID_NUMBER}"
        response = self.send_command('ADD_POINT', *args)
        split = response.split(' ')
        cmd = split.pop(0)
        vals = [float(x) for x in split]
        assert cmd == 'ADD_POINT', f"Expected ADD_POINT, got {cmd}"
        assert len(vals) == self._ndim, f"Expected {self._ndim} args, got {len(vals)}"
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
        print(response)
        split = response.split(' ')
        assert split[0] == 'LIMITS', f"Expected LIMITS, got {split[0]}"
        limits = [tuple([float(l) for l in lim.split(',')]) for lim in split[1:]]
        assert len(limits) == self.ndim, f"Expected {self._ndim} limits, got {len(limits)}"
        self._limits = limits
        return limits
    
    def QUEUE(self) -> List[Tuple[float]]:
        """ get the queue of the scan and store it in self.queue

        Returns:
            queue: list of tuples of floats
        """
        response = self.send_command('QUEUE')
        split = response.split(' ')
        assert split[0] == 'QUEUE', f"Expected QUEUE, got {split[0]}"
        # queue_size = int()
        return split[1]

    def NDIM(self) -> int:
        """ get the number of dimensions and store it in self.ndim 
        
        Returns:
            ndim: number of dimensions
        """
        response = self.send_command('NDIM')
        split = response.split(' ')
        assert split[0] == 'NDIM', f"Expected NDIM, got {split[0]}"
        self._ndim = int(split[1])
        return self._ndim
    
    def FILENAME(self) -> str:
        """ get the filename of the scan and store it in self.filename
        
        Returns:
            filename: filename of the scan
        """
        response = self.send_command('FILENAME')
        split = response.split(' ')
        assert split[0] == 'FILENAME', f"Expected FILENAME, got {split[0]}"
        self._filename = split[1]
        return self._filename
    
    def CURRENT_POS(self) -> List[float]:
        """ get the current position 


        Returns:
            dict: {axis: position}
        """
        response = self.send_command('CURRENT_POS')
        split = response.split(' ')
        assert split[0] == 'CURRENT_POS', f"Expected CURRENT_POS, got {split[0]}"
        current_pos = {str(p[0]):float(p[1]) for p in [x.split(':') for x in split[1:]]}
        assert len(current_pos) == self.ndim, f"Expected {self.ndim} positions, got {len(current_pos)}"
        self._current_pos = current_pos
        return current_pos 

    def STEP_SIZE(self) -> List[float]:
        """ get the step size of the scan and store it in self.step_size
        
        Returns:
            step_size: list of floats
        """
        response = self.send_command('STEP_SIZE')
        split = response.split(' ')
        assert split[0] == 'STEP_SIZE', f"Expected STEP_SIZE, got {split[0]}"
        step_size = [float(s) for s in split[1:]]
        assert len(step_size) == self.ndim, f"Expected {self.ndim} step sizes, got {len(step_size)}"
        self._step_size = step_size
        return step_size

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

    def MEASURE(self):
        """ Measure the current position
        
        Returns:
            ack: MEASURE
        """
        message = self.send_command('MEASURE')
        vals = message.strip('\r\n').split(' ')
        msg_code = vals[0]
        vals = [v for v in vals[1:] if len(v) >0]
        match msg_code:
            case 'ERROR':
                return message, None
            case 'NO_DATA':
                return message, None
            case 'MEASURE':
                n_pos = int(vals[0])
                pos = np.asarray(vals[1:n_pos+1], dtype=float)
                data = np.asarray(vals[n_pos+1:], dtype=float)
                # data = data.reshape(640,400)
                return pos, data
            case _:
                self.logger.warning(f'Unknown message code: {msg_code}')
                return message, None