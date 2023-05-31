from typing import List,Tuple,Sequence,Union
import time
import asyncio
from pathlib import Path

import xarray as xr
# import dataloader as dl

from .TCP import TCPServer
from .file import SGM4FileReader


class VirtualSGM4(TCPServer):

    MOTOR_SPEED = 300 # um/s

    def __init__(
            self,
            ip: str,
            port: int,
            ndim: int = 2,
            filename: Union[str, Path] = 'test.txt',
            limits: List[Tuple[float]] = None,
            verbose: bool = True,
            dwell_time: float = 0.1,
    ) -> None:
        super().__init__(ip, port)
        self.queue = []
        self.status = 'IDLE' # TODO: implement status
        self.ndim = ndim
        self.limits = limits if limits is not None else [(10_000, 10_000)] * self.ndim
        self.current_pos = [0] * self.ndim
        self.verbose = verbose
        self.dwell_time = dwell_time
        self.wait_at_queue_empty = False
        self.filename = filename

    def position_is_allowed(self, axis: int, target: float) -> bool:
        """ Check if the target position is allowed for the specified axis.
        """
        return self.limits[axis*self.ndim] <= target <= self.limits[axis*self.ndim+1]

    async def measure(self) -> float:
        """ Fake measuring the current position.
        """
        # wait for the dwell time
        await asyncio.sleep(self.dwell_time)
        # self.last_measure = 0.0
        return 0.0

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
        self.log('moving from {} to {} took {:.3f} seconds'.format(old_pos, position, time.time()-t0))

    async def move_axis(self, axis: int, target: float) -> None:
        """ Move the specified axis to the specified target position.

        Args:
            axis: The axis to move.
            target: The target position.
        """
        assert axis in range(self.ndim), f'Invalid axis {axis}'
        if not self.position_is_allowed(axis, target):
            self.log(f'Invalid target {target} for axis {axis}')
        delay = abs(target - self.current_pos[axis]) / self.MOTOR_SPEED
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
        self.log('Starting scan...')
        while True:
            if len(self.queue) == 0:
                if not self.wait_at_queue_empty:
                    self.log('queue is empty, stopping scan')
                    break
                self.log('queue is empty, waiting is {}...'.format(self.wait_at_queue_empty), end='\r')
                await asyncio.sleep(1)
                continue
            next_pos = self.queue.pop(0)
            self.log(f'Moving to {next_pos}')
            await self.go_to_position(next_pos)
            _ = await self.measure()


        self.log('Scan finished')

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
            self.log(f'Sending answer "{answer}"')
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
        pos_str = ' '.join([str(x) for x in self.current_pos])
        return f'CURRENT_POS {pos_str}'
    
    def QUEUE(self) -> str:
        return f'QUEUE {self.queue}'
    
    def STATUS(self) -> str:
        return f'STATUS {self.status}'
    
    def FILENAME(self) -> str:
        return f'FILENAME {self.filename}'

    def ERROR(self, error: str) -> str:
        return f'ERROR {error}'
    

class FileSGM4(SGM4FileReader, VirtualSGM4):

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
        self.log(f'Waiting for {self.dwell_time} seconds')
        await asyncio.sleep(self.dwell_time)
        self.measured.loc[{self.dims[i]: position[i] for i in range(self.ndim)}] = value
        return value
    

if __name__ == '__main__':

    
    vm = VirtualSGM4(
        'localhost', 
        12345, 
        ndim = 2, 
        limits=[-10000,10000,-10000,10000], 
        verbose=True
    )
    t0 = time.time()
    lin_pts = [
        (0,0),
        # (100,0),(200,0),(300,0),(400,0),(500,0),
        # (0,100),(100,100),(200,100),(300,100),(400,100),(500,100),
        # (0,200),(100,200),(200,200),(300,200),(400,200),(500,200),
        # (0,300),(100,300),(200,300),(300,300),(400,300),(500,300),
    ]

    vm.queue=lin_pts
    vm.wait_at_queue_empty = True

    vm.run()
    # print('All done. Quitting...')
