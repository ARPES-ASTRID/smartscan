import socket
from pathlib import Path
from typing import List, Tuple, Sequence, Union

from .TCP import send_tcp_message

class Controller:
    """ Controller for the SGM4

    Args:
        host: host to connect to
        port: port to connect to
        checksum: add a checksum to the message
        buffer_size: size of the buffer to use
        verbose: print out extra information
        timeout: timeout for the connection
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
        self.filename = self.FILENAME()
        self.current_pos = self.CURRENT_POS()

    def parse_h5_file(self, filename:str | Path) -> None:
        """Read the h5 file and get the scan info

        assert the info found in the file matches the info found in the SGM4


        Args:
            filename: path to the h5 file

        Returns:
            None
        """
        raise NotImplementedError
        

    def LIMITS(self) -> List[Tuple[float]]:
        """ get the limits of the scan """
        response = self.send_command('LIMITS')
        split = response.split(' ')
        assert split[0] == 'LIMITS', f"Expected LIMITS, got {split[0]}"
        return [tuple(lim.split(',')) for lim in split[1:]]
    
    def NDIM(self) -> int:
        """ get the number of dimensions """
        response = self.send_command('NDIM')
        split = response.split(' ')
        assert split[0] == 'NDIM', f"Expected NDIM, got {split[0]}"
        return int(split[1])
    
    def FILENAME(self) -> str:
        """ get the filename of the scan """
        response = self.send_command('FILENAME')
        split = response.split(' ')
        assert split[0] == 'FILENAME', f"Expected FILENAME, got {split[0]}"
        return split[1]
    
    def CURRENT_POS(self) -> List[float]:
        """ get the current position """
        response = self.send_command('CURRENT_POS')
        split = response.split(' ')
        assert split[0] == 'CURRENT_POS', f"Expected CURRENT_POS, got {split[0]}"
        return [float(x) for x in split[1:]]
    
    def ADD_POINT(self, *args) -> None:
        """ add a point to the scan queue """
        assert len(args) == self.ndim, f"Expected {self.ndim} args, got {len(args)}"
        assert all(a != self.INVALID_NUMBER for a in args), f"DO NOT move to {self.INVALID_NUMBER}"
        response = self.send_command('ADD_POINT', *args)
        split = response.split(' ')
        assert split[0] == 'ADD_POINT', f"Expected ADD_POINT, got {split[0]}"
        for v in split[1:]:
            try: # TODO: implement warning if answer hits out of range.
                if float(v) == self.INVALID_NUMBER:
                    Warning("The point sent was out of range. It was ignored...")
                return False
            except  ValueError:
                pass
        return True
    