from typing import Any, Union
import logging
import socket
import hashlib
import time

import numpy as np
from sympy import Q
from PyQt6 import QtCore, QtGui, QtWidgets




class TCPManager(QtCore.QObject):

    def __init__(self, parent=None, settings: dict=None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger(f"{__name__}.TCPManager")
        self.logger.debug("init TCPManager")
        self.p = parent

        self.host = self.settings['host']
        self.port = self.settings['port']
        self.buffer_size = self.settings['buffer_size']
        self.timeout = self.settings.get('timeout',1.0)
        self.checksum = self.settings.get('checksum',False)
        self.CLRF = self.settings.get('CLRF',True)
        
        self.logger.debug(f"TCP settings: {self.settings}")

    
    @property
    def settings(self) -> dict:
        """Get the settings."""
        return self.p.settings['TCP']   

    @QtCore.pyqtSlot()
    def send_tcp_message(self, msg:str) -> str:
        """ send a message to a host and port and return the response

        Args:
            host: host to connect to
            port: port to connect to
            msg: message to send
            checksum: add a checksum to the message
            buffer_size: size of the buffer to use
            verbose: print out extra information
            timeout: timeout for the connection: # TODO: implement timeout
            CLRF: add a carriage return and line feed to the message
            
        Returns:
            data: response from host
        """

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            self.logger.debug(f"TPC Connecting to {self.host}:{self.port}")
            s.connect((self.host, self.port))
            if self.checksum:
                msg = add_checksum(msg)
                self.logger.debug(f"TCP Sending message with checksum: {msg}")
            else:
                self.logger.debug(f"TCP Sending message: {msg}")
            if self.CLRF:
                msg += "\r\n"
            if len(msg) > self.buffer_size:
                raise ValueError(f'Message is too long. {len(msg)}/{self.buffer_size}')
            s.sendall(msg.encode())
            self.logger.debug("TCP Waiting for response")
            data = s.recv(self.buffer_size).decode()
            datastr = data[:30] + '...' if len(data) > 30 else data
            self.logger.debug(f"TCP Received: {datastr}")
            data = remove_checksum(data)
    
    @QtCore.pyqtSlot()
    def ask(self, msg:str) -> str:
        """ send a command to the device and return the parsed response
        
        Args:
            msg: message to send
        
        Returns:
            response: parsed response from device
        """
        response_str = self.send_tcp_message(msg)
        response: list[str] = response_str.strip("\r\n").split(" ")
        response_code: str = response[0]
        response_values: list[str] = [v for v in msg[1:] if len(v) > 0]

        self.logger.debug(f"{msg} answer: {msg_code}: {len(response)/1024:,.1f} kB")
        return response_code, response_values
    
    def read_data(self) -> tuple[str, None] | tuple[np.ndarray,np.ndarray] | None:
        """Get data from SGM4.

        Returns:
            tuple[str, None] | tuple[NDArray[Any], NDArray[Any]] | None:
                - tuple[str, None] if there was an error
                - tuple[NDArray[Any], NDArray[Any]] if there was no error
                - None if no data was received
        """
        self.logger.debug("Fetching data...")

        res, vals = self.ask("MEASURE")
        if res == 'MEASURE':
            n_pos = int(vals[0])
            pos = np.asarray(vals[1 : n_pos + 1], dtype=float)
            data = np.asarray(vals[n_pos + 1 :], dtype=float)
            data = data.reshape(self.remote.spectrum_shape)
            return pos, data
        else:
            return False

        match res:
            case "ERROR":
                self.logger.error(message)
                return message, None
            case "NO_DATA":
                self.logger.debug(f"No data received: {message}")
                return message, None
            case "MEASURE":

            case _:
                self.logger.warning(f"Unknown message code: {msg_code}")
                return message, None
    
def calculate_checksum(message: str) -> str:
    """
    Calculates the checksum of a message using sha256.

    Args:
        message (str): The message to calculate the checksum of.

    Returns:
        str: The checksum of the message.
    """
    hash_object = hashlib.sha256(message.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig


def add_checksum(message: str) -> str:
    """ add a checksum to the message """
    return message + f'||{calculate_checksum(message)}'


def remove_checksum(message: str) -> str:
    """ remove the checksum from the message and verify it"""
    split = message.split('||')
    if len(split) == 1:
        return message
    else:
        message, checksum = split
        if calculate_checksum(message) == checksum:
            return message
        else:
            raise ValueError('Checksum mismatch')
    

def check_checksum(message: str) -> bool:
    """ check the checksum of the message """
    message, checksum = message.split('||')
    return calculate_checksum(message) == checksum
