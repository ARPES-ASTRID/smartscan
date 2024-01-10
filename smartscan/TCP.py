import asyncio
from asyncio.streams import StreamReader, StreamWriter
import socket
from abc import ABC, abstractmethod
import hashlib
import logging

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


def send_tcp_message(
        host:str,
        port:int|str,
        msg:str,
        checksum:bool=False,
        buffer_size:int=1024,
        verbose:bool=False,
        timeout:float=1.0,
        CLRF:bool=True,
        logger=None,
    ) -> str:
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
    if logger is None:
        logger = logging.getLogger('send_tcp_message')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        logger.debug(f"TPC Connecting to {host}:{port}")
        s.connect((host, port))
        if checksum:
            msg = add_checksum(msg)
            logger.debug(f"TCP Sending message with checksum: {msg}")
        else:
            logger.debug(f" TCP Sending message: {msg}")
        if CLRF:
            msg += "\r\n"
        if len(msg) > buffer_size:
            raise ValueError(f'Message is too long. {len(msg)}/{buffer_size}')
        s.sendall(msg.encode())
        logger.debug("Waiting for response")
        data = s.recv(buffer_size).decode()
        logger.debug(f"Received: {data}")
        data = remove_checksum(data)
        
    return data.strip("\r\n")


class TCPServer:

    def __init__(
            self, 
            host: str, 
            port: int, 
            checksum: bool = False, 
            verbose: bool = True,
            timeout: float = 0.1,
            message_size: int = 1024*1024*16,
            ) -> None:
        self.verbose = verbose
        self.checksum = checksum
        self.host = host
        self.port = port
        self.server = None
        self.message_size = message_size

    def log(self, message: str, end='\n') -> None:
        """
        Print a message to the console.

        Args:
            message (str): The message to print.
        """
        if self.logger is not None:
            self.logger.info(message)
        elif self.verbose:
            print(message,end=end)

    async def handle_client(self, reader: StreamReader, writer: StreamWriter):
        """
        Handle communication with a client.

        Args:
            reader (StreamReader): The reader object for receiving data from the client.
            writer (StreamWriter): The writer object for sending data to the client.
        """
        client_address = writer.get_extra_info('peername')
        self.log(f'New connection from {client_address}')

        while True:
            data = await reader.read(self.message_size) 
            if not data:
                break
            if self.checksum and '||' in data.decode('utf-8'):
                # Verify the checksum
                message, recieved_checksum = data.decode('utf-8').split('||')
                calculated_checksum = calculate_checksum(data.decode('utf-8'))
                if recieved_checksum != calculated_checksum:
                    self.log(f'Checksum mismatch: {recieved_checksum} != {calculated_checksum}')
                    continue
            elif self.checksum:
                self.log('No checksum found')
                continue
            else:
                message = data.decode('utf-8')
            self.log(f'Received message: {message}')
            response = self.parse_message(message)
            # Send a response
            writer.write(response.encode('utf-8'))
            await writer.drain()

        self.log(f'Connection from {client_address} closed')
        writer.close()

    @abstractmethod
    def parse_message(self, message: str):
        """
        Parse a message received from the client.

        Args:
            message (str): The message to parse.

        Returns:
            str: The response to send to the client.
        """
        response = f'Received message "{message}"\n'
        self.log(response, end='')
        return response
        
    async def tcp_loop(self):
        """
        Start the TCP server.
        """
        self.server = await asyncio.start_server(
            self.handle_client, self.host, self.port)

        print(f'TCP server is listening on {self.host}:{self.port}...')

        async with self.server:
            await self.server.serve_forever()

    async def all_loops(self):
        """
        Start all the loops.
        """
        loop_methods = [getattr(self, method)() for method in dir(self) if method.endswith('_loop')]
        await asyncio.gather(*loop_methods)

    def close(self):
        """
        Close the TCP server.
        """
        self.server.close()

    def run(self):
        """
        Run the TCP server.
        """
        loop = asyncio.get_event_loop()
        try:
            loop.run_until_complete(self.all_loops())
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()


class TCPClient:
    def __init__(
            self, 
            host: str, 
            port: int,
            checksum:bool=False,
            end: str = '\r\n',
            verbose:bool=True,
            timeout:float=1.0,
            buffer_size:int=1024*1024*8,
            ) -> None:
        self.host = host
        self.port = port
        self.reader = None
        self.writer = None
        self.verbose = verbose
        self.timeout = timeout
        self.checksum = checksum
        self.buffer_size = buffer_size
        self.end = end

    async def connect(self):
        """
        Connect to the TCP server.
        """
        self.reader, self.writer = await asyncio.open_connection(
            self.host, self.port, limit=self.buffer_size)

    async def send_message(self, message: str):
        """
        Send a message to the TCP server.

        Args:
            message (str): The message to send.
        """
        if self.checksum:
            message = add_checksum(message)
        if self.end:
            message += self.end
        self.writer.write(message.encode('utf-8'))
        await self.writer.drain()

    async def receive_message(self) -> str:
        """
        Receive a message from the TCP server.

        Returns:    
            str: The message received from the TCP server.
        """
        data = await self.reader.read(self.buffer_size)#, timeout=self.timeout)
        if self.checksum and '||' in data.decode('utf-8'):
            # Verify the checksum
            message, recieved_checksum = data.decode('utf-8').split('||')
            calculated_checksum = calculate_checksum(data.decode('utf-8'))
            if recieved_checksum != calculated_checksum:
                self.log(f'Checksum mismatch: {recieved_checksum} != {calculated_checksum}')
                return None
        elif self.checksum:
            self.log('No checksum found')
            return None
        else:
            message = data.decode('utf-8')
        return message

    def close(self):
        """
        Close the connection to the TCP server.
        """
        self.writer.close()
    
    def __del__(self):
        try:
            self.close()
        except:
            pass


if __name__ == '__main__':
    server = TCPServer('localhost', 12345)
    server.run()
    # # add user inputs when running in console
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--host', type=str, default='localhost')
    # parser.add_argument('--port', type=int, default=12345)
    # parser.add_argument('--ndim', type=int, default=3)
    # parser.add_argument('--verbose', action='store_true')
    # parser.add_argument('--message', type=str, default='END')
    # args = parser.parse_args()


    # client = TCPClient(args.host, args.port)
    # asyncio.run(client.connect())
    # asyncio.run(client.send_message(args.message))
    # message = asyncio.run(client.receive_message())
    # print(f'Received message: {message}')
