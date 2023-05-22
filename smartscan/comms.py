from typing import Tuple, List, Literal, Union
import time
import socket
import hashlib
import numpy as np

def send_tcp_messge(host,port,msg):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((host, port))
        s.sendall(msg.encode())
        data = s.recv(1024)
    return data

def calculate_checksum(message: str) -> str:
    """
    Calculates the SHA256 checksum of a given message.

    Parameters:\n        -----------
    message : str
        The message to calculate the checksum for
    
    Returns:
    --------
    The SHA256 checksum of the given message.
    """
    hash_object = hashlib.sha256(message.encode())
    hex_dig = hash_object.hexdigest()
    return hex_dig

class TCPMessenger:
    """
    A class for sending and receiving messages through TCP.

    Attributes:
    -----------
    ip : str
        IP address to bind the socket to
    port : int
        Port number to bind the socket to
    socket : socket.socket
        TCP socket used for communication
    connection : socket object
        Connection object returned from socket.accept()
    address : tuple
        Address of the connected socket

    Methods:
    --------
    calculate_checksum(message: str) -> str
        Calculates the SHA256 checksum of a given message
    connect(ip: str, port: int) -> None
        Connects to another TCP server
    send_message(message: str) -> None
        Sends a message through the TCP socket
    receive_message() -> str
        Receives a message through the TCP socket and verifies its checksum
    close() -> None
        Closes the TCP connection
    """

    def __init__(
            self, 
            ip: str, 
            port: int, 
            CLRF:bool=True,
            verbose:bool=False,
            checksum:bool=False,
        ) -> None:
        """
        Initializes a TCPMessenger instance.

        Parameters:
        -----------
        ip : str
            IP address to bind the socket to
        port : int
            Port number to bind the socket to
        """
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.ip, self.port))
        self.socket.listen(1)
        self.connection = None
        self.address = None
        self.CLRF = CLRF
        self.verbose = verbose
        self.checksum = checksum

    def log(self,msg:str):
        if self.verbose:
            print(msg)

    def ping(self, timeout:int) -> bool:
        t0 = time.time()
        self.send_message('ping\r\n')
        ans = self.receive_message(timeout=timeout)
        dt = time.time() - t0
        print(f'{self.ip}:{self.port} answered in {dt/1000:.3f} ms') 
        
    def connect(self, ip: str, port: int) -> None:
        """
        Connects to another TCP server.

        Parameters:
        -----------
        ip : str
            The IP address to connect to
        port : int
            The port number to connect to
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((ip, port))
        self.log(f'Connected to {ip},{port}')

    def send_message(self, message: str) -> None:        
        """
        Sends a message through the TCP socket.

        Parameters:
        -----------
        message : str
            The message to send through the TCP socket
        """
        if self.checksum:
            checksum = calculate_checksum(message)
            message = f"{message}||{checksum}"
        if self.CLRF:
            message += "\r\n"
        self.socket.sendall(message.encode())
        self.log(f'Sent message: {message}')

    def receive_message(self, msg_size = None) -> str:
        """
        Receives a message through the TCP socket and verifies its checksum.

        Returns:
        --------
        If the message checksum is correct, return the message.
        Otherwise, return an error message.
        """
        self.connection, self.address = self.socket.accept()
        if msg_size is None:
            msg_size=1024
        data = self.connection.recv(msg_size)
        message = data.decode()
        if "||" in message:
            message, checksum = message.split("||")
            if calculate_checksum(message) != checksum:
                self.connection.sendall("error checksum_mismatch")
                return "Invalid message received - checksum mismatch"
        
            
        return message
        
    def close(self) -> None:
        """
        Closes the TCP connection.
        """
        if self.connection:
            self.connection.close()
        self.socket.close()
        self.log(f'Connection closed')

    def __del__(self) -> None:
        self.close()


class VirtualSGM4(TCPMessenger):
    # this class acts as a server which checks weather a mesage is valid
    # and returns a fake answer

    def __init__(
            self, 
            ip:str,
            port:Union[int,str],
            position_limits:List[Tuple[float]] = None,
            verbose:bool=True,
        ) -> None:
        super().__init__(ip=ip,port=port,verbose=verbose)

    def start_server(self,) -> None:
        print(f'Starting server at {self.ip}:{self.port}')
        while True:
            msg = self.receive_message()
            if msg == 'ping':
                print('recieved ping')
                self.send_message('pong')  
            elif msg == 'quit':
                print('recieved quit')
                self.send_message('quitting')
                break
            else:
                print(f'recieved message {msg}')
                # check that the message has this structure "i f t" where:
                # i is an integer between 0 and 2
                # f and t are numbers
                try:
                    i,f,t = msg.split(' ')
                    i = int(i)
                    f = float(f)
                    t = float(t)
                    if i not in [0,1,2]:
                        raise ValueError
                    self.send_message('moving to {f} {t}')
                except ValueError:
                    print('invalid message')
                    self.send_message('invalid message')
                

def recieve_tcp_message(host, port):
    """
    Recieves a message through the TCP socket and verifies its checksum.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        print(f'Listening at {host}:{port}')
        s.bind((host, port))
        s.listen()
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                data = conn.recv(1024)
                print(f'recieved: {data}')
                if not data or data == b'quit':
                    break
                conn.sendall(data)
            
    
if __name__ == '__main__':
    ip = 'localhost'
    port = 65432
    recieve_tcp_message(ip,port)
