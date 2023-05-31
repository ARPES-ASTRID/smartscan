

class SGM4Scanner:

    def __init__(
            self,
            ip: str,
            port: int,
            verbose: bool = True,
            timeout: float = 1.0,
            buffer_size: int = 1024,
            ) -> None:
        self.commander = SGM4Commander(ip, port, verbose, timeout, buffer_size)
        self.verbose = verbose
        self.file = 