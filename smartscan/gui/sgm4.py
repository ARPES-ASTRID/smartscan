import logging
import time

import numpy as np
from PyQt5 import QtCore

from smartscan.sgm4commands import SGM4Commands


class DataFetcher(QtCore.QObject):

    new_data = QtCore.pyqtSignal(np.ndarray, np.ndarray)
    status = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)
    
    def __init__(self, settings) -> None:
        super().__init__()
        self.logger = logging.getLogger("DataFetcher")
        self.logger.debug("Created DataFetcher")

        self.settings = settings

        self.host = self.settings['TCP']['host']
        self.port = self.settings['TCP']['port']
        self.buffer_size = self.settings['TCP'].get('buffer_size',1024*1024*8)
        self.timeout = self.settings['TCP'].get('timeout',1.0)
        self.checksum = self.settings['TCP'].get('checksum',False)
        self.CLRF = self.settings['TCP'].get('CLRF',True)

        self.sgm4 = SGM4Commands(
            host=self.host,
            port=self.port,
            buffer_size=self.buffer_size,
            timeout=self.timeout,
            checksum=self.checksum,
        )

        self.logger.debug(f"TCP settings: {self.settings}")
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(self.settings['core'].get('fetch_data_clock',50)) # in ms
        self.timer.timeout.connect(self.update)

    @QtCore.pyqtSlot()
    def update(self) -> None:
        # self.fetch_data()
        pass

    @QtCore.pyqtSlot()
    def fetch_data(self) -> None:
        """Fetch data from SGM4."""
        self.logger.debug("Fetching data...")
        try:
            pos, data = self.sgm4.MEASURE()
            if isinstance(pos, str):
                err = pos.split(' ')[0]
                if err == 'NO_DATA':
                    self.logger.debug(f"Received no data: {err}")
                else:
                    self.logger.error(f"Error fetching data: {err}")
                    self.error.emit(err)
            else:    
                self.logger.info(f"Received data: {pos}, {data.shape}")
                self.new_data.emit(pos,data)
        except Exception as e:
            self.logger.error(f"{type(e)} while fetching data: {e}")
            self.error.emit(str(e))

    def run(self) -> None:
        """Run the data fetcher."""
        self.logger.debug("Starting data fetcher.")
        self.sgm4.START()
        self.logger.info("Waiting 2 seconds for SMG4 intialization...")
        time.sleep(2)
        self.logger.debug("Starting timer")
        self.timer.start()
        self.status.emit("Running")

    def stop(self) -> None:
        """Stop the data fetcher."""
        self.logger.debug("Stopping data fetcher.")
        self.timer.stop()
        self.sgm4.END()
        self.finished.emit()
        self.status.emit("Stopped")

    def __del__(self) -> None:
        """Delete the data fetcher."""
        self.logger.debug("Deleting data fetcher.")
        try:
            self.stop()
            self.timer.deleteLater()
        except AttributeError:
            pass
        self.finished.emit()
        self.logger.debug("Data fetcher deleted.")