from PyQt6 import QtCore, QtGui, QtWidgets

from .core import SmartScanManager

class SmartScanMainWindow(QtCore.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SmartScan")
        self.setWindowIcon(QtGui.QIcon("icons/logo256.png"))
        self.resize(800, 600)
        self.move(300, 300)

        self.scan_manager, self.scan_manager_thread = self.init_scan_manager()

    def init_scan_manager(self) -> SmartScanManager:
        """ init scan manager """
        manager = SmartScanManager()
        manager.new_raw_data.connect(self.on_raw_data)
        manager.new_processed_data.connect(self.on_reduced_data)
        manager.error.connect(self.on_thread_error)
        manager_thread = QtCore.QThread()
        manager.moveToThread(manager_thread)
        manager_thread.start()
        return manager, manager_thread


