from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np

from smarstcan.gui.core import SmartScanManager

class SmartScanMainWindow(QtCore.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SmartScan")
        self.setWindowIcon(QtGui.QIcon("icons/logo256.png"))
        self.resize(800, 600)
        self.move(300, 300)

        self.scan_manager, self.scan_manager_thread = self.init_scan_manager()
        self.plot_widget = self.init_plot_widget()

    def init_scan_manager(self) -> SmartScanManager:
        """ init scan manager """
        manager = SmartScanManager()
        manager.new_raw_data.connect(self.on_raw_data)
        manager.new_reduced_data.connect(self.on_reduced_data)
        manager.status.connect(self.statusBar().showMessage)
        manager.new_hyperparameters.connect(self.on_new_hyperparameters)
        manager.new_points.connect(self.on_new_points)
        manager.finished.connect(self.on_finished)

        manager.error.connect(self.on_thread_error)
        manager_thread = QtCore.QThread()
        manager.moveToThread(manager_thread)
        manager_thread.start()
        return manager, manager_thread
    
    @QtCore.pyqtSlot(str)
    def on_thread_error(self, error: str) -> None:
        """Handle errors from the scan manager thread."""
        self.statusBar().showMessage(error)

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def on_raw_data(self, pos: np.ndarray, data: np.ndarray) -> None:
        """Handle new raw data from the scan manager thread."""
        self.statusBar().showMessage(f"Received raw data: {pos.shape}, {data.shape}")

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def on_reduced_data(self, pos: np.ndarray, data: np.ndarray) -> None:
        """Handle new reduced data from the scan manager thread."""
        self.statusBar().showMessage(f"Received reduced data: {pos.shape}, {data.shape}")

    @QtCore.pyqtSlot(dict)
    def on_new_hyperparameters(self, hyperparameters: dict) -> None:
        """Handle new hyperparameters from the scan manager thread."""
        self.statusBar().showMessage(f"Received new hyperparameters: {hyperparameters}")

    @QtCore.pyqtSlot(np.ndarray)
    def on_new_points(self, points: np.ndarray) -> None:
        """Handle new points from the scan manager thread."""
        self.statusBar().showMessage(f"Received new points: {points.shape}")

    @QtCore.pyqtSlot()
    def on_finished(self) -> None:
        """Handle finished signal from the scan manager thread."""
        self.statusBar().showMessage(f"Scan finished.")

    def init_plot_widget(self) -> QtWidgets.QWidget:
        """ init plot widget """
        plot_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(plot_widget)
        return plot_widget

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle close event."""
        self.scan_manager_thread.quit()
        self.scan_manager_thread.wait()
        event.accept()


class SmartScanApp(QtWidgets.QApplication):
    def __init__(self, argv: list) -> None:
        super().__init__(argv)
        self.main_window = SmartScanMainWindow()
        self.main_window.show()

if __name__ == "__main__":
    import sys

    app = SmartScanApp(sys.argv)
    sys.exit(app.exec_())


