import logging

from PyQt5 import QtCore, QtGui, QtWidgets

import numpy as np
import yaml

from smartscan.gui.plot import MainPlotWidget

from .core import SmartScanManager, Settings


class SmartScanMainWindow(QtWidgets.QMainWindow):
    def __init__(self, settings) -> None:
        super().__init__()
        self.logger = logging.getLogger("SmartScanMainWindow")
        self.logger.debug("Created SmartScanMainWindow")

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('ready')

        self.setWindowTitle("SmartScan")
        self.setWindowIcon(QtGui.QIcon("icons/logo256.png"))
        self.resize(1600,1000)
        self.move(300, 300)

        # # set the cool dark theme and other plotting settings
        # self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())

        # pg.setConfigOption('background', (25, 35, 45))
        # pg.setConfigOption('foreground', 'w')
        # pg.setConfigOptions(antialias=True)

        self.settings = settings

        self.scan_manager, self.scan_manager_thread = self.init_scan_manager()
        self.plot_widget = self.init_plot_widget()

        self.create_menu()
        self.create_main_frame()
        self.create_status_bar()

        # self.update_ui()


    # Create the GUI interface
    def create_menu(self) -> None:
        self.file_menu = self.menuBar().addMenu("&File")

        quit_action = self.create_action("&Quit", slot=self.close, 
            shortcut="Ctrl+Q", tip="Close the application")

        self.add_actions(self.file_menu, 
            (quit_action,))

        self.help_menu = self.menuBar().addMenu("&Help")

        about_action = self.create_action("&About", 
            shortcut='F1', slot=self.on_about, tip='About the demo')

        self.add_actions(self.help_menu, (about_action,))

    def init_plot_widget(self) -> QtWidgets.QWidget:
        """ init plot widget """
        plot_widget = MainPlotWidget(parent=self, settings=self.settings)
        self.setCentralWidget(plot_widget)
        return plot_widget

    def create_main_frame(self) -> None:
        self.main_frame = QtWidgets.QWidget()
        self.main_frame.setFocus()
        self.setCentralWidget(self.main_frame)

        self.create_main_layout()

    def create_main_layout(self) -> None:
        self.main_layout = QtWidgets.QHBoxLayout()
        self.main_frame.setLayout(self.main_layout)

        self.create_control_frame()
        self.create_plot_frame()
    
    def create_control_frame(self) -> None:
        self.control_frame = QtWidgets.QFrame()
        self.control_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.control_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_layout.addWidget(self.control_frame)

        self.control_frame_layout = QtWidgets.QVBoxLayout()
        self.control_frame.setLayout(self.control_frame_layout)

        self.start_button = QtWidgets.QPushButton('Start')
        self.start_button.clicked.connect(self.on_start)
        self.control_frame_layout.addWidget(self.start_button)
        
        self.stop_button = QtWidgets.QPushButton('Stop')
        self.stop_button.clicked.connect(self.on_stop)
        self.control_frame_layout.addWidget(self.stop_button)

    def create_plot_frame(self) -> None:
        self.plot_frame = QtWidgets.QFrame()
        self.plot_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plot_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.main_layout.addWidget(self.plot_frame)

        self.plot_frame_layout = QtWidgets.QVBoxLayout()
        self.plot_frame.setLayout(self.plot_frame_layout)
        
        self.plot_widget = MainPlotWidget(parent=self, settings=self.settings)
        self.plot_frame_layout.addWidget(self.plot_widget)

    def create_status_bar(self) -> None:
        self.status_text = QtWidgets.QLabel("This is a demo")
        self.statusBar().addWidget(self.status_text, 1)

    def create_action(
            self, 
            text: str, 
            slot=None, 
            shortcut=None, 
            icon=None, 
            tip=None, 
            checkable=False, 
            # signal="triggered()"
        ) -> QtWidgets.QAction:
        action = QtWidgets.QAction(text, self)
        if icon is not None:
            action.setIcon(QtGui.QIcon(icon))
        if shortcut is not None:
            action.setShortcut(shortcut)
        if tip is not None:
            action.setToolTip(tip)
            action.setStatusTip(tip)
        if slot is not None:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
        return action
    
    def add_actions(
            self, 
            target: QtWidgets.QWidget, 
            actions: list
        ) -> None:
        for action in actions:
            if action is None:
                target.addSeparator()
            else:
                target.addAction(action)

    def on_about(self) -> None:
        msg = """ 
            This is a demo
            """
        QtWidgets.QMessageBox.about(self, "About the demo", msg.strip())

    # Initialize the scan manager

    @QtCore.pyqtSlot()
    def on_start(self) -> None:
        self.logger.debug('Starting scan')
        if self.scan_manager_thread is None:
            self.scan_manager, self.scan_manager_thread = self.init_scan_manager()
        if self.scan_manager_thread.isRunning():
            self.logger.warning('Scan manager thread is already running')
            return
        # self.init_scan_manager()
        self.scan_manager_thread.start()
        self.logger.info('Started scan')

    @QtCore.pyqtSlot()
    def on_stop(self) -> None:
        self.logger.debug('Stopping scan')
        self.scan_manager_thread.quit()
        self.scan_manager_thread.wait()
        self.scan_manager_thread.terminate()
        self.scan_manager = None
        self.scan_manager_thread = None
        self.logger.info('Stopped scan')

    def init_scan_manager(self) -> tuple[SmartScanManager, QtCore.QThread]:
        """ init scan manager """
        manager = SmartScanManager(settings=self.settings)
        manager.new_raw_data.connect(self.on_raw_data)
        manager.new_reduced_data.connect(self.on_reduced_data)
        manager.status.connect(self.statusBar().showMessage)
        manager.new_hyperparameters.connect(self.on_new_hyperparameters)
        manager.new_points.connect(self.on_new_points)
        manager.finished.connect(self.on_finished)

        manager.error.connect(self.on_thread_error)
        manager_thread = QtCore.QThread()
        manager.moveToThread(manager_thread)
        manager_thread.started.connect(manager.start)
        manager_thread.finished.connect(manager.stop)
        # manager_thread.start()
        return manager, manager_thread
    
    @QtCore.pyqtSlot(str)
    def on_thread_error(self, error: str) -> None:
        """Handle errors from the scan manager thread."""
        self.logger.error(error)
        self.statusBar().showMessage(error)

    @QtCore.pyqtSlot(dict)
    def on_raw_data(self, data_dict:dict) -> None:
        """Handle new raw data from the scan manager thread."""
        pos = data_dict['pos']
        data = data_dict['data']
        n = data_dict['data_counter']
        self.statusBar().showMessage(f"Received raw data #{n} | {pos} {data.shape}")

    @QtCore.pyqtSlot(dict)
    def on_reduced_data(self, data_dict:dict) -> None:
        """Handle new reduced data from the scan manager thread."""
        pos = data_dict['pos']
        data = data_dict['data']
        n = data_dict['data_counter']        
        self.statusBar().showMessage(f"Received processed data #{n} | {pos} {data.shape}")

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
        self.statusBar().showMessage("Scan finished.")

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """Handle close event."""
        if self.scan_manager_thread is not None:
            self.logger.info('Killing scan manager thread')
            self.scan_manager_thread.quit()
            self.scan_manager_thread.wait()
        event.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(event)

    def close(self) -> None:
        self.logger.info('Closing MainWindow')
        super().close()
        self.logger.info('Closed MainWindow')


class SmartScanApp(QtWidgets.QApplication):
    def __init__(self, argv: list, settings=None) -> None:
        super().__init__(argv)
        self.logger = logging.getLogger("SmartScanApp")
        self.settings = Settings(settings)
        self.logger.debug("init SmartScanApp")
        self.main_window = SmartScanMainWindow(settings=self.settings)
        self.main_window.show()


if __name__ == "__main__":
    import sys

    app = SmartScanApp(sys.argv)
    sys.exit(app.exec_())


