""" plotting widgets and controls go here"""
from asyncio import Task
import logging

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget
import numpy as np
import pyqtgraph as pg

from smartscan.gui.core import SmartScanManager, Settings

class MainPlotWidget(QtWidgets.QWidget):

    def __init__(self, parent: QWidget | None, settings: Settings=None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger("MainPlotWidget")
        self.logger.debug("Initializing MainPlotWidget")
        # self.setGeometry(QtCore.QRect(100, 100, 1200, 800))
        self.setMinimumSize(800, 300)
        # stretch to fill the whole panel
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.settings = settings

        self.tasks = self.settings['tasks'].keys()
        self.task_panels: list[TaskPanel] = []

        self.init_ui()

        # self.timer = QtCore.QTimer()
        # self.timer.timeout.connect(self.update_plots)
        # self.timer.start(1000)

    @QtCore.pyqtSlot(dict)
    def on_new_plot_dict(self, data_dict:dict) -> None:
        """ updates the plots with new data
        
        Args:
            data_dict (dict): dictionary with data to plot
        """
        self.aqf_panel.update_plot(data_dict)
        for task, panel in zip(self.tasks, self.task_panels):
            panel.update_plots(data_dict[task])
        self.logger.info(f"Plots #{data_dict['data_counter']} updated")

    def init_ui(self) -> None:
        self.tasks_panel = QtWidgets.QWidget(self)
        self.tasks_panel_layout = QtWidgets.QVBoxLayout(self.tasks_panel)
        # self.tasks_panel.setGeometry(QtCore.QRect(0, 0, 800, 600))
        self.tasks_panel.setObjectName("tasks_panel")
        # self.tasks_panel.setStyleSheet("background-color: rgb(25, 35, 45);")
        self.tasks_panel.setContentsMargins(0, 0, 0, 0)
        for task in self.tasks:
            panel = TaskPanel(self.tasks_panel, task)
            self.tasks_panel_layout.addWidget(panel)
            self.task_panels.append(panel)        
        self.tasks_panel.setLayout(self.tasks_panel_layout)

        self.aqf_panel = AquisitionFunctionPanel(self)
        self.aqf_panel.setGeometry(QtCore.QRect(0, 600, 800, 200))
        self.aqf_panel.setObjectName("aqf_panel")
        self.aqf_panel.setStyleSheet("background-color: rgb(25, 35, 45);")

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.tasks_panel)
        self.layout.addWidget(self.aqf_panel)


class AquisitionFunctionPanel(QtWidgets.QGroupBox):
    """ creates a panel to view the aquisition function.

    The panel consists of a image area in a QBox and a label
    """

    def __init__(self, parent: QWidget | None) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger("AquisitionFunctionPanel")
        self.logger.debug("Initializing AquisitionFunctionPanel")
        # set minimum size
        self.setMinimumSize(800, 300)
        # stretch to fill the whole panel
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # set title
        self.setTitle("Aquisition Function")

        self.init_ui()

    def init_ui(self) -> None:
        self.setObjectName("aqf_panel")
        self.image =  pg.ImageView(
            self,
            view=pg.PlotItem(
                labels={'left': ('y', 'um'), 'bottom': ('x', 'um')},
                range=QtCore.QRectF(-50,-50,100,100),
            ),
            name="Aquisition Function"
        )
        # add color
        self.image.setColorMap(pg.ColorMap(
            [0, 0.5, 1],
            [(0, 0, 0), (255, 255, 255), (255, 0, 0)]
        ))
        self.image.ui.histogram.show()
        self.image.ui.roiBtn.show()
        self.image.ui.menuBtn.show()
        self.scatter = pg.ScatterPlotItem(
            pen=None,
            symbol='o',
            symbolSize=10,
            symbolBrush=(255, 0, 0, 255)
        )
        self.image.getView().addItem(self.scatter)

        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.image)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot(dict)
    def update_plot(self, data_dict:dict) -> None:
        """ updates the plots with new data
        
        Args:
            data_dict (dict): dictionary with data to plot
        """
        self.image.setImage(data_dict["acquisition_function"])
        # self.scatter.setData(
        #     data_dict["data"]["positions"][:, 0], 
        #     data_dict["data"]["positions"][:, 1]
        # )


class TaskPanel(QtWidgets.QGroupBox):
    """ creates a panel to view a task. 
    
    The panel consists of two plot areas in a QBox and a label
    """
    
    def __init__(self, parent: QWidget | None, task: str) -> None:
        super().__init__(parent)
        self.logger = logging.getLogger("TaskPanel")
        self.logger.debug("Initializing TaskPanel")
        # set minimum size
        self.setMinimumSize(800, 300)
        # stretch to fill the whole panel
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # set title
        self.setTitle

        self.task = task
        self.init_ui()

    def init_ui(self) -> None:
        self.setObjectName("task_panel")
        self.setTitle(self.task)

        self.mean_plot_widget = pg.ImageView(
            self, 
            view=pg.PlotItem(labels={'left': ('y', 'um'), 'bottom': ('x', 'um')}),
            name="Posterior Mean"
        )
        self.mean_plot_widget.ui.histogram.hide()
        self.mean_plot_widget.ui.roiBtn.hide()
        self.mean_plot_widget.ui.menuBtn.hide()
        self.var_plot_widget = pg.ImageView(
            self, 
            view=pg.PlotItem(labels={'left': ('y', 'um'), 'bottom': ('x', 'um')}),
            name="Posterior Variance"
        )
        self.var_plot_widget.ui.histogram.hide()
        self.var_plot_widget.ui.roiBtn.hide()
        self.var_plot_widget.ui.menuBtn.hide()

        # make the plot widgets stretch to fill the panel, each taking half of the space
        self.mean_plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self.var_plot_widget.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        # self.mean_plot_widget.setMinimumSize(400, 300)
        # self.var_plot_widget.setMinimumSize(400, 300)
        self.mean_plot_widget.ui.roiBtn.hide()
        self.mean_plot_widget.ui.menuBtn.hide()
        self.var_plot_widget.ui.roiBtn.hide()
        self.var_plot_widget.ui.menuBtn.hide()


        self.layout = QtWidgets.QHBoxLayout(self)
        self.layout.addWidget(self.mean_plot_widget)
        self.layout.addWidget(self.var_plot_widget)
        self.setLayout(self.layout)

    @QtCore.pyqtSlot(dict)
    def update_plots(self, data_dict:dict) -> None:
        """ updates the plots with new data
        
        Args:
            data_dict (dict): dictionary with data to plot
        """
        self.mean_plot_widget.setImage(data_dict["mean"])
        self.var_plot_widget.setImage(data_dict["variance"])



if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    settings = Settings('scan_settings.yaml')
    window = MainPlotWidget(None, settings)
    window.show()
    sys.exit(app.exec_())