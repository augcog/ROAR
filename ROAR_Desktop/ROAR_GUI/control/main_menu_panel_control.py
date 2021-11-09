from ROAR_Desktop.ROAR_GUI.view.main_menu_panel import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from ROAR_Desktop.ROAR_GUI.control.gui_utilities import BaseWindow
from ROAR_Desktop.ROAR_GUI.control.jetson_config_panel_control import JetsonConfigWindow
from ROAR_Desktop.ROAR_GUI.control.simulation_config_panel_control import SimConfigWindow
from pathlib import Path


class MainMenuWindow(BaseWindow):
    def __init__(
            self, app: QtWidgets.QApplication, sim_config_json_file_path: Path, jetson_config_json_file_path: Path,
            **kwargs,
    ):
        super().__init__(
            app=app, UI=Ui_MainWindow, **kwargs,
        )
        self.kwargs = kwargs
        self.kwargs["sim_config_json_file_path"] = sim_config_json_file_path
        self.kwargs["jetson_config_json_file_path"] = jetson_config_json_file_path

    def set_listener(self):
        super(MainMenuWindow, self).set_listener()
        self.ui.pushbtn_simconfig.clicked.connect(self.btn_simconfig_clicked)
        self.ui.pushbtn_jetsonconfig.clicked.connect(self.btn_jetsonconfig_clicked)

    def btn_simconfig_clicked(self):
        self.auto_wire_window(target_window=SimConfigWindow)

    def btn_jetsonconfig_clicked(self):
        self.auto_wire_window(target_window=JetsonConfigWindow)
