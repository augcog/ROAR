from ROAR_Desktop.ROAR_GUI.control.gui_utilities import BaseWindow, ConfigWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from ROAR_Desktop.ROAR_GUI.view.agent_config_panel import Ui_AgentConfigWindow
from ROAR.configurations.configuration import Configuration as AgentConfigModel
from ROAR_Desktop.ROAR_GUI.control.control_panel_control import ControlPanelWindow
from pathlib import Path


class AgentConfigWindow(ConfigWindow):
    def __init__(self,
                 app: QtWidgets.QApplication,
                 **kwargs):
        super().__init__(app,
                         UI=Ui_AgentConfigWindow,
                         config_json_file_path=kwargs["agent_config_json_file_path"],
                         ConfigModel=AgentConfigModel,
                         NextWindowClass=ControlPanelWindow, **kwargs)

