from ROAR_Desktop.ROAR_GUI.control.gui_utilities import ConfigWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from ROAR_Desktop.ROAR_GUI.view.jetson_config_panel import Ui_JetsonConfigWindow
from ROAR_Desktop.ROAR_GUI.control.agent_config_panel import AgentConfigWindow
from ROAR_Jetson.configurations.configuration import Configuration as JetsonConfigModel
from pprint import pprint
import json
from typing import Dict, Union
from pathlib import Path


class JetsonConfigWindow(ConfigWindow):
    def __init__(self, app, **kwargs):
        super(JetsonConfigWindow, self).__init__(app=app,
                                                 UI=Ui_JetsonConfigWindow,
                                                 config_json_file_path=kwargs["jetson_config_json_file_path"],
                                                 ConfigModel=JetsonConfigModel,
                                                 NextWindowClass=AgentConfigWindow,
                                                 **kwargs
                                                 )

    def pushButton_confirm(self):
        self.kwargs["agent_config_json_file_path"] = self.config.agent_config_path
        super(JetsonConfigWindow, self).pushButton_confirm()
