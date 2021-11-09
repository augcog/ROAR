import logging
from pathlib import Path
import sys, os
from PyQt5 import QtWidgets
from ROAR_Desktop.ROAR_GUI.control.main_menu_panel_control import MainMenuWindow


class GUILauncher:
    def __init__(self, sim_config_json_file_path: Path, jetson_config_json_file_path: Path, debug=False):
        self.logger = logging.getLogger(__name__)
        self.sim_config_json_file_path = sim_config_json_file_path
        self.jetson_config_json_file_path = jetson_config_json_file_path
        if debug:
            logging.basicConfig(level=logging.DEBUG)
        logging.basicConfig(level=logging.INFO)
        logging.basicConfig(level=logging.WARNING)
        logging.basicConfig(level=logging.ERROR)
        logging.basicConfig(level=logging.CRITICAL)

    def run(self, add_debug_path: bool = False):
        if add_debug_path:
            sys.path.append(Path(os.getcwd()).parent.parent.as_posix())

        app = QtWidgets.QApplication(sys.argv)
        _ = MainMenuWindow(app, sim_config_json_file_path=self.sim_config_json_file_path,
                           jetson_config_json_file_path=self.jetson_config_json_file_path)
        app.exec()
