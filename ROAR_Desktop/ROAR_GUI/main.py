from PyQt5 import QtWidgets
import sys
from pathlib import Path
import os
import logging

try:
    from .gui_launcher import GUILauncher
except:
    from gui_launcher import GUILauncher

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - [%(name)s] '
                               '- %(message)s',
                        level=logging.DEBUG)
    launcher = GUILauncher(debug=True,
                           sim_config_json_file_path=Path(os.getcwd()).parent.parent /
                                                     "ROAR_Sim" / "configurations" / "configuration.json",
                           jetson_config_json_file_path=Path(os.getcwd()).parent.parent / "ROAR_Jetson" /
                                                        "configurations" / "configuration.json")
    launcher.run(add_debug_path=True)
