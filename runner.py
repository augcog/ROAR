from ROAR_Desktop.ROAR_GUI.gui_launcher import GUILauncher
import logging
from pathlib import Path
import os

if __name__ == "__main__":
    logging.basicConfig(format='[%(asctime)s] - [%(levelname)s] - [%(name)s] '
                               '- %(message)s',
                        level=logging.DEBUG)
    launcher = GUILauncher(debug=True,
                           sim_config_json_file_path=Path("./ROAR_Sim/configurations/configuration.json"),
                           jetson_config_json_file_path=Path("./ROAR_Jetson/configurations/configuration.json"))
    launcher.run()
