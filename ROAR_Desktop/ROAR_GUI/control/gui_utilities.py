from PyQt5 import QtWidgets  # type: ignore
from PyQt5.QtWidgets import QApplication  # type: ignore
from pathlib import Path
import logging
from abc import abstractmethod
from pydantic import BaseModel
from PyQt5 import QtGui, QtCore
from typing import Tuple
from typing import Dict, Union, Optional
import json
import os


class BaseWindow(QtWidgets.QMainWindow):
    def __init__(
            self,
            app: QApplication,
            UI,
            show=True,
            **kwargs
    ):
        """
        Args:
            app: QApplication
            UI: a callable that represents a class. ex: Ui_MainWindow in view/mainwindow_ui.py
        """
        super().__init__()
        self.logger = logging.getLogger("GUI")
        self.kwargs = kwargs
        self.app = app
        self.ui = UI()
        self.dialogs = list()
        try:
            self.ui.setupUi(self)
        except AttributeError:
            raise AttributeError(
                "Given UI {} does not have setupUi function. Please see documentation".format(
                    UI,
                ),
            )
        self.set_listener()
        if show:
            self.show()

    @abstractmethod
    def set_listener(self):
        self.ui.actionQuit.triggered.connect(self.close)

    def auto_wire_window(self, target_window):
        target_app = target_window(self.app, **self.kwargs)
        self.dialogs.append(target_app)
        target_app.show()
        self.hide()
        target_app.show()
        target_app.closeEvent = self.app_close_event

    # rewires annotation_app's closing event
    def app_close_event(self, close_event):
        self.show()


class ConfigWindow(BaseWindow):
    def __init__(self, app: QtWidgets.QApplication, UI,
                 config_json_file_path: Path, ConfigModel,
                 NextWindowClass, **kwargs):
        super(ConfigWindow, self).__init__(app, UI, **kwargs)
        self.NextWindowClass = NextWindowClass
        self.config = ConfigModel()
        self.config_json_file_path = config_json_file_path
        self.fill_config_list()

    def fill_config_list(self):
        model_info: Dict[str, Union[str, int, float, bool]] = dict()
        self.config.parse_file(self.config_json_file_path)
        for key_name, entry in self.config.dict().items():
            if type(entry) in [str, int, float, bool]:
                model_info[key_name] = entry

        for name, entry in model_info.items():
            self.add_entry_to_settings_gui(name=name,
                                           value=entry)

    def update_config_model(self):
        curr_config = self.config.dict()
        for key_name in curr_config.keys():
            line_edit: Optional[QtWidgets.QLineEdit] = self.findChild(QtWidgets.QLineEdit, key_name)
            if line_edit is not None:
                curr_config[key_name] = line_edit.text()
        self.config = self.config.parse_obj(curr_config)

    def add_entry_to_settings_gui(self, name: str, value: Union[str, int, float, bool]):
        input_field = QtWidgets.QLineEdit()
        input_field.setText(str(value))
        input_field.setObjectName(name)  # set the object name so that we can find it later
        if "path" not in name:
            label = QtWidgets.QLabel()
            label.setText(name)
            self.ui.formLayout.addRow(label, input_field)
        else:
            tool_button = QtWidgets.QToolButton()
            tool_button.setText(name)
            tool_button.clicked.connect(lambda: self.onToolButtonClicked(name))  # use call back
            self.ui.formLayout.addRow(tool_button, input_field)

    def onToolButtonClicked(self, name):
        line_edit: Optional[QtWidgets.QLineEdit] = self.findChild(QtWidgets.QLineEdit, name)
        path = Path(line_edit.text())
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(parent=self, caption='Open file',
                                                         directory=path.parent.as_posix())
        if fname:
            if Path(fname).is_file():
                line_edit.setText(fname)

    def set_listener(self):
        super(ConfigWindow, self).set_listener()
        self.ui.pushButton_confirm.clicked.connect(self.pushButton_confirm)
        self.ui.actionSave.triggered.connect(self.action_save)

    def action_save(self):
        self.update_config_model()
        simulation_config_file = self.config_json_file_path.open('w')
        content = json.dumps(self.config.dict(), indent=2)
        simulation_config_file.write(content)
        simulation_config_file.close()
        self.logger.info(f"Configuration saved to {self.config_json_file_path}")

    def pushButton_confirm(self):
        self.update_config_model()
        self.auto_wire_window(self.NextWindowClass)


class KeyboardControl:
    def __init__(self, throttle_increment=0.1, steering_increment=0.1):
        self.logger = logging.getLogger(__name__)
        self._steering_increment = steering_increment
        self._throttle_increment = throttle_increment
        self.steering = 0.0
        self.throttle = 0.0
        self.logger.debug("Keyboard Control Initiated")

    def parse_events(self, event: QtGui.QKeyEvent) -> Tuple[bool, float, float]:
        """
        parse a keystoke event
        Args:
            event: Qt Keypress event

        Returns:
            Tuple bool, throttle, steering
            boolean states whether quit is pressed. VehicleControl by default has throttle = 0, steering =
        """
        key_pressed = event.key()
        if key_pressed == QtCore.Qt.Key_Q:
            return False, 0, 0
        else:
            self._parse_vehicle_keys(key_pressed)
            return True, self.throttle, self.steering

    def _parse_vehicle_keys(self, key: int) -> Tuple[float, float]:
        """
        Parse a single key press and set the throttle & steering
        Args:
            keys: array of keys pressed. If pressed keys[PRESSED] = 1

        Returns:
            None
        """
        if key == QtCore.Qt.Key_W:
            self.throttle = min(self.throttle + self._throttle_increment, 1)

        elif key == QtCore.Qt.Key_S:
            self.throttle = max(self.throttle - self._throttle_increment, -1)

        if key == QtCore.Qt.Key_A:
            self.steering = max(self.steering - self._steering_increment, -1)

        elif key == QtCore.Qt.Key_D:
            self.steering = min(self.steering + self._steering_increment, 1)

        self.throttle, self.steering = round(self.throttle, 5), round(self.steering, 5)
        self.logger.info((self.throttle, self.steering))
        return self.throttle, self.steering
