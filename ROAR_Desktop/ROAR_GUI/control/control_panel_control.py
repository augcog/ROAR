from ROAR_Desktop.ROAR_GUI.control.gui_utilities import BaseWindow, KeyboardControl
from PyQt5 import QtCore, QtGui, QtWidgets
from ROAR_Desktop.ROAR_GUI.view.control_panel import Ui_ControlPanelWindow
from typing import Optional
import cv2
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

cap = cv2.VideoCapture(0)


class ControlPanelWindow(BaseWindow):
    def __init__(self, app: QtWidgets.QApplication, **kwargs):
        super().__init__(app, Ui_ControlPanelWindow, **kwargs)
        self.disply_width, self.display_height = 640, 480
        self.front_camera_image_label: Optional[QtWidgets.QLabel] = None
        self.rear_camera_image_label: Optional[QtWidgets.QLabel] = None
        self.init_ui()
        self.front_camera_image_thread = Thread(self)
        self.front_camera_image_thread.changePixmap.connect(self.set_front_camera_image)
        self.keyboard_control = KeyboardControl()

        # vehicle control variables
        self.throttle = 0
        self.steering = 0

        # self.rear_camera_image_thread = Thread(self)
        # self.rear_camera_image_thread.changePixmap.connect(self.set_rear_camera_image)

        # self.start_poll_images()

    def init_ui(self):
        self.front_camera_image_label = QtWidgets.QLabel(self)
        self.rear_camera_image_label = QtWidgets.QLabel(self)
        self.ui.visualization_layout.addWidget(self.front_camera_image_label)
        # self.ui.visualization_layout.addWidget(self.rear_camera_image_label)

    def start_poll_images(self):
        self.front_camera_image_thread.start()
        # self.rear_camera_image_thread.start()

    def set_listener(self):
        super(ControlPanelWindow, self).set_listener()
        self.ui.stop_pushButton.clicked.connect(self.stop_pressed)
        self.ui.start_pushButton.clicked.connect(self.start_pressed)
        self.ui.configuration_pushButton.clicked.connect(self.configuration_pressed)

    def stop_pressed(self):
        self.logger.info("Stopping")

    def start_pressed(self):
        self.logger.info("Starting")
        # detect which mode (simulation/jetson) am i in

        # initialize the right runner

        # initialize thread that pull image from agent

        # update keyboard control

    def configuration_pressed(self):
        self.logger.error("Configuration window not implemented yet")

    @pyqtSlot(QImage)
    def set_front_camera_image(self, image):
        self.front_camera_image_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(QImage)
    def set_rear_camera_image(self, image):
        self.rear_camera_image_label.setPixmap(QPixmap.fromImage(image))

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.front_camera_image_thread.exit(returnCode=0)
        # self.rear_camera_image_thread.exit(returnCode=0)
        super(ControlPanelWindow, self).closeEvent(a0)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        self.keyboard_control.parse_events(event)
        self.update_controls(new_throttle=self.keyboard_control.throttle,
                             new_steering=self.keyboard_control.steering)

    def update_controls(self, new_throttle, new_steering):
        self.throttle = new_throttle
        self.steering = new_steering
        self.ui.steering_value.setValue(self.steering)
        self.ui.throttle_value.setValue(self.throttle)
        self.keyboard_control.throttle = new_throttle
        self.keyboard_control.steering = new_steering


class Thread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, mw):
        super(Thread, self).__init__()

    def run(self):
        global cap  # TODO hack as of now, needs to be changed
        while True:
            ret, frame = cap.read()
            if ret:
                # https://stackoverflow.com/a/55468544/6622587
                rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape
                bytesPerLine = ch * w
                convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
