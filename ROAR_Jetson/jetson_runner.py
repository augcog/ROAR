from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR_Jetson.jetson_vehicle import Vehicle as JetsonVehicle
from typing import Optional, Tuple

from ROAR_Jetson.arduino_cmd_sender import ArduinoCommandSender

from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import Transform
from Bridges.jetson_bridge import JetsonBridge
import logging
import pygame
from ROAR_Jetson.jetson_keyboard_control import JetsonKeyboardControl
import numpy as np
from ROAR_Jetson.configurations.configuration import Configuration as JetsonConfig
from ROAR_Jetson.arduino_receiver import ArduinoCommandReceiver
import serial
import sys
from pathlib import Path
import cv2
from ROAR_Jetson.camera_d_t import RealsenseD435iAndT265 as D435AndT265


class JetsonRunner:
    """
    In charge of maintaining the state of the game.
    Drive command for agent to move next step
    Drive command for jetson to issue next command
    Update PyGame visualizations and controls parsing
    """

    def __init__(self, agent: Agent, jetson_config: JetsonConfig):
        self.jetson_vehicle: JetsonVehicle = JetsonVehicle()
        self.jetson_config = jetson_config
        self.agent = agent
        self.jetson_bridge = JetsonBridge()
        self.logger = logging.getLogger("Jetson Runner")
        self.display: Optional[pygame.display] = None
        self.serial: Optional[serial.Serial] = None
        self.transform = Transform()
        self.controller = JetsonKeyboardControl()
        self.d435_and_t265: Optional[D435AndT265] = None

        if jetson_config.initiate_pygame:
            self.setup_pygame()
        self.setup_serial()
        self.setup_jetson_vehicle()
        self.jetson_vehicle.start_parts()

        self.auto_pilot = True
        self.pygame_initiated = False
        self.logger.info("Jetson Vehicle Connected and Initialized. All hardware running")

    def setup_pygame(self):
        """
        Initiate pygame
        Returns:

        """
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.jetson_config.pygame_display_width,
                                                self.jetson_config.pygame_display_height),
                                               pygame.HWSURFACE | pygame.DOUBLEBUF)
        self.pygame_initiated = True
        self.logger.debug("PyGame initiated")

    def setup_serial(self):
        try:
            if 'win' in sys.platform:
                self.serial = serial.Serial(port=self.jetson_config.win_serial_port,
                                            baudrate=self.jetson_config.baud_rate,
                                            timeout=self.jetson_config.arduino_timeout,  # read timeout
                                            writeTimeout=self.jetson_config.write_timeout)
            else:
                self.serial = serial.Serial(port=self.jetson_config.unix_serial_port,
                                            baudrate=self.jetson_config.baud_rate,
                                            timeout=self.jetson_config.arduino_timeout,
                                            writeTimeout=self.jetson_config.write_timeout)
        except Exception as e:
            self.logger.error(f"Unable to establish serial connection: {e}")

    def setup_jetson_vehicle(self):
        """
        Add component to JetsonVehicle instance
        Returns:
            None
        """
        if self.jetson_config.use_arduino:
            self._setup_arduino()
            print("arduino setted up")

        if self.jetson_config.use_t265 and self.jetson_config.use_t265:
            self._setup_d435i_and_t265()
            print("camera set up done")

    def _setup_arduino(self):
        self.arduino_command_sender = ArduinoCommandSender(serial=self.serial,
                                                           jetson_vehicle=self.jetson_vehicle,
                                                           servo_steering_range=[self.jetson_config.theta_min,
                                                                                 self.jetson_config.theta_max],
                                                           servo_throttle_range=[self.jetson_config.motor_min,
                                                                                 self.jetson_config.motor_max]
                                                           )
        self.arduino_command_receiver = ArduinoCommandReceiver(s=self.serial)
        self.jetson_vehicle.add(self.arduino_command_sender)
        self.jetson_vehicle.add(self.arduino_command_receiver)

    def _setup_d435i_and_t265(self):
        self.d435_and_t265 = D435AndT265()
        self.jetson_vehicle.add(self.d435_and_t265)
        self.agent.front_rgb_camera.intrinsics_matrix = self.d435_and_t265.rgb_camera_intrinsics
        self.agent.front_rgb_camera.distortion_coefficient = self.d435_and_t265.rgb_camera_distortion_coefficients
        self.agent.front_depth_camera.intrinsics_matrix = self.d435_and_t265.depth_camera_intrinsics
        self.agent.front_depth_camera.distortion_coefficient = self.d435_and_t265.depth_camera_distortion_coefficients

        self.logger.info("D435 and T265 cam set up complete")

    def start_game_loop(self, use_manual_control=False):
        self.logger.info("Starting Game Loop")
        try:
            self.agent.start_module_threads()

            clock = pygame.time.Clock()
            should_continue = True
            while should_continue:
                clock.tick_busy_loop(60)

                # pass throttle and steering into the bridge
                sensors_data, vehicle = self.convert_data()

                vehicle_control = VehicleControl()
                if self.auto_pilot:
                    vehicle_control = self.agent.run_step(sensors_data=sensors_data, vehicle=vehicle)

                should_continue, manual_vehicle_control = self.update_pygame(clock=clock)
                if use_manual_control:
                    vehicle_control = manual_vehicle_control

                self.jetson_vehicle.throttle = vehicle_control.throttle
                self.jetson_vehicle.steering = vehicle_control.steering
        except KeyboardInterrupt:
            self.logger.info("Keyboard Interrupt detected. Safely quitting")
        except Exception as e:
            self.logger.error(f"Something bad happened: [{e}]")
        finally:
            self.on_finish()

    def on_finish(self):
        self.jetson_vehicle.stop_parts()
        self.agent.shutdown_module_threads()
        exit(0)  # forced closure

    def convert_data(self) -> Tuple[SensorsData, Vehicle]:
        sensor_data: SensorsData = self.jetson_bridge.convert_sensor_data_from_source_to_agent(
            source={
                "front_rgb": self.d435_and_t265.color_image if self.d435_and_t265 is not None else None,
                "rear_rgb": None,
                "front_depth": self.d435_and_t265.depth_image if self.d435_and_t265 is not None else None,
                "imu_data": None,
                "location": self.d435_and_t265.location if self.d435_and_t265 is not None else [0, 0, 0],
                "rotation": self.d435_and_t265.rotation if self.d435_and_t265 is not None else [0, 0, 0],
                "velocity": self.d435_and_t265.velocity if self.d435_and_t265 is not None else [0, 0, 0]
            })
        vehicle: Vehicle = self.jetson_bridge.convert_vehicle_from_source_to_agent(
            source=self.jetson_vehicle
        )
        vehicle.transform = Transform(location=sensor_data.location, rotation=sensor_data.rotation)
        vehicle.velocity = sensor_data.velocity

        return sensor_data, vehicle

    def update_pygame(self, clock) -> Tuple[bool, VehicleControl]:
        """
        Update the pygame window, including parsing keypress
        Args:
            clock: pygame clock
        Returns:
            bool - whether to continue the game
            VehicleControl - the new VehicleControl cmd by the keyboard
        """
        if self.display is not None and self.agent.front_rgb_camera.data is not None:
            array: np.ndarray = self.agent.front_rgb_camera.data.copy()[:, :, ::-1]
            surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
            self.display.blit(surface, (0, 0))
        pygame.display.flip()
        return self.controller.parse_events(clock=clock)
