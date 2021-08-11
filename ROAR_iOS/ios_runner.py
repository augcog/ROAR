from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import Transform
from ROAR_iOS.config_model import iOSConfig
from Bridges.ios_bridge import iOSBridge
import logging
import pygame
from typing import Optional, Tuple
from ROAR_iOS.manual_control import ManualControl
from ROAR_iOS.depth_cam_streamer import DepthCamStreamer
from ROAR_iOS.rgb_camera_streamer import RGBCamStreamer
from ROAR_iOS.transform_streamer import TransformStreamer
from ROAR_iOS.control_streamer import ControlStreamer
import numpy as np
import cv2
import qrcode
from ROAR.utilities_module.utilities import get_ip
import time


class iOSRunner:
    def __init__(self, agent: Agent, ios_config: iOSConfig):
        self.agent = agent
        self.ios_config = ios_config
        self.ios_bridge = iOSBridge()
        self.pygame_display_width = 800
        self.pygame_display_height = 640
        self.logger = logging.getLogger("iOS Runner")
        self.display: Optional[pygame.display] = None

        self.controller = ManualControl(max_throttle=self.ios_config.max_throttle,
                                        max_steering=self.ios_config.max_steering)

        self.setup_pygame()
        self.world_cam_streamer = RGBCamStreamer(host=self.ios_config.ios_ip_addr,
                                                 port=self.ios_config.ios_port,
                                                 name=self.ios_config.world_cam_route_name,
                                                 resize=(self.pygame_display_height,
                                                         self.pygame_display_width)
                                                 )
        self.depth_cam_streamer = DepthCamStreamer(host=self.ios_config.ios_ip_addr,
                                                   port=self.ios_config.ios_port,
                                                   name=self.ios_config.depth_cam_route_name,
                                                   threaded=True,
                                                   update_interval=0.1,
                                                   )
        self.transform_streamer = TransformStreamer(host=self.ios_config.ios_ip_addr,
                                                    port=self.ios_config.ios_port,
                                                    name=self.ios_config.transform_route_name)
        self.control_streamer = ControlStreamer(host=self.ios_config.ios_ip_addr,
                                                port=self.ios_config.ios_port,
                                                name=self.ios_config.control_route_name)

        self.last_control_time = time.time()
        self.logger.info("iOS Runner Initialized")

    def auto_configure_screen_size(self, mode_idx: Optional[int] = 0):
        modes = pygame.display.list_modes()  # big -> small
        self.pygame_display_width, self.pygame_display_height = 1080, 640

    def setup_pygame(self):
        """
        Initiate pygame
        Returns:

        """
        pygame.init()
        pygame.font.init()
        self.auto_configure_screen_size(mode_idx=-1)
        self.display = pygame.display.set_mode((self.pygame_display_width,
                                                self.pygame_display_height))
        self.logger.debug("PyGame initiated")

    def show_qr_code(self):
        cv2.imshow("qr code", np.array(qrcode.make(f"{get_ip()}").convert('RGB')))
        cv2.waitKey(1000)
        cv2.destroyWindow("qr code")

    def start_game_loop(self, auto_pilot=False):
        self.logger.info("Starting Game loop")
        self.agent.add_threaded_module(self.transform_streamer)
        if self.ios_config.ar_mode is False:
            self.agent.add_threaded_module(self.depth_cam_streamer)
        self.agent.add_threaded_module(self.world_cam_streamer)
        try:
            self.agent.start_module_threads()

            clock = pygame.time.Clock()
            should_continue = True
            while should_continue:
                clock.tick_busy_loop(60)
                should_continue, control = self.update_pygame(clock=clock)
                sensor_data, vehicle = self.convert_data()
                agent_control = self.agent.run_step(vehicle=vehicle,
                                                    sensors_data=sensor_data)
                if auto_pilot:
                    control = self.ios_bridge.convert_control_from_agent_to_source(agent_control)

                control.throttle = np.clip(control.throttle, -self.ios_config.max_throttle, self.ios_config.max_throttle)
                control.steering = np.clip(control.steering, -self.ios_config.max_steering,
                                           self.ios_config.max_steering)
                self.control_streamer.send(control)

        except Exception as e:
            self.logger.error(f"Something bad happend {e}")
        finally:
            self.on_finish()

    def convert_data(self):
        try:
            if self.ios_config.ar_mode and self.world_cam_streamer.curr_image is not None:
                frame = self.world_cam_streamer.curr_image
            else:
                frame = cv2.rotate(self.world_cam_streamer.curr_image, cv2.ROTATE_90_CLOCKWISE)
            sensor_data: SensorsData = \
                self.ios_bridge.convert_sensor_data_from_source_to_agent(
                    {
                        "front_rgb": frame,
                        "front_depth": self.depth_cam_streamer.curr_image,
                    }
                )
            vehicle = self.ios_bridge.convert_vehicle_from_source_to_agent(
                {
                    "transform": self.transform_streamer.transform,
                }
            )
            current_time = time.time()
            diff = current_time - self.last_control_time
            vehicle.velocity.x = (self.agent.vehicle.transform.location.x - vehicle.transform.location.x) / diff
            vehicle.velocity.y = (self.agent.vehicle.transform.location.y - vehicle.transform.location.y) / diff
            vehicle.velocity.z = (self.agent.vehicle.transform.location.z - vehicle.transform.location.z) / diff
            self.last_control_time = current_time
            return sensor_data, vehicle
        except Exception as e:
            self.logger.error(f"Cannot convert data: {e}")
            return SensorsData(), Vehicle()

    def on_finish(self):
        self.logger.info("Finishing...")
        self.control_streamer.send(VehicleControl())
        self.agent.shutdown_module_threads()

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
            frame = cv2.flip(cv2.rotate(self.agent.front_rgb_camera.data.copy(), cv2.ROTATE_90_COUNTERCLOCKWISE), 0)
            reshaped = cv2.resize(frame, (self.pygame_display_height, self.pygame_display_width))
            data = cv2.cvtColor(reshaped, cv2.COLOR_BGR2RGB)
            pygame.surfarray.blit_array(self.display, data)
        pygame.display.flip()
        return self.controller.parse_events(clock=clock)
