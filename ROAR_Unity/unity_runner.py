from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import Transform
from ROAR_iOS.config_model import iOSConfig
from Bridges.ios_bridge import iOSBridge
import logging
import pygame
from typing import Optional, Tuple
from ROAR_Unity.unity_server import UnityServer
from ROAR_iOS.depth_cam_streamer import DepthCamStreamer
from ROAR_iOS.rgb_camera_streamer import RGBCamStreamer
from ROAR_iOS.veh_state_streamer import VehicleStateStreamer
from ROAR_iOS.control_streamer import ControlStreamer
import numpy as np
import cv2
import time


class iOSUnityRunner:
    def __init__(self, agent: Agent, ios_config: iOSConfig):
        self.agent = agent
        self.ios_config = ios_config
        self.ios_bridge = iOSBridge()
        self.pygame_display_width = self.ios_config.pygame_display_width
        self.pygame_display_height = self.ios_config.pygame_display_height
        self.logger = logging.getLogger("iOS Runner")
        self.unity_server = UnityServer(host="127.0.0.1", port=8009)
        self.world_cam_streamer = RGBCamStreamer(pc_port=8001,
                                                 name="world_rgb_streamer",
                                                 update_interval=0.01,
                                                 threaded=True)
        self.depth_cam_streamer = DepthCamStreamer(name=self.ios_config.depth_cam_route_name,
                                                   threaded=True,
                                                   update_interval=0.01,
                                                   pc_port=8002
                                                   )
        self.veh_state_streamer = VehicleStateStreamer(pc_port=8003,
                                                       threaded=True,
                                                       name="vehicle_state_streamer",
                                                       update_interval=0.01)
        self.control_streamer = ControlStreamer(pc_port=8004,
                                                threaded=False,
                                                name="control_streamer")

        self.front_cam_display_size: Tuple[int, int] = (100, 480)
        self.front_cam_offsets = (self.pygame_display_width // 2 - self.front_cam_display_size[1] // 2, 0)

        s = (self.pygame_display_height, self.pygame_display_width)
        self.green_overlay_pts = [
            np.array([[0, s[0]],  # lower left
                      [30 * s[1] // 100, 60 * s[0] // 100],  # upper left
                      [70 * s[1] // 100, 60 * s[0] // 100],  # upper right
                      [s[1], s[0]]  # lower right
                      ],
                     np.int32).reshape((-1, 1, 2))
        ]
        self.yellow_overlay_pts = [
            np.array([[0, s[0]],
                      [20 * s[1] // 100, 74 * s[0] // 100],  # upper left
                      [80 * s[1] // 100, 74 * s[0] // 100],  # upper right
                      [s[1], s[0]]
                      ],
                     np.int32).reshape((-1, 1, 2))
        ]
        self.red_overlay_pts = [
            np.array([[0, s[0]],  # lower left
                      [10 * s[1] // 100, 88 * s[0] // 100],  # upper left
                      [90 * s[1] // 100, 88 * s[0] // 100],  # upper right
                      [s[1], s[0]]  # lower right
                      ],
                     np.int32)
        ]

        self.last_control_time = time.time()

        # smoothen control
        # TODO optimize this smoothening
        self.should_smoothen_control = False
        self.prev_control = VehicleControl()
        self.steering_smoothen_factor_forward = 100
        self.steering_smoothen_factor_backward = 10
        self.throttle_smoothen_factor = 100

        self.logger.info("iOS Runner Initialized")

    def start_game_loop(self, auto_pilot=False):
        self.logger.info("Starting Game loop")
        self.control_streamer.connect()
        if self.ios_config.ar_mode:
            self.world_cam_streamer.connect()
            self.agent.add_threaded_module(self.world_cam_streamer)
        else:
            self.world_cam_streamer.connect()
            self.depth_cam_streamer.connect()
            self.veh_state_streamer.connect()
            self.agent.add_threaded_module(self.world_cam_streamer)
            self.agent.add_threaded_module(self.depth_cam_streamer)
            self.agent.add_threaded_module(self.veh_state_streamer)
        try:
            self.unity_server.startAsync()
            self.agent.start_module_threads()

            clock = pygame.time.Clock()
            should_continue = True
            while should_continue:
                clock.tick_busy_loop(60)
                sensor_data, vehicle = self.convert_data()
                agent_control = self.agent.run_step(vehicle=vehicle,
                                                    sensors_data=sensor_data)

                control = self.unity_server.get_control()
                if auto_pilot:
                    control = self.ios_bridge.convert_control_from_agent_to_source(agent_control)
                control.throttle = np.clip(control.throttle, -self.ios_config.max_throttle,
                                           self.ios_config.max_throttle)

                control.steering = np.clip(control.steering + self.ios_config.steering_offset,
                                           -self.ios_config.max_steering,
                                           self.ios_config.max_steering)
                if self.should_smoothen_control:
                    self.smoothen_control(control)
                if self.ios_config.invert_steering:
                    control.steering = -1 * control.steering

                if control.brake:
                    control.throttle = -0.1
                self.control_streamer.send(control)
                if self.agent.front_rgb_camera.data is not None:
                    self.unity_server.update_frame(self.agent.front_rgb_camera.data.copy())

        except Exception as e:
            self.logger.error(f"Something bad happend {e}")
        finally:
            self.on_finish()

    def smoothen_control(self, control: VehicleControl):
        if abs(control.throttle) > abs(self.prev_control.throttle) and self.prev_control.throttle > 0.15:
            # ensure slower increase, faster decrease. 0.15 barely drives the car
            control.throttle = (self.prev_control.throttle * self.throttle_smoothen_factor + control.throttle) / \
                               (self.throttle_smoothen_factor + 1)
        if abs(control.steering) < abs(self.prev_control.steering):
            # slowly turn back
            control.steering = (
                                           self.prev_control.steering * self.steering_smoothen_factor_backward + control.steering) / \
                               (self.steering_smoothen_factor_backward + 1)
        elif abs(control.steering) < abs(self.prev_control.steering):
            control.steering = (self.prev_control.steering * self.steering_smoothen_factor_forward + control.steering) / \
                               (self.steering_smoothen_factor_backward + 1)

        self.prev_control = control
        return control

    def convert_data(self):
        try:
            if self.ios_config.ar_mode and self.world_cam_streamer.curr_image is not None:
                front_rgb = cv2.rotate(self.world_cam_streamer.curr_image, cv2.ROTATE_90_CLOCKWISE)
            else:
                front_rgb = cv2.rotate(self.world_cam_streamer.curr_image, cv2.ROTATE_90_CLOCKWISE)

            sensor_data: SensorsData = \
                self.ios_bridge.convert_sensor_data_from_source_to_agent(
                    {
                        "front_rgb": front_rgb,
                        "front_depth": self.depth_cam_streamer.curr_image,
                    }
                )
            vehicle = self.ios_bridge.convert_vehicle_from_source_to_agent(
                {
                    "transform": self.veh_state_streamer.transform,
                    "velocity": self.veh_state_streamer.velocity
                }
            )
            vehicle.control = self.control_streamer.control_tx

            return sensor_data, vehicle
        except Exception as e:
            self.logger.error(f"Cannot convert data: {e}")
            return SensorsData(), Vehicle()

    def on_finish(self):
        self.logger.info("Finishing...")
        for i in range(20):
            self.control_streamer.send(VehicleControl())
        self.agent.shutdown_module_threads()
        self.unity_server.shutdown()
