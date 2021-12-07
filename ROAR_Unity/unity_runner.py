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
from ROAR_iOS.glove_controller import GloveControl
from ROAR_iOS.depth_cam_streamer import DepthCamStreamer
from ROAR_iOS.rgb_camera_streamer import RGBCamStreamer
from ROAR_iOS.control_streamer import ControlStreamer
import numpy as np
import cv2
import time
from ROAR_iOS.veh_state_streamer import VehicleStateStreamer
from ROAR_iOS.brake import Brake
from ROAR_Unity.unity_server2 import *


class iOSUnityRunner:
    def __init__(self, agent: Agent, ios_config: iOSConfig, is_unity=False):
        self.agent = agent
        self.is_auto = False
        self.ios_config = ios_config
        self.ios_bridge = iOSBridge()
        self.pygame_display_width = self.ios_config.pygame_display_width
        self.pygame_display_height = self.ios_config.pygame_display_height

        self.is_unity = is_unity
        # self.unity_server = UnityServer(host="127.0.0.1", port=8009)

        if self.is_unity:
            self.logger = logging.getLogger("iOS Unity Runner")
        else:
            self.logger = logging.getLogger("iOS Runner")
        self.display: Optional[pygame.display] = None
        self.should_display_system_status = self.ios_config.should_display_system_status
        self.controller = GloveControl(ios_config=ios_config) \
            if self.ios_config.should_use_glove else ManualControl(ios_config=ios_config)

        if self.is_unity is False:
            self.setup_pygame()
            self.unity_rgb_streamer = None
            self.unity_veh_state_streamer = None
            self.unity_control_streamer = None
        else:
            self.unity_rgb_streamer = UnityRGBServer(name="unity_rgb_server",
                                                     host="127.0.0.1",
                                                     port=8001,
                                                     threaded=True,
                                                     update_interval=0.01)
            self.unity_veh_state_streamer = UnityVehicleStateServer(name="unity_veh_state_server",
                                                                    host="127.0.0.1",
                                                                    port=8003,
                                                                    threaded=True,
                                                                    update_interval=0.01)
            self.unity_control_streamer = UnityControlServer(name="unity_control_server",
                                                             host="127.0.0.1",
                                                             port=8004,
                                                             threaded=True)

        self.world_cam_streamer = RGBCamStreamer(ios_address=self.ios_config.ios_ip_addr,
                                                 port=8001,
                                                 name="world_rgb_streamer",
                                                 update_interval=0.05,
                                                 threaded=True)
        self.depth_cam_streamer = DepthCamStreamer(ios_address=self.ios_config.ios_ip_addr,
                                                   name=self.ios_config.depth_cam_route_name,
                                                   threaded=True,
                                                   update_interval=0.05,
                                                   port=8002
                                                   )
        self.veh_state_streamer = VehicleStateStreamer(ios_address=self.ios_config.ios_ip_addr,
                                                       port=8003,
                                                       name="VehicleStateStreamer",
                                                       update_interval=0.05,
                                                       max_vel_buffer=5,
                                                       threaded=True)
        self.control_streamer = ControlStreamer(ios_address=self.ios_config.ios_ip_addr,
                                                port=8004,
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

        self.braker = Brake(kp=0.01, kd=0, ki=0, k_incline=0.015, max_brake=0.16)

        self.pitch_offset: Optional[float] = None

        if self.is_unity:
            self.logger.info("iOS Unity Runner Initialized")
        else:
            self.logger.info("iOS Runner Initialized")

    def setup_pygame(self):
        """
        Initiate pygame
        Returns:

        """
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode((self.pygame_display_width,
                                                self.pygame_display_height))
        self.logger.debug("PyGame initiated")

    def start_game_loop(self, auto_pilot=False):
        self.is_auto = auto_pilot
        self.logger.info("Starting Game loop")
        try:
            clock = pygame.time.Clock()
            should_continue = True

            if self.is_unity:
                self.agent.add_threaded_module(self.unity_rgb_streamer)
                # self.agent.add_threaded_module(self.unity_veh_state_streamer)
                # self.agent.add_threaded_module(self.unity_control_streamer)

            self.agent.add_threaded_module(self.world_cam_streamer)
            self.agent.add_threaded_module(self.depth_cam_streamer)
            self.agent.add_threaded_module(self.veh_state_streamer)
            self.agent.start_module_threads()

            is_manual_toggled = False
            while should_continue:
                clock.tick_busy_loop(60)
                if self.pitch_offset is None:
                    if -80 > self.agent.vehicle.transform.rotation.pitch > -100:
                        self.pitch_offset = -90 - self.agent.vehicle.transform.rotation.pitch
                    elif self.agent.vehicle.transform.rotation.pitch == 0:
                        pass  # this means that first data has not been received yet
                    else:
                        self.logger.error(f"Erroneous pitch reading: {self.agent.vehicle.transform.rotation.pitch}. "
                                          f"Please recalibrate your phone")
                else:
                    self.agent.vehicle.transform.rotation.pitch = self.agent.vehicle.transform.rotation.pitch + self.pitch_offset

                if self.is_unity:
                    # if unity, overwrite the control with what you get from unity.
                    control = self.unity_control_streamer.vehicle_control
                else:
                    should_continue, control, is_manual_toggled = self.update_pygame(clock=clock)

                if is_manual_toggled:
                    self.is_auto = False if self.is_auto else True
                sensor_data, vehicle = self.convert_data()

                agent_control = self.agent.run_step(vehicle=vehicle,
                                                    sensors_data=sensor_data)

                if self.is_auto:
                    # if autonomous mode, overwrite the previous controls with agent's control
                    control = self.ios_bridge.convert_control_from_agent_to_source(agent_control)

                # smooth and clip the steering and throttle
                if self.should_smoothen_control:
                    self.smoothen_control(control)
                if self.ios_config.invert_steering:
                    control.steering = -1 * control.steering
                if control.brake:
                    control = self.braker.run_step(control=control, vehicle=vehicle)
                control.throttle = np.clip(control.throttle, self.ios_config.max_reverse_throttle,
                                           self.ios_config.max_forward_throttle)
                control.steering = np.clip(control.steering + self.ios_config.steering_offset,
                                           -self.ios_config.max_steering,
                                           self.ios_config.max_steering)
                self.control_streamer.send(control)

                if self.agent.front_rgb_camera.data is not None and self.is_unity:
                    self.unity_rgb_streamer.update_image(self.world_cam_streamer.curr_image,
                                                         self.world_cam_streamer.intrinsics)
                    self.unity_veh_state_streamer.update_state(
                        x=self.veh_state_streamer.transform.location.x,
                        y=self.veh_state_streamer.transform.location.y,
                        z=self.veh_state_streamer.transform.location.z,
                        roll=self.veh_state_streamer.transform.rotation.roll,
                        pitch=self.veh_state_streamer.transform.rotation.pitch,
                        yaw=self.veh_state_streamer.transform.rotation.yaw,
                        vx=self.veh_state_streamer.velocity.x,
                        vy=self.veh_state_streamer.velocity.y,
                        vz=self.veh_state_streamer.velocity.z,
                        ax=self.veh_state_streamer.acceleration.x,
                        ay=self.veh_state_streamer.acceleration.y,
                        az=self.veh_state_streamer.acceleration.z,
                        gx=self.veh_state_streamer.gyro.x,
                        gy=self.veh_state_streamer.gyro.y,
                        gz=self.veh_state_streamer.gyro.z,
                        recv_time=self.veh_state_streamer.recv_time,
                    )
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
            rear_rgb = None
            if self.ios_config.ar_mode and self.world_cam_streamer.curr_image is not None:
                front_rgb = cv2.rotate(self.world_cam_streamer.curr_image, cv2.ROTATE_90_CLOCKWISE)
            else:
                front_rgb = cv2.rotate(self.world_cam_streamer.curr_image, cv2.ROTATE_90_CLOCKWISE)

            sensor_data: SensorsData = \
                self.ios_bridge.convert_sensor_data_from_source_to_agent(
                    {
                        "front_rgb": front_rgb,
                        "front_depth": self.depth_cam_streamer.curr_image,
                        "rear_rgb": rear_rgb
                    }
                )
            vehicle = self.ios_bridge.convert_vehicle_from_source_to_agent(
                {
                    "transform": self.veh_state_streamer.transform,
                    "velocity": self.veh_state_streamer.velocity,
                    "acceleration": self.veh_state_streamer.acceleration
                }
            )
            vehicle.control = self.control_streamer.control_tx
            if self.depth_cam_streamer.intrinsics is not None:
                self.agent.front_depth_camera.intrinsics_matrix = self.depth_cam_streamer.intrinsics
            if self.world_cam_streamer.intrinsics is not None:
                self.agent.front_rgb_camera.intrinsics_matrix = self.world_cam_streamer.intrinsics
            return sensor_data, vehicle
        except Exception as e:
            self.logger.error(f"Cannot convert data: {e}")
            return SensorsData(), Vehicle()

    def on_finish(self):
        self.logger.info("Finishing...")
        for i in range(10):
            self.control_streamer.send(VehicleControl())
        self.agent.shutdown_module_threads()
        if self.is_unity:
            self.unity_rgb_streamer.shutdown()
            self.unity_veh_state_streamer.shutdown()
            self.unity_control_streamer.shutdown()
        self.logger.info("Finished Peacefully, please ignore the output error")

    def update_pygame(self, clock) -> Tuple[bool, VehicleControl, bool]:
        """
        Update the pygame window, including parsing keypress
        Args:
            clock: pygame clock
        Returns:
            bool - whether to continue the game
            VehicleControl - the new VehicleControl cmd by the keyboard
        """
        if self.display is not None:
            frame = self.generate_current_frame(self.agent.front_rgb_camera.data, self.agent.rear_rgb_camera.data)
            if frame is not None:
                frame = self.impose_reference_line(frame)
                frame: np.ndarray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
                pygame.surfarray.blit_array(self.display, frame)
                pygame.display.flip()
        pygame.display.flip()
        return self.controller.parse_events(clock=clock)

    def generate_current_frame(self,
                               world_cam_data: Optional[np.ndarray] = None,
                               face_cam_data: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        frame: Optional[np.ndarray] = None
        overlay_frame: Optional[np.ndarray] = None
        if world_cam_data is not None:
            s = world_cam_data.shape
            height = 3 * s[1] // 4
            min_y = s[0] - height - self.controller.vertical_view_offset
            max_y = s[0] - self.controller.vertical_view_offset
            display_view = world_cam_data[min_y:max_y, :]
            frame = cv2.resize(display_view, dsize=(self.pygame_display_width, self.pygame_display_height))

        if face_cam_data is not None:
            overlay_frame = cv2.resize(face_cam_data,
                                       (self.front_cam_display_size[1], self.front_cam_display_size[0]))
        if overlay_frame is not None and frame is not None:
            x_offset = self.front_cam_offsets[0]
            y_offset = self.front_cam_offsets[1]
            frame[y_offset:y_offset + overlay_frame.shape[0],
            x_offset:x_offset + overlay_frame.shape[1]] = overlay_frame
        if self.should_display_system_status:
            self.display_system_status(frame)
        return frame

    def impose_reference_line(self, frame: np.ndarray):
        frame = cv2.polylines(frame, self.green_overlay_pts, isClosed=True, color=(0, 255, 0), thickness=2)
        frame = cv2.polylines(frame, self.yellow_overlay_pts, isClosed=True, color=(0, 255, 255), thickness=2)
        frame = cv2.polylines(frame, self.red_overlay_pts, isClosed=True, color=(0, 0, 255), thickness=2)

        return frame

    def display_system_status(self, frame: np.ndarray):
        if frame is not None:
            frame = cv2.putText(img=frame, text=f"{self.agent.vehicle.transform.location.__str__()}", org=(20, 20),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            frame = cv2.putText(img=frame, text=f"{self.agent.vehicle.transform.rotation.__str__()}", org=(20, 40),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            frame = cv2.putText(img=frame, text=f"vx: {round(self.agent.vehicle.velocity.x, 3)}, "
                                                f"vy: {round(self.agent.vehicle.velocity.y, 3)}, "
                                                f"vz: {round(self.agent.vehicle.velocity.z, 3)}", org=(20, 60),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            frame = cv2.putText(img=frame, text=f"ax: {round(self.agent.vehicle.acceleration.x, 3)}, "
                                                f"ay: {round(self.agent.vehicle.acceleration.y, 3)}, "
                                                f"az: {round(self.agent.vehicle.acceleration.z, 3)}", org=(20, 80),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)
            frame = cv2.putText(img=frame, text=f"gx: {round(self.veh_state_streamer.gyro.x, 3)}, "
                                                f"gy: {round(self.veh_state_streamer.gyro.y, 3)}, "
                                                f"gz: {round(self.veh_state_streamer.gyro.z, 3)}", org=(20, 100),
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

            frame = cv2.putText(img=frame,
                                text=f"Auto = {self.is_auto} | {self.control_streamer.control_tx}",
                                org=(20, frame.shape[0] - 20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6,
                                color=(0, 255, 0),  # BGR
                                thickness=1, lineType=cv2.LINE_AA)

        return frame
