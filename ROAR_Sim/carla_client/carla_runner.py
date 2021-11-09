import carla
from carla import ColorConverter as cc
from ROAR_Sim.carla_client.util.sensors import CollisionSensor
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
import logging
import pygame
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
from Bridges.carla_bridge import CarlaBridge
from ROAR_Sim.carla_client.util.hud import HUD
from ROAR_Sim.carla_client.util.world import World
from ROAR_Sim.carla_client.util.keyboard_control import KeyboardControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR_Sim.configurations.configuration import import_carla
from pathlib import Path
from typing import List, Dict, Any
from ROAR.utilities_module.vehicle_models import VehicleControl
import json
from typing import Optional
import numpy as np
import cv2
from threading import Thread


class CarlaRunner:

    def __init__(self,
                 carla_settings: CarlaConfig,
                 agent_settings: AgentConfig,
                 npc_agent_class,
                 competition_mode=False,
                 start_bbox: np.ndarray = np.array([5, -5, 0, 13, 5, 50]),
                 lap_count=10):
        """

        Args:
            carla_settings: CarlaConfig instance
            agent_settings: AgentConfig instance
            npc_agent_class: an agent class
            competition_mode: [Optional] True/False
            start_bbox: [Optional] array of [minx, miny, minz, maxx, maxy, maxz].
                        [5, -5, 0, 13, 5, 50] is the bbox for easy_map.
                        [-815, 20, -760, -770, 120, -600] is the bbox for berkeley_minor_map
            lap_count: [Optional] total lap count

        """
        self.carla_settings = carla_settings
        self.agent_settings = agent_settings
        self.carla_bridge = CarlaBridge()
        self.npc_agent_class = npc_agent_class
        self.world = None
        self.client = None
        self.controller = None
        self.display = None
        self.agent = None

        self.npc_agents: Dict[npc_agent_class, Any] = {}
        self.agent_collision_counter = 0

        self.competition_mode = competition_mode
        self.start_bbox = start_bbox
        self.lap_count = lap_count
        self.completed_lap_count = 0
        self.sensor_data = SensorsData()
        self.vehicle_state = Vehicle()

        self.start_simulation_time: Optional[float] = None
        self.start_vehicle_position: Optional[np.array] = None
        self.end_simulation_time: Optional[float] = None
        self.end_vehicle_position: Optional[np.array] = None

        self.logger = logging.getLogger(__name__)
        self.timestep_counter = 0

    def set_carla_world(self) -> Vehicle:
        """
        Initiating the vehicle with loading messages
        Returns:
            Vehicle Information
        """

        try:
            pygame.init()
            pygame.font.init()
            self.logger.debug(f"Connecting to {self.carla_settings.host}: "
                              f"{self.carla_settings.port}")

            self.client = carla.Client(self.carla_settings.host,
                                       self.carla_settings.port)
            if not self.check_version(client=self.client):
                self.logger.error(f"Version Mismatch: Client = {self.client.get_client_version()}, "
                                  f"Server = {self.client.get_server_version()}. \n"
                                  f"HINT: Please change carla_version to either 0.9.9 or 0.9.10 "
                                  f"in ROAR_Sim.configurations.carla_version.txt")
                exit(1)

            if self.carla_settings.should_visualize_with_pygame is True:
                self.display = pygame.display.set_mode(
                    (self.carla_settings.width, self.carla_settings.height),
                    pygame.HWSURFACE | pygame.DOUBLEBUF)

            self.logger.debug(f"Setting HUD")
            hud = HUD(self.carla_settings.width, self.carla_settings.height)

            self.logger.debug("Setting up world")

            self.world = World(carla_world=self.client.get_world(), hud=hud,
                               carla_settings=self.carla_settings,
                               agent_settings=self.agent_settings)

            if self.carla_settings.should_spawn_npcs:
                self.spawn_npcs()

            self.logger.debug(f"Connecting to Keyboard controls")
            self.controller = KeyboardControl(world=self.world,
                                              carla_setting=self.carla_settings
                                              )
            self.logger.debug("All settings done")

            return self.carla_bridge. \
                convert_vehicle_from_source_to_agent(self.world.player)

        except Exception as e:
            self.logger.error(
                f"Unable to initiate the world due to error: {e}")
            raise e

    def start_game_loop(self,
                        agent,
                        use_manual_control=False,
                        starting_lap_count=0):
        """Start running the vehicle and stop when finished running
        the track"""

        self.agent = agent
        lap_count = starting_lap_count
        has_entered_bbox = False
        should_restart_lap = False
        try:
            self.logger.debug("Initiating game")
            self.agent.start_module_threads()
            clock = pygame.time.Clock()
            self.start_simulation_time = self.world.hud.simulation_time
            self.start_vehicle_position = self.agent.vehicle.transform.location.to_array()

            while True:

                # make sure the program does not run above 60 frames per second
                # this allow proper synchrony between server and client
                clock.tick_busy_loop(60)
                should_continue, carla_control = self.controller.parse_events(client=self.client,
                                                                              world=self.world,
                                                                              clock=clock)

                self.agent_collision_counter = self.get_num_collision()

                if self.competition_mode:
                    is_currently_in_bbox = self.is_within_start_finish_bbox(
                        curr_pos=self.agent.vehicle.transform.location.to_array())
                    if has_entered_bbox is True and is_currently_in_bbox is False:
                        has_entered_bbox = False
                    elif has_entered_bbox is False and is_currently_in_bbox is True:
                        has_entered_bbox = True
                        lap_count += 1
                        if lap_count > self.lap_count:
                            break
                        else:
                            self.logger.info(f"Going onto Lap {lap_count} out of {self.lap_count}")

                    if len(self.world.collision_sensor.history) > 0:
                        should_restart_lap = True

                    if should_restart_lap:
                        should_continue = False

                # check for exiting condition
                if should_continue is False:
                    break

                self.world.tick(clock)
                self.world.render(display=self.display)
                if self.carla_settings.should_visualize_with_pygame is True:
                    pygame.display.flip()
                self.fetch_data_async()
                sensor_data, new_vehicle = self.sensor_data.copy(), self.vehicle_state.copy()

                if self.carla_settings.save_semantic_segmentation and self.world.semantic_segmentation_sensor_data:
                    Thread(target=lambda: self.world.semantic_segmentation_sensor_data.save_to_disk((Path(
                        "./data/output") / "ss" / f"frame_{self.agent.time_counter}.png").as_posix(),
                                                                                                    cc.CityScapesPalette),
                           args=()).start()

                if self.carla_settings.should_spawn_npcs:
                    self.execute_npcs_step()

                if self.agent_settings.enable_autopilot:
                    if self.agent is None:
                        raise Exception(
                            "In autopilot mode, but no agent is defined.")
                    agent_control = self.agent.run_step(vehicle=new_vehicle,
                                                        sensors_data=sensor_data)
                    if not use_manual_control:
                        carla_control = self.carla_bridge. \
                            convert_control_from_agent_to_source(agent_control)
                self.world.player.apply_control(carla_control)

                self.timestep_counter += 1

            self.completed_lap_count = lap_count - 1
        except Exception as e:
            self.logger.error(f"Error happened, exiting safely. Error: {e}")
        finally:
            if self.competition_mode and should_restart_lap:
                self.restart_on_lap(agent=agent,
                                    use_manual_control=use_manual_control,
                                    starting_lap_count=lap_count - 1)
            else:
                self.on_finish()

    def restart_on_lap(self, agent, use_manual_control: bool, starting_lap_count: int):
        self.logger.info(f"Restarting on Lap {starting_lap_count}")
        self.on_finish()
        self.set_carla_world()
        agent.__init__(vehicle=agent.vehicle, agent_settings=agent.agent_settings)
        self.start_game_loop(agent=agent, use_manual_control=use_manual_control,
                             starting_lap_count=starting_lap_count)

    def on_finish(self):
        self.logger.debug("Ending Game")
        if self.agent is not None:
            self.agent.shutdown_module_threads()
            self.end_vehicle_position = self.agent.vehicle.transform.location.to_array()
        else:
            self.end_vehicle_position = self.start_vehicle_position
        if self.world is not None:
            self.end_simulation_time = self.world.hud.simulation_time
            self.world.destroy()
            self.logger.debug("All actors are destroyed")
        try:
            pygame.quit()
        except Exception as e:
            self.logger.debug(
                f"Cannot quit pygame normally, force quitting. Error: {e}")
        self.logger.debug("Game ended")

    def is_within_start_finish_bbox(self, curr_pos: np.ndarray) -> bool:
        min_bounding_box = self.start_bbox[:3]
        max_bounding_box = self.start_bbox[3:]
        return all(np.logical_and(min_bounding_box < curr_pos, curr_pos < max_bounding_box))

    def fetch_data_async(self):
        t = Thread(target=self.convert_data, args=())
        t.start()

    def convert_data(self):
        """
        Convert data from source to agent

        Returns:
            sensors_data: sensor data for agent
            new_vehicle: the current player's vehicle state

        """
        try:
            self.sensor_data: SensorsData = \
                self.carla_bridge.convert_sensor_data_from_source_to_agent(
                    {
                        "front_rgb": None if self.world.front_rgb_sensor_data is None
                        else self.world.front_rgb_sensor_data,
                        "rear_rgb": None if self.world.rear_rgb_sensor_data is None
                        else self.world.rear_rgb_sensor_data,
                        "front_depth":
                            None if self.world.front_depth_sensor_data is None else
                            self.world.front_depth_sensor_data,
                        "imu": self.world.imu_sensor
                    }
                )
            if self.world.player.is_alive:
                self.vehicle_state = self.carla_bridge.convert_vehicle_from_source_to_agent(self.world.player)
        except Exception as e:
            print("Error", e)
            self.logger.error(e)

    def execute_npcs_step(self):
        # TODO this can be parallelized
        try:
            for agent, actor in self.npc_agents.items():
                new_vehicle = self.carla_bridge.convert_vehicle_from_source_to_agent(actor)
                curr_control: VehicleControl = agent.run_step(sensors_data=SensorsData(), vehicle=new_vehicle)
                carla_control = self.carla_bridge.convert_control_from_agent_to_source(curr_control)
                actor.apply_control(carla_control)
        except Exception as e:
            self.logger.error(f"Failed to execute step for NPC. "
                              f"Error: {e}")

    def spawn_npcs(self):
        # parse npc file
        npc_config_file_path = Path(self.carla_settings.npc_config_file_path)
        assert npc_config_file_path.exists(), f"NPC file path {npc_config_file_path.as_posix()} does not exist"
        npc_configs = json.load(npc_config_file_path.open('r'))

        npc_configs: List[AgentConfig] = [AgentConfig.parse_obj(config) for config in npc_configs]

        self.world.spawn_npcs(npc_configs)
        self.npc_agents = {
            self.npc_agent_class(vehicle=actor, agent_settings=npc_config): actor for actor, npc_config in
            self.world.npcs_mapping.values()
        }

    def get_num_collision(self):
        collision_sensor: CollisionSensor = self.world.collision_sensor
        return len(collision_sensor.history)

    def check_version(self, client):
        return ("0.9.9" in client.get_server_version()) == ("0.9.9" in client.get_client_version())
