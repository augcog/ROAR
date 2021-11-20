import carla
from carla import ColorConverter as cc
import logging
import random
import sys
from Bridges.carla_bridge import CarlaBridge
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR_Sim.carla_client.util.hud import HUD

from ROAR_Sim.carla_client.util.utilities import CarlaCarColor, \
    CarlaCarColors, get_actor_display_name
from ROAR_Sim.carla_client.util.sensors import CollisionSensor, \
    GnssSensor, LaneInvasionSensor, IMUSensor, RadarSensor
from ROAR_Sim.carla_client.util.camera_manager import CameraManager
from ROAR.configurations.configuration import Configuration as AgentConfig
import weakref
from typing import List, Dict, Tuple, Any


class World(object):
    """An World that holds all display settings"""

    def __init__(self, carla_world: carla.World,
                 hud: HUD, carla_settings: CarlaConfig,
                 agent_settings: AgentConfig):
        """Create a World with the given carla_world, head-up display and
        server setting."""

        self.logger = logging.getLogger(__name__)
        self.carla_settings: CarlaConfig = carla_settings
        self.agent_settings: AgentConfig = agent_settings
        self.carla_world: carla.World = carla_world
        self.clean_spawned_all_actors()
        self.actor_role_name = carla_settings.role_name
        try:
            self.map = self.carla_world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print(
                '  Make sure it exists, has the same name of your town, '
                'and is correct.')
            sys.exit(1)
        self.hud = hud
        self.carla_bridge = CarlaBridge()
        self._spawn_point_id = agent_settings.spawn_point_id
        self._actor_filter = carla_settings.carla_vehicle_blueprint_filter
        self._car_color = carla_settings.car_color
        self._gamma = carla_settings.gamma
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.radar_sensor = None
        self.camera_manager = None
        self.recording_enabled = False
        self.time_counter = 0

        self.front_left_depth_sensor = None
        self.front_right_depth_sensor = None
        self.front_rgb_sensor = None
        self.front_depth_sensor = None
        self.rear_rgb_sensor = None
        self.semantic_segmentation_sensor = None

        self.recording_start = 0
        # set weather
        self.logger.debug("Setting Weather")
        self.set_weather(
            carla_settings.carla_weather.to_carla_weather_params())

        # set player
        self.logger.debug("Setting Player")
        self.player = self.spawn_actor(
            actor_filter=self._actor_filter,
            player_role_name=self.actor_role_name,
            color=self._car_color,
            spawn_point_id=self._spawn_point_id,
        )
        # set camera
        self.logger.debug("Setting Camera")
        self.set_camera()

        # set sensor
        self.logger.debug("Setting Default Sensor")
        self.set_sensor()

        # set custom sensor
        self.logger.debug("Setting Custom Sensor")
        self.set_custom_sensor()
        self.front_left_depth_sensor_data = None
        self.front_right_depth_sensor_data = None
        self.front_rgb_sensor_data = None
        self.front_depth_sensor_data = None
        self.rear_rgb_sensor_data = None
        self.semantic_segmentation_sensor_data = None

        # spawn npc
        self.npcs_mapping: Dict[str, Tuple[Any, AgentConfig]] = {}

        settings = self.carla_world.get_settings()
        settings.synchronous_mode = self.carla_settings.synchronous_mode
        settings.no_rendering_mode = self.carla_settings.no_rendering_mode
        if self.carla_settings.synchronous_mode:
            settings.fixed_delta_seconds = self.carla_settings.fixed_delta_seconds
        self.carla_world.apply_settings(settings)
        self.carla_world.on_tick(hud.on_world_tick)
        self.logger.debug("World Initialized")

    def spawn_actor(self, actor_filter: str = "vehicle.tesla.model3",
                    player_role_name: str = "npc",
                    color: CarlaCarColor = CarlaCarColors.GREY,
                    spawn_point_id: int = random.choice(list(range(8)))):
        """Set up a hero-named player with Grey Tesla Model3 Vehicle """

        blueprint = self.carla_world.get_blueprint_library().find(actor_filter)
        blueprint.set_attribute("role_name", player_role_name)
        if blueprint.has_attribute("color"):
            blueprint.set_attribute("color", color.to_string())
        if blueprint.has_attribute("is_invincible"):
            self.logger.debug("TESLA IS INVINCIBLE")
            blueprint.set_attribute("is_invincible", "true")
        try:
            actor = \
                self.carla_world.spawn_actor(blueprint,
                                             self.map.get_spawn_points()[
                                                 spawn_point_id])
            return actor
        except Exception as e:
            raise ValueError(f"Cannot spawn actor at ID [{spawn_point_id}]. "
                             f"Error: {e}")

    def set_camera(self, cam_index: int = 0, cam_pos_index: int = 0):
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def set_sensor(self):
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)

    def toggle_radar(self):
        if self.radar_sensor is None:
            self.radar_sensor = RadarSensor(self.player)
        elif self.radar_sensor.sensor is not None:
            self.radar_sensor.sensor.destroy()
            self.radar_sensor = None

    def tick(self, clock):
        self.time_counter += 1
        self.hud.tick(self, clock)
        if self.carla_settings.synchronous_mode:
            self.carla_world.tick()

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def set_weather(self, new_weather: carla.WeatherParameters):
        self.carla_world.weather = new_weather

    def set_custom_sensor(self):
        Attachment = carla.AttachmentType
        self._destroy_custom_sensors()
        self.front_left_depth_sensor = self._spawn_custom_sensor(
            blueprint_filter="sensor.camera.depth",
            transform=carla.Transform(
                location=carla.Location(x=1.8, y=0, z=1.7),
                rotation=carla.Rotation(pitch=0, yaw=-30, roll=0)
            ),
            attachment=Attachment.Rigid,
            attributes={
                "fov": self.agent_settings.front_rgb_cam.fov,
            })

        self.front_right_depth_sensor = self._spawn_custom_sensor(
            blueprint_filter="sensor.camera.depth",
            transform=carla.Transform(
                location=carla.Location(x=1.4, y=0, z=1.7),
                rotation=carla.Rotation(pitch=0, yaw=30, roll=0)
            ),
            attachment=Attachment.Rigid,
            attributes={
                "fov": self.agent_settings.front_rgb_cam.fov,
            })

        self.front_rgb_sensor = self._spawn_custom_sensor(
            blueprint_filter="sensor.camera.rgb",
            transform=self.carla_bridge.convert_transform_from_agent_to_source(
                self.agent_settings.front_rgb_cam.transform),
            attachment=Attachment.Rigid,
            attributes={
                "fov": self.agent_settings.front_rgb_cam.fov,
            })
        self.front_depth_sensor = self._spawn_custom_sensor(
            blueprint_filter="sensor.camera.depth",
            transform=self.carla_bridge.convert_transform_from_agent_to_source(
                self.agent_settings.front_depth_cam.transform),
            attachment=Attachment.Rigid,
            attributes={
                "fov": self.agent_settings.front_depth_cam.fov,
            })
        self.rear_rgb_sensor = \
            self._spawn_custom_sensor(
                blueprint_filter="sensor.camera.rgb",
                transform=self.carla_bridge.
                    convert_transform_from_agent_to_source(
                    self.agent_settings.rear_rgb_cam.transform),
                attachment=Attachment.Rigid,
                attributes={
                    "fov":
                        self.agent_settings.rear_rgb_cam.fov,
                })

        if self.carla_settings.save_semantic_segmentation:
            self.semantic_segmentation_sensor = self._spawn_custom_sensor(
                blueprint_filter="sensor.camera.semantic_segmentation",
                transform=self.carla_bridge.convert_transform_from_agent_to_source(
                    self.agent_settings.front_depth_cam.transform
                ),
                attachment=Attachment.Rigid,
                attributes={"fov": self.agent_settings.front_depth_cam.fov},
            )

        weak_self = weakref.ref(self)
        self.front_left_depth_sensor.listen(
            lambda image: World._parse_front_left_depth_sensor_image(
                weak_self=weak_self, image=image))

        self.front_right_depth_sensor.listen(
            lambda image: World._parse_front_right_depth_sensor_image(
                weak_self=weak_self, image=image))

        self.front_rgb_sensor.listen(
            lambda image: World._parse_front_rgb_sensor_image(
                weak_self=weak_self, image=image))
        self.front_depth_sensor.listen(
            lambda image: World._parse_front_depth_sensor_image(
                weak_self=weak_self, image=image))
        self.rear_rgb_sensor.listen(lambda image:
                                    World._parse_rear_rgb_sensor_image(
                                        weak_self=weak_self, image=image))
        if self.carla_settings.save_semantic_segmentation:
            self.semantic_segmentation_sensor.listen(lambda image: World._parse_semantic_segmentation_image(
                weak_self=weak_self, image=image
            ))

    def _spawn_custom_sensor(self, blueprint_filter: str,
                             transform: carla.Transform,
                             attachment: carla.AttachmentType,
                             attributes: dict):
        blueprint = self.carla_world.get_blueprint_library(). \
            find(blueprint_filter)
        for key, val in attributes.items():
            if blueprint.has_attribute(key):
                blueprint.set_attribute(key, str(val))
            else:
                self.logger.error(f"Unable to set attribute [{key}] "
                                  f"for blueprint [{blueprint_filter}]")
        # self.logger.debug(f"Spawning {blueprint} at {transform}")
        return self.carla_world.spawn_actor(blueprint, transform,
                                            self.player, attachment)

    def _destroy_custom_sensors(self):
        if self.front_rgb_sensor is not None:
            self.front_rgb_sensor.destroy()

        if self.front_depth_sensor is not None:
            self.front_depth_sensor.destroy()

        if self.rear_rgb_sensor is not None:
            self.rear_rgb_sensor.destroy()

        if self.semantic_segmentation_sensor is not None:
            self.semantic_segmentation_sensor.destroy()
        if self.front_left_depth_sensor is not None:
            self.front_left_depth_sensor.destroy()
        if self.front_right_depth_sensor is not None:
            self.front_right_depth_sensor.destroy()

    @staticmethod
    def _parse_front_left_depth_sensor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self.front_left_depth_sensor_data = image

    @staticmethod
    def _parse_front_right_depth_sensor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self.front_right_depth_sensor_data = image

    @staticmethod
    def _parse_front_rgb_sensor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        self.front_rgb_sensor_data = image

    @staticmethod
    def _parse_front_depth_sensor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # image.convert(cc.Raw)
        self.front_depth_sensor_data = image

    @staticmethod
    def _parse_rear_rgb_sensor_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        self.rear_rgb_sensor_data = image

    @staticmethod
    def _parse_semantic_segmentation_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        # image.convert(cc.CityScapesPalette)
        self.semantic_segmentation_sensor_data = image

    def spawn_npcs(self, npc_configs: List[AgentConfig]):
        for npc_config in npc_configs:
            self.logger.debug(f"Spawning NPC [{npc_config.name}]")
            try:
                npc = self.spawn_actor(spawn_point_id=npc_config.spawn_point_id)
                self.npcs_mapping[npc_config.name] = (npc, npc_config)
            except Exception as e:
                self.logger.error(f"Failed to Spawn NPC {'default'}."
                                  f"Error: {e}")

    def destroy(self):
        self.logger.debug(f"destroying all actors belonging to "
                          f"[{self.actor_role_name}] in this world")
        # if self.radar_sensor is not None:
        #     self.toggle_radar()
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.imu_sensor.sensor,
            self.player,
        ]
        for actor in actors:
            if actor is not None:
                actor.destroy()

        self._destroy_custom_sensors()
        for npc, _ in self.npcs_mapping.values():
            npc.destroy()

        self.clean_spawned_all_actors()

    def clean_spawned_all_actors(self):
        """
        This function is to clean all actors that are not traffic light/signals
        Returns:

        """
        self.carla_world.tick()
        for actor in self.carla_world.get_actors():
            if "traffic" not in actor.type_id and "spectator" not in actor.type_id:
                actor.destroy()
        self.carla_world.tick()
