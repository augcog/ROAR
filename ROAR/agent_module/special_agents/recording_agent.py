from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import VehicleControl, Vehicle
import logging
from pathlib import Path
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import LoopSimpleWaypointFollowingLocalPlanner


class RecordingAgent(Agent):
    def __init__(self, target_speed=20, **kwargs):
        super().__init__(**kwargs)
        # ensure recording status is ON
        self.agent_settings.save_sensor_data = True
        super().__init__(**kwargs)
        self.target_speed = target_speed
        self.logger = logging.getLogger("Recording Agent")
        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1),
                                            throttle_boundary=(0, 1),
                                            target_speed=target_speed)
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        self.local_planner = LoopSimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1)
        # self.occupancy_map = OccupancyGridMap(absolute_maximum_map_size=1000,
        #                                       world_coord_resolution=1,
        #                                       occu_prob=0.99,
        #                                       max_points_to_convert=5000,
        #                                       threaded=True,
        #                                       should_save=self.agent_settings.save_sensor_data,
        #                                       agent=self)
        # self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self,
        #                                                       threaded=True,
        #                                                       max_detectable_distance=0.3,
        #                                                       max_points_to_convert=10000,
        #                                                       min_obstacle_height=2)
        # self.add_threaded_module(self.obstacle_from_depth_detector)
        # self.add_threaded_module(self.occupancy_map)
        self.option = "obstacle_coords"
        self.lap_count = 0

    def run_step(self, sensors_data: SensorsData,
                 vehicle: Vehicle) -> VehicleControl:
        super(RecordingAgent, self).run_step(sensors_data=sensors_data, vehicle=vehicle)
        # self.transform_history.append(self.vehicle.transform)

        # control = self.local_planner.run_in_series()
        # if self.kwargs.get(self.option, None) is not None:
        #     points = self.kwargs[self.option]
        #     self.occupancy_map.update_async(points)
        #     self.occupancy_map.visualize(transform=self.vehicle.transform, view_size=(200, 200))
        # return VehicleControl(throttle=0.2, steering=0)

