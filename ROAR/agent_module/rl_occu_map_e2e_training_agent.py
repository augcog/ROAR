from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.data_structures_models import SensorsData
from ROAR.utilities_module.vehicle_models import Vehicle, VehicleControl
from ROAR.perception_module.obstacle_from_depth import ObstacleFromDepth
from ROAR.utilities_module.occupancy_map import OccupancyGridMap
from pathlib import Path
from ROAR.control_module.pid_controller import PIDController
from ROAR.planning_module.behavior_planner.behavior_planner import BehaviorPlanner
from ROAR.planning_module.mission_planner.waypoint_following_mission_planner import WaypointFollowingMissionPlanner
from ROAR.planning_module.local_planner.loop_simple_waypoint_following_local_planner import \
    LoopSimpleWaypointFollowingLocalPlanner
from ROAR.planning_module.local_planner.simple_waypoint_following_local_planner import \
    SimpleWaypointFollowingLocalPlanner


class RLOccuMapE2ETrainingAgent(Agent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.route_file_path = Path(self.agent_settings.waypoint_file_path)
        self.pid_controller = PIDController(agent=self, steering_boundary=(-1, 1), throttle_boundary=(0, 1))
        self.mission_planner = WaypointFollowingMissionPlanner(agent=self)
        # initiated right after mission plan

        self.behavior_planner = BehaviorPlanner(agent=self)
        self.traditional_local_planner = LoopSimpleWaypointFollowingLocalPlanner(
            agent=self,
            controller=self.pid_controller,
            mission_planner=self.mission_planner,
            behavior_planner=self.behavior_planner,
            closeness_threshold=1.5
        )

        self.occupancy_map = OccupancyGridMap(absolute_maximum_map_size=1000,
                                              world_coord_resolution=1,
                                              occu_prob=0.99,
                                              max_points_to_convert=10000,
                                              threaded=True)
        self.obstacle_from_depth_detector = ObstacleFromDepth(agent=self,
                                                              threaded=True,
                                                              max_detectable_distance=0.5,
                                                              max_points_to_convert=20000,
                                                              min_obstacle_height=2)
        self.add_threaded_module(self.obstacle_from_depth_detector)
        self.add_threaded_module(self.occupancy_map)

    def run_step(self, sensors_data: SensorsData, vehicle: Vehicle) -> VehicleControl:
        super(RLOccuMapE2ETrainingAgent, self).run_step(sensors_data, vehicle)
        self.traditional_local_planner.run_in_series()
        self.transform_history.append(self.vehicle.transform)
        option = "obstacle_coords"  # ground_coords, point_cloud_obstacle_from_depth
        if self.kwargs.get(option, None) is not None:
            points = self.kwargs[option]
            self.occupancy_map.update_async(points)

        if self.kwargs.get("control") is None:
            return VehicleControl()
        else:
            return self.kwargs.get("control")
