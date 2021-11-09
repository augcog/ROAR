try:
    from ROAR_Gym.envs.roar_env import ROAREnv
except:
    from ROAR_Gym.ROAR_Gym.envs.roar_env import ROAREnv

from ROAR.utilities_module.vehicle_models import VehicleControl
from ROAR.agent_module.agent import Agent
from ROAR.utilities_module.vehicle_models import Vehicle
from typing import Tuple
import numpy as np
from typing import List, Any
import gym
import math
from collections import OrderedDict

class ROARPIDEnv(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        # action_space = speed, long_k, lat_k

        self.action_space = gym.spaces.Box(low=np.array([0, 0, 0, 0, 0, 0, 0]),
                                           high=np.array([200, 1, 1, 1, 1, 1, 1]), dtype=np.float64)
        # observation_space = curr_speed, curr_transform, next_waypoint
        self.observation_space = gym.spaces.Box(low=np.array([-200,
                                                              -1000, -1000, -1000, -360, -360, -360,
                                                              -1000, -1000, -1000, -360, -360, -360,
                                                              ]),
                                                high=np.array([200,
                                                               1000, 1000, 1000, 360, 360, 360,
                                                               1000, 1000, 1000, 360, 360, 360]),
                                                dtype=np.float64)
        self._prev_speed = 0

    def step(self, action: List[float]) -> Tuple[np.ndarray, float, bool, dict]:
        """

        Args:
            action: array of [target_speed,
                              long_kp, long_kd, long_ki,
                              lat_kp, lat_kd, lat_ki]

        Returns:

        """
        assert type(action) == list or type(action) == np.ndarray, f"Action is of type {type(action)}"
        assert len(action) == 7, f"Action of shape {np.shape(action)} is not correct"
        self._prev_speed = Vehicle.get_speed(self.agent.vehicle)

        target_speed = action[0]
        long_k_p, long_k_d, long_k_i = action[1], action[2], action[3]
        lat_k_p, lat_k_d, lat_k_i = action[4], action[5], action[6]
        self.agent.kwargs["long_k_p"] = long_k_p
        self.agent.kwargs["long_k_d"] = long_k_d
        self.agent.kwargs["long_k_i"] = long_k_i
        self.agent.kwargs["target_speed"] = target_speed
        self.agent.kwargs["lat_k_p"] = lat_k_p
        self.agent.kwargs["lat_k_d"] = lat_k_d
        self.agent.kwargs["lat_k_i"] = lat_k_i

        agent, reward, is_done, other_info = super(ROARPIDEnv, self).step(action=action)
        obs = self._get_obs()
        return obs, reward, is_done, other_info

    def _get_obs(self) -> Any:
        curr_speed = np.array([Vehicle.get_speed(self.agent.vehicle)])
        curr_transform = self.agent.vehicle.transform.to_array()
        if len(self.agent.local_planner.way_points_queue) > 0:
            next_waypoint_transform = self.agent.local_planner.way_points_queue[0].to_array()
        else:
            next_waypoint_transform = curr_transform
        return np.append(np.append(curr_speed, curr_transform), next_waypoint_transform)

    def get_reward(self) -> float:
        reward: float = 0.0
        target_speed = self.agent.kwargs["target_speed"]
        current_speed = Vehicle.get_speed(self.agent.vehicle)

        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            reward += -1000000
        else:
            # made a right step without collision
            reward = 10.0  # base reward for making a successful step

        # if the agent is able to complete a lap, reward heavy
        if self.agent.is_done:
            reward += 1000000

        # if the agent tries to turn too much, penalize
        if abs(self.agent.vehicle.control.steering) > 0.3:
            # prevent it from over steering
            reward -= 50

        if current_speed < 50 or target_speed < 50:
            # i can definitely go through all the track above 50 km/h
            reward -= 1000
        elif current_speed > self._prev_speed:
            reward += 10
        return reward

    def _get_info(self) -> dict:
        info_dict = OrderedDict()
        info_dict["reward"] = self.get_reward()

        info_dict["speed"] = Vehicle.get_speed(self.agent.vehicle)
        info_dict["target_speed"] = self.agent.kwargs["target_speed"]

        info_dict["throttle"] = self.agent.vehicle.control.throttle
        info_dict["steering"] = self.agent.vehicle.control.steering

        info_dict["long_kp"] = self.agent.kwargs["long_k_p"]
        info_dict["long_kd"] = self.agent.kwargs["long_k_d"]
        info_dict["long_ki"] = self.agent.kwargs["long_k_i"]
        info_dict["lat_kp"] = self.agent.kwargs["lat_k_p"]
        info_dict["lat_kd"] = self.agent.kwargs["lat_k_d"]
        info_dict["lat_ki"] = self.agent.kwargs["lat_k_i"]
        return info_dict

    def reset(self) -> Any:
        super(ROARPIDEnv, self).reset()
        return self._get_obs()
