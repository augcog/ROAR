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
from gym.spaces import Discrete, Box
import cv2

# Define the discrete action space
DISCRETE_ACTIONS = {
    0: [0.0, 0.0],  # Coast
    1: [0.0, -0.5],  # Turn Left
    2: [0.0, 0.5],  # Turn Right
    3: [1.0, 0.0],  # Forward
    4: [-0.5, 0.0],  # Brake
    5: [1.0, -0.5],  # Bear Left & accelerate
    6: [1.0, 0.5],  # Bear Right & accelerate
    7: [-0.5, -0.5],  # Bear Left & decelerate
    8: [-0.5, 0.5],  # Bear Right & decelerate
}
FRAME_STACK = 4
CONFIG = {
    "x_res": 80,
    "y_res": 80
}


class ROAREnvE2E(ROAREnv):
    def __init__(self, params):
        super().__init__(params)
        self.action_space = Discrete(len(DISCRETE_ACTIONS))
        self.observation_space = Box(-1, 1, shape=(FRAME_STACK, CONFIG["x_res"], CONFIG["y_res"]), dtype=np.float32)
        self.prev_speed = 0
        self.prev_dist_to_strip = 0

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:
        obs = []
        rewards = []

        for i in range(FRAME_STACK):
            self.agent.kwargs["control"] = VehicleControl(throttle=DISCRETE_ACTIONS[action][0],
                                                          steering=DISCRETE_ACTIONS[action][1])
            ob, reward, is_done, info = super(ROAREnvE2E, self).step(action)
            obs.append(ob)
            rewards.append(reward)
            if is_done:
                break
        self.render()
        return np.array(obs), sum(rewards), self._terminal(), {"reward": sum(rewards),
                                                               "action": DISCRETE_ACTIONS[action]}

    def _terminal(self) -> bool:
        if self.carla_runner.get_num_collision() > self.max_collision_allowed:
            return True
        else:
            return False

    def get_reward(self) -> float:
        # prep for reward computation
        reward = 0
        curr_dist_to_strip = self.agent.curr_dist_to_strip

        # reward computation
        reward += 0.5 * (Vehicle.get_speed(self.agent.vehicle) - self.prev_speed)
        reward += abs(self.agent.vehicle.control.steering)
        reward += np.clip(self.prev_dist_to_strip - curr_dist_to_strip, -10, 10)
        reward -= self.carla_runner.get_num_collision()

        # log prev info for next reward computation
        self.prev_speed = Vehicle.get_speed(self.agent.vehicle)
        self.prev_dist_to_strip = curr_dist_to_strip
        return reward

    def _get_obs(self) -> np.ndarray:
        # star edited this: it's better to set the view_size directly instead of doing resize
        data = self.agent.occupancy_map.get_map(transform=self.agent.vehicle.transform,
                                                view_size=(CONFIG["x_res"], CONFIG["y_res"]),
                                                arbitrary_locations=self.agent.bbox.get_visualize_locs(size=20),
                                                arbitrary_point_value=0.5
                                                )
        # data = cv2.resize(occu_map, (CONFIG["x_res"], CONFIG["y_res"]), interpolation=cv2.INTER_AREA)
        cv2.imshow("data", data) # uncomment to show occu map
        cv2.waitKey(1)
        return data  # height x width x 1 array

    def reset(self) -> Any:
        super(ROAREnvE2E, self).reset()
        return self.agent.front_depth_camera.data
