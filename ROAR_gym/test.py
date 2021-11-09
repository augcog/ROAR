import os
import sys
from pathlib import Path

sys.path.append(Path(os.getcwd()).parent.as_posix())
import gym
import ROAR_Gym
from ROAR_Sim.configurations.configuration import Configuration as CarlaConfig
from ROAR.configurations.configuration import Configuration as AgentConfig
from ROAR.agent_module.agent import Agent
from ROAR.agent_module.pid_agent import PIDAgent

def main():
    # Set gym-carla environment
    agent_config = AgentConfig.parse_file(Path("configurations/agent_configuration.json"))
    carla_config = CarlaConfig.parse_file(Path("configurations/carla_configuration.json"))

    params = {
        "agent_config": agent_config,
        "carla_config": carla_config,
        "ego_agent_class": PIDAgent
    }
    env = gym.make('roar-pid-v0', params=params)
    obs: Agent = env.reset()
    counter = 1
    while True:
        action = [1.0, 0.0]
        curr_step = env.bbox_step(action)
        obs: Agent = curr_step[0]
        reward: float = curr_step[1]
        is_done: bool = curr_step[2]
        info: dict = curr_step[3]


        env.render()

        if counter % 200 == 0:
            # test reset ability
            env.reset()
            counter = 1
        if is_done:
            print("IS DONE")
            obs: Agent = env.reset()
        counter += 1


if __name__ == '__main__':
    main()
