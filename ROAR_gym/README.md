# ROAR_Gym

## Introduction
Gym designed for the ROAR Simulation, purposed for ease of extension into specially purposed environment for RL or other advanced research. 


## Setup
1. Install [gym](https://gym.openai.com/docs/#installation)
2. Install [stablebaseline](https://stable-baselines.readthedocs.io/en/master/guide/install.html)
3. Ensure that tensorflow version is correct  (as of writing, stablebaseline support only Tensorflow <1.15)
4. Read through the code in `ROAR_Gym/roar_env.py` to understand what variables are in your disposal
5. Create your own environment, as in the `roar_pid_env.py` example, and also following the [gym tutorial](https://github.com/openai/gym/blob/master/docs/creating-environments.md)

### Todo:
1. more documentation
2. clearer action items on how to inherit and the design.

