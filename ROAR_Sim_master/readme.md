# ROAR Carla Simlulation

## Introduction
PENDING 

## Quick Start
Please visit [https://augcog.github.io/ROAR-Sim/](https://augcog.github.io/ROAR-Sim/)

## For running with Jetson Nano & the vehicle

#### Pre-checks
1. Make sure Jetson Nano is powered
2. Make sure Arduino is plugged into the Jetson nano
3. Make sure Intel Realsense is plugged into the Jetson Nano
4. Make sure the ESC is turned on. (It should be Green)

#### Setup software
1. clone this repo
    - `git clone --recursive https://github.com/augcog/ROAR-Sim.git`
    - (we are working on changing it to a different repo so that file organization makes more sense) 
2. install dependencies
    - `cd ROAR-Sim` 
    - `pip3 install -r requirements.txt`
        - you might need to install a few other packages manually
        - You should be able to Google and install them, we are working on a all-in-one script that does this
3. Run it
    - `python3 runner_jetson.py`
    
#### Side notes:
You should run setup everything on the Jetson Nano. And then use [X11vnc](https://tecadmin.net/setup-x11vnc-server-on-ubuntu-linuxmint/) or similar service to remote control it. 

However, setting all the software on your local computer should also work. Just make sure to hook plug the Camera & Arduino to the computer that you are running the code on.
Platforms tested:

    - IOS 10.15
    - Windows 10
    - Ubuntu 18.04
    - Jetson Nano with Ubuntu 18.04 Aarch64