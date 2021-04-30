# **Documentation for ROAR-Simulator**

![](images/ROAR_Car_1.jpg)

## Quick Links

* If you are new to the project visit [Quick Start](#quick-start)
* If you are curious about ROAR Competition at Berkeley visit [Berkeley ROAR](https://vivecenter.berkeley.edu/research1/roar/)
* If you are curious about Carla visit [Carla Simulator](https://carla.org)
* For more information regarding DeCal Course visit [Roar Decal](https://roar-decal.github.io/ROAR.html).



## Quick Start
**Platforms Tested:** Ubuntu 18.04, Windows 10
    
**Approximate Time:** ~10 minutes    

#### Windows
1. Clone the repo
    - `git clone --recursive https://github.com/augcog/ROAR-Sim.git`
2. Download Carla Server package
    - [https://drive.google.com/drive/folders/1FlkNCNENaC3qrTub7mqra7EH7iSbRNiI](https://drive.google.com/drive/folders/1FlkNCNENaC3qrTub7mqra7EH7iSbRNiI)
    - put it OUTSIDE of the `ROAR-Sim` folder, doesn't matter where
3. Check your file directory, it should be:
    - `ROAR-Sim`
        - `data`
            - `easy_map_waypoints.txt`
        - `ROAR_simulation`
        - `runner.py`
        - ... other files and folders
4. Create virtual environment and install dependencies
    - `conda create -n ROAR python=3.7`
    - `conda activate ROAR`
    - `pip install -r requirements.txt`
5. Enjoy
    - Double click the `CarlaUE4.exe` file in the Carla Server package to launch the server
    - `python runner.py`
        
#### Linux
Same as Windows, in step 6, just type in `./CarlaUE4.sh` to start the server
    
## Contribute To ROAR Guide
### Communication
Before starting a new contribution, it is important to let the community know what you plan on doing. This serves several purposes.

1. It lets everyone know that you are going to start work on a feature or bug fix so that no one else does the same and duplicates your effort.
2. It lets anyone who may already be working on the feature or bug fix know you intend to work on it, so they can tell you and you don't duplicate their effort.
3. It gives others a chance to provide more information about what the feature or bug fix might need or how it may need to be implemented.

You can let the community know by first opening an Issue on Github. An admin will tag a related Pull Request if this is a duplicated issue

### Documentation Style
We use [mkdocs](https://www.mkdocs.org/) and [mkdocstrings](https://github.com/pawamoy/mkdocstrings) to automatically generate documentation. 
This means that we require all Python code documentation to be written in Google Style

The recommended method for enabling automatic Google Docstring framework generation is through PyCharm. Here's a [tutorial](https://intellij-support.jetbrains.com/hc/en-us/community/posts/360000218290-Configure-google-docstring) on how to enable this feature in PyCharm 

### Pull Request Style
We ask that you fill out the pull request template as indicated in Github, to provide as much details as possible. 

### Issue Style
We ask that you fill out the correct issue template as indicated on Github. 



## FAQ
1. If you see an error such as 
`
WARNING: sensor object went out of the scope but the sensor is still alive in the simulation: Actor 69 (sensor.other.collision) ` 

    - Please restart the server


2. If you see `ERROR: Something bad happened. Safely exiting. Error:time-out of 2000ms while waiting for the simulator, make sure the simulator is ready and connected to 127.0.0.1:2000`
    - Make sure your server has launched
    
3. If you see error such as 
`ERROR: name 'agent' is not defined. Need to restart Server` or `ERROR: Cannot spawn actor at ID [1]. Error: Spawn failed because of collision at spawn position. Need to restart Server`
    - Just restart the server

4. My computer is getting very hot
    - Yeah, this is normal. If it gets too hot, just turn off the server and let it cool down for a minute. 
    - Our suggestion is that when you are writing code, just turn the server off
5. The simulation is very laggy
    - One method to mitigate this is to start the simulator with the `-quality-level=Low` flag
        - For example:
            - `./CarlaUE4.sh -quality-level=Low` on linux
            - `./CarlaUE4.exe -quality-level=Low` on windows
    - Another method is to turn off the display (this will just make it SLIGHTLY faster), but this is only available on Linux
        - `DISPLAY= ./CarlaUE4.sh -opengl`
    - Last method is to understand and tryout Carla's [Synchronized Mode](https://carla.readthedocs.io/en/0.9.9/adv_synchrony_timestep/#client-server-synchrony)
        - You may modify the default values at `ROAR_simulation/carla_client/carla_settings.py`
            - `fixed_delta_seconds`
            - `no_rendering_mode`
            - `synchronoous_mode`
            
    