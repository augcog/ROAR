# Robot Open Autonomous Racing (ROAR)
### To Contribute
- Please click the [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) button on the upper right corner and submit a pull request to master branch.
- Please follow suggested guidelines on Pull Request. 

### Quick Start
1. clone the repo
    - `git clone --recursive https://github.com/YOURUSERNAME/ROAR.git`

2. Create virtual environment with *python3.7*
    - `conda create -n ROAR python=3.7`
    - `conda activate ROAR`

3. Install Dependency
    - For simulator
        - `cd ROAR_Sim`
        - `pip install -r requirements.txt`
        - Download Simulator distribution for your OS
            - https://drive.google.com/drive/folders/13JSejJED31cZHBbfIz_gyxxPmiqABOJj?usp=sharing
    - For actual vehicle wired to your computer
        - `cd ROAR_Jetson`
        - `pip install -r requirements.txt`
    - For actual vehicle running on Jetson Nano
        - `cd ROAR_Jetson`
        - `sudo apt-get install python-dev libsdl1.2-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev`
        - `pip3 install -r requirements_jetson_nano.txt`


4. Enjoy
    - For Simulator
        - `python runner_sim.py`
    - For physical car
        - `python runner_jetson.py` or `python3 runner_jetson.py`
