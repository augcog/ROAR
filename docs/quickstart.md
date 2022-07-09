*Note: for Mac users, please dual boot as Windows 10.*

0.  Download Simulator distribution for your OS:
    - [evGrandPrix Link](https://drive.google.com/drive/u/1/folders/1uI-GlpC3h81EelPpS0xqDb8VWKf8sBwk)
    - [Berkeley Major Link](https://drive.google.com/file/d/14Gsemfq_nL8Ga1Nf8-5DVQYd4C4nRHM6/view)
    - [Berkeley Minor Link](https://drive.google.com/drive/folders/1ejKIOp8_vXaTroA7prcCDrfQet9WL-oD?usp=sharing)
    - [Easy map Link](https://drive.google.com/drive/folders/13JSejJED31cZHBbfIz_gyxxPmiqABOJj?usp=sharing)
         
1.  Fork the Repo
    - Please click the [Fork](https://docs.github.com/en/free-pro-team@latest/github/getting-started-with-github/fork-a-repo) button on the upper right corner. Note you would need to sign in to your GitHub account first, and your account will be referred to below as `YOURUSERNAME`.

2. clone the repo
    - `git clone --recursive https://github.com/YOURUSERNAME/ROAR.git`
        - replace `YOURUSERNAME` with your github username.

3. Create virtual environment with *python3.8*
    - To run Berkeley_Major_Map and evGrandPrix_Map, create *python3.8* environment:
        - `conda create -n ROAR_major python=3.8`
        - `conda activate ROAR_major`
    - To run Berkeley_Minor_Map and Easy_Map, create *python3.7* environment:
        - `conda create -n ROAR_minor python=3.7`
        - `conda activate ROAR_minor`
    
4. Install Dependency
    - General Dependency
        - `pip install -r requirements.txt` in the ROAR folder
        - If any specific package version cannot be installed, install a downgrade version or install lastest avaiable version:
            - `pip install -r requirements_general.txt`

5. To run simulator:
    - open the carla map
    - run simulation script runner_sim.py:
        - `python runner_sim.py`




**   To run on a physcial car (The jetson code is out of date and out of support, it is included here just for reference):    
        
- For actual vehicle wired to your computer
    - `cd ROAR_Jetson`
    - `pip install -r requirements.txt`
        
- For actual vehicle running on Jetson Nano
    - `cd ROAR_Jetson`
    - `sudo apt-get install python-dev libsdl1.2-dev libsdl-image1.2-dev libsdl-mixer1.2-dev libsdl-ttf2.0-dev libsdl1.2-dev libsmpeg-dev python-numpy subversion libportmidi-dev ffmpeg libswscale-dev libavformat-dev libavcodec-dev libfreetype6-dev`
    - `pip3 install -r requirements_jetson_nano.txt`
        
- Run script:
    - `python runner_jetson.py` or `python3 runner_jetson.py`




