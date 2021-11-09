# Vive Tracker Setup Guide

### For first time
*Tutorials adapted from [roadtovr.com](https://www.roadtovr.com/how-to-use-the-htc-vive-tracker-without-a-vive-headset/)*
- install dependency
    - `pip install openvr`
- Configure Steam and SteamVR
    - Change `Steam -> settings -> Account -> Beta participation` from `None` to `Steam Beta Update`
    - Locate the following configuration file and open it with a text editor: 
        - `<Steam Directory>/steamapps/common/SteamVR/resources/settings/default.vrsettings`
        - change `"requireHmd" : true` to `"requireHmd" : false`
    - Restart SteamVR
- See Quick Start
    
### Quick Start
- Ensure Vive Tracker Dongle plugged in
    - It has to be in the station that comes with it, do NOT plug the usb directly into the computer, it wouldn't work
    - Make sure Vive Tracker still has battery
- Run `python vive_tracker_test.py`
- Note, if it doesn't work the first time, just give it some more try. It will work eventually.
