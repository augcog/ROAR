# Jetson-Control

**Important**: Before you run anything, make sure your car is alleviated from the surface, i.e., make sure the wheels of the car does not touch anything.

**Note**: You need to first clone this repository and checkout to branch [fix-jetson-control](https://github.com/augcog/ROAR_Jetson/tree/fix_jetson_control) in your Jetson Nano (Since you are reading this markdown file, I will assume you already did so).

**Demo**: Here is the link to the video: [Jetson Control demo](https://youtu.be/1LLeuwiGe3c).

## Arduino
To run the Jetson-Control from the terminal, you need to update Arduino code first. Below are the specific instructions:

1. Connect LiPo battery and make sure the voltage is 5.1 volts 

2. Open up [`ROAR_NODE_V4.ino`](https://github.com/augcog/ROAR_Jetson/blob/fix_jetson_control/Arduino/ROAR_node/ROAR_NODE_V4/ROAR_NODE_V4.ino) inside ArduinoIDE

3. Select `Arduino Due(Programming Port)` in `Tools -> Board: "..."` (if you are not able to see `Arduino Due` in the Board, follow pert's instructions under this topic: [Installing Arduino SAM boards on Arduino IDE for Arm64](https://forum.arduino.cc/index.php?topic=572898.0); if you do not want to mess up with Jetson, you can choose to upload Arduino code from other computer)

4. Select `/dev/ttyACM0(Arduino Due(Programming Port))` in `Tools -> Port -> "..."` (the specific port name may varies, but make sure you select the Programming Port)

5. Click `Upload` button (the righthand arrow) to upload the code to the board

6. Check the front wheels of the car are in neutral position (if not, then you are in `TESTING_STATE`, change the first line of `ROAR_NODE_V4.ino` to `bool TESTING_STATE = false` and re-upload the code)

## Python

Now you are ready to run the Jetson-Control from the terminal. Below are the specific instructions:

1. Open up TRAXXAS XL5 ESC (the blue box on the side of the car)

2. Turn on your radio controller and try throttle/steering (make sure they are function correctly, if not, follow the instructions here: [How to Calibrate Traxxas Electronic Speed Controls](https://youtu.be/ix-J85uRFjE))

3. **Important: Turn off your radio controller**

4. In your terminal, make sure you are inside the repository, i.e., `cd ROAR_Jetson`

5. Run `python3 roar-vr.py -c`

6. After you see `Starting vehicle...`, type in two numbers between [-1, 1] (with space in between), and hit `enter` on your keyboard, then you should see the termianl reports your comand and the wheels of your car start changing (Note: the current program will ignore the identical commands to the previous ones)

## Known Issues
- when the input throttle is small, the car is not responding
- when the new steering angle is close to the previous one, the car is not responding



