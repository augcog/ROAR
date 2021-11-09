from serial import Serial
import logging
import time
import numpy as np
from typing import List, Tuple, Optional
try:
    from ROAR_Jetson.part import Part
    from ROAR_Jetson.jetson_vehicle import Vehicle
except:
    from part import Part
    from jetson_vehicle import Vehicle
MOTOR_MAX = 2000
MOTOR_MIN = 1000
MOTOR_NEUTRAL = 1500
THETA_MAX = 2000
THETA_MIN = 1000


class ArduinoCommandSender(Part):
    """
    Responsible for translating Agent Throttle and Steering to Servo (motor on the race car) RPM and issue the command
    """

    def __init__(self, serial: Serial, jetson_vehicle:Vehicle, min_command_time_gap: float = 0.1, agent_throttle_range: Optional[List] = None,
                 agent_steering_range: Optional[List] = None, servo_throttle_range: Optional[List] = None,
                 servo_steering_range: Optional[List] = None, throttle_reversed=False):
        """
        Initialize parameters.

        Args:
            min_command_time_gap: minimum command duration between two commands
        """
        super().__init__(name="ArduinoCMDSender")
        self.throttle_reversed = throttle_reversed
        if agent_steering_range is None:
            agent_steering_range = [-1, 1]
        if agent_throttle_range is None:
            agent_throttle_range = [-1, 1]
        if servo_throttle_range is None:
            servo_throttle_range = [MOTOR_MIN, MOTOR_MAX]
        if servo_steering_range is None:
            servo_steering_range = [THETA_MIN, THETA_MAX]

        self.serial = serial

        self.prev_throttle = 1500  # record previous throttle, set to neutral initially 
        self.prev_steering = 1500  # record previous steering, set to neutral initially
        self.last_cmd_time = None
        # time in seconds between two commands to avoid killing the arduino
        self.min_command_time_gap = min_command_time_gap
        self.agent_throttle_range = agent_throttle_range
        self.agent_steering_range = agent_steering_range
        self.servo_throttle_range = servo_throttle_range
        self.servo_steering_range = servo_steering_range

        self.jetson_vehicle: Optional[Vehicle] = jetson_vehicle

        self.forward_mode = True
        self.logger = logging.getLogger("Jetson CMD Sender")
        self.logger.debug("Jetson CMD Sender Initialized")

    def run_step(self):
        if self.jetson_vehicle is not None:
            throttle, steering = self.jetson_vehicle.throttle, self.jetson_vehicle.steering
            if self.serial is not None:
                if self.last_cmd_time is None:
                    self.last_cmd_time = time.time()
                elif time.time() - self.last_cmd_time > self.min_command_time_gap:
                    self.send_cmd(throttle=throttle, steering=steering)
                    self.last_cmd_time = time.time()

    def send_cmd(self, throttle, steering):
        """
        Step 1: maps the cmd from agent_steering_range and agent_throttle_range to servo ranges
        Args:
            throttle: new throttle, in the range of agent_throttle_range
            steering: new steering, in the range of agent_steering_range

        Returns:
            None
        """
        if self.forward_mode and throttle < 0:
            # servo
            self.logger.debug("Switching to R")
            self.forward_mode = False
        if self.forward_mode is False and throttle > 0:
            self.logger.debug("Switching to D")
            self.forward_mode = True

        if self.throttle_reversed:
            throttle = -1 * throttle
        self.prev_throttle = throttle
        self.prev_steering = steering
        throttle_send, steering_send = self.map_control(throttle, steering)
        try:
            self.send_cmd_helper(new_throttle=throttle_send, new_steering=steering_send)
        except KeyboardInterrupt as e:
            self.logger.debug("Interrupted Using Keyboard")
            exit(0)
        except Exception as e:
            self.logger.error(f"Something bad happened {e}")

    def send_cmd_helper(self, new_throttle, new_steering):
        """
        Actually send the command
        Args:
            new_throttle: new throttle, in the range of servo_throttle_range
            new_steering: new steering, in the range of servo_steering_range

        Returns:

        """
        # if self.prev_throttle != new_throttle or self.prev_steering != new_steering:
        serial_msg = '({},{})'.format(new_throttle, new_steering)
        self.logger.debug(f"Sending [{serial_msg.rstrip()}]")
        self.serial.write(serial_msg.encode('ascii'))

    def shutdown(self):
        """
        Ensure the device is shut down properly by sending neutral cmd 5 times
        Returns:

        """
        self.logger.debug('Shutting down')
        for i in range(5):
            self.logger.debug("Sending Neutral Command for safe shutdown")
            self.send_cmd_helper(new_throttle=1500, new_steering=1500)

    def map_control(self, throttle, steering) -> Tuple[int, int]:
        """
        Maps control from agent ranges to servo ranges
        Args:
            throttle: new throttle, in the range of agent_throttle_range
            steering: new steering, in the range of agent_steering_range

        Returns:
            Tuple of throttle and steering in servo ranges
        """
        return (int(np.interp(x=throttle,
                              xp=self.agent_throttle_range,
                              fp=self.servo_throttle_range)),
                int(np.interp(x=steering,
                              xp=self.agent_steering_range,
                              fp=self.servo_steering_range)))


if __name__ == '__main__':
    serial = Serial(port="COM6",
                    baudrate=9600,
                    timeout=0.5,
                    writeTimeout=0.5)
    sender = ArduinoCommandSender(serial=serial, jetson_vehicle=Vehicle())
    while True:
        val = input("Enter two values, split by [,]: ")
        throttle, steerting = val.split(",")
        sender.send_cmd(float(throttle), float(steerting))

