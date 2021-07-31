from pygame import *
import logging
import pygame
from ROAR.utilities_module.vehicle_models import VehicleControl


class ManualControl:
    def __init__(self, throttle_increment=0.05, steering_increment=0.05):
        self.logger = logging.getLogger(__name__)
        self._steering_increment = steering_increment
        self._throttle_increment = throttle_increment

        try:
            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.logger.info(f"Joystick [{self.joystick.get_name()}] detected, Using Joytick")
            self.use_joystick = True
        except Exception as e:
            self.logger.info("No joystick detected. Plz use your keyboard instead")

        self.steering = 0.0
        self.throttle = 0.0
        self.logger.debug("Keyboard Control Initiated")

    def parse_events(self, clock: pygame.time.Clock):
        """
        parse a keystoke event
        Args:
            clock: pygame clock

        Returns:
            Tuple bool, and vehicle control
            boolean states whether quit is pressed. VehicleControl by default has throttle = 0, steering =
        """
        events = pygame.event.get()
        key_pressed = pygame.key.get_pressed()
        for event in events:
            if event.type == pygame.QUIT or key_pressed[K_q] or key_pressed[K_ESCAPE]:
                return False, VehicleControl()

        if self.use_joystick:
            self._parse_joystick()
        else:
            self._parse_vehicle_keys(key_pressed)
        return True, VehicleControl(throttle=self.throttle, steering=self.steering)

    def _parse_joystick(self):
        self.steering = self.joystick.get_axis(0)
        self.throttle = -self.joystick.get_axis(1)

    def _parse_vehicle_keys(self, keys):
        """
        Parse a single key press and set the throttle & steering
        Args:
            keys: array of keys pressed. If pressed keys[PRESSED] = 1
        Returns:
            None
        """
        if keys[K_UP] or keys[K_w]:
            self.throttle = min(self.throttle + self._throttle_increment, 1)

        elif keys[K_DOWN] or keys[K_s]:
            self.throttle = max(self.throttle - self._throttle_increment, -1)
        else:
            self.throttle = 0

        if keys[K_LEFT] or keys[K_a]:
            self.steering = max(self.steering - self._steering_increment, -1)

        elif keys[K_RIGHT] or keys[K_d]:
            self.steering = min(self.steering + self._steering_increment, 1)
        else:
            self.steering = 0

        self.throttle, self.steering = round(self.throttle, 5), round(self.steering, 5)
