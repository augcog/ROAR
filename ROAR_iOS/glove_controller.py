import pygame
from ROAR_iOS.config_model import iOSConfig
from ROAR.utilities_module.vehicle_models import VehicleControl
import requests
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
from collections import deque
from tqdm import tqdm
import pandas as pd
from pygame import *
import time
import pygame.time as pygame_time


def get_acceleration(host, port) -> Optional[List[float]]:
    try:
        r = requests.get(url=f"http://{host}:{port}", timeout=0.1)
        content = r.content
        content = content.decode('utf-8')
        return [float(c) for c in content.split(',')]
    except TimeoutError as e:
        print("Timed out")
        return None
    except Exception as e:
        print(e)
        return None


def caliberateHelper(host, port, N=10) -> pd.DataFrame:
    input("Please \n"
          "   1. Wear the glove \n"
          "   2. Lay your hand flat on a flat surface \n"
          "   3. Please do not move while caliberating \n"
          "   4. Press Enter with the other hand when ready \n")

    counter = N
    lat_deque = deque(maxlen=counter)
    lon_deque = deque(maxlen=counter)
    for _ in tqdm(range(counter, 0, -1)):
        acc = get_acceleration(host, port)
        yaw, roll, pitch = acc_to_rpy(*acc)
        roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
        lat_deque.append(roll)
        lon_deque.append(pitch)
        time.sleep(0.02)
    data = np.array([lat_deque, lon_deque])
    df = pd.DataFrame(data=data.T, columns=["lat", "lon"])
    return df


def caliberate(host, port):
    is_caliberated = False
    while is_caliberated is False:
        try:
            df = caliberateHelper(host, port, 10)
        except:
            pass
        if all(df.std() < 0.15):
            return np.asarray(df.mean())
    return None


def acc_to_rpy(x, y, z):
    roll = np.arctan(x / np.sqrt(y * y + z * z))
    pitch = np.arctan(y / np.sqrt(x * x + z * z))
    yaw = np.arctan(np.sqrt(x * x + y * y) / z)
    return roll, pitch, yaw


def process_val(val, offset, min_val, max_val):
    return np.interp(np.clip(val - offset, min_val, max_val),
                     [min_val, max_val],
                     [-1, 1]
                     )


class GloveControl:
    def __init__(self, ios_config: iOSConfig):
        self.ios_config = ios_config
        self.host = ios_config.glove_ip_addr
        self.port = 81

        self.vertical_view_offset = 0
        result = caliberate(self.host, self.port)
        self.steering_offset = -2 if result is None else result[0]
        self.throttle_offset = -1
        if result is None:
            self.throttle_offset = -1
        else:
            if result[1] <= 0:
                self.throttle_offset = result[1] + 90
            else:
                self.throttle_offset = result[1] - 90

        self.start_taking_input = False

    def parse_events(self, clock: pygame_time.Clock):
        key_pressed = pygame.key.get_pressed()
        throttle = 0
        steering = 0
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT or key_pressed[K_q] or key_pressed[K_ESCAPE]:
                return False, VehicleControl()
        if key_pressed[K_o] is True:
            self.start_taking_input = ~self.start_taking_input
        acc = get_acceleration(self.host, self.port)
        if acc is not None:
            yaw, roll, pitch = acc_to_rpy(*acc)
            roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
            pitch, yaw = int(pitch), int(yaw)
            if pitch <= 0:
                new_pitch = pitch + 90
            else:
                new_pitch = pitch - 90

            steering = -1 * process_val(roll, offset=self.steering_offset, min_val=-60, max_val=60)
            throttle = process_val(new_pitch, offset=self.throttle_offset, min_val=-60, max_val=60)
        return True, VehicleControl(throttle=throttle, steering=steering)
