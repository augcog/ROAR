import requests
import time
from datetime import datetime
from typing import Optional, Tuple, List
import numpy as np
from collections import deque
from tqdm import tqdm
import pandas as pd


def get_acceleration(host, port) -> Optional[List[float]]:
    try:
        r = requests.get(url=f"http://{host}:{port}")
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
        df = caliberateHelper(host, port, 100)
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


if __name__ == '__main__':
    host = "192.168.1.30"
    port = 81
    result = caliberate(host, port)
    if result is not None:
        lat_mean, lon_mean = result
    steering_offset = -2 if result is None else result[0]
    throttle_offset = -1
    if result is None:
        throttle_offset = -1
    else:
        if result[1] <= 0:
            throttle_offset = result[1] + 90
        else:
            throttle_offset = result[1] - 90

    while True:
        acc = get_acceleration(host, port)
        if acc is not None:
            yaw, roll, pitch = acc_to_rpy(*acc)
            roll, pitch, yaw = np.rad2deg([roll, pitch, yaw])
            pitch, yaw = int(pitch), int(yaw)
            if pitch <= 0:
                new_pitch = pitch + 90
            else:
                new_pitch = pitch - 90

            steering = -1 * process_val(roll, offset=steering_offset, min_val=-60, max_val=60)
            throttle = process_val(new_pitch, offset=throttle_offset, min_val=-60, max_val=60)
            print(throttle, steering)
        time.sleep(0.02)
