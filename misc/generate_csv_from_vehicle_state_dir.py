import time
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import plotly.graph_objects as go
from glob import glob
import math
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import os
from collections import defaultdict
from collections import OrderedDict
from datetime import datetime


def generate_csv_from_vehicle_state_dir(directory: Path, regex: str = "frame_*"):
    paths = sorted(glob((directory / regex).as_posix()), key=os.path.getmtime)

    log = OrderedDict()

    for p in paths:
        data = np.load(p)
        file_name = Path(p).name.split(".")[0][6:]
        log[datetime.strptime(file_name, '%m_%d_%Y_%H_%M_%S_%f')] = data

    df = pd.DataFrame(
        data=log.values(),
        columns="x,y,z,roll,pitch,yaw,vx,vy,vz,ax,ay,az,throttle,steering".split(",")
    )
    df.insert(0, "time", log.keys())
    writer = ExcelWriter(f'{time.time()} analysis output.xlsx')
    df.to_excel(writer, 'Sheet1', index=False)
    writer.save()
    print(df.head())
    print("Data written")


if __name__ == '__main__':
    generate_csv_from_vehicle_state_dir(Path("/Users/michaelwu/Desktop/projects/ROAR/data/output/vehicle_state"))
