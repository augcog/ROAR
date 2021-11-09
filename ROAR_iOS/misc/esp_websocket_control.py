import urllib.request
import numpy as np
import cv2
import websocket
import requests
import time


def main(host):
    controlWS = websocket.WebSocket()
    controlWS.connect(f"ws://{host}:81/control")
    # ws.connect(f"ws://{host}:81/cam")
    while True:
        # r = requests.get(f"http://{host}:81/")
        # print(r.content)
        # ws.send("(1500,1400)")
        data = controlWS.recv()
        print(data)
        time.sleep(0.025)
        # print("Sent")


if __name__ == "__main__":
    host = "192.168.1.38"
    main(host=host)
