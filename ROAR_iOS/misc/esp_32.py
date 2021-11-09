import urllib.request
import cv2
import numpy as np
import websocket
import time
import cv2
import requests
import numpy as np
from threading import Thread


def main(host):
    ws = websocket.WebSocket()
    ws.connect(f"ws://{host}:81/control")
    while True:
        start = time.time()
        # need to make this part asynchronous
        imgResp = urllib.request.urlopen(f'http://{host}/cam-lo.jpg')
        imgNp = np.frombuffer(imgResp.read(), dtype=np.uint8)
        img = cv2.imdecode(imgNp, -1)
        # all the opencv processing is done here
        cv2.imshow('test', img)
        if ord('q') == cv2.waitKey(1):
            exit(0)

        # ws.send(f"({1500},{1500})")
        # res = input("Enter throttle, steering: ")
        # res = "1500,1500"
        # throttle, steering = res.split(",")
        # urllib.request.urlopen(f'http://{host}/cmd/({throttle},{steering})')
        time.sleep(0.025)
        print(f"Frames: {1 / (time.time() - start)}")


if __name__ == "__main__":
    host = "192.168.1.38"
    main(host=host)
