import websocket
import cv2
import time
import _thread
import numpy as np

cap = cv2.VideoCapture(0)

from simple_websocket_server import WebSocketServer, WebSocket


class SimpleEcho(WebSocket):
    def handle(self):
        print(self.data)
        ret, frame = cap.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        ret, encoded_frame = cv2.imencode('.jpg', img=frame)
        self.send_message(encoded_frame)
        print("Image sent")

    def connected(self):
        print(self.address, 'connected')

    def handle_close(self):
        print(self.address, 'closed')


def main():
    server = WebSocketServer('127.0.0.1', 8009, SimpleEcho)
    server.serve_forever()
    # show_cam()

def show_cam():
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow("frame", frame)
            print(np.shape(frame))
            k = cv2.waitKey(1)
            if k == ord('q') or k == 27:
                break


if __name__ == '__main__':
    main()
