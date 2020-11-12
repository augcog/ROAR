# MIT License
# Copyright (c) 2019 JetsonHacks
# See LICENSE for OpenCV license and additional information

# https://docs.opencv.org/3.3.1/d7/d8b/tutorial_py_face_detection.html
# On the Jetson Nano, OpenCV comes preinstalled
# Data files are in /usr/sharc/OpenCV
import numpy as np
import cv2
import pyrealsense2 as rs
from multiprocessing import Process

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 30fps 
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen

def gstreamer_pipeline(capture_width=3280, capture_height=2464, output_width=224, output_height=224, framerate=21, flip_method=0) :   
        return ('nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=%d, height=%d, format=(string)NV12, framerate=(fraction)%d/1 ! nvvidconv flip-method=%d ! nvvidconv ! video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! videoconvert ! appsink' % (
                capture_width, capture_height, framerate, flip_method, output_width, output_height))

client_ip='192.168.1.74'

def send():
    cap_send =cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width =1280,
                capture_height =720,
                output_width=1280,
                output_height=720,
                framerate=30,
                flip_method=0),
            cv2.CAP_GSTREAMER)
    
    #client_ip='10.0.0.79'
    out_send = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720, framerate=(fraction)30/1 ! videoconvert ! video/x-raw, format=(string)I420 ! omxh264enc control-rate=2 bitrate=8000000 ! video/x-h264, stream-format=byte-stream ! rtph264pay mtu=1400 ! udpsink host=%s port=5000 sync=false async=false'%(client_ip),cv2.CAP_GSTREAMER,0,30,(1280,720), True)

    if not cap_send.isOpened() or not out_send.isOpened():
        print('VideoCapture or VideoWriter not opened')
        exit(0)

    while True:
        ret,frame = cap_send.read()

        if not ret:
            print('empty frame')
            break

        out_send.write(frame)

        #cv2.imshow('send', frame)
        #if cv2.waitKey(1)&0xFF == ord('q'):
            #break

    cap_send.release()
    out_send.release()

def sendrs():
    out_send = cv2.VideoWriter('appsrc ! videoconvert ! video/x-raw, format=(string)BGRx, width=(int)1280, height=(int)720, framerate=(fraction)30/1 ! videoconvert ! video/x-raw, format=(string)I420 ! omxh264enc control-rate=2 bitrate=4000000 ! video/x-h264, stream-format=byte-stream ! rtph264pay mtu=1400 ! udpsink host=%s port=5001 sync=false async=false'%(client_ip),cv2.CAP_GSTREAMER,0,30,(1280,720), True)

    pipe = rs.pipeline()
    cfg = rs.config()
    #cfg.enable_stream(rs.stream.gyro)
    cfg.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30) # color camera
    pipe.start(cfg)
    while True:
        frames = pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        #cv2.imshow('send1', img)
        #if cv2.waitKey(1)&0xFF == ord('q'):
            #break

        out_send.write(img)
    out_send.release()

if __name__ == '__main__':
    #send()
    #sendrs()
    s = Process(target=send)
    r = Process(target=sendrs)
    s.start()
    r.start()
    s.join()
    r.join()

    cv2.destroyAllWindows()
