import json
import requests
import cv2

while True:
    r = requests.get('http://192.168.1.38:81/stream', stream=True)
    if r.encoding is None:
        r.encoding = 'utf-8'

    print(r.content)