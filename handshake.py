import logging
from ROAR.utilities_module.utilities import get_ip
import qrcode
import cv2
import numpy as np
import socket


def showIPUntilAckUDP():
    img = np.array(qrcode.make(f"{get_ip()}").convert('RGB'))
    success = False
    addr = None

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(0.1)

    try:
        s.bind((get_ip(), 8008))
        while success is False:
            try:
                cv2.imshow("Scan this code to connect to phone", img)
                k = cv2.waitKey(1) & 0xff
                seg, addr = s.recvfrom(1024)  # this command might timeout

                if k == ord('q') or k == 27:
                    s.close()
                    break
                addr = addr
                success = True
                for i in range(10):
                    print("data sent")
                    s.sendto(b"hi", addr)
            except socket.timeout as e:
                logging.info(f"Please tap on the ip address to scan QR code. ({get_ip()}:{8008}). {e}")
    except Exception as e:
        logging.error(f"Unable to bind socket: {e}")
    finally:
        s.close()
        cv2.destroyWindow("Scan this code to connect to phone")
    return success, addr[0]


logging.basicConfig(format='%(levelname)s - %(asctime)s - %(name)s '
                           '- %(message)s',
                    datefmt="%H:%M:%S",
                    level=logging.DEBUG)
print(showIPUntilAckUDP())