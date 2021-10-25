import cv2
import numpy as np
from typing import Tuple, Optional


def detectHSV(img: np.ndarray,
              min_red_radius=70, max_red_radius=100,
              min_green_radius=70, max_green_radius=100, should_show=False) -> Tuple[Optional[list], Optional[list]]:
    cimg = img.copy()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # color range
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 90, 90])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([90, 255, 255])

    # mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    maskr = mask2  # cv2.add(mask1, mask2)
    kernel = np.ones((5, 5), np.uint8)
    maskr = cv2.erode(maskr, kernel, iterations=1)
    maskg = cv2.inRange(hsv, lower_green, upper_green)

    # hough circle detect
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, 80, param1=80, param2=10,
                                 minRadius=min_red_radius, maxRadius=max_red_radius)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, 60,
                                 param1=50, param2=10, minRadius=min_green_radius, maxRadius=max_green_radius)
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))
        for i in r_circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))
        for i in g_circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 0, 0), 2)
    if should_show:
        cv2.imshow('detected', cimg)
        # cv2.imshow('maskr', maskr)

    return r_circles, g_circles


def main():
    cap = cv2.VideoCapture(0)
    should_continue = True
    while should_continue:
        ret, frame = cap.read()
        if ret:
            red_circles, green_circles = detectHSV(frame, should_show=True)
            has_red_circles, has_green_circles = red_circles is not None, green_circles is not None
            print(f"has_red_light = {has_red_circles} | has_green_light = {has_green_circles}")

        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            should_continue = False


if __name__ == '__main__':
    main()
