from typing import Any
import matplotlib.pyplot as plt
from ROAR.perception_module.detector import Detector
import numpy as np
import cv2
from ROAR.agent_module.agent import Agent
import collections, math


class LaneDetector(Detector):
    def __init__(self, agent: Agent, **kwargs):
        super().__init__(agent, **kwargs)

    def run_in_series(self, **kwargs) -> Any:
        im = self.agent.front_rgb_camera.data
        # it's my code
        pt1_sum_ri = (0, 0)
        pt2_sum_ri = (0, 0)
        pt1_avg_ri = (0, 0)
        count_posi_num_ri = 0

        pt1_sum_le = (0, 0)
        pt2_sum_le = (0, 0)
        pt1_avg_le = (0, 0)

        count_posi_num_le = 0
        size_im = cv2.resize(im, dsize=(640, 480))  # VGA resolution
        roi = size_im[240:480, 108:532]  # [380:430, 330:670]   [y:y+b, x:x+a]
        roi_im = cv2.resize(roi, (424, 240))  # (a of x, b of y)
        Blur_im = cv2.bilateralFilter(roi_im, d=-1, sigmaColor=5, sigmaSpace=5)

        edges = cv2.Canny(Blur_im, 50, 100)
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180.0, threshold=25, minLineLength=10, maxLineGap=20)

        if lines is None:  # in case HoughLinesP fails to return a set of lines
            # make sure that this is the right shape [[ ]] and ***not*** []
            lines = [[0, 0, 0, 0]]
        else:
            for line in lines:

                x1, y1, x2, y2 = line[0]

                if x2 == x1:
                    a = 1
                else:
                    a = x2 - x1

                b = y2 - y1

                radi = b / a  # 라디안 계산
                # print('radi=', radi)

                theta_atan = math.atan(radi) * 180.0 / math.pi
                # print('theta_atan=', theta_atan)

                pt1_ri = (x1 + 108, y1 + 240)
                pt2_ri = (x2 + 108, y2 + 240)
                pt1_le = (x1 + 108, y1 + 240)
                pt2_le = (x2 + 108, y2 + 240)

                if theta_atan > 20.0 and theta_atan < 90.0:
                    # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 255, 0), 2)
                    # print('live_atan=', theta_atan)

                    count_posi_num_ri += 1

                    pt1_sum_ri = self.sumMatrix(pt1_ri, pt1_sum_ri)
                    # pt1_sum = pt1 + pt1_sum
                    # print('pt1_sum=', pt1_sum)

                    pt2_sum_ri = self.sumMatrix(pt2_ri, pt2_sum_ri)
                    # pt2_sum = pt2 + pt2_sum
                    # print('pt2_sum=', pt2_sum)

                if theta_atan < -20.0 and theta_atan > -90.0:
                    # cv2.line(size_im, (x1+108, y1+240), (x2+108, y2+240), (0, 0, 255), 2)
                    # print('live_atan=', theta_atan)

                    count_posi_num_le += 1

                    pt1_sum_le = self.sumMatrix(pt1_le, pt1_sum_le)
                    # pt1_sum = pt1 + pt1_sum
                    # print('pt1_sum=', pt1_sum)

                    pt2_sum_le = self.sumMatrix(pt2_le, pt2_sum_le)
            pt1_avg_ri = pt1_sum_ri // np.array(count_posi_num_ri)
            pt2_avg_ri = pt2_sum_ri // np.array(count_posi_num_ri)
            pt1_avg_le = pt1_sum_le // np.array(count_posi_num_le)
            pt2_avg_le = pt2_sum_le // np.array(count_posi_num_le)

            x1_avg_ri, y1_avg_ri = pt1_avg_ri
            x2_avg_ri, y2_avg_ri = pt2_avg_ri

            a_avg_ri = ((y2_avg_ri - y1_avg_ri) / (x2_avg_ri - x1_avg_ri))
            b_avg_ri = (y2_avg_ri - (a_avg_ri * x2_avg_ri))

            pt2_y2_fi_ri = 480

            if a_avg_ri > 0:
                pt2_x2_fi_ri = int((pt2_y2_fi_ri - b_avg_ri) // a_avg_ri)
            else:
                pt2_x2_fi_ri = 0

            pt2_fi_ri = (pt2_x2_fi_ri, pt2_y2_fi_ri)
            x1_avg_le, y1_avg_le = pt1_avg_le
            x2_avg_le, y2_avg_le = pt2_avg_le

            a_avg_le = ((y2_avg_le - y1_avg_le) / (x2_avg_le - x1_avg_le))
            b_avg_le = (y2_avg_le - (a_avg_le * x2_avg_le))

            pt1_y1_fi_le = 480
            if a_avg_le < 0:
                pt1_x1_fi_le = int((pt1_y1_fi_le - b_avg_le) // a_avg_le)
            else:
                pt1_x1_fi_le = 0

            pt1_fi_le = (pt1_x1_fi_le, pt1_y1_fi_le)
            # print('pt1_fi_le=', pt1_fi_le)

            # print('pt1_avg_ri=', pt1_sum_ri)
            # print('pt2_fi_ri=', pt2_fi_ri)
            # print('pt1_fi_le=', pt1_fi_le)
            # print('pt2_avg_le=', pt2_sum_le)
            #################################################

            #################################################
            # lane painting
            # right-----------------------------------------------------------
            # cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_avg_ri), (0, 255, 0), 2) # right lane
            cv2.line(size_im, tuple(pt1_avg_ri), tuple(pt2_fi_ri), (0, 255, 0), 2)  # right lane
            # left-----------------------------------------------------------
            # cv2.line(size_im, tuple(pt1_avg_le), tuple(pt2_avg_le), (0, 255, 0), 2) # left lane
            cv2.line(size_im, tuple(pt1_fi_le), tuple(pt2_avg_le), (0, 255, 0), 2)  # left lane
            #################################################

            #################################################
            # possible lane
            # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
            # cv2.fillConvexPoly(size_im, FCP, color=(255, 242, 213)) # BGR
            #################################################
            # FCP_img = np.zeros(shape=(480, 640, 3), dtype=np.uint8) + 0
            # # FCP = np.array([pt1_avg_ri, pt2_avg_ri, pt1_avg_le, pt2_avg_le])
            # # FCP = np.array([(100,100), (100,200), (200,200), (200,100)])
            # FCP = np.array([pt2_avg_le, pt1_fi_le, pt2_fi_ri, pt1_avg_ri])
            # cv2.fillConvexPoly(FCP_img, FCP, color=(255, 242, 213))  # BGR
            # alpha = 0.9
            # size_im = cv2.addWeighted(size_im, alpha, FCP_img, 1 - alpha, 0)
            cv2.imshow("im", size_im)
            cv2.waitKey(1)
    def run_in_threaded(self, **kwargs):
        pass

    @staticmethod
    def sumMatrix(A, B):
        A = np.array(A)
        B = np.array(B)
        answer = A + B
        return answer.tolist()
