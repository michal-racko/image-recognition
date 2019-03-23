import cv2
import numpy as np


class PointFinder(object):
    """
    Finds coordinates of points in the image
    """

    def __init__(self):
        self.frame = None
        self.point_centres = None

        self.min_point_area = 3

    def set_frame(self,
                  frame):
        self.frame = frame

    def detect_points(self):
        contours, hierarchy = cv2.findContours(
            self.frame,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_NONE)

        x_coords = []
        y_coords = []

        for c in contours:
            area = cv2.contourArea(c)

            if area < self.min_point_area:
                continue

            mom = cv2.moments(c)

            x_coords.append(int(mom['m10'] / mom['m00']))
            y_coords.append(int(mom['m01'] / mom['m00']))

        return np.vstack((np.array(x_coords), np.array(y_coords))).T
