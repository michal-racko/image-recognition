import cv2
import numpy as np

from abc import ABCMeta, abstractmethod


class FrameProcessor(object):
    """
    Abstraction on image processing
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        self.frame = None

    def set_frame(self, frame):
        self.frame = frame

    @abstractmethod
    def get_results(self):
        pass


class Preprocessor(FrameProcessor):
    """
    Performs first changes to the frame
    """

    def __init__(self,
                 kernel=(5, 5)):
        super().__init__()

        self.kernel = kernel

    def get_results(self):
        return cv2.GaussianBlur(self.frame, self.kernel, 0)


class ColorMask(FrameProcessor):
    """
    Returns a mask of pixels with hsv values in the given range
    """

    def __init__(self):
        super().__init__()

        self.min_hsv = None
        self.max_hsv = None

    def set_thresholds(self,
                       min_hsv: np.ndarray,
                       max_hsv: np.ndarray):
        """
        Redefines the thresholds
        """
        self.min_hsv = min_hsv
        self.max_hsv = max_hsv

    def get_results(self):
        if self.min_hsv is None or self.max_hsv is None:
            raise Exception('Thresholds must be defined first')

        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)

        return cv2.inRange(hsv, self.min_hsv, self.max_hsv)
