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
                 kernel=(9, 9)):
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

        mask = cv2.inRange(hsv, self.min_hsv, self.max_hsv)

        kernel = np.ones((5, 5), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask


class Movement(FrameProcessor):
    """
    Returns an image with nonzero values where objects have changed
    """

    def __init__(self):
        super().__init__()

        self.prior_frame = None

        self.delay = 2  # n frames
        self.i = 0

    def get_results(self):
        if self.prior_frame is None:  # return a white mask if first frame
            diff = cv2.inRange(self.frame,
                               np.array([0, 0, 0]),
                               np.array([180, 255, 255]))

        else:
            diff = cv2.subtract(self.frame,
                                self.prior_frame)
        if self.i == self.delay:
            self.prior_frame = self.frame
            self.i = 0
        else:
            self.i += 1

        return diff
