import cv2
import time


class VideoReader(object):
    """
    An iterator for the given video file
    """

    def __init__(self,
                 filepath: str,
                 scaling_factor=2):
        self.cap = cv2.VideoCapture(filepath)

        self.scaling_factor = scaling_factor

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()

        if ret:
            return cv2.resize(frame,
                              None,
                              fx=self.scaling_factor,
                              fy=self.scaling_factor,
                              interpolation=cv2.INTER_AREA)
        else:
            raise StopIteration
