import cv2
import numpy as np

from tools.video_reader import VideoReader
from tools.frame_processor import Preprocessor, ColorMask


video_path = 'data/02_20190227_190000_2_2_2_2.mp4'


if __name__ == '__main__':
    video = VideoReader(video_path)

    preprocessor = Preprocessor()

    color_mask = ColorMask()
    color_mask.set_thresholds(
        min_hsv=np.array([78, 158, 124]),
        max_hsv=np.array([138, 255, 255])
    )

    for frame in video:
        preprocessor.set_frame(frame)
        tmp_frame = preprocessor.get_results()

        color_mask.set_frame(tmp_frame)
        mask = color_mask.get_results()

        cv2.imshow('recording', mask)
        cv2.waitKey(80)

    cv2.destroyAllWindows()
