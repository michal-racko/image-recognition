import cv2
import numpy as np

from tools.video_reader import VideoReader
from tools.frame_processor import Preprocessor, ColorMask, Movement
from tools.shape_finder import PointFinder, PlayerFinder
from tools.target_finder import TargetFinder


video_path = 'data/02_20190227_190000_2_2_2.mp4'

show_points = False
show_figures = False
show_target = True


if __name__ == '__main__':
    video = VideoReader(video_path)

    first_frame = next(video)

    preprocessor = Preprocessor()

    color_mask = ColorMask()
    color_mask.set_thresholds(
        min_hsv=np.array([70, 40, 160]),
        max_hsv=np.array([120, 255, 255])
    )

    movement = Movement()

    point_finder = PointFinder()
    player_finder = PlayerFinder()
    target_finder = TargetFinder(first_frame.shape[0], first_frame.shape[1])

    for frame in video:
        preprocessor.set_frame(frame)
        tmp_frame = preprocessor.get_results()

        color_mask.set_frame(tmp_frame)
        mask = color_mask.get_results()

        movement.set_frame(tmp_frame)
        motion = movement.get_results()

        point_finder.set_frame(mask)
        point_coords = point_finder.detect_points()

        if show_points:
            for pc in point_coords:
                cv2.circle(tmp_frame, (pc[0], pc[1]), 5, (0, 0, 255), -1)

        player_coords = player_finder.cluster(point_coords)

        if show_figures:
            for pc in player_coords:
                cv2.circle(tmp_frame, (pc[0], pc[1]), 10, (0, 255, 0), -1)

        target = target_finder.search(player_coords)

        if show_target:
            if len(target) != 0:
                cv2.circle(tmp_frame, (target[0], target[1]), 25, (0, 0, 255), 2)

        cv2.imshow('mask', mask)
        cv2.imshow('recording', tmp_frame)

        cv2.waitKey(100)

    cv2.destroyAllWindows()
