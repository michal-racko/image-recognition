import cv2

from tools.video_reader import VideoReader
from tools.interface import Interface


video_path = 'data/02_20190227_190000_2_2_2.mp4'


if __name__ == '__main__':
    video = VideoReader(video_path)

    interface = Interface()

    interface.read_config('config.json')
    interface.set_target_group('blue')

    for frame in video:
        interface.set_frame(frame)

        target_coords = interface.get_target_coordinates()

        print('target_coords', target_coords)

        interface.show_results()

        cv2.waitKey(1000)

    cv2.destroyAllWindows()
