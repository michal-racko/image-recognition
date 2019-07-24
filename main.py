import tkinter as tk
import cv2

from tools.video_reader import VideoReader
from tools.interface import Interface


video_path = 'data/02_20190227_190000_2_2_2.mp4'

variables = []

video = VideoReader(video_path)

interface = Interface()

interface.read_config('config.json')

current_group = None

root = tk.Tk()


def run():
    video = VideoReader(video_path)

    for frame in video:
        interface.set_frame(frame)

        interface.get_target_coordinates()

        if abs(interface.global_coords[0]) >= 700:
            interface.change_azimuthal_direction()

        cv2.waitKey(80)

        root.update()


# button functions (geoups must be defined in the config file):


def select_blue():
    interface.set_target_group('blue')

    print('Blue group selected')


def select_green():
    interface.set_target_group('green')

    print('Green group selected')


if __name__ == '__main__':
    # Basic GUI:
    select_blue()

    myBtn = tk.Button(root, bd=5, text="Open the program", command=run)
    myBtn.grid()

    config = interface.config

    blue = tk.Button(root, bd=5, text='Blue', command=select_blue)
    blue.grid()

    green = tk.Button(root, bd=5, text='Green', command=select_green)
    green.grid()

    Quit = tk.Button(root, bd=5, text='Quit', command=root.destroy)
    Quit.grid()

    root.mainloop()
    cv2.destroyAllWindows()
