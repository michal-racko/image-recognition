import os
import cv2
import json

import numpy as np

from tools.frame_processor import Preprocessor, ColorMask, Movement
from tools.shape_finder import PointFinder, PlayerFinder
from tools.target_finder import TargetFinder


class Interface(object):
    def __init__(self):
        self.frame = None

        self.preprocessor = Preprocessor()
        self.color_mask = ColorMask()
        self.point_finder = PointFinder()
        self.player_finder = PlayerFinder()
        self.target_finder = TargetFinder()

        self.point_coords = None
        self.player_coords = None
        self.target_coords = None

        self.config = None

        self.known_groups = None

        self.centre_coords = np.zeros(2)

        self.target_group_set = False

        self.show_points = False  # All points which passed the mask
        self.show_players = False  # All players found
        self.show_target = True  # The player to be targeted
        self.show_motion_vector = True  # Show which direction to point the camera

    def set_frame(self, frame):
        self.frame = frame

        image_dimensions = self.frame.shape

        self.centre_coords = np.array(
            [int(image_dimensions[1] / 2.), int(image_dimensions[0] / 2.)])

    def read_config(self, path):
        if not os.path.isfile(path):
            raise Exception('No config found at: ' + path)

        with open(path, 'r') as config_file:
            json_string = config_file.read()

            self.config = json.loads(json_string)

        self.known_groups = self.config['groups']

    def set_target_group(self, group_name: str):
        """
        Sets which group of players are to be targetet.
        Each group must be defined in the config.
        """
        if self.config is None:
            raise Exception('A config file must be read first')

        if group_name not in self.known_groups:
            raise Exception('Group ' + group_name + ' not found in the config')

        self.color_mask.set_thresholds(
            min_hsv=np.array(self.known_groups[group_name]['min_hsv']),
            max_hsv=np.array(self.known_groups[group_name]['max_hsv'])
        )

        self.target_group_set = True

    def _find_players(self):
        """
        Finds positions of players within the current frame
        """
        self.preprocessor.set_frame(self.frame)
        tmp_frame = self.preprocessor.get_results()

        self.color_mask.set_frame(tmp_frame)
        mask = self.color_mask.get_results()

        self.point_finder.set_frame(mask)
        self.point_coords = self.point_finder.detect_points()

        self.player_coords = self.player_finder.cluster(self.point_coords)

    def _find_target(self):
        """
        Finds coordinates of a target. The coordinate
        system starts at pixel 0, 0
        """
        self.target_coords = self.target_finder.search(self.player_coords)

    def get_target_coordinates(self) -> np.ndarray:
        if not self.target_group_set:
            raise Exception('Target group must be defined first')

        self._find_players()
        self._find_target()

        if self.target_coords is None or len(self.target_coords) == 0:
            self.target_coords = None
            return []

        target_motion_vector = np.array([
            self.target_coords[0] - self.centre_coords[0],
            self.centre_coords[1] - self.target_coords[1]
        ])

        target_motion_vector = target_motion_vector / \
            np.sqrt(sum(target_motion_vector ** 2))

        n_steps = self.config['n_steps']

        target_motion_vector[0] = int(target_motion_vector[0] * n_steps)
        target_motion_vector[1] = int(target_motion_vector[1] * n_steps)

        return target_motion_vector

    def show_results(self):
        """
        Shows results in the current frame.
        """
        if self.show_players and self.player_coords is not None and len(self.player_coords) != 0:
            for pc in self.player_coords:
                cv2.circle(self.frame, (pc[0], pc[1]), 10, (0, 255, 0), -1)

        if self.show_points and self.point_coords is not None and len(self.point_coords) != 0:
            for pc in self.point_coords:
                cv2.circle(self.frame, (pc[0], pc[1]), 5, (0, 0, 255), -1)

        if self.show_target and self.target_coords is not None and len(self.target_coords) != 0:
            cv2.circle(
                self.frame, (self.target_coords[0], self.target_coords[1]), 25, (0, 0, 255), 2)

        if self.show_motion_vector and self.target_coords is not None:
            cv2.line(self.frame, (self.centre_coords[0], self.centre_coords[1]),
                     (self.target_coords[0], self.target_coords[1]), (0, 0, 255), 2)

        cv2.circle(self.frame, (0, 0), 25, (0, 255, 0), -1)

        cv2.imshow('Targeting', self.frame)
