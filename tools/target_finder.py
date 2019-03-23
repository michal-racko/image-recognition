import numpy as np

from scipy import spatial


class TargetFinder(object):
    def __init__(self,
                 image_size_x: int,
                 image_size_y: int):
        self.last_target = np.array(
            [int(image_size_x / 2.), int(image_size_y / 2.)])

    def search(self,
               coords: np.ndarray):
        if len(coords) == 0:
            return []

        tree = spatial.KDTree(coords)

        dist, indices = tree.query(self.last_target)

        target = coords[indices]

        self.last_target = target

        return target
