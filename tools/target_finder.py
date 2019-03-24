import numpy as np

from scipy import spatial


class TargetFinder(object):
    def __init__(self):
        self.last_target = np.zeros(2)

    def search(self,
               coords: np.ndarray):
        if len(coords) == 0:
            return []

        tree = spatial.KDTree(coords)

        dist, indices = tree.query(self.last_target)

        target = coords[indices]

        self.last_target = target

        return target
