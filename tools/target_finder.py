import numpy as np

from scipy import spatial


class TargetFinder(object):
    """
    Searches target closest to the one from the previous image.
    """

    def __init__(self):
        self.last_target = np.zeros(2)

    def search(self,
               coords: np.ndarray):
        """
        Takes coordinates of all targets found.

        Returns coordinates of the target which is closest to 
        the one selected on the previous image.
        """
        if len(coords) == 0:
            return []

        if len(coords.shape) != 2:
            return []

        tree = spatial.KDTree(coords)

        dist, indices = tree.query(self.last_target)

        target = coords[indices]

        self.last_target = target

        return target
