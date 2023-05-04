from initials import *


class Frame:
    """
    Class / structure for saving information about a single frame.
    """

    def __init__(self, image=None):
        self.image = image
        self.id = None
        self.keypoints = None
        self.descriptors = None
        self.features = None

    def __repr__(self):
        return repr('Frame %d' % (
            self.id))
