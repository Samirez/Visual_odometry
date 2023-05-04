from initials import *


class TrackedPoint:
    def __init__(self, point, descriptor, color, feature_id, point_id=None):
        self.point = point
        self.descriptor = descriptor
        self.color = color
        self.feature_id = feature_id
        self.point_id = point_id

    def __repr__(self):
        return repr('Point %5d %s (%8.2f %8.2f %8.2f)' % (
            self.point_id,
            self.feature_id,
            self.point[0],
            self.point[1],
            self.point[2]))
