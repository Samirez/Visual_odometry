from initials import *


class Observation:
    def __init__(self, point_id, camera_id, image_coordinates):
        self.point_id = point_id
        self.camera_id = camera_id
        self.image_coordinates = image_coordinates

    def __repr__(self):
        return repr("Observation - point %d - camera %d (%f %f)" % (
            self.point_id,
            self.camera_id,
            self.image_coordinates[0],
            self.image_coordinates[1]))
