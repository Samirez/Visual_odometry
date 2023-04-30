from libs import *
from frame import Frame

class FrameGenerator():
    def __init__(self, detector):
        self.next_image_counter = 0
        self.detector = detector

    def make_frame(self, image) -> Frame:
        """
        Create a frame by extracting features from the provided image.

        This method should only be called once for each image.
        Each of the extracted features will be assigned a unique
        id, whic will help with tracking of individual features 
        later in the pipeline.
        """
        # Create a frame and assign it a unique id.
        frame = Frame(image)
        frame.id = self.next_image_counter
        self.next_image_counter += 1

        # Extract features
        frame.keypoints, frame.descriptors = self.detector.detectAndCompute(
                frame.image, None)
        enumerated_features = enumerate(
                zip(frame.keypoints, frame.descriptors))

        # Save features in a list with the following elements
        # keypoint, descriptor, feature_id
        # where the feature_id refers to the image id and the feature 
        # number.
        frame.features = [Feature(keypoint, descriptor, (frame.id, idx)) 
                for (idx, (keypoint, descriptor))
                in enumerated_features]

        return frame