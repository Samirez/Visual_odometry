from initials import *
from image_and_keypoints import ImageAndKeypoints
from image_pair import ImagePair
from map import Map


class TriangulatePointsFromTwoImages():
    def __init__(self):
        pass

    @Timer(text="load_images {:.4f}")
    def load_images(self, filename_one, filename_two):
        # Load images
        self.img1 = cv2.imread(filename_one)
        self.img2 = cv2.imread(filename_two)

    def run(self, filename_one, filename_two):
        self.load_images(filename_one, filename_two)
        detector = "ORB"

        image1 = ImageAndKeypoints(detector)
        image1.set_image(self.img1)
        image1.detect_keypoints()

        image2 = ImageAndKeypoints(detector)
        image2.set_image(self.img2)
        image2.detect_keypoints()

        pair12 = ImagePair(detector)
        pair12.set_images(image1, image2)
        pair12.standard_pipeline()
        pair12.show_estimated_camera_motion()
        pair12.visualise_points_in_3d_with_plotly()

        map = Map()

        visualization12 = pair12.visualize_filtered_matches()
        cv2.imwrite("./output/images/filtered_matches.png", visualization12)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return
