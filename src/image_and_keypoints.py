from initials import *


class ImageAndKeypoints():
    def __init__(self, detector_name):
        if detector_name == "ORB":
            # Use ORB features
            self.detector = cv2.ORB_create()
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use SIFT features
            self.detector = cv2.SIFT_create()
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Camera parameters are taken from a manual calibration 
        # of the used camera.
        # See the file ../input/camera_calibration_extended.txt
        self.cameraMatrix = np.array([[2676.1051390718389, 0., -35.243952918157035],
                                      [0., 704.01349597, -279.58562078697361],
                                      [0., 0., 1.]])
        self.distCoeffs = np.array([[0.0097935857180804498,
                                     -0.021794052829051412,
                                     0.0046443590741258711,
                                     -0.0045664024579022498,
                                     0.017776502734846815]])

        # Values from ../input/toys2/calibration.xml
        self.cameraMatrix = np.array([[835.69, 0., 1008 / 2 + 61.6],
                                      [0., 827, 756 / 2 - 9.4],
                                      [0., 0., 1.]])

        # Factor for rescaling images (usually used to downscale images)
        self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1

    def set_image(self, image):
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dim = (width, height)

        self.image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    def detect_keypoints(self):
        # find keypoints and descriptors with the selected feature detector
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.image, None)
        self.kp_colors = []
        for kp in self.keypoints:
            point = kp.pt
            color = self.image[int(point[1]), int(point[0])]
            self.kp_colors.append(color)
