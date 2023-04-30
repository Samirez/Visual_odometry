import cv2
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import time
import plotly.graph_objects as go
from codetiming import Timer


# Code borrowed from the following sources
# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html
# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/master/da/de9/tutorial_py_epipolar_geometry.html
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# https://www.morethantechnical.com/2016/10/17/structure-from-motion-toy-lib-upgrades-to-opencv-3/


def isRotationMatrix(R) :
    # Checks if a matrix is a valid rotation matrix.
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R) :
    """Calculates rotation matrix to euler angles

    The result is the same as MATLAB except the order
    of the euler angles ( x and z are swapped ).
    """

    assert(isRotationMatrix(R))
    
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
    
    singular = sy < 1e-6

    if not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])


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
        self.cameraMatrix = np.array([[2676.1051390718389,   0.,             -35.243952918157035],
                                      [   0.,              704.01349597,    -279.58562078697361],
                                      [   0.,                0.,              1.]])
        self.distCoeffs = np.array([[ 0.0097935857180804498, 
                                     -0.021794052829051412, 
                                      0.0046443590741258711, 
                                     -0.0045664024579022498, 
                                      0.017776502734846815]])

        # Values from ../input/toys2/calibration.xml
        self.cameraMatrix = np.array([[835.69,   0.,   1008 / 2 + 61.6 ],
                                [  0.,         827, 756 / 2 - 9.4 ],
                                [0., 0., 1.]])


        # Factor for rescaling images (usually used to downscale images)
        self.scale_factor = 1
        self.cameraMatrix *= self.scale_factor
        self.cameraMatrix[2, 2] = 1


    def set_image(self, image):
        width = int(image.shape[1] * self.scale_factor)
        height = int(image.shape[0] * self.scale_factor)
        dim = (width, height)
          
        self.image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


    def detect_keypoints(self):
        # find keypoints and descriptors with the selected feature detector
        self.keypoints, self.descriptors = self.detector.detectAndCompute(self.image, None)
        self.kp_colors = []
        for kp in self.keypoints:
            point = kp.pt
            color = self.image[int(point[1]), int(point[0])]
            self.kp_colors.append(color)


class ImagePair():
    def __init__(self, detector_name):
        if detector_name == "ORB":
            # Use ORB features
            self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        else:
            # Use SIFT features
            self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)


    def set_images(self, img1, img2):
        self.image1 = img1
        self.image2 = img2

 
    @Timer(text="match_detected_keypoints {:.4f}")
    def match_detected_keypoints(self):
        # Match descriptors.
        self.matches = self.bf.match(self.image1.descriptors, self.image2.descriptors)

        points1_temp = []
        points2_temp = []
        match_indices_temp = []
        color1_temp = []
        # ratio test as per Lowe's paper
        for idx, m in enumerate(self.matches):
            points2_temp.append(self.image2.keypoints[m.trainIdx].pt)
            points1_temp.append(self.image1.keypoints[m.queryIdx].pt)
            match_indices_temp.append(idx)
            color1_temp.append(self.image1.kp_colors[m.queryIdx])

        self.points1_1to2 = np.float32(points1_temp)
        self.points2_1to2 = np.float32(points2_temp)
        self.kp_colors = np.uint8(color1_temp)
        self.match_indices_1to2 = np.int32(match_indices_temp)


    @Timer(text="determine_essential_matrix {:.4f}")
    def determine_essential_matrix(self):
        confidence = 0.99
        ransacReprojecThreshold = 1
        self.essential_matrix_1to2, mask = cv2.findEssentialMat(
                self.points1_1to2, 
                self.points2_1to2, 
                self.image1.cameraMatrix, 
                cv2.FM_RANSAC, 
                confidence,
                ransacReprojecThreshold)

        # We select only inlier points
        self.points1_1to2 = self.points1_1to2[mask.ravel()==1]
        self.points2_1to2 = self.points2_1to2[mask.ravel()==1]
        self.match_indices_1to2 = self.match_indices_1to2[mask.ravel()==1]
        self.kp_colors = self.kp_colors[mask.ravel() == 1]


    @Timer(text="estimate_camera_movement {:.4f}")
    def estimate_camera_movement(self):
        retval, self.R, self.t, mask = cv2.recoverPose(self.essential_matrix_1to2, self.points1_1to2, self.points2_1to2, self.image1.cameraMatrix)


    @Timer(text="reconstruct_3d_points {:.4f}")
    def reconstruct_3d_points(self):
        self.null_projection_matrix = self.image1.cameraMatrix @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.projection_matrix = self.image1.cameraMatrix @ np.hstack((self.R.T, self.t))

        self.points3d_reconstr = cv2.triangulatePoints(
                self.projection_matrix, self.null_projection_matrix,
                self.points1_1to2.T, self.points2_1to2.T) 
        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]


    def get_validated_matches(self):
        # Construct list with all points from matches.
        self.points1_temp = []
        self.points2_temp = []

        self.filtered_matches_1to2 = []
        for index in self.match_indices_1to2:
            m = self.matches[index]
            self.points2_temp.append(self.image2.keypoints[m.trainIdx].pt)
            self.points1_temp.append(self.image1.keypoints[m.queryIdx].pt)
            self.filtered_matches_1to2.append(m)

        self.points1_filtered = np.float32(self.points1_temp)
        self.points2_filtered = np.float32(self.points2_temp)


    def show_estimated_camera_motion(self):
        print("Estimated direction of translation between images")
        print(self.t)
        print("Estimated rotation of camera between images in degrees")
        print(rotationMatrixToEulerAngles(self.R) * 180 / math.pi)


    def visualize_filtered_matches(self):
        visualization = cv2.drawMatches(self.image1.image,
            self.image1.keypoints,
            self.image2.image,
            self.image2.keypoints,
            self.filtered_matches_1to2,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return visualization


    def visualise_points_in_3d_with_plotly(self):
        # The reconstructed points appear to be mirrored.
        # This is changed here prior to visualization
        # TODO: Figure out why this is the case?
        pitch = 0*19.7 / 180 * np.pi
        transform = np.array(
                [[1, 0, 0, 0],
                 [0, np.cos(pitch), np.sin(pitch), 0], 
                 [0, -np.sin(pitch), np.cos(pitch), 0], 
                 [0, 0, 0, 1]])

        print("self.points3d_reconstr.shape")
        print(self.points3d_reconstr.shape)
        point3dtemp = transform @ self.points3d_reconstr

        xs = point3dtemp[0]
        ys = point3dtemp[1]
        zs = point3dtemp[2]

        formatted_colors = [f'rgb({R}, {G}, {B})' for R, G, B in self.kp_colors]
        plotlyfig = go.Figure(data=[go.Scatter3d(
            x=xs, y=ys, z=-zs, mode='markers', 
            marker=dict(
                    size=3,
                    color=formatted_colors,                # set color to an array/list of desired values
                    colorscale='Viridis',   # choose a colorscale
                    opacity=0.8
                )
            )],
            layout={
                "title": "3D points from triangulation"}
            )

        # Show estimated camera positions
        camera2x = self.t[0, 0]
        camera2y = self.t[1, 0]
        camera2z = self.t[2, 0]
        plotlyfig.add_trace(
            go.Scatter3d(
                x=[0, camera2x], 
                y=[0, camera2y], 
                z=[0, -camera2z], mode='markers', 
                marker=dict(
                        size=10,
                        color=['red', 'green']
                    )
                )
            )

        xptp = np.hstack((xs, 0, camera2x)).ptp()
        yptp = np.hstack((ys, 0, camera2y)).ptp()
        zptp = np.hstack((zs, 0, -camera2z)).ptp()
        plotlyfig.update_layout(scene=dict(
            aspectmode='data',
            aspectratio=go.layout.scene.Aspectratio(
               x=xptp, y=xptp/yptp, z=zptp/xptp)
            ))

        plotlyfig.show()


    @Timer(text="standard_pipeline {:.4f}")
    def standard_pipeline(self):
        self.match_detected_keypoints()
        self.determine_essential_matrix()
        self.estimate_camera_movement()
        self.get_validated_matches()
        self.reconstruct_3d_points()



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

        visualization12 = pair12.visualize_filtered_matches()
        cv2.imwrite("./output/images/filtered_matches.png", visualization12)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return 


# Parse command line args



# Notes to my self
# The order of images seems to be important for the reconstruction. This should be investigated.

    
def main():
    
    filelocation = "./output/images/"
    frame1 = filelocation + 'DJI_0199_1200.jpg'
    frame2 = filelocation + 'DJI_0199_1250.jpg'

    TPFTI = TriangulatePointsFromTwoImages()
    TPFTI.run(frame1, frame2)
    
    
    
    # cv2.imshow("frame1",frame1)
    # cv2.imshow("frame2",frame2)
    # cv2.triangulatePoints(frame1, frame2)



if __name__ == "__main__":
    main()
