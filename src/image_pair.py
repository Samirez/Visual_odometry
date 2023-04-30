from initials import *


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

# class ImagePair():
#     """
#     Class for working with image pairs.
#     """
#     def __init__(self, frame1, frame2, matcher, camera_matrix):
#         self.frame1 = frame1
#         self.frame2 = frame2
#         self.matcher = matcher
#         self.camera_matrix = camera_matrix


#     def match_features(self):
#         temp = self.matcher.match(
#                 self.frame1.descriptors, 
#                 self.frame2.descriptors)
#         # Make a list with the following values
#         # - feature 1 id
#         # - feature 2 id
#         # - image coordinate 1
#         # - image coordinate 2
#         # - match distance
#         self.raw_matches = [
#                 Match(self.frame1.features[match.queryIdx].feature_id, 
#                     self.frame2.features[match.trainIdx].feature_id,
#                     self.frame1.features[match.queryIdx].keypoint.pt, 
#                     self.frame2.features[match.trainIdx].keypoint.pt, 
#                     self.frame1.features[match.queryIdx].descriptor, 
#                     self.frame2.features[match.trainIdx].descriptor,
#                     match.distance, np.random.random((3))) 
#                 for idx, match
#                 in enumerate(temp)]


#         # Perform a very crude filtering of the matches
#         self.filtered_matches = [match
#                 for match
#                 in self.raw_matches
#                 if match.distance < 1130]


#     def visualize_matches(self, matches):
#         h, w, _ = self.frame1.image.shape
#         # Place the images next to each other.
#         vis = np.concatenate((self.frame1.image, self.frame2.image), axis=1)

#         # Draw the matches
#         for match in matches:
#             start_coord = (int(match.keypoint1[0]), int(match.keypoint1[1]))
#             end_coord = (int(match.keypoint2[0] + w), int(match.keypoint2[1]))
#             thickness = 1
#             color = list(match.color * 256)
#             vis = cv2.line(vis, start_coord, end_coord, color, thickness)

#         return vis


#     def determine_essential_matrix(self, matches):
#         points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

#         confidence = 0.99
#         ransacReprojecThreshold = 1
#         self.essential_matrix, mask = cv2.findEssentialMat(
#                 points_in_frame_1,
#                 points_in_frame_2, 
#                 self.camera_matrix, 
#                 cv2.FM_RANSAC, 
#                 confidence,
#                 ransacReprojecThreshold)

#         inlier_matches = [match 
#                 for match, inlier in zip(matches, mask.ravel() == 1)
#                 if inlier]

#         return inlier_matches


#     def get_image_points(self, matches):
#         points_in_frame_1 = np.array(
#                 [match.keypoint1 for match in matches], dtype=np.float64)
#         points_in_frame_2 = np.array(
#                 [match.keypoint2 for match in matches], dtype=np.float64)
#         return points_in_frame_1, points_in_frame_2


#     def estimate_camera_movement(self, matches):
#         points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

#         retval, self.R, self.t, mask = cv2.recoverPose(
#                 self.essential_matrix, 
#                 points_in_frame_1, 
#                 points_in_frame_2, 
#                 self.camera_matrix)
#         self.relative_pose = np.eye(4)
#         self.relative_pose[:3, :3] = self.R
#         self.relative_pose[:3, 3] = self.t.T[0]

#         print("relative movement in image pair")
#         print(self.relative_pose)


#     def reconstruct_3d_points(self, matches, 
#             first_projection_matrix = None, 
#             second_projection_matrix = None):
#         identify_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
#         estimated_transform = np.hstack((self.R.T, -self.R.T @ self.t))

#         self.null_projection_matrix = self.camera_matrix @ identify_transform
#         self.projection_matrix = self.camera_matrix @ estimated_transform

#         if first_projection_matrix is not None:
#             self.null_projection_matrix = self.camera_matrix @ first_projection_matrix
#         if second_projection_matrix is not None:
#             self.projection_matrix = self.camera_matrix @ second_projection_matrix

#         points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

#         self.points3d_reconstr = cv2.triangulatePoints(
#                 self.projection_matrix, 
#                 self.null_projection_matrix,
#                 points_in_frame_1.T, 
#                 points_in_frame_2.T) 

#         # Convert back to unit value in the homogeneous part.
#         self.points3d_reconstr /= self.points3d_reconstr[3, :]

#         self.matches_with_3d_information = [
#                 Match3D(match.featureid1, match.featureid2, 
#                     match.keypoint1, match.keypoint2, 
#                     match.descriptor1, match.descriptor2, 
#                     match.distance, match.color,
#                     (self.points3d_reconstr[0, idx],
#                         self.points3d_reconstr[1, idx],
#                         self.points3d_reconstr[2, idx]))
#                 for idx, match 
#                 in enumerate(matches)]
        
#         print("Reconstructed points")
#         print(self.points3d_reconstr.transpose().shape)
#         print(self.points3d_reconstr.transpose())