import cv2
import numpy as np
import glob
import OpenGL.GL as gl
from display import Display3D
import collections
import g2opy as g2o

np.set_printoptions(precision=4, suppress=True)

Feature = collections.namedtuple('Feature', 
        ['keypoint', 'descriptor', 'feature_id'])
Match = collections.namedtuple('Match', 
        ['featureid1', 'featureid2', 
            'keypoint1', 'keypoint2', 
            'descriptor1', 'descriptor2', 
            'distance', 'color'])
Match3D = collections.namedtuple('Match3D', 
        ['featureid1', 'featureid2', 
            'keypoint1', 'keypoint2', 
            'descriptor1', 'descriptor2', 
            'distance', 'color', 
            'point'])
MatchWithMap = collections.namedtuple('MatchWithMap', 
        ['featureid1', 'featureid2', 
            'imagecoord', 'mapcoord', 
            'descriptor1', 'descriptor2', 
            'distance'])



class FrameGenerator():
    def __init__(self, detector):
        self.next_image_counter = 0
        self.detector = detector

    def make_frame(self, image):
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



class Frame():
    """
    Class / structure for saving information about a single frame.
    """
    def __init__(self, image = None):
        self.image = image
        self.id = None
        self.keypoints = None
        self.descriptors = None
        self.features = None

    def __repr__(self):
        return repr('Frame %d' % (
            self.id))



class TrackedPoint():
    def __init__(self, point, descriptor, color, feature_id, point_id = None):
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



class TrackedCamera():
    def __init__(self, R, t, frame_id, camera_id = None, fixed = False):
        self.R = R
        self.t = t
        self.frame_id = frame_id
        self.camera_id = camera_id
        self.fixed = fixed

    def pose(self):
        if self.t.shape == (3, 1):
            # TODO catch this earlier
            self.t = self.t.T[0]
        ret = np.eye(4)
        ret[:3, :3] = self.R
        ret[:3, 3] = self.t
        return ret

    def __repr__(self):
        return repr("Camera %d [%s] %s (%f %f %f) %s" % (self.camera_id,
            self.frame_id, 
            self.fixed,
            self.t[0],
            self.t[1],
            self.t[2],
            self.R))



def quarternion_to_rotation_matrix(q):
    """
    The formula for converting from a quarternion to a rotation 
    matrix is taken from here:
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/index.htm
    """
    qw = q.w()
    qx = q.x()
    qy = q.y()
    qz = q.z()
    R11 = 1 - 2*qy**2 - 2*qz**2	
    R12 = 2*qx*qy - 2*qz*qw
    R13 = 2*qx*qz + 2*qy*qw
    R21 = 2*qx*qy + 2*qz*qw
    R22 = 1 - 2*qx**2 - 2*qz**2
    R23 = 2*qy*qz - 2*qx*qw
    R31 = 2*qx*qz - 2*qy*qw
    R32 = 2*qy*qz + 2*qx*qw 
    R33 = 1 - 2*qx**2 - 2*qy**2
    R = np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])
    return R



class Observation():
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



class ImagePair():
    """
    Class for working with image pairs.
    """
    def __init__(self, frame1, frame2, matcher, camera_matrix):
        self.frame1 = frame1
        self.frame2 = frame2
        self.matcher = matcher
        self.camera_matrix = camera_matrix


    def match_features(self):
        temp = self.matcher.match(
                self.frame1.descriptors, 
                self.frame2.descriptors)
        # Make a list with the following values
        # - feature 1 id
        # - feature 2 id
        # - image coordinate 1
        # - image coordinate 2
        # - match distance
        self.raw_matches = [
                Match(self.frame1.features[match.queryIdx].feature_id, 
                    self.frame2.features[match.trainIdx].feature_id,
                    self.frame1.features[match.queryIdx].keypoint.pt, 
                    self.frame2.features[match.trainIdx].keypoint.pt, 
                    self.frame1.features[match.queryIdx].descriptor, 
                    self.frame2.features[match.trainIdx].descriptor,
                    match.distance, np.random.random((3))) 
                for idx, match
                in enumerate(temp)]


        # Perform a very crude filtering of the matches
        self.filtered_matches = [match
                for match
                in self.raw_matches
                if match.distance < 1130]


    def visualize_matches(self, matches):
        h, w, _ = self.frame1.image.shape
        # Place the images next to each other.
        vis = np.concatenate((self.frame1.image, self.frame2.image), axis=1)

        # Draw the matches
        for match in matches:
            start_coord = (int(match.keypoint1[0]), int(match.keypoint1[1]))
            end_coord = (int(match.keypoint2[0] + w), int(match.keypoint2[1]))
            thickness = 1
            color = list(match.color * 256)
            vis = cv2.line(vis, start_coord, end_coord, color, thickness)

        return vis


    def determine_essential_matrix(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        confidence = 0.99
        ransacReprojecThreshold = 1
        self.essential_matrix, mask = cv2.findEssentialMat(
                points_in_frame_1,
                points_in_frame_2, 
                self.camera_matrix, 
                cv2.FM_RANSAC, 
                confidence,
                ransacReprojecThreshold)

        inlier_matches = [match 
                for match, inlier in zip(matches, mask.ravel() == 1)
                if inlier]

        return inlier_matches


    def get_image_points(self, matches):
        points_in_frame_1 = np.array(
                [match.keypoint1 for match in matches], dtype=np.float64)
        points_in_frame_2 = np.array(
                [match.keypoint2 for match in matches], dtype=np.float64)
        return points_in_frame_1, points_in_frame_2


    def estimate_camera_movement(self, matches):
        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        retval, self.R, self.t, mask = cv2.recoverPose(
                self.essential_matrix, 
                points_in_frame_1, 
                points_in_frame_2, 
                self.camera_matrix)
        self.relative_pose = np.eye(4)
        self.relative_pose[:3, :3] = self.R
        self.relative_pose[:3, 3] = self.t.T[0]

        print("relative movement in image pair")
        print(self.relative_pose)


    def reconstruct_3d_points(self, matches, 
            first_projection_matrix = None, 
            second_projection_matrix = None):
        identify_transform = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        estimated_transform = np.hstack((self.R.T, -self.R.T @ self.t))

        self.null_projection_matrix = self.camera_matrix @ identify_transform
        self.projection_matrix = self.camera_matrix @ estimated_transform

        if first_projection_matrix is not None:
            self.null_projection_matrix = self.camera_matrix @ first_projection_matrix
        if second_projection_matrix is not None:
            self.projection_matrix = self.camera_matrix @ second_projection_matrix

        points_in_frame_1, points_in_frame_2 = self.get_image_points(matches)

        self.points3d_reconstr = cv2.triangulatePoints(
                self.projection_matrix, 
                self.null_projection_matrix,
                points_in_frame_1.T, 
                points_in_frame_2.T) 

        # Convert back to unit value in the homogeneous part.
        self.points3d_reconstr /= self.points3d_reconstr[3, :]

        self.matches_with_3d_information = [
                Match3D(match.featureid1, match.featureid2, 
                    match.keypoint1, match.keypoint2, 
                    match.descriptor1, match.descriptor2, 
                    match.distance, match.color,
                    (self.points3d_reconstr[0, idx],
                        self.points3d_reconstr[1, idx],
                        self.points3d_reconstr[2, idx]))
                for idx, match 
                in enumerate(matches)]


        
class Map():
    def __init__(self):
        self.clean()

    def clean(self):
        self.next_id_to_use = 0
        self.points = []
        self.cameras = []
        self.observations = []
        self.camera_matrix = None

    def increment_id(self):
        t = self.next_id_to_use
        self.next_id_to_use += 1
        return t
    
    def add_camera(self, camera):
        camera.camera_id = self.increment_id()
        self.cameras.append(camera)
        return camera

    def add_point(self, point):
        point.point_id = self.increment_id()
        self.points.append(point)
        return point


    def calculate_reprojection_error_for_point(self, point, camera_pose, image_coord):
        #print("calculate_reprojection_error_for_point")
        #print(point)
        #print(camera_pose)
        #print(image_coord)
        temp = camera_pose @ point
        #print(temp)
        projected_image_coord = self.camera_matrix @ temp[0:3]
        #print(projected_image_coord)
        projected_image_coord /= projected_image_coord[2]
        dx = projected_image_coord[0] - image_coord[0]
        dy = projected_image_coord[1] - image_coord[1]
        return dx**2 + dy**2


    def remove_observations_with_reprojection_errors_above_threshold(self, threshold = 100):
        camera_dict = {}
        for camera in self.cameras:
            camera_dict[camera.camera_id] = camera
        point_dict = {}
        for point in self.points:
            point_dict[point.point_id] = point
        total_error = 0
        temp_observations = []
        for observation in self.observations:
            camera = camera_dict[observation.camera_id]
            point = point_dict[observation.point_id]
            point = np.array(point.point)
            point = np.hstack((point, 1))
            point = np.array([point]).T
            point_in_cam_coords = camera.pose() @ point
            t = self.camera_matrix @ point_in_cam_coords[0:3, :]
            t = t / t[2, 0]
            dx = t[0] - observation.image_coordinates[0]
            dy = t[1] - observation.image_coordinates[1]
            sqerror = np.abs(dx*dx) + np.abs(dy*dy)
            if sqerror < threshold:
                temp_observations.append(observation)

        self.observations = temp_observations


    def calculate_reprojection_error(self, threshold = 10):
        camera_dict = {}
        for camera in self.cameras:
            camera_dict[camera.camera_id] = camera
        point_dict = {}
        for point in self.points:
            point_dict[point.point_id] = point
        total_error = 0
        for observation in self.observations:
            camera = camera_dict[observation.camera_id]
            point = point_dict[observation.point_id]
            point = np.array(point.point)
            point = np.hstack((point, 1))
            point = np.array([point]).T
            point_in_cam_coords = camera.pose() @ point
            t = self.camera_matrix @ point_in_cam_coords[0:3, :]
            t = t / t[2, 0]
            dx = t[0] - observation.image_coordinates[0]
            dy = t[1] - observation.image_coordinates[1]
            sqerror = np.abs(dx*dx) + np.abs(dy*dy)
            if sqerror > threshold:
                print("high reprojection error: %f" % sqerror)
                print(observation)
            total_error += sqerror

        return total_error


    def show_total_reprojection_error(self):
        total_error = self.calculate_reprojection_error()
        print("calculated reprojection error")
        print("total error: %f" % total_error)


    def optimize_map(self, postfix = ""):
        optimizer = g2o.SparseOptimizer()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        optimizer.set_algorithm(solver)

        # Define camera parameters
        print(self.camera_matrix)
        #focal_length = 1000
        focal_length = self.camera_matrix[0, 0]
        #principal_point = (320, 240)
        principal_point = (self.camera_matrix[0, 2], self.camera_matrix[1, 2])
        baseline = 0
        cam = g2o.CameraParameters(focal_length, principal_point, baseline)
        cam.set_id(0)
        optimizer.add_parameter(cam)

        camera_vertices = {}
        for camera in self.cameras:
            # Use the estimated pose of the second camera based on the 
            # essential matrix.
            pose = g2o.SE3Quat(camera.R, camera.t)

            # Set the poses that should be optimized.
            # Define their initial value to be the true pose
            # keep in mind that there is added noise to the observations afterwards.
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_id(camera.camera_id)
            v_se3.set_estimate(pose)
            v_se3.set_fixed(camera.fixed)
            optimizer.add_vertex(v_se3)
            camera_vertices[camera.camera_id] = v_se3
            #print("camera id: %d" % camera.camera_id)

        point_vertices = {}
        for point in self.points:
            # Add 3d location of point to the graph
            vp = g2o.VertexPointXYZ()
            vp.set_id(point.point_id)
            vp.set_marginalized(True)
            # Use positions of 3D points from the triangulation
            point_temp = np.array(point.point, dtype=np.float64)
            vp.set_estimate(point_temp)
            optimizer.add_vertex(vp)
            point_vertices[point.point_id]= vp


        for observation in self.observations:
            # Add edge from first camera to the point
            edge = g2o.EdgeProjectXYZ2UV()

            # 3D point
            edge.set_vertex(0, point_vertices[observation.point_id]) 
            # Pose of first camera
            edge.set_vertex(1, camera_vertices[observation.camera_id]) 
            
            edge.set_measurement(observation.image_coordinates)
            edge.set_information(np.identity(2))
            edge.set_robust_kernel(g2o.RobustKernelHuber())

            edge.set_parameter_id(0, 0)
            optimizer.add_edge(edge)

        print('num vertices:', len(optimizer.vertices()))
        print('num edges:', len(optimizer.edges()))

        print('Performing full BA:')
        optimizer.initialize_optimization()
        optimizer.set_verbose(True)
        optimizer.optimize(40)
        optimizer.save("test.g2o");

        for idx, camera in enumerate(self.cameras):
            t = camera_vertices[camera.camera_id].estimate().translation()
            self.cameras[idx].t = t
            q = camera_vertices[camera.camera_id].estimate().rotation()
            self.cameras[idx].R = quarternion_to_rotation_matrix(q)

        for idx, point in enumerate(self.points):
            p = point_vertices[point.point_id].estimate()
            # It is important to copy the point estimates.
            # Otherwise I end up with some memory issues.
            # self.points[idx].point = p
            self.points[idx].point = np.copy(p)


    def remove_camera_from_map(self, camera_to_remove):
        print("Removing camera with id %s" % camera_to_remove.camera_id)
        print(camera_to_remove)
        self.cameras.remove(camera_to_remove)

        list_of_camera_ids = set([camera.camera_id
                for camera 
                in self.cameras])
        #print("list_of_camera_ids")
        #print(list_of_camera_ids)

        # Only keep observations associated with an active camera
        filtered_observations = [observation
                for observation 
                in self.observations
                if observation.camera_id in list_of_camera_ids]

        self.observations = filtered_observations

        # Count number of observations of each point
        point_observations = collections.defaultdict(lambda: 0)
        for observation in self.observations:
            point_observations[observation.point_id] += 1

        #print(point_observations)
        points_to_remove = []
        for key, value in point_observations.items():
            if value < 2:
                points_to_remove.append(key)

        #print(points_to_remove)
        self.points = [point
                for point
                in self.points
                if point.point_id not in points_to_remove]

        list_of_point_ids = set([point.point_id
                for point 
                in self.points])

        # Only keep observations associated with points on the map
        self.observations = [observation
                for observation 
                in self.observations
                if observation.point_id in list_of_point_ids]


    def show_map_statistics(self):
        print("Map statistics")
        print("Number of points in map: %d" % len(self.points))
        print("Number of cameras in map: %d" % len(self.cameras))
        print("Number of observations in map: %d" % len(self.observations))
        print("Total reprojection error: %f" % self.calculate_reprojection_error())
        #self.show_number_of_observations_per_point()
        self.show_observation_matrix()


    def show_number_of_observations_per_point(self):
        observations_per_point = collections.defaultdict(lambda: 0)
        for observation in self.observations:
            observations_per_point[observation.point_id] += 1

        counts = collections.defaultdict(lambda: 0)
        for count in observations_per_point.values():
            counts[count] += 1
        for key in sorted(counts.keys()):
            print(key, counts[key])

        # Make dict with all points in the current map
        mappointdict = {}
        for point in self.points:
            mappointdict[point.point_id] = point

        for idx, obs in enumerate(observations_per_point.keys()):
            print(mappointdict[obs])
            print(observations_per_point[obs])
            if idx > 20:
                break


    def show_observation_matrix(self):
        """
        The observation matrix contains the number of features that 
        have been matched between two cameras.
        """

        # Make it easy to access all observations of a single point.
        observations_of_point = collections.defaultdict(lambda: [])
        for observation in self.observations:
            observations_of_point[observation.point_id].append(observation)

        # For each point in the map, extract the cameras that have
        # observed the point. Make all combinations of these cameras
        # and increment the associated entries in the observation matrix.
        obs_matrix = collections.defaultdict(lambda: 
                collections.defaultdict(lambda: 0))
        for point in self.points:
            cameras_that_see_point = [observation.camera_id
                    for observation 
                    in observations_of_point[point.point_id]]
            combinations = [(cam1, cam2)
                    for cam1 
                    in cameras_that_see_point
                    for cam2 
                    in cameras_that_see_point
                    if cam1 < cam2]
            for (cam1, cam2) in combinations:
                obs_matrix[cam1][cam2] += 1
        for cam1, value1 in obs_matrix.items():
            for cam2, value2 in obs_matrix[cam1].items():
                print("%4d %4d %4d" % (cam1, cam2, value2))



    def limit_number_of_camera_in_map(self, max_camera_number):
        print("limit_number_of_camera_in_map")
        for camera in self.cameras:
            print(camera)
        if len(self.cameras) > max_camera_number:
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("Should clean up map")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
            print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")

            self.remove_camera_from_map(self.cameras[0])



class VisualSlam:
    def __init__(self, inputdirectory, feature = "ORB"):
        self.input_directory = inputdirectory

        # Use ORB features
        #self.detector = cv2.ORB_create()
        self.detector = cv2.SIFT_create()
        #self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        
        self.frame_generator = FrameGenerator(self.detector)
        self.list_of_frames = []
        self.map = Map()
        self.three_dim_viewport = Display3D()

        self.feature_mapper = {}
        self.feature_history = {}


    def set_camera_matrix(self):
        self.camera_matrix = np.array([[2676, 0., 3840 / 2 - 35.24], 
            [0.000000000000e+00, 2676., 2160 / 2 - 279],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        #self.camera_matrix = np.array([[835, 0., 1008 / 2 + 61], 
        #    [0.000000000000e+00, 835, 756 / 2 - 9],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])
        #self.camera_matrix = np.array([[1750, 0., 1920 / 2 - 10], 
        #    [0.000000000000e+00, 1750, 1080 / 2 + 32],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


        # Calibration from input/KITTY_sequence_1/calib.txt
        #self.camera_matrix = np.array([[7.070912e+02, 0.e+00, 6.018873e+02], 
        #    [0.000000000000e+00, 7.070912000000e+02, 1.831104000000e+02],
        #    [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])


        #self.scale_factor = 0.2
        self.scale_factor = 0.3
        #self.scale_factor = 1
        self.camera_matrix *= self.scale_factor
        self.camera_matrix[2, 2] = 1


    def add_to_list_of_frames(self, image):
        frame = self.frame_generator.make_frame(image)
        self.list_of_frames.append(frame)


    def initialize_map(self, image_pair):
        self.map.clean()
        self.map.camera_matrix = self.camera_matrix

        ip = image_pair

        for match in ip.matches_with_3d_information:
            self.feature_mapper[match.featureid2] = match.featureid1
            # Save all features matches in a dictionary for easy 
            # access later on.
            self.feature_history[match.featureid1] = match

        # We have only seen two frame, to initialize a map.
        first_camera = TrackedCamera(
                np.eye(3), 
                np.zeros((3)), 
                ip.frame1.id,
                fixed = True)
        second_camera = TrackedCamera(
                ip.R, 
                ip.t.T[0],
                ip.frame2.id,
                fixed = False)
        first_camera = self.map.add_camera(first_camera)
        second_camera = self.map.add_camera(second_camera)

        for match in ip.matches_with_3d_information:
            tp = TrackedPoint(
                    match.point, 
                    match.descriptor1, 
                    match.color, 
                    match.featureid1)
            tp = self.map.add_point(tp)

            observation1 = Observation(tp.point_id, 
                    first_camera.camera_id, 
                    match.keypoint1)
            observation2 = Observation(tp.point_id, 
                    second_camera.camera_id, 
                    match.keypoint2)

            self.map.observations.append(observation1)
            self.map.observations.append(observation2)

        self.map.remove_observations_with_reprojection_errors_above_threshold(100)
        self.map.optimize_map()
        print("Hej")
        #self.map.calculate_reprojection_error(0.1)
        self.map.remove_observations_with_reprojection_errors_above_threshold(1)
        print("Hej")


    def track_feature_back_in_time(self, featureid):
        while featureid in self.feature_mapper:
            featureid_old = featureid
            featureid = self.feature_mapper[featureid]
            #print(featureid, " <- ", featureid_old)
        return featureid


    def add_new_match_to_map(self, match, first_camera, second_camera):
        tp = TrackedPoint(
                match.point, 
                match.descriptor1, 
                match.color, 
                match.featureid1)
        tp = self.map.add_point(tp)

        observation1 = Observation(
                tp.point_id, 
                first_camera.camera_id, 
                match.keypoint1)
        observation2 = Observation(
                tp.point_id, 
                second_camera.camera_id, 
                match.keypoint2)

        self.map.observations.append(observation1)
        self.map.observations.append(observation2)
        #print(observation1)
        #print(observation2)


    def add_new_observation_of_existing_point(
            self, featureid, match, camera):

        try:
            tp = self.mappointdict[featureid]
            observation = Observation(tp.point_id, 
                    camera.camera_id, 
                    match.keypoint2)
            self.map.observations.append(observation)
            #print(observation)
        except Exception as e:
            print("Exception in add_new_observation_of_existing_point")
            print(e)
            print(self.mappointdict)
            print("-----------")


    def add_point_observation_to_map(self, match, first_camera, second_camera):
        try:
            # Check to see if the point is in the map already
            featureid = self.track_feature_back_in_time(match.featureid2)

            if featureid in self.mappointdict:
                self.add_new_observation_of_existing_point(
                        featureid, match, second_camera)
            else:
                pass
                self.add_new_match_to_map(match, first_camera, second_camera)
        except Exception as e:
            print("Error in add_point_observation_to_map")
            print(match)
            print(e)


    def add_information_to_map(self):
        # Make dict with all points in the current map
        self.mappointdict = {}
        for point in self.map.points:
            self.mappointdict[point.feature_id] = point

        # Update map with more points
        camera_dict = {}
        for item in self.map.cameras:
            camera_dict[item.frame_id] = item
            print(item)

        ip = self.current_image_pair
        essential_matches = ip.determine_essential_matrix(ip.filtered_matches)
        projection_matrix_one = camera_dict[ip.frame2.id].pose()
        projection_matrix_two = camera_dict[ip.frame1.id].pose()

        ip.reconstruct_3d_points(
                essential_matches, 
                projection_matrix_one[0:3, :], 
                projection_matrix_two[0:3, :])

        first_camera = camera_dict[ip.frame1.id]
        second_camera = camera_dict[ip.frame2.id]
        for match in ip.matches_with_3d_information:
            if np.linalg.norm(match.point) > 50:
                continue
            self.add_point_observation_to_map(match, first_camera, second_camera)

        #self.map.calculate_reprojection_error()


    def update_feature_mapper(self):
        for match in self.current_image_pair.matches_with_3d_information:
            self.feature_mapper[match.featureid2] = match.featureid1
            # Save all features matches in a dictionary for easy 
            # access later on.
            self.feature_history[match.featureid1] = match


    def estimate_current_camera_position(self, current_frame):
        self.update_feature_mapper()

        # Make dict with all points in the current map
        self.mappointdict = {}
        for point in self.map.points:
            self.mappointdict[point.feature_id] = point


        if len(self.list_of_frames) < 3:
            return

        # Trace track of all matches in current_image_pair
        matches_with_map = []
        for match in self.current_image_pair.matches_with_3d_information:
            feature_id = match.featureid2
            while feature_id in self.feature_mapper:
                feature_id = self.feature_mapper[feature_id]

            if feature_id in self.mappointdict:
                image_feature = self.feature_history[match.featureid1]
                map_feature = self.mappointdict[feature_id]
                t = MatchWithMap(image_feature.featureid2, 
                        map_feature.feature_id, 
                        image_feature.keypoint2,
                        map_feature.point, 
                        image_feature.descriptor2, 
                        map_feature.descriptor, 
                        0)   # TODO: Set to the proper feature distance
                matches_with_map.append(t)

        print("Matches with map")
        print(len(matches_with_map))
        image_coords = [match.imagecoord 
                for match 
                in matches_with_map]

        map_coords = [match.mapcoord
                for match 
                in matches_with_map]

        try: 
            retval, rvec, tvec, inliers = cv2.solvePnPRansac(
                    np.array(map_coords), 
                    np.array(image_coords), 
                    self.camera_matrix, 
                    np.zeros(4)) 

            if retval:
                self.mappointdict = {}
                for point in self.map.points:
                    self.mappointdict[point.feature_id] = point

                print("Succesfully estimated the camera position")
                R, _ = cv2.Rodrigues(rvec)
                camera = TrackedCamera(
                        R,
                        tvec,
                        current_frame.id, 
                        fixed = False)
                camera = self.map.add_camera(camera)

                self.add_information_to_map()
            else:
                print("Failed to estimate the camera position")
                print("####################")


        except Exception as e:
            print("Position estimation failed")
            print(e)
            pass

        self.freeze_nonlast_cameras()
        self.print_camera_details()
        self.map.optimize_map()
        self.map.remove_observations_with_reprojection_errors_above_threshold(1)
        #self.unfreeze_cameras()
        #self.map.optimize_map()
                
        
    def freeze_nonlast_cameras(self):
        for idx, camera in enumerate(self.map.cameras):
            self.map.cameras[idx].fixed = True
        self.map.cameras[-1].fixed = False
        if len(self.map.cameras) > 2:
            self.map.cameras[-2].fixed = False


    def print_camera_details(self):
        for camera in self.map.cameras:
            print(camera)


    def unfreeze_cameras(self, number_of_fixed_cameras = 5):
        for idx, camera in enumerate(self.map.cameras):
            if idx > number_of_fixed_cameras:
                self.map.cameras[idx].fixed = False
            else:
                self.map.cameras[idx].fixed = True


    def match_current_and_previous_frame(self):
        if len(self.list_of_frames) < 2:
            return

        frame1 = self.list_of_frames[-2]
        frame2 = self.list_of_frames[-1]
        self.current_image_pair = ImagePair(frame1, frame2, self.bf, self.camera_matrix)
        self.current_image_pair.match_features()
        essential_matches = self.current_image_pair.determine_essential_matrix(self.current_image_pair.filtered_matches)
        self.current_image_pair.estimate_camera_movement(essential_matches)
        self.current_image_pair.reconstruct_3d_points(essential_matches)

        if len(self.list_of_frames) == 2:
            self.initialize_map(self.current_image_pair)

        image_to_show = self.current_image_pair.visualize_matches(essential_matches)
        cv2.imshow("matches", image_to_show)

        self.estimate_current_camera_position(frame2)
        self.three_dim_viewport.set_points_to_draw(self.map.points, self.map.cameras)

        self.freeze_nonlast_cameras()
        self.print_camera_details()

        self.map.limit_number_of_camera_in_map(8)


    def show_map_points(self, message):
        print(message)
        for element in self.map.points:
            print(element)


    def process_frame(self, frame):
        self.add_to_list_of_frames(frame)
        self.match_current_and_previous_frame()
        return frame


    def run(self):
        list_of_files = glob.glob("%s/*.jpg" % self.input_directory)
        list_of_files.sort()
        for idx, filename in enumerate(list_of_files):
            print(filename)
            img = cv2.imread(filename)

            scale_percent = 30 # percent of original size
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)

            img = cv2.resize(img, dim)
            frame = self.process_frame(img)
            self.map.show_map_statistics()

            #cv2.imshow("test", frame);
            k = cv2.waitKey(400000)
            if k == ord('q'):
                break
            if k == ord('p'):
                k = cv2.waitKey(100000)
            if k == ord('b'):
                # Perform bundle adjusment
                self.unfreeze_cameras(5)
                self.print_camera_details()
                self.map.optimize_map()
                self.freeze_nonlast_cameras()
                self.print_camera_details()

        while True:
            k = cv2.waitKey(100)
            if k == ord('q'):
                break


vs = VisualSlam(r"input/frames")
#vs = VisualSlam(r"input/frames2")
#vs = VisualSlam(r"/home/hemi/Nextcloud/Work/01_teaching_courses/2021-02-01_LSDP_Large_scale_drone_perception/materials/experiments/visual_slam/test")
vs.set_camera_matrix()
vs.run()

