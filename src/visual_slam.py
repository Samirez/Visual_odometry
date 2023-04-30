from libs import *

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

        self.feature_mapper = {}
        self.feature_history = {}


    def set_camera_matrix(self):
        self.camera_matrix = np.array([[2676.1051390718389, 0., 3840 / 2 - 35.243952918157035], 
            [0.000000000000e+00, 279.58562078697361, 2160 / 2 - 279.58562078697361],
            [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00]])

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
                image_pair.frame1.image,
                fixed = True)
        second_camera = TrackedCamera(
                ip.R, 
                ip.t.T[0],
                ip.frame2.id,
                image_pair.frame2.image,
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

        #self.map.remove_observations_with_reprojection_errors_above_threshold(100)
        self.map.optimize_map()
        #self.map.calculate_reprojection_error(0.1)
        #self.map.remove_observations_with_reprojection_errors_above_threshold(1)


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


    def add_new_observation_of_existing_point(
            self, featureid, match, camera):

        try:
            tp = self.mappointdict[featureid]
            observation = Observation(tp.point_id, 
                    camera.camera_id, 
                    match.keypoint2)
            self.map.observations.append(observation)
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


    def estimate_current_camera_position(self, current_frame: Frame):
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
                        current_frame.image, 
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

        self.freeze_nonlast_cameras()
        self.print_camera_details()

        self.map.limit_number_of_camera_in_map(18)
        cv2.waitKey(100)


        # Testing the new visualization framework.
        viewport = ThreeDimViewer.ThreeDimViewer()
        viewport.vertices = [point.point
                    for point
                    in self.map.points]
        viewport.colors = [point.color
                    for point 
                    in self.map.points]

        viewport.cameras = [camera
                    for camera
                    in self.map.cameras]
        viewport.main()


    def show_map_points(self, message):
        print(message)
        for element in self.map.points:
            print(element)


    def process_frame(self, frame):
        self.add_to_list_of_frames(frame)
        self.match_current_and_previous_frame()
        return frame