from initials import *

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
    
    def add_camera(self, camera: TrackedCamera) -> TrackedCamera:
        camera.camera_id = self.increment_id()
        self.cameras.append(camera)
        return camera

    def add_point(self, point: TrackedPoint) -> TrackedPoint:
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
        solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
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
        if(len(self.observations) > 0):
            print("Mean reprojection error: %f" % (self.calculate_reprojection_error() / len(self.observations)))
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