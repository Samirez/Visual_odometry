from triangulate import *
from visual_slam import *

    def run(self):
        list_of_files = glob.glob("%s/*.jpg" % self.input_directory)
        list_of_files.sort()
        if len(list_of_files) == 0:
            print("No images found in the specified directory")
            return
        for idx, filename in enumerate(list_of_files):
            print(filename)
            img = cv2.imread(filename)

            width = int(img.shape[1] * self.scale_factor)
            height = int(img.shape[0] * self.scale_factor)
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


vs = VisualSlam("./input/images")
vs.set_camera_matrix()
vs.run()
