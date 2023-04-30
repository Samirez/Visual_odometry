from triangulate_points_from_images import TriangulatePointsFromTwoImages
from visual_slam import VisualSlam

def main():
    
    filelocation = "./input/images/"
    frame1 = filelocation + 'DJI_0199_1200.jpg'
    frame2 = filelocation + 'DJI_0199_1250.jpg'

    TPFTI = TriangulatePointsFromTwoImages()
    TPFTI.run(frame1, frame2)
    
    # vs = VisualSlam("./input/images")
    # vs.set_camera_matrix()
    # vs.run()
    
if __name__ == "__main__":
    main()



