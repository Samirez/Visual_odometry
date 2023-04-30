from initials import *
from triangulate_points_from_images import TriangulatePointsFromTwoImages








# Parse command line args



# Notes to my self
# The order of images seems to be important for the reconstruction. This should be investigated.

    
def main():
    
    filelocation = "./input/images/"
    frame1 = filelocation + 'DJI_0199_1200.jpg'
    frame2 = filelocation + 'DJI_0199_1250.jpg'

    TPFTI = TriangulatePointsFromTwoImages()
    TPFTI.run(frame1, frame2)
    
    
    
    # cv2.imshow("frame1",frame1)
    # cv2.imshow("frame2",frame2)
    # cv2.triangulatePoints(frame1, frame2)



if __name__ == "__main__":
    main()
