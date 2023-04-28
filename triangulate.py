import cv2
import numpy as np

    
    
def main():
    filelocation = "./output/images/"
    frame1 = cv2.imread(filelocation + 'DJI_0199_1200.jpg')
    frame2 = cv2.imread(filelocation + 'DJI_0199_1300.jpg')
    
    cv2.imshow("frame1",frame1)
    cv2.imshow("frame2",frame2)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
