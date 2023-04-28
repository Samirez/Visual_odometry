import cv2
import numpy as np
import os

def main():
    check_directories()
    video_name = './input/video/DJI_0199.MOV'
    output_images = './output/images/'
    cap = cv2.VideoCapture(video_name)
    exclude = 1200
    save_frame = 0
    while True:
        ret, frame = cap.read()
        if exclude > 0:
            exclude -= 1
        if save_frame % 50 == 0 and exclude == 0:
            cv2.imwrite(output_images + 'DJI_0199_{}.jpg'.format(save_frame), frame)
        k = cv2.waitKey(1) & 0xFF
        if not ret or k == ord('q') or k == 27:
            break
        cv2.imshow('frame', frame)
        save_frame += 1
    
    cap.release()
    cv2.destroyAllWindows()

def check_directories():
    if not os.path.exists('./input/video'):
        os.makedirs('./input/video')
    if not os.path.exists('./output/images'):
        os.makedirs('./output/images')
    
if __name__ == '__main__':
    main()