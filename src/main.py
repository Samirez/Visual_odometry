from visual_slam import VisualSlam

def main():
    vs = VisualSlam("./input/images")
    vs.set_camera_matrix()
    vs.run()
    
if __name__ == "__main__":
    main()