from multiprocessing import Process, Queue
import pangolin
import OpenGL.GL as gl
import numpy as np

class Display3D(object):
  def __init__(self):
    self.state = None
    self.points_to_draw = np.array([[]])
    self.point_colors = np.array([[]])
    self.poses_to_draw = []
    self.q = Queue()
    self.vp = Process(target=self.viewer_thread, args=(self.q,))
    self.vp.daemon = True
    self.vp.start()

  def viewer_thread(self, q):
    self.viewer_init(1024, 768)
    while 1:
      self.viewer_refresh(q)

  def viewer_init(self, w, h):
    pangolin.CreateWindowAndBind('Map Viewer', w, h)
    gl.glEnable(gl.GL_DEPTH_TEST)

    self.scam = pangolin.OpenGlRenderState(
      pangolin.ProjectionMatrix(w, h, 420, 420, w//2, h//2, 0.2, 10000),
      pangolin.ModelViewLookAt(0, -1, -2,
                               0, 0, 10,
                               0, -1, 0))
    self.handler = pangolin.Handler3D(self.scam)

    # Create Interactive View in window
    self.dcam = pangolin.CreateDisplay()
    self.dcam.SetBounds(0.0, 1.0, 0.0, 1.0, w/h)
    self.dcam.SetHandler(self.handler)
    # hack to avoid small Pangolin, no idea why it's *2
    self.dcam.Resize(pangolin.Viewport(0,0,w*2,h*2))
    self.dcam.Activate()




  def viewer_refresh(self, q):
    while not q.empty():
      self.points_to_draw, self.point_colors, self.poses_to_draw = q.get()

    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    gl.glClearColor(0.0, 0.0, 0.0, 1.0)
    self.dcam.Activate(self.scam)

    #pangolin.glDrawColouredCube(1)
    # Draw feature points with red dots.
    gl.glPointSize(4)
    gl.glColor3f(1.0, 0.0, 0.0)
    if self.points_to_draw is not None:
        pangolin.DrawPoints(self.points_to_draw, self.point_colors)

    # Test of drawing manual points.
    #gl.glColor3f(1.0, 1.0, 1.0)
    #pangolin.DrawPoints(np.array([[0, 0, 0], [10, 0, 0], [0, 20, 0], [0, 0, 40]]))

    gl.glColor3f(1.0, 1.0, 1.0)
    if self.poses_to_draw is not None:
        for pose in self.poses_to_draw:
            pangolin.DrawCamera(pose, 1, 0.5, 0.8)

    pangolin.FinishFrame()
    return 


  def set_points_to_draw(self, points, cameras):
      np_points = np.array([point.point 
              for point in points])
      np_colors = np.array([point.color
              for point in points])
      np_poses = np.array([np.linalg.inv(camera.pose())
              for camera in cameras])
      self.q.put((np_points, np_colors, np_poses))


  def paint(self, map_in, poses_in):
    return 
    if self.q is None:
      return

    poses, pts, colors = [], [], []
    for pose in poses_in:
      print(pose.pose())
      # invert pose for display only
      #poses.append(np.linalg.inv(pose.pose()))
      poses.append(pose.pose())
    for p in map_in:
      pts.append(p.point)
      colors.append((255, 0, 0))
    self.q.put((np.array(poses), np.array(pts), np.array(colors)/256.0))


