import pygame
import pygame.locals as pgl
import numpy as np
from PIL import Image # type: ignore

import OpenGL.GL as gl # type: ignore
import OpenGL.GLU as glu # type: ignore

from icecream import ic # type: ignore

verticies = (
    (1, -1, -1),
    (1, 1, -1),
    (-1, 1, -1),
    (-1, -1, -1),
    (1, -1, 1),
    (1, 1, 1),
    (-1, -1, 1),
    (-1, 1, 1)
    )

edges = (
    (0,1),
    (0,3),
    (0,4),
    (2,1),
    (2,3),
    (2,7),
    (6,3),
    (6,4),
    (6,7),
    (5,1),
    (5,4),
    (5,7)
    )


def CubeLines():
    gl.glColor3f(1.0, 1.0, 1.0)
    gl.glBegin(gl.GL_LINES)
    for edge in edges:
        for vertex in edge:
            gl.glVertex3fv(verticies[vertex])
    gl.glEnd()


class ThreeDimViewer:
    # Code for camera movement taken from 
    # https://stackoverflow.com/a/56609894/185475
    def __init__(self):
        pygame.init()
        self.initialize_window()
        self.set_opengl_settings()
        self.set_initial_camera_position()
        self.initialize_state_variables()

        self.vertices = []
        self.colors = []
        self.cameras = []
        self.camera_texture_ids = []


    def initialize_window(self):
        self.display = (800, 600)
        self.screen = pygame.display.set_mode(self.display, pgl.DOUBLEBUF|pgl.OPENGL|pgl.RESIZABLE)
        gl.glMatrixMode(gl.GL_PROJECTION)
        horizontal_field_of_view = 75
        near_clipping_distance = 0.01
        far_clipping_distance = 500.0
        glu.gluPerspective(horizontal_field_of_view, 
                           (self.display[0]/self.display[1]), 
                           near_clipping_distance, 
                           far_clipping_distance)


    def set_opengl_settings(self):
        gl.glEnable(gl.GL_DEPTH_TEST)

        gl.glEnable(gl.GL_LIGHT0)
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, [1.0, 1.0, 1.0, 1])
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, [1.0, 1.0, 1.0, 1])


    def set_initial_camera_position(self):
        gl.glMatrixMode(gl.GL_MODELVIEW)
        glu.gluLookAt(0, -8, 0, 0, 0, 0, 0, 0, 1)
        gl.glLoadIdentity()
        # Let the viewport point in direction of the camera
        gl.glRotatef(180, 1.0, 0.0, 0.0)
        # Move the viewport a bit behind and above the camera
        gl.glTranslatef(0, 0.5, 1.0)


    def initialize_state_variables(self):
        self.run = True
        self.terminate = False
        self.paused = False


    def handle_pygame_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    ic(gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX))
                if event.key == pygame.K_SPACE:
                    self.run = False
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    self.run = False
                    self.terminate = True
                if event.key == pygame.K_PAUSE or event.key == pygame.K_p:
                    self.paused = not self.paused
                    pygame.mouse.set_pos(self.displayCenter) 

            if not self.paused: 
                if event.type == pygame.MOUSEMOTION:
                    self.mouseMove = [event.pos[i] - self.displayCenter[i] for i in range(2)]
                pygame.mouse.set_pos(self.displayCenter)  


    def update_camera_position(self):
        viewMatrix = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)

        # More or less magic to make the camera movements work decently.
        gl.glPushMatrix()
        gl.glLoadIdentity()
        gl.glRotatef(0.1*self.mouseMove[1], 1.0, 0.0, 0.0)
        gl.glRotatef(0.1*self.mouseMove[0], 0.0, 1.0, 0.0)
        self.mouseMove = (0, 0)
        gl.glMultMatrixf(viewMatrix)
        viewMatrix = gl.glGetFloatv(gl.GL_MODELVIEW_MATRIX)
        gl.glPopMatrix()
        gl.glLoadIdentity()
        self.move_based_on_key_inputs()
        gl.glMultMatrixf(viewMatrix)


    def move_based_on_key_inputs(self):
        keypress = pygame.key.get_pressed()
        move_scale = 0.1
        if keypress[pygame.K_LSHIFT]:
            # Reduce speed of movement to one tenth of the standard speed.
            move_scale *= 0.1
        if keypress[pygame.K_w]:
            gl.glTranslatef(0, 0, move_scale)
        if keypress[pygame.K_s]:
            gl.glTranslatef(0, 0, -move_scale)
        if keypress[pygame.K_d]:
            gl.glTranslatef(-move_scale, 0, 0)
        if keypress[pygame.K_a]:
            gl.glTranslatef(move_scale, 0, 0)
        if keypress[pygame.K_q]:
            gl.glTranslatef(0, move_scale, 0)
        if keypress[pygame.K_e]:
            gl.glTranslatef(0, -move_scale, 0)


    def draw_vertices(self):
        gl.glPushMatrix()
        gl.glColor3f(1.0, 0.0, 1.0)
        gl.glPointSize(5)
        gl.glBegin(gl.GL_POINTS)
        for vertex, color in zip(self.vertices, self.colors):
            r, g, b = color
            gl.glColor4f(r, g, b, 1)
            gl.glVertex3fv(vertex)
        gl.glEnd()
        gl.glPopMatrix()


    def draw_cameras(self):
        for counter, camera in enumerate(self.cameras):
            gl.glPushMatrix()
            gl.glMultMatrixf(np.linalg.inv(camera.pose()).transpose())

            # Draw lines from camera center to top of image plane
            gl.glLineWidth(1)
            gl.glBegin(gl.GL_LINES)
            gl.glColor3f(1, 1, 1)
            gl.glVertex3fv((0, 0, 0))
            gl.glVertex3fv((0.4, -0.2, 0.4))
            gl.glVertex3fv((-0.4, -0.2, 0.4))
            gl.glVertex3fv((0, 0, 0))
            gl.glEnd()

            # Draw image plane
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture_ids[counter])
            gl.glEnable(gl.GL_TEXTURE_2D);
            gl.glBegin(gl.GL_QUADS);
            gl.glTexCoord2f(1.0, 1.0)
            gl.glVertex3fv((0.4, -0.2, 0.4))
            gl.glTexCoord2f(1.0, 0.0)
            gl.glVertex3fv((0.4, 0.2, 0.4))
            gl.glTexCoord2f(0.0, 0.0)
            gl.glVertex3fv((-0.4, 0.2, 0.4))
            gl.glTexCoord2f(0.0, 1.0)
            gl.glVertex3fv((-0.4, -0.2, 0.4))
            gl.glEnd()
            gl.glDisable(gl.GL_TEXTURE_2D);

            gl.glPopMatrix()


    def loadTextures(self):
        for counter, camera in enumerate(self.cameras):
            textureData = Image.fromarray(camera.frame)
            textureData = textureData.tobytes('raw', 'BGRX', 0, -1)
            width = camera.frame.shape[1]
            height = camera.frame.shape[0]

            gl.glEnable(gl.GL_TEXTURE_2D)
            self.camera_texture_ids.append(gl.glGenTextures(1))

            gl.glBindTexture(gl.GL_TEXTURE_2D, self.camera_texture_ids[-1])
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, width, height,
                         0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, textureData)

            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameterf(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)


    def main(self):
        self.displayCenter = [self.screen.get_size()[i] // 2 for i in range(2)]
        self.mouseMove = [0, 0]
        pygame.mouse.set_pos(self.displayCenter)

        self.loadTextures()

        self.up_down_angle = 0
        while self.run:
            self.handle_pygame_events()
            self.update_camera_position()
            self.update_view()
            pygame.display.flip()
            pygame.time.wait(10)

        pygame.quit()
        if self.terminate:
            quit()


    def update_view(self):
        gl.glClear(gl.GL_COLOR_BUFFER_BIT|gl.GL_DEPTH_BUFFER_BIT)

        gl.glPushMatrix()

        self.draw_vertices()
        self.draw_cameras()

        CubeLines()
        gl.glPopMatrix()



if __name__ == "__main__":
    tdv = ThreeDimViewer()
    tdv.vertices = ((1, 2, 3), (1, 2, 2), (1, 2, 1), (1, 1, 2))
    tdv.colors = ((0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 0))
    tdv.main()

