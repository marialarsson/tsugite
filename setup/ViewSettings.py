import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders

class ViewSettings:
    def __init__(self):

        # Visibility variables
        self.hidden_a = False
        self.hidden_b = False
        self.show_hidden_lines = True
        self.show_arrows = True

        # Joint opening variables
        self.open_joint = False
        self.open_start_time = None
        self.close_start_time = None
        self.open_start_dist = 0
        self.close_start_dist = 0
        self.distance = 0

        # Mouse rotation variables
        self.xrot, self.yrot = 0.8, 0.4
        self.xrot0, self.yrot0 = self.xrot, self.yrot
        self.xstart = self.ystart = 0.0
        self.dragged = False

    def set_joint_opening_distance(self,geometry):
        if self.open_joint:
            if self.open_start_time==None:
                self.open_start_time = glfw.get_time()
                self.close_start_time = None
            self.distance = self.open_start_dist + 0.4 * (glfw.get_time()-self.open_start_time)
            if self.distance>geometry.component_size:
                self.distance = geometry.component_size
            self.close_start_dist = self.distance
        else:
            if self.close_start_time==None:
                self.close_start_time = glfw.get_time()
                self.open_start_time = None
            self.distance = self.close_start_dist - 0.4 * (glfw.get_time()-self.close_start_time)
            if self.distance <0: self.distance = 0
            self.open_start_dist = self.distance

    def update_rotation(self, window):
        # Rotate view by dragging
        if self.dragged:
            xpos, ypos = glfw.get_cursor_pos(window)
            ratio = 0.001
            ydiff = ratio*(xpos-self.xstart)
            xdiff = ratio*(ypos-self.ystart)
            self.xrot = self.xrot0 + xdiff
            self.yrot = self.yrot0 + ydiff

    def start_rotation(self, window):
        self.xstart, self.ystart = glfw.get_cursor_pos(window)
        self.dragged = True
        self.xrot0 = self.xrot
        self.yrot0 = self.yrot

    def end_rotation(self):
        self.dragged = False
