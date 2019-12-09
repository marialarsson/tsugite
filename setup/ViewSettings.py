import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import random

class ViewSettings:
    def __init__(self):
        # Visibility variables
        self.hidden = [False,False,False,False,False,False]
        self.show_hidden_lines = True
        self.show_arrows = True
        self.show_milling_path = False

        # Joint opening variables
        self.open_joint = False
        self.open_start_time = None
        self.close_start_time = None
        self.open_start_dist = 0
        self.close_start_dist = 0
        self.open_ratio = 0

        # Mouse rotation variables
        self.xrot, self.yrot = 0.8, 0.4
        self.xrot0, self.yrot0 = self.xrot, self.yrot
        self.xstart = self.ystart = 0.0
        self.dragged = False

        # Colors of unbridged components
        col_yellow = [1.0, 1.0, 0.6]
        col_turkoise = [0.6, 1.0, 1.0]
        col_pink =  [1.0, 0.6, 1.0]
        col_orange = [1.0, 0.8, 0.6]
        col_purple = [0.8,0.8,1.0]
        col_grey = [0.95,0.95,0.95]
        self.unbridge_colors = [[col_yellow,col_turkoise],[col_pink,col_orange],[col_purple,col_grey]]

    def set_joint_opening_distance(self,noc):
        if self.open_joint: # open joint
            if self.open_start_time==None:
                self.open_start_time = glfw.get_time()
                self.close_start_time = None
            self.open_ratio = self.open_start_dist + 1.2 * (glfw.get_time()-self.open_start_time)
            if self.open_ratio>=noc-1:
                self.open_ratio=noc-1
            self.close_start_dist = self.open_ratio
        else: # close joint
            if self.close_start_time==None:
                self.close_start_time = glfw.get_time()
                self.open_start_time = None
            self.open_ratio = self.close_start_dist - 1.2 * (glfw.get_time()-self.close_start_time)
            if self.open_ratio<0: self.open_ratio = 0
            self.open_start_dist = self.open_ratio

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
