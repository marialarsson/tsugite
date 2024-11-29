#!/usr/bin/env python3

import sys
import numpy as np
import time
import math
import os

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from PyQt5.QtOpenGL import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
from math import tan, pi
from Types import Types
from Geometries import Geometries
from ViewSettings import ViewSettings
from Show import Show
from _mainWindow import get_untitled_filename

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QGLWidget.__init__(self, parent)
        # self.setMinimumSize(800, 800)
        self.setMouseTracking(True)
        self.click_time = time.time()
        self.x = 0
        self.y = 0

    # def __init__(self, parent=None):
    #     fmt = QGLFormat()
    #     fmt.setVersion(2, 1)
    #     fmt.setProfile(QGLFormat.CoreProfile)
    #     fmt.setSampleBuffers(True)

    #     self.parent = parent
    #     QGLWidget.__init__(self, fmt, parent)

    #     # self.parent = parent
    #     # QGLWidget.__init__(self, parent)
    #     # self.setMinimumSize(10, 10)
    #     # self.setMaximumSize(10000, 10000)
    #     #self.setMinimumSize(800, 800)

    #     self.setMouseTracking(True)
    #     self.click_time = time.time()
    #     self.x = 0
    #     self.y = 0    


    from _GLWidget import initializeGL
    # from _GLWidget import resizeGL
    def resizeGL(self, w, h):
        def perspective(fovY, aspect, zNear, zFar):
            fH =tan(fovY / 360. * pi) * zNear
            fW = fH * aspect
            glFrustum(-fW, fW, -fH, fH, zNear, zFar)

        # oratio = self.width() /self.height()
        ratio = 1.267
        
 

        if h * ratio > w:
            h = round(w / ratio)

        else:
            w = round(h * ratio)


        # print(" widget resizeGL")        
        # print(w)
        # print(h)
        
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        perspective(45.0, ratio, 1, 1000)
        glMatrixMode(GL_MODELVIEW)
        self.width = w
        self.height = h
        self.wstep = int(0.5+w/5)
        self.hstep = int(0.5+h/4)



        
    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        # glViewport(0,0,self.width-self.wstep,self.height)
        glLoadIdentity()

        self.show.update()
        # ortho = np.multiply(np.array((-2, +2, -2, +2), dtype=float), self.zoomFactor)
	    # glOrtho(ortho[0], ortho[1], ortho[2], ortho[3], 4.0, 15.0)

        glViewport(0,0,self.width-self.wstep,self.height)
        # glLoadIdentity()
        # Color picking / editing
        # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling
        if not self.type.mesh.select.state==2 and not self.type.mesh.select.state==12: # Draw back buffer colors
            #print(self.x,self.y,self.height)
            self.show.pick(self.x,self.y,self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        elif self.type.mesh.select.state==2: # Editing joint geometry
            self.type.mesh.select.edit([self.x,self.y], self.show.view.xrot, self.show.view.yrot, w=self.width, h=self.height)
        elif self.type.mesh.select.state==12: # Editing timber orientation/position
            self.type.mesh.select.move([self.x,self.y], self.show.view.xrot, self.show.view.yrot)

        
        # Display main geometry
        self.show.end_grains()
        if self.show.view.show_feedback:
            self.show.unfabricatable()
            self.show.nondurable()
            self.show.unconnected()
            self.show.unbridged()
            self.show.checker()
            self.show.arrows()
            show_area=False #<--replace by checkbox...
            if show_area:
                self.show.area()
        self.show.joint_geometry()

        if self.type.mesh.select.suggstate>=0:
            index=self.type.mesh.select.suggstate
            if len(self.type.sugs)>index: self.show.difference_suggestion(index)

        # Display editing in action
        self.show.selected()
        self.show.moving_rotating()
        
        # Display milling paths
        self.show.milling_paths()

        # Suggestions
        if self.show.view.show_suggestions:
            for i in range(len(self.type.sugs)):
                hquater = self.height/4
                wquater = self.width/5
                glViewport(self.width-self.wstep,self.height-self.hstep*(i+1),self.wstep,self.hstep)
                glLoadIdentity()
                if i==self.type.mesh.select.suggstate:
                    glEnable(GL_SCISSOR_TEST)
                    glScissor(self.width-self.wstep,self.height-self.hstep*(i+1),self.wstep,self.hstep)
                    glClearDepth(1.0)
                    glClearColor(0.9, 0.9, 0.9, 1.0) #light grey
                    glClear(GL_COLOR_BUFFER_BIT)
                    glDisable(GL_SCISSOR_TEST)
                self.show.joint_geometry(mesh=self.type.sugs[i],lw=2,hidden=False)

                
    # def resizeGL(self, w, h):
    #     # setup viewport, projection etc.:
    #     glViewport(0,0,w,h)
    #     self.show.update()

    # resize_cb = pyqtSignal(int,int)
    # def resizeGL(self, width, height):
    #     if height == 0: height = 1
    #     self.resize_cb.emit(width,height)
        


    def mousePressEvent(self, e):
        print("mouse_pressed")
        if e.button() == Qt.LeftButton:
            if time.time()-self.click_time<0.2:
                self.show.view.open_joint = not self.show.view.open_joint
            elif self.type.mesh.select.state==0: #face hovered
                self.type.mesh.select.start_pull([self.parent.scaling*e.x(),self.parent.scaling*e.y()])
            elif self.type.mesh.select.state==10: #body hovered
                self.type.mesh.select.start_move([self.parent.scaling*e.x(),self.parent.scaling*e.y()],h=self.height)
            #SUGGESTION PICK
            elif self.type.mesh.select.suggstate>=0:
                index = self.type.mesh.select.suggstate
                if len(self.type.sugs)>index:
                    self.type.mesh = Geometries(self.type,hfs=self.type.sugs[index].height_fields)
                    self.type.sugs = []
                    self.type.combine_and_buffer_indices()
                    self.type.mesh.select.suggstate=-1
            #GALLERY PICK -- not implemented currently
            #elif type.mesh.select.gallstate>=0:
            #    index = type.mesh.select.gallstate
            #    if index<len(type.gals):
            #        type.mesh = Geometries(type,hfs=type.gals[index].height_fields)
            #        type.gals = []
            #        view_opt.gallery=False
            #        type.gallary_start_index = -20
            #        type.combine_and_buffer_indices()
            else: self.click_time = time.time()
        elif e.button() == Qt.RightButton:
            print("start rot")
            self.show.view.start_rotation_xy(self.parent.scaling*e.x(),self.parent.scaling*e.y())


    def mouseMoveEvent(self, e):
        self.x = self.parent.scaling*e.x()
        self.y = self.parent.scaling*e.y()
        if self.show.view.dragged:
            self.show.view.update_rotation_xy(self.x,self.y)

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.type.mesh.select.state==2: #face pulled
                self.type.mesh.select.end_pull()
            elif self.type.mesh.select.state==12: #body moved
                self.type.mesh.select.end_move()
        elif e.button() == Qt.RightButton:
            self.show.view.end_rotation()
    def minimumSizeHint(self):
        return QSize(50, 50)
    def sizeHint(self):        
        # print("resize Hint!")
        return QSize(800, 800)
    # def resizeEvent(self, event):
    #     print(' widget resizeEvent')
        # self.resizeGL(self.width(), self.height())
            # self.resize(self.width, self.height)
    
            

class MovieSplashScreen(QSplashScreen):

    def __init__(self, movie, parent = None):
        
        
        movie.jumpToFrame(0)
        pixmap = QPixmap(movie.frameRect().size())
   
        QSplashScreen.__init__(self, pixmap)
        self.movie = movie
        self.movie.frameChanged.connect(self.repaint)
    
    def showEvent(self, event):
        self.movie.start()
    
    def hideEvent(self, event):
        self.movie.stop()
    
    def paintEvent(self, event):
    
        painter = QPainter(self)
        pixmap = self.movie.currentPixmap()
        self.setMask(pixmap.mask())
        painter.drawPixmap(0, 0, pixmap)


    
    def sizeHint(self):
        return self.movie.scaledSize()


class mainWindow(QMainWindow):

    def __init__(self, *args):
        super(mainWindow, self).__init__(*args)
        self.scaling = self.devicePixelRatioF()

        loadUi('Tsugite.ui', self)
        self.setupUi()

        self.title = "Tsugite"
        self.filename = get_untitled_filename("Untitled","tsu","_")
        self.setWindowTitle(self.filename.split(os.sep)[-1]+" - "+self.title)
        self.setWindowIcon(QIcon("tsugite_icon.png"))

        self.glWidget = GLWidget(self)

        self.H_layout.addWidget(self.glWidget)
        # self.setCentralWidget(self.glWidget)
        
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("To open and close the joint: PRESS 'Open/close joint' button or DOUBLE-CLICK anywhere inside the window.")

        timer = QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()


    from _mainWindow import setupUi
    from _mainWindow import open_close_joint
    from _mainWindow import set_feedback_view
    from _mainWindow import change_sliding_axis
    from _mainWindow import change_number_of_timbers
    from _mainWindow import change_resolution
    from _mainWindow import set_angle_of_intersection
    from _mainWindow import set_timber_X
    from _mainWindow import set_timber_Y
    from _mainWindow import set_timber_Z
    from _mainWindow import set_all_timber_same
    from _mainWindow import randomize_geometry
    from _mainWindow import clear_geometry
    from _mainWindow import set_milling_bit_diameter
    from _mainWindow import set_fab_tolerance
    from _mainWindow import set_fab_speed
    from _mainWindow import set_fab_spindlespeed
    from _mainWindow import set_milling_path_axis_alginement
    from _mainWindow import set_incremental
    from _mainWindow import set_interpolation
    from _mainWindow import set_millingpath_view
    from _mainWindow import export_gcode
    from _mainWindow import set_gcode_as_standard
    from _mainWindow import set_nccode_as_standard
    from _mainWindow import set_sbp_as_standard
    from _mainWindow import new_file
    from _mainWindow import open_file
    from _mainWindow import save_file
    from _mainWindow import save_file_as
    from _mainWindow import show_hide_hidden_lines
    from _mainWindow import show_hide_timbers
    from _mainWindow import show_all_timbers
    from _mainWindow import set_standard_rotation
    from _mainWindow import set_closest_plane_rotation
    from _mainWindow import set_ui_values

    def keyPressEvent(self, e):
        if e.key()==Qt.Key_Shift:
            self.glWidget.type.mesh.select.shift = True
            self.glWidget.type.mesh.select.refresh = True

    def keyReleaseEvent(self, e):
        if e.key()==Qt.Key_Shift:
            self.glWidget.type.mesh.select.shift = False
            self.glWidget.type.mesh.select.refresh = True

    # def resizeEvent(self, event):
        # print('  resizeEvent')

    #     # self.resize(self.width(), self.height())
    #     self.glWidget.resizeEvent(event)
  

#deal with dpi
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
app = QApplication(sys.argv)
movie = QMovie("tsugite_loading_3d.gif")

splash = MovieSplashScreen(movie)

splash.show()

start = time.time()

while movie.state() == QMovie.Running and time.time() < start + 1:
    app.processEvents()
#screen = app.screens()[0]
#dpi = screen.physicalDotsPerInch()

window = mainWindow()
window.show()
splash.finish(window)
sys.exit(app.exec_())
