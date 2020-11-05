#!/usr/bin/env python3

import sys

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from PyQt5.QtOpenGL import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

#Other
import sys
import numpy as np
import time
import math
import os
import cv2

#My files
from Types import Types
from Geometries import Geometries
from ViewSettings import ViewSettings
from Show import Show

def get_untitled_filename(name,ext,sep):
    # list of all filenames with specified extension in the current directory
    extnames = []
    for item in os.listdir():
        items = item.split(".")
        if len(items)>1 and items[1]==ext:
            extnames.append(items[0])
    # if the name already exists, append seperator and number
    fname = name
    cnt = 1
    while fname in extnames:
        fname = name+sep+str(cnt)
        cnt+=1
    # add path and extension, return
    fname = os.getcwd()+os.sep+fname+"."+ext
    return fname

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(800, 800)
        self.setMouseTracking(True)
        self.click_time = time.time()
        self.x = 0
        self.y = 0

    def initializeGL(self):
        self.qglClearColor(QColor(255, 255, 255))
        glEnable(GL_DEPTH_TEST)                  # enable depth testing
        sax = self.parent.findChild(QComboBox, "comboSLIDE").currentIndex()
        dim = self.parent.findChild(QSpinBox, "spinBoxRES").value()
        ang = self.parent.findChild(QDoubleSpinBox, "spinANG").value()
        dx = self.parent.findChild(QDoubleSpinBox, "spinDX").value()
        dy = self.parent.findChild(QDoubleSpinBox, "spinDY").value()
        dz = self.parent.findChild(QDoubleSpinBox, "spinDZ").value()
        dia = self.parent.findChild(QDoubleSpinBox, "spinDIA").value()
        tol = self.parent.findChild(QDoubleSpinBox, "spinTOL").value()
        spe = self.parent.findChild(QSpinBox, "spinSPEED").value()
        spi = self.parent.findChild(QSpinBox, "spinSPINDLE").value()
        aax = self.parent.findChild(QComboBox, "comboALIGN").currentIndex()
        inc = self.parent.findChild(QCheckBox, "checkINC").isChecked()
        fin = self.parent.findChild(QCheckBox, "checkFIN").isChecked()
        if self.parent.findChild(QRadioButton, "radioGCODE").isChecked(): ext = "gcode"
        elif self.parent.findChild(QRadioButton, "radioNC").isChecked(): ext = "nc"
        elif self.parent.findChild(QRadioButton, "radioSBP").isChecked(): ext = "sbp"
        self.type = Types(self,fs=[[[2,0]],[[2,1]]],sax=sax,dim=dim,ang=ang, td=[dx,dy,dz], fabtol=tol, fabdia=dia, fspe=spe, fspi=spi, fabext=ext, align_ax=aax, incremental=inc, finterp=fin)
        self.show = Show(self,self.type)

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        self.wstep = int(0.5+width/5)
        self.hstep = int(0.5+height/4)
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glViewport(0,0,self.width-self.wstep,self.height)
        glLoadIdentity()

        self.show.update()

        if not self.show.view.gallery:
            glViewport(0,0,self.width-self.wstep,self.height)
            glLoadIdentity()
            # Color picking / editing
            # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling
            if not self.type.mesh.select.state==2 and not self.type.mesh.select.state==12: # Draw back buffer colors
                self.show.pick(self.x,self.y,self.height)
                glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            elif self.type.mesh.select.state==2: # Edit joint geometry
                self.type.mesh.select.edit([self.x,self.y], self.show.view.xrot, self.show.view.yrot, w=self.width, h=self.height)
            elif self.type.mesh.select.state==12: # Edit timber orientation/position
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
        else:
            print("gallery mode")

    def mousePressEvent(self, e):
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
            #GALLERY PICK
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

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("To open and close the joint: PRESS 'Open/close joint' button or DOUBLE-CLICK anywhere inside the window.")

        timer = QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()

    def setupUi(self):
        #get opengl window size
        self.x_range = [10,500]
        self.y_range = [10,500]

        #---Design
        self.findChild(QPushButton, "buttonOPEN").clicked.connect(self.open_close_joint)
        self.findChild(QCheckBox, "checkFEED").stateChanged.connect(self.set_feedback_view)
        #suggestions
        self.findChild(QComboBox, "comboSLIDE").currentTextChanged.connect(self.change_sliding_axis)
        self.findChild(QSpinBox, "spinBoxNUM").valueChanged.connect(self.change_number_of_timbers)
        self.findChild(QSpinBox, "spinBoxRES").valueChanged.connect(self.change_resolution)
        self.findChild(QDoubleSpinBox, "spinANG").valueChanged.connect(self.set_angle_of_intersection)
        self.findChild(QCheckBox, "checkCUBE").stateChanged.connect(self.set_all_timber_same)
        self.findChild(QDoubleSpinBox, "spinDX").valueChanged.connect(self.set_timber_X)
        self.findChild(QDoubleSpinBox, "spinDY").valueChanged.connect(self.set_timber_Y)
        self.findChild(QDoubleSpinBox, "spinDZ").valueChanged.connect(self.set_timber_Z)
        self.findChild(QPushButton, "buttonR").clicked.connect(self.randomize_geometry)
        self.findChild(QPushButton, "buttonC").clicked.connect(self.clear_geometry)
        #gallery
        #---Fabrication
        self.findChild(QDoubleSpinBox, "spinDIA").valueChanged.connect(self.set_milling_bit_diameter)
        self.findChild(QDoubleSpinBox, "spinTOL").valueChanged.connect(self.set_fab_tolerance)
        self.findChild(QSpinBox, "spinSPEED").valueChanged.connect(self.set_fab_speed)
        self.findChild(QSpinBox, "spinSPINDLE").valueChanged.connect(self.set_fab_spindlespeed)
        self.findChild(QComboBox, "comboALIGN").currentTextChanged.connect(self.set_milling_path_axis_alginement)
        self.findChild(QCheckBox, "checkINC").stateChanged.connect(self.set_incremental)
        self.findChild(QCheckBox, "checkFIN").stateChanged.connect(self.set_interpolation)
        self.findChild(QPushButton, "buttonPATH").clicked.connect(self.set_millingpath_view)
        self.findChild(QPushButton, "buttonGCODE").clicked.connect(self.export_gcode)
        self.findChild(QRadioButton, "radioGCODE").toggled.connect(self.set_gcode_as_standard)
        self.findChild(QRadioButton, "radioNC").toggled.connect(self.set_nccode_as_standard)
        self.findChild(QRadioButton, "radioSBP").toggled.connect(self.set_sbp_as_standard)
        #---MENU
        #---File
        self.findChild(QAction, "actionNew").triggered.connect(self.new_file)
        self.findChild(QAction, "actionOpen").triggered.connect(self.open_file)
        self.findChild(QAction, "actionSave").triggered.connect(self.save_file)
        self.findChild(QAction, "actionSaveas").triggered.connect(self.save_file_as)
        self.findChild(QAction, "actionImage").triggered.connect(self.save_screenshot)
        #---View
        self.findChild(QAction, "actionHIDDEN").triggered.connect(self.show_hide_hidden_lines)
        self.findChild(QAction, "actionA").triggered.connect(self.show_hide_timbers)
        self.findChild(QAction, "actionB").triggered.connect(self.show_hide_timbers)
        self.findChild(QAction, "actionC").triggered.connect(self.show_hide_timbers)
        self.findChild(QAction, "actionD").triggered.connect(self.show_hide_timbers)
        self.findChild(QAction, "actionALL").triggered.connect(self.show_all_timbers)
        self.findChild(QAction, "actionAXO").triggered.connect(self.set_standard_rotation)
        self.findChild(QAction, "actionPLN").triggered.connect(self.set_closest_plane_rotation)


    @pyqtSlot()
    def open_close_joint(self):
        self.glWidget.show.view.open_joint = not self.glWidget.show.view.open_joint

    @pyqtSlot()
    def set_feedback_view(self):
        bool = self.findChild(QCheckBox, "checkFEED").checkState()
        self.glWidget.show.view.show_feedback = bool

    @pyqtSlot()
    def change_sliding_axis(self):
        ax = self.findChild(QComboBox, "comboSLIDE").currentIndex()
        bool, msg = self.glWidget.type.update_sliding_direction(ax)
        print("msg")
        #QMessageBox = ...

    @pyqtSlot()
    def change_number_of_timbers(self):
        val = self.findChild(QSpinBox, "spinBoxNUM").value()
        self.glWidget.type.update_number_of_components(val)

    @pyqtSlot()
    def change_resolution(self):
        val = self.findChild(QSpinBox, "spinBoxRES").value()
        add = val-self.glWidget.type.dim
        self.glWidget.type.update_dimension(add)

    @pyqtSlot()
    def set_angle_of_intersection(self):
        val = self.findChild(QDoubleSpinBox, "spinANG").value()
        self.glWidget.type.update_angle(val)

    @pyqtSlot()
    def set_timber_X(self):
        val = self.findChild(QDoubleSpinBox, "spinDX").value()
        mp = self.glWidget.show.view.show_milling_path
        if mp: self.glWidget.type.create_and_buffer_vertices(milling_path=True)
        if self.findChild(QCheckBox, "checkCUBE").isChecked():
            self.glWidget.type.update_timber_width_and_height([0,1,2],val,milling_path=mp)
            self.findChild(QDoubleSpinBox, "spinDY").setValue(val)
            self.findChild(QDoubleSpinBox, "spinDZ").setValue(val)
        else:
            self.glWidget.type.update_timber_width_and_height([0],val,milling_path=mp)

    @pyqtSlot()
    def set_timber_Y(self):
        val = self.findChild(QDoubleSpinBox, "spinDY").value()
        mp = self.glWidget.show.view.show_milling_path
        if self.findChild(QCheckBox, "checkCUBE").isChecked():
            self.glWidget.type.update_timber_width_and_height([0,1,2],val,milling_path=mp)
            self.findChild(QDoubleSpinBox, "spinDX").setValue(val)
            self.findChild(QDoubleSpinBox, "spinDZ").setValue(val)
        else:
            self.glWidget.type.update_timber_width_and_height([1],val,milling_path=mp)

    @pyqtSlot()
    def set_timber_Z(self):
        val = self.findChild(QDoubleSpinBox, "spinDZ").value()
        mp = self.glWidget.show.view.show_milling_path
        if self.findChild(QCheckBox, "checkCUBE").isChecked():
            self.glWidget.type.update_timber_width_and_height([0,1,2],val,milling_path=mp)
            self.findChild(QDoubleSpinBox, "spinDX").setValue(val)
            self.findChild(QDoubleSpinBox, "spinDY").setValue(val)
        else:
            self.glWidget.type.update_timber_width_and_height([2],val,milling_path=mp)

    @pyqtSlot()
    def set_all_timber_same(self):
        mp = self.glWidget.show.view.show_milling_path
        if self.findChild(QCheckBox, "checkCUBE").isChecked():
            val = self.glWidget.type.real_tim_dims[0]
            self.glWidget.type.update_timber_width_and_height([0,1,2],val,milling_path=mp)
            self.findChild(QDoubleSpinBox, "spinDY").setValue(val)
            self.findChild(QDoubleSpinBox, "spinDZ").setValue(val)

    @pyqtSlot()
    def randomize_geometry(self):
        self.glWidget.type.mesh.randomize_height_fields()

    @pyqtSlot()
    def clear_geometry(self):
        self.glWidget.type.mesh.clear_height_fields()

    @pyqtSlot()
    def set_milling_bit_diameter(self):
        val = self.findChild(QDoubleSpinBox, "spinDIA").value()
        self.glWidget.type.fab.real_dia = val
        self.glWidget.type.fab.rad = 0.5*self.glWidget.type.fab.real_dia-self.glWidget.type.fab.tol
        self.glWidget.type.fab.dia = 2*self.glWidget.type.fab.rad
        self.glWidget.type.fab.vdia = self.glWidget.type.fab.dia/self.glWidget.type.ratio
        self.glWidget.type.fab.vrad = self.glWidget.type.fab.rad/self.glWidget.type.ratio
        if self.glWidget.show.view.show_milling_path:
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)

    @pyqtSlot()
    def set_fab_tolerance(self):
        val = self.findChild(QDoubleSpinBox, "spinTOL").value()
        self.glWidget.type.fab.tol = val
        self.glWidget.type.fab.rad = 0.5*self.glWidget.type.fab.real_dia-self.glWidget.type.fab.tol
        self.glWidget.type.fab.dia = 2*self.glWidget.type.fab.rad
        self.glWidget.type.fab.vdia = self.glWidget.type.fab.dia/self.glWidget.type.ratio
        self.glWidget.type.fab.vrad = self.glWidget.type.fab.rad/self.glWidget.type.ratio
        self.glWidget.type.fab.vtol = self.glWidget.type.fab.tol/self.glWidget.type.ratio
        if self.glWidget.show.view.show_milling_path:
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)

    @pyqtSlot()
    def set_fab_speed(self):
        val = self.findChild(QSpinBox, "spinSPEED").value()
        self.glWidget.type.fab.speed = val


    @pyqtSlot()
    def set_fab_spindlespeed(self):
        val = self.findChild(QSpinBox, "spinSPINDLE").value()
        self.glWidget.type.fab.spindlespeed = val

    @pyqtSlot()
    def set_milling_path_axis_alginement(self):
        val = self.findChild(QComboBox, "comboALIGN").currentIndex()
        self.glWidget.type.fab.align_ax = val

    @pyqtSlot()
    def set_incremental(self):
        val = self.findChild(QCheckBox, "checkINC").isChecked()
        self.glWidget.type.incremental = val

    @pyqtSlot()
    def set_interpolation(self):
        val = self.findChild(QCheckBox, "checkFIN").isChecked()
        self.glWidget.type.fab.interp = val

    @pyqtSlot()
    def set_millingpath_view(self):
        self.glWidget.show.view.show_milling_path = not self.glWidget.show.view.show_milling_path
        bool = self.glWidget.show.view.show_milling_path
        self.glWidget.type.create_and_buffer_vertices(milling_path=bool)
        self.glWidget.type.combine_and_buffer_indices(milling_path=bool)

    @pyqtSlot()
    def export_gcode(self):
        if not self.glWidget.show.view.show_milling_path:
            self.glWidget.show.view.show_milling_path = True
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)
        self.glWidget.type.fab.export_gcode(filename_tsu=self.filename)

    @pyqtSlot()
    def set_gcode_as_standard(self):
        bool = self.findChild(QRadioButton, "radioGCODE").isChecked()
        if bool: self.glWidget.type.fab.ext = "gcode"

    @pyqtSlot()
    def set_nccode_as_standard(self):
        bool = self.findChild(QRadioButton, "radioNC").isChecked()
        if bool: self.glWidget.type.fab.ext = "nc"

    @pyqtSlot()
    def set_sbp_as_standard(self):
        bool = self.findChild(QRadioButton, "radioSBP").isChecked()
        if bool: self.glWidget.type.fab.ext = "sbp"

    @pyqtSlot()
    def new_file(self):
        self.filename = get_untitled_filename("Untitled","tsu","_")
        self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
        self.glWidget.show.view.show_milling_path=False
        self.glWidget.type.reset()
        self.set_ui_values()
        self.show_all_timbers()

    @pyqtSlot()
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(filter="Tsugite files (*.tsu)")
        if filename!='':
            self.filename = filename
            self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
            self.findChild(QCheckBox, "checkCUBE").setChecked(False)
            self.glWidget.type.open(self.filename)
            self.set_ui_values()

    @pyqtSlot()
    def save_file(self):
        self.glWidget.type.save(self.filename)

    @pyqtSlot()
    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(filter="Tsugite files (*.tsu)")
        if filename!='':
            self.filename = filename
            self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
            self.glWidget.type.save(self.filename)

    @pyqtSlot()
    def save_screenshot(self):
        img_filename, _ = QFileDialog.getSaveFileName(filter="Image file (*.png)")
        if img_filename!='':
            image_buffer = glReadPixels(0, 0, self.glWidget.width, self.glWidget.height, GL_RGB, GL_UNSIGNED_BYTE)
            image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(self.glWidget.height, self.glWidget.width, 3)
            image = np.flip(image,axis=0)
            image = np.flip(image,axis=2)
            cv2.imwrite(img_filename, image)
            print("Saved screenshot to",img_filename)

    @pyqtSlot()
    def show_hide_hidden_lines(self):
        self.glWidget.show.view.show_hidden_lines = self.findChild(QAction, "actionHIDDEN").isChecked()

    @pyqtSlot()
    def show_hide_timbers(self):
        names = ["A","B","C","D"]
        for i,item in enumerate(names):
            bool = self.findChild(QAction, "action"+names[i]).isChecked()
            self.glWidget.show.view.hidden[i] = not bool

    @pyqtSlot()
    def show_all_timbers(self):
        names = ["A","B","C","D"]
        for i,item in enumerate(names):
            self.findChild(QAction, "action"+names[i]).setChecked(True)
            self.glWidget.show.view.hidden[i] = False

    @pyqtSlot()
    def set_standard_rotation(self):
        self.glWidget.show.view.xrot = 0.8
        self.glWidget.show.view.yrot = 0.4

    @pyqtSlot()
    def set_closest_plane_rotation(self):
        xrot = self.glWidget.show.view.xrot
        yrot = self.glWidget.show.view.yrot
        nang = 0.5*math.pi
        xrot = round(xrot/nang,0)*nang
        yrot = round(yrot/nang,0)*nang
        self.glWidget.show.view.xrot = xrot
        self.glWidget.show.view.yrot = yrot

    def set_ui_values(self):
        self.findChild(QComboBox, "comboSLIDE").setCurrentIndex(self.glWidget.type.sax)
        self.findChild(QSpinBox, "spinBoxNUM").setValue(self.glWidget.type.noc)
        self.findChild(QSpinBox, "spinBoxRES").setValue(self.glWidget.type.dim)
        self.findChild(QDoubleSpinBox, "spinANG").setValue(self.glWidget.type.ang)
        self.findChild(QDoubleSpinBox, "spinDX").setValue(self.glWidget.type.real_tim_dims[0])
        self.findChild(QDoubleSpinBox, "spinDY").setValue(self.glWidget.type.real_tim_dims[1])
        self.findChild(QDoubleSpinBox, "spinDZ").setValue(self.glWidget.type.real_tim_dims[2])
        if np.max(self.glWidget.type.real_tim_dims)==np.min(self.glWidget.type.real_tim_dims):
            self.findChild(QCheckBox, "checkCUBE").setChecked(True)
        else: self.findChild(QCheckBox, "checkCUBE").setChecked(False)
        self.findChild(QDoubleSpinBox, "spinDIA").setValue(self.glWidget.type.fab.real_dia)
        self.findChild(QDoubleSpinBox, "spinTOL").setValue(self.glWidget.type.fab.tol)
        self.findChild(QSpinBox, "spinSPEED").setValue(self.glWidget.type.fab.speed)
        self.findChild(QSpinBox, "spinSPINDLE").setValue(self.glWidget.type.fab.spindlespeed)
        self.findChild(QCheckBox, "checkINC").setChecked(self.glWidget.type.incremental)
        self.findChild(QCheckBox, "checkFIN").setChecked(self.glWidget.type.fab.interp)
        self.findChild(QComboBox, "comboALIGN").setCurrentIndex(self.glWidget.type.fab.align_ax)
        if self.glWidget.type.fab.ext=="gcode":
            self.findChild(QRadioButton, "radioGCODE").setChecked(True)
        elif self.glWidget.type.fab.ext=="sbp":
            self.findChild(QRadioButton, "radioSBP").setChecked(True)
        elif self.glWidget.type.fab.ext=="nc":
            self.findChild(QRadioButton, "radioNC").setChecked(True)

    def keyPressEvent(self, e):
        if e.key()==Qt.Key_Shift:
            self.glWidget.type.mesh.select.shift = True
            self.glWidget.type.mesh.select.refresh = True

    def keyReleaseEvent(self, e):
        if e.key()==Qt.Key_Shift:
            self.glWidget.type.mesh.select.shift = False
            self.glWidget.type.mesh.select.refresh = True

#deal with dpi
QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True) #enable highdpi scaling
QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True) #use highdpi icons
app = QApplication(sys.argv)
#screen = app.screens()[0]
#dpi = screen.physicalDotsPerInch()
window = mainWindow()
window.show()
sys.exit(app.exec_())
