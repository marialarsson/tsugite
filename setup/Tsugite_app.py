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
import pyrr
import numpy as np
import time
import math
import os
import cv2

#My files
from Types import Types
from Geometries import Geometries
from ViewSettings import ViewSettings
from Display import Display

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
    fname = os.getcwd()+"\\"+fname+"."+ext
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
        glEnable(GL_SCISSOR_TEST)
        sax = self.parent.findChild(QComboBox, "comboSLIDE").currentIndex()
        dim = self.parent.findChild(QSpinBox, "spinBoxRES").value()
        ang = self.parent.findChild(QDoubleSpinBox, "spinANG").value()
        w = self.parent.findChild(QDoubleSpinBox, "spinW").value()
        d = self.parent.findChild(QDoubleSpinBox, "spinD").value()
        rad = 0.5*self.parent.findChild(QDoubleSpinBox, "spinDIA").value()
        tol = self.parent.findChild(QDoubleSpinBox, "spinTOL").value()
        self.type = Types(fs=[[[2,0]],[[2,1]]],sax=sax,dim=dim,ang=ang, wd=[w,d], fabtol=tol, fabrad=rad)
        self.display = Display(self.type)

    def resizeGL(self, width, height):
        self.width = width
        self.height = height
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        self.display.update()

        # Gallery

        # Color picking / editing
        # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling
        if not self.type.mesh.select.state==2 and not self.type.mesh.select.state==12: # Draw back buffer colors
            self.display.pick(self.x,self.y,self.height)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        elif self.type.mesh.select.state==2: # Edit joint geometry
            self.type.mesh.select.edit([self.x,self.y], self.display.view.xrot, self.display.view.yrot, w=self.width, h=self.height)
        elif self.type.mesh.select.state==12: # Edit timber orientation/position
            self.type.mesh.select.move([self.x,self.y], self.display.view.xrot, self.display.view.yrot)

        # Display main geometry
        self.display.end_grains()
        if self.display.view.show_feedback:
            self.display.unfabricatable()
            self.display.nondurable()
            self.display.unconnected()
            self.display.unbridged()
            self.display.checker()
            self.display.arrows()
            show_area=False #<--replace by checkbox...
            if show_area:
                self.display.area()
        self.display.joint_geometry()

        # Display editing in action
        self.display.selected()
        self.display.moving_rotating()

        # Display milling paths
        self.display.milling_paths()

        # Suggestions
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            if time.time()-self.click_time<0.2:
                self.display.view.open_joint = not self.display.view.open_joint
            elif self.type.mesh.select.state==0: #face hovered
                self.type.mesh.select.start_pull([e.x(),e.y()])
            elif self.type.mesh.select.state==10: #body hovered
                self.type.mesh.select.start_move([e.x(),e.y()])
            #SUGGESTION PICK
            #elif self.type.mesh.select.suggstate>=0:
            #    index = type.mesh.select.suggstate
            #    if len(type.sugs)>index:
            #        type.mesh = Geometries(type,hfs=type.sugs[index].height_fields)
            #        type.sugs = []
            #        type.combine_and_buffer_indices()
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
            self.display.view.start_rotation_xy(e.x(),e.y())

    def mouseMoveEvent(self, e):
        self.x = e.x()
        self.y = e.y()
        if self.display.view.dragged:
            self.display.view.update_rotation_xy(e.x(),e.y())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.LeftButton:
            if self.type.mesh.select.state==2: #face pulled
                self.type.mesh.select.end_pull()
            elif self.type.mesh.select.state==12: #body moved
                self.type.mesh.select.end_move()
        elif e.button() == Qt.RightButton:
            self.display.view.end_rotation()

    def keyPressEvent(self, e):
        if e.key()==Qt.Key_S:
            print('S')

class mainWindow(QMainWindow):

    def __init__(self, *args):
        super(mainWindow, self).__init__(*args)

        loadUi('Tsugite.ui', self)
        self.setupUi()

        self.title = "Tsugite"
        self.filename = get_untitled_filename("Untitled","tsu","_")
        self.setWindowTitle(self.filename.split("\\")[-1]+" - "+self.title)
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
        self.findChild(QDoubleSpinBox, "spinW").valueChanged.connect(self.set_timber_width)
        self.findChild(QDoubleSpinBox, "spinD").valueChanged.connect(self.set_timber_depth)
        self.findChild(QPushButton, "buttonR").clicked.connect(self.randomize_geometry)
        self.findChild(QPushButton, "buttonC").clicked.connect(self.clear_geometry)
        #gallery
        #---Fabrication
        self.findChild(QDoubleSpinBox, "spinDIA").valueChanged.connect(self.set_milling_bit_diameter)
        self.findChild(QDoubleSpinBox, "spinTOL").valueChanged.connect(self.set_fab_tolerance)
        self.findChild(QSpinBox, "spinPATHROT").valueChanged.connect(self.set_milling_path_extra_rotation)
        self.findChild(QCheckBox, "checkPATH").stateChanged.connect(self.set_millingpath_view)
        self.findChild(QPushButton, "buttonGCODE").clicked.connect(self.export_gcode)
        #---Menu
        self.findChild(QAction, "actionNew").triggered.connect(self.new_file)
        self.findChild(QAction, "actionOpen").triggered.connect(self.open_file)
        self.findChild(QAction, "actionSave").triggered.connect(self.save_file)
        self.findChild(QAction, "actionSaveas").triggered.connect(self.save_file_as)
        self.findChild(QAction, "actionImage").triggered.connect(self.save_screenshot)

    @pyqtSlot()
    def open_close_joint(self):
        self.glWidget.display.view.open_joint = not self.glWidget.display.view.open_joint

    @pyqtSlot()
    def set_feedback_view(self):
        bool = self.findChild(QCheckBox, "checkFEED").checkState()
        self.glWidget.display.view.show_feedback = bool

    @pyqtSlot()
    def change_sliding_axis(self):
        ax = self.findChild(QComboBox, "comboSLIDE").currentIndex()
        self.glWidget.type.update_sliding_direction(ax)

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
    def set_timber_width(self):
        val = self.findChild(QDoubleSpinBox, "spinW").value()
        self.glWidget.type.update_timber_width_and_height(val,self.glWidget.type.real_comp_depth)

    @pyqtSlot()
    def set_timber_depth(self):
        val = self.findChild(QDoubleSpinBox, "spinD").value()
        self.glWidget.type.update_timber_width_and_height(self.glWidget.type.real_comp_width,val)

    @pyqtSlot()
    def randomize_geometry(self):
        self.glWidget.type.mesh.randomize_height_fields()

    @pyqtSlot()
    def clear_geometry(self):
        self.glWidget.type.mesh.clear_height_fields()

    @pyqtSlot()
    def set_milling_bit_diameter(self):
        val = self.findChild(QDoubleSpinBox, "spinDIA").value()
        self.glWidget.type.fab.rad = 0.5*val-self.glWidget.type.fab.tol
        self.glWidget.type.fab.dia = 2*self.glWidget.type.fab.rad
        self.glWidget.type.fab.vrad = self.glWidget.type.fab.rad/self.glWidget.type.fab.ratio
        self.glWidget.type.fab.vdia = self.glWidget.type.fab.dia/self.glWidget.type.fab.ratio
        if self.glWidget.display.view.show_milling_path:
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)

    @pyqtSlot()
    def set_fab_tolerance(self):
        val = self.findChild(QDoubleSpinBox, "spinTOL").value()
        self.glWidget.type.fab.tol = val
        self.glWidget.type.fab.rad = self.glWidget.type.fab.rad_real #milling bit radius in mm
        self.glWidget.type.fab.rad -= self.glWidget.type.fab.tol
        self.glWidget.type.fab.dia = 2*self.glWidget.type.fab.rad
        self.glWidget.type.fab.vrad = self.glWidget.type.fab.rad/self.glWidget.type.fab.ratio
        self.glWidget.type.fab.vdia = self.glWidget.type.fab.dia/self.glWidget.type.fab.ratio
        if self.glWidget.display.view.show_milling_path:
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)

    def set_milling_path_extra_rotation(self):
        val = self.findChild(QSpinBox, "spinPATHROT").value()
        self.glWidget.type.fab.extra_rot = math.radians(val)

    @pyqtSlot()
    def set_millingpath_view(self):
        bool = self.findChild(QCheckBox, "checkPATH").checkState()
        self.glWidget.display.view.show_milling_path = bool
        if bool:
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)

    @pyqtSlot()
    def export_gcode(self):
        if not self.glWidget.display.view.show_milling_path:
            self.glWidget.display.view.show_milling_path = True
            self.glWidget.type.create_and_buffer_vertices(milling_path=True)
            self.glWidget.type.combine_and_buffer_indices(milling_path=True)
        self.glWidget.type.fab.export_gcode(filename_tsu=self.filename)

    @pyqtSlot()
    def new_file(self):
        self.filename = get_untitled_filename("Untitled","tsu","_")
        #check so that untitled is not already used, if so add 1 , 2, 3 etc.
        self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
        rad = 0.5*self.findChild(QDoubleSpinBox, "spinDIA").value()
        tol = self.findChild(QDoubleSpinBox, "spinTOL").value()
        self.glWidget.type = Types(fs=[[[2,0]],[[2,1]]],sax=0,dim=3,ang=90.0, wd=[44,44], fabtol=tol, fabrad=rad)
        self.set_ui_values()

    @pyqtSlot()
    def open_file(self):
        filename, _ = QFileDialog.getOpenFileName(filter="Tsugite files (*.tsu)")
        if filename!='':
            self.filename = filename
            self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
            self.glWidget.type.mesh.open(self.filename)
            self.set_ui_values()

    @pyqtSlot()
    def save_file(self):
        self.glWidget.type.mesh.save(self.filename)

    @pyqtSlot()
    def save_file_as(self):
        filename, _ = QFileDialog.getSaveFileName(filter="Tsugite files (*.tsu)")
        if filename!='':
            self.filename = filename
            self.setWindowTitle(self.filename.split("/")[-1]+" - "+self.title)
            self.glWidget.type.mesh.save(self.filename)

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

    def set_ui_values(self):
        self.findChild(QComboBox, "comboSLIDE").setCurrentIndex(self.glWidget.type.sax)
        self.findChild(QSpinBox, "spinBoxNUM").setValue(self.glWidget.type.noc)
        self.findChild(QSpinBox, "spinBoxRES").setValue(self.glWidget.type.dim)
        self.findChild(QDoubleSpinBox, "spinANG").setValue(self.glWidget.type.ang)
        self.findChild(QDoubleSpinBox, "spinW").setValue(self.glWidget.type.real_comp_width)
        self.findChild(QDoubleSpinBox, "spinD").setValue(self.glWidget.type.real_comp_depth)

app = QApplication(sys.argv)
window = mainWindow()
window.show()
sys.exit(app.exec_())
