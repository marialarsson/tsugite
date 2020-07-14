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

#My files
from Types import Types
from Geometries import Geometries
from ViewSettings import ViewSettings
from Display import Display

def create_color_shaders():
    vertex_shader = """
    #version 330
    #extension GL_ARB_explicit_attrib_location : require
    #extension GL_ARB_explicit_uniform_location : require
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    layout(location = 2) in vec2 inTexCoords;
    layout(location = 3) uniform mat4 transform;
    layout(location = 4) uniform mat4 translate;
    layout(location = 5) uniform vec3 myColor;
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* translate* vec4(position, 1.0f);
        newColor = myColor;
        outTexCoords = inTexCoords;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;
    in vec2 outTexCoords;
    out vec4 outColor;
    uniform sampler2D samplerTex;
    void main()
    {
        outColor = vec4(newColor, 1.0);
    }
    """
    # Compiling the shaders
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    return shader

def init_shader(shader,view_opt):
    glUseProgram(shader)
    rot_x = pyrr.Matrix44.from_x_rotation(view_opt.xrot)
    rot_y = pyrr.Matrix44.from_y_rotation(view_opt.yrot)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)

class GLWidget(QGLWidget):
    def __init__(self, parent=None):
        self.parent = parent
        self.view = ViewSettings() #view
        QGLWidget.__init__(self, parent)
        self.setMinimumSize(500, 500)

    def initializeGL(self):
        self.qglClearColor(QColor(255, 255, 255))
        glEnable(GL_DEPTH_TEST)                  # enable depth testing
        glEnable(GL_SCISSOR_TEST)
        self.type = Types(fs=[[[2,0]],[[2,1]]],sax=2,dim=3,ang=90, wd=[30,30])
        self.display = Display(self.type,self.view)
        self.shader_col = create_color_shaders()
        init_shader(self.shader_col, self.view)

    def resizeGL(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        #calculate joint opening distance
        #if (self.view.open_joint and self.view.open_ratio<self.type.noc-1) or (not self.view.open_joint and self.view.open_ratio>0):
        #    self.view.set_joint_opening_distance(self.type.noc)
        self.display.joint_geometry(self.type.mesh)

class mainWindow(QMainWindow):

    def __init__(self, *args):
        super(mainWindow, self).__init__(*args)

        loadUi('Tsugite.ui', self)
        self.setupUi()

        self.glWidget = GLWidget(self)
        self.H_layout.addWidget(self.glWidget)

        timer = QTimer(self)
        timer.setInterval(20)   # period, in milliseconds
        timer.timeout.connect(self.glWidget.updateGL)
        timer.start()


    def setupUi(self):
        self.findChild(QSlider, "sliderOPEN").valueChanged.connect(self.set_opening_distance)
        #feedback
        #suggestions
        self.findChild(QComboBox, "comboSLIDE").currentTextChanged.connect(self.change_sliding_axis)
        self.findChild(QSpinBox, "spinBoxNUM").valueChanged.connect(self.change_number_of_timbers)
        self.findChild(QSpinBox, "spinBoxRES").valueChanged.connect(self.change_resolution)
        self.findChild(QDoubleSpinBox, "spinANG").valueChanged.connect(self.set_angle_of_intersection)
        #timber dimensions
        self.findChild(QPushButton, "buttonR").clicked.connect(self.randomize_geometry)
        self.findChild(QPushButton, "buttonC").clicked.connect(self.clear_geometry)
        #gallery
        #---
        #milling bit diameter
        #tolerances
        #show milling path
        #export gcode

    @pyqtSlot()
    def set_opening_distance(self):
        val = self.findChild(QSlider, "sliderOPEN").value()
        self.glWidget.view.set_absolute_joint_opening_distance(self.glWidget.type.noc,val)

    @pyqtSlot()
    def change_sliding_axis(self):
        ax = self.findChild(QComboBox, "comboSLIDE").currentIndex()
        self.glWidget.type.update_sliding_direction(ax)

    @pyqtSlot()
    def set_angle_of_intersection(self):
        val = self.findChild(QDoubleSpinBox, "spinANG").value()
        self.glWidget.type.update_angle(val)

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
    def randomize_geometry(self):
        self.glWidget.type.mesh.randomize_height_fields()

    @pyqtSlot()
    def clear_geometry(self):
        self.glWidget.type.mesh.clear_height_fields()



app = QApplication(sys.argv)
window = mainWindow()
window.show()
sys.exit(app.exec_())
