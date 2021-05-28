from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.uic import *
from PyQt5.QtOpenGL import *

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

#My files
from Types import Types
from Show import Show

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
