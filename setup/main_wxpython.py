import wx
from wx import glcanvas
from OpenGL.GL import *
import OpenGL.GL.shaders
from pyrr import Matrix44, matrix44, Vector3
import time, sys
from Geometries import Geometries
import ctypes

vertex_shader = """
# version 330
in layout(location = 0) vec3 positions;
in layout(location = 1) vec3 colors;
out vec3 newColor;
uniform mat4 rotate;
uniform mat4 translate;
uniform mat4 vp;
void main(){
    gl_Position = vp * translate * rotate * vec4(positions, 1.0);
    newColor = colors;
}
"""

fragment_shader = """
# version 330
in vec3 newColor;
out vec4 outColor;
void main(){
    outColor = vec4(newColor, 1.0);
}
"""

def draw_geometry_with_excluded_area(show_geos,screen_geos):
    glDisable(GL_DEPTH_TEST)
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
    glEnable(GL_STENCIL_TEST)
    glStencilFunc(GL_ALWAYS,1,1)
    glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
    for geo in show_geos:
        glDrawElements(geo[0], geo[1], GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo[2]))
    glEnable(GL_DEPTH_TEST)
    glStencilFunc(GL_EQUAL,1,1)
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
    for geo in screen_geos:
        glDrawElements(geo[0], geo[1], GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo[2]))
    glDisable(GL_STENCIL_TEST)
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
    for geo in show_geos:
        glDrawElements(geo[0], geo[1], GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo[2]))

class OpenGLCanvas(glcanvas.GLCanvas):
    def __init__(self, parent):
        self.size = (1600, 1600)
        self.aspect_ratio = self.size[0] / self.size[1]
        #ctypes.windll.shcore.SetProcessDpiAwareness(1)
        #dispAttrs = wx.glcanvas.GLAttributes()
        #dispAttrs.PlatformDefaults().MinRGBA(8, 8, 8, 8).DoubleBuffer().Depth(32).EndList()
        glcanvas.GLCanvas.__init__(self, parent, -1, size=self.size)
        self.context = glcanvas.GLContext(self)
        self.SetCurrent(self.context)
        self.init = False
        self.rotate = False
        self.rot_y = Matrix44.identity()
        self.geometry = None
        self.rot_loc = None
        self.trans_loc = None
        self.trans_x, self.trans_y, self.trans_z = 0.0, 0.0, 0.0
        self.translate = Matrix44.identity()
        self.combined_matrix = Matrix44.identity()
        self.hide_hidden_lines = False
        self.hide_component_a = False
        self.hide_component_b = False
        self.auto_rotate = False

        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_SIZE, self.OnResize)

    def OnResize(self, event):
        size = self.GetClientSize()
        glViewport(0, 0, size.width, size.height)

    def OnPaint(self, event):
        wx.PaintDC(self)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()

    def InitGL(self):

        glEnable(GL_POLYGON_OFFSET_FILL)

        #glViewport(100,100, 400, 400)

        self.geometry = Geometries()

        shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                  OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

        view = matrix44.create_from_translation(Vector3([0.0, 0.0, -2.0]))
        projection = matrix44.create_perspective_projection_matrix(45.0, self.aspect_ratio, 0.1, 100.0)

        vp = matrix44.multiply(view, projection)

        glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)

        glUseProgram(shader)
        glEnable(GL_DEPTH_TEST)

        vp_loc = glGetUniformLocation(shader, "vp")
        glUniformMatrix4fv(vp_loc, 1, GL_FALSE, vp)

        self.rot_loc = glGetUniformLocation(shader, "rotate")
        self.trans_loc = glGetUniformLocation(shader, "translate")

    def OnDraw(self):

        print("Drawing screen...")
        print("Open:",self.geometry.open_joint)
        print("Type",self.geometry.joint_type)
        print("Slide",self.geometry.sliding_direction)
        print("...")

        glClearColor(1.0, 1.0, 1.0, 1.0)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

        glPolygonOffset(1.0,1.0)

        self.translate = matrix44.create_from_translation(Vector3([self.trans_x, self.trans_y, self.trans_z]))
        self.combined_matrix = matrix44.multiply(self.rot_y, self.translate)

        if self.auto_rotate:
            ct = time.clock()
            self.rot_y = Matrix44.from_y_rotation(ct)
            glUniformMatrix4fv(self.rot_loc, 1, GL_FALSE, self.rot_y)
            glUniformMatrix4fv(self.trans_loc, 1, GL_FALSE, self.translate)
            self.Refresh()
        else:
            glUniformMatrix4fv(self.rot_loc, 1, GL_FALSE, self.rot_y)
            glUniformMatrix4fv(self.trans_loc, 1, GL_FALSE, self.translate)

        iA = self.geometry.ifA + self.geometry.ifeA + self.geometry.ilA
        iB = self.geometry.ifB + self.geometry.ifeB + self.geometry.ilB

        ### Draw end grain faces (hidden in depth by full geometry) ###
        a0 = [GL_QUADS,self.geometry.ifeA,self.geometry.ifA]
        b0 = [GL_QUADS,self.geometry.ifeB,iA+self.geometry.ifB]
        a1 = [GL_QUADS,self.geometry.ifA,0]
        b1 = [GL_QUADS,self.geometry.ifB,iA]
        G0 = []
        G1 = []
        if not self.hide_component_a:
            G0.append(a0)
            G1.append(a1)
        if not self.hide_component_b:
            G0.append(b0)
            G1.append(b1)
        if not self.hide_component_a or not self.hide_component_b:
            draw_geometry_with_excluded_area(G0,G1)

        ### Draw lines HIDDEN by other component ###
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(1)
        glLineStipple(3, 0xAAAA) # dashed line
        glEnable(GL_LINE_STIPPLE)
        # Component A
        if self.hide_component_a==False and self.hide_hidden_lines==False:
            glClear(GL_DEPTH_BUFFER_BIT)
            G0 = [[GL_LINES, self.geometry.ilA, self.geometry.ifA+self.geometry.ifeA]]
            G1 = [[GL_QUADS, self.geometry.ifA+self.geometry.ifeA, 0]]
            draw_geometry_with_excluded_area(G0,G1)
        # Component B
        if not self.hide_component_b and not self.hide_hidden_lines:
            glClear(GL_DEPTH_BUFFER_BIT)
            G0 = [[GL_LINES, self.geometry.ilB, iA+self.geometry.ifB+self.geometry.ifeB]]
            G1 = [[GL_QUADS, self.geometry.ifB+self.geometry.ifeB, iA]]
            draw_geometry_with_excluded_area(G0,G1)
        glPopAttrib()

        ### Draw visible lines ###
        glLineWidth(3)
        glClear(GL_DEPTH_BUFFER_BIT)
        a0 = [GL_LINES, self.geometry.ilA, self.geometry.ifA+self.geometry.ifeA]
        b0 = [GL_LINES, self.geometry.ilB, iA+self.geometry.ifB+self.geometry.ifeB]
        a1 = [GL_QUADS, self.geometry.ifA+self.geometry.ifeA, 0]
        b1 = [GL_QUADS, self.geometry.ifB+self.geometry.ifeB, iA]
        G0 = []
        G1 = []
        if not self.hide_component_a:
            G0.append(a0)
            G1.append(a1)
        if not self.hide_component_b:
            G0.append(b0)
            G1.append(b1)
        if not self.hide_component_a or not self.hide_component_b:
            draw_geometry_with_excluded_area(G0,G1)

        ### Draw dashed lines when joint is open ###
        if self.geometry.open_joint and not self.hide_component_a and not self.hide_component_b:
            glPushAttrib(GL_ENABLE_BIT)
            glLineWidth(2)
            glLineStipple(1, 0x00FF)
            glEnable(GL_LINE_STIPPLE)
            G0 = [[GL_LINES, self.geometry.iopen, iA+iB]]
            a1 = [GL_QUADS, self.geometry.ifA+self.geometry.ifeA, 0]
            b1 = [GL_QUADS, self.geometry.ifB+self.geometry.ifeB, iA]
            G1 = [a1,b1]
            draw_geometry_with_excluded_area(G0,G1)
            glPopAttrib()

        self.SwapBuffers()

class MyPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.SetBackgroundColour("#ffffff")

        # all the widgets
        # the OpenGL canvas
        self.canvas = OpenGLCanvas(self)

        f = 2

        # Text label
        font = wx.Font(f*10, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
        self.title = wx.StaticText(self, -1, label="View\n------", pos=(f*810, f*10))
        self.title.SetFont(font)

        # Checkboxes for viewing options
        self.tick_open = wx.CheckBox(self, -1, pos=(f*810, f*50), label="Open joint")
        self.tick_hidden = wx.CheckBox(self, -1, pos=(f*810, f*70), label="Hide hidden lines")
        self.tick_a = wx.CheckBox(self, -1, pos=(f*810, f*90), label="Hide component A")
        self.tick_b = wx.CheckBox(self, -1, pos=(f*810, f*110), label="Hide component B")
        self.tick_rot = wx.CheckBox(self, -1, pos=(f*810, f*130), label="Rotate")

        # Text label INPUT
        self.title = wx.StaticText(self, -1, label="Input\n------", pos=(f*810, f*170))
        self.title.SetFont(font)

        # Radio button to change between joint types
        self.title = wx.StaticText(self, -1, label="Joint type:", pos=(f*810, f*210))
        self.title.SetFont(font)
        self.rad_I = wx.RadioButton(self, -1, label="I", pos=(f*810, f*230),style = wx.RB_GROUP)
        self.rad_L = wx.RadioButton(self, -1, label="L", pos=(f*840, f*230))
        self.rad_T = wx.RadioButton(self, -1, label="T", pos=(f*870, f*230))
        self.rad_X = wx.RadioButton(self, -1, label="X", pos=(f*900, f*230))

        # Sliding direction
        self.title = wx.StaticText(self, -1, label="Sliding direction:", pos=(f*810, f*260))
        self.title.SetFont(font)
        self.rad_up = wx.RadioButton(self, -1, label="Up", pos=(f*810, f*280), style = wx.RB_GROUP)
        self.rad_right = wx.RadioButton(self, -1, label="Right", pos=(f*870, f*280))

        # Buttons to update height field
        self.title = wx.StaticText(self, -1, label="Edit height field:", pos=(f*810, f*310))
        self.title.SetFont(font)
        self.btn_Y = wx.Button(self, -1, label="Y", pos=(f*810, f*330), size=(f*30, f*30))
        self.btn_U = wx.Button(self, -1, label="U", pos=(f*810, f*360), size=(f*30, f*30))
        self.btn_I = wx.Button(self, -1, label="I", pos=(f*810, f*390), size=(f*30, f*30))
        self.btn_H = wx.Button(self, -1, label="H", pos=(f*840, f*330), size=(f*30, f*30))
        self.btn_J = wx.Button(self, -1, label="J", pos=(f*840, f*360), size=(f*30, f*30))
        self.btn_K = wx.Button(self, -1, label="K", pos=(f*840, f*390), size=(f*30, f*30))
        self.btn_B = wx.Button(self, -1, label="B", pos=(f*870, f*330), size=(f*30, f*30))
        self.btn_N = wx.Button(self, -1, label="N", pos=(f*870, f*360), size=(f*30, f*30))
        self.btn_M = wx.Button(self, -1, label="M", pos=(f*870, f*390), size=(f*30, f*30))

        # Button to optimize joint geometry
        self.clr_btn = wx.Button(self, -1, label="Clear", pos=(f*810, f*435), size=(f*90, f*25))
        self.ran_btn = wx.Button(self, -1, label="Randomize", pos=(f*810, f*465), size=(f*90, f*25))
        self.opt_btn = wx.Button(self, -1, label="Optimize", pos=(f*810, f*495), size=(f*90, f*25))


        # Text label
        self.title = wx.StaticText(self, -1, label="Output\n------", pos=(f*810, f*560))
        self.title.SetFont(font)
        self.title = wx.StaticText(self, -1, label="Friction: ...", pos=(f*810, f*600))
        self.title.SetFont(font)
        self.title = wx.StaticText(self, -1, label="Sliding directions: ...", pos=(f*810, f*620))
        self.title.SetFont(font)
        self.title = wx.StaticText(self, -1, label="Fabrication time: ...", pos=(f*810, f*640))
        self.title.SetFont(font)

        # all the event bindings
        self.Bind(wx.EVT_BUTTON, self.optimize, self.opt_btn)
        self.Bind(wx.EVT_BUTTON, self.updateY, self.btn_Y)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_U)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_I)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_H)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_J)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_K)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_B)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_N)
        self.Bind(wx.EVT_BUTTON, self.update, self.btn_M)
        self.Bind(wx.EVT_RADIOBUTTON, self.typeI, self.rad_I)
        self.Bind(wx.EVT_RADIOBUTTON, self.typeL, self.rad_L)
        self.Bind(wx.EVT_RADIOBUTTON, self.typeT, self.rad_T)
        self.Bind(wx.EVT_RADIOBUTTON, self.typeX, self.rad_X)
        self.Bind(wx.EVT_RADIOBUTTON, self.slide_up, self.rad_up)
        self.Bind(wx.EVT_RADIOBUTTON, self.slide_rgt, self.rad_right)
        self.Bind(wx.EVT_CHECKBOX, self.hidden_lines, self.tick_hidden)
        self.Bind(wx.EVT_CHECKBOX, self.component_A, self.tick_a)
        self.Bind(wx.EVT_CHECKBOX, self.component_B, self.tick_b)
        self.Bind(wx.EVT_CHECKBOX, self.rotate, self.tick_rot)
        self.Bind(wx.EVT_CHECKBOX, self.open, self.tick_open)

    def rotate(self, event):
        self.canvas.auto_rotate = self.tick_rot.GetValue()
        self.canvas.Refresh()

    def open(self, event):
        self.canvas.geometry.open_joint = self.tick_open.GetValue()
        Geometries.create_and_buffer_vertices(self.canvas.geometry)
        self.canvas.Refresh()

    def hidden_lines(self,event):
        self.canvas.hide_hidden_lines = self.tick_hidden.GetValue()
        self.canvas.Refresh()

    def component_A(self,event):
        self.canvas.hide_component_a = self.tick_a.GetValue()
        self.canvas.Refresh()

    def component_B(self,event):
        self.canvas.hide_component_b = self.tick_b.GetValue()
        self.canvas.Refresh()

    def typeI(self,event):
        if self.rad_I.GetValue()==True:
            self.canvas.geometry.joint_type = "I"
            Geometries.create_and_buffer_indicies(self.canvas.geometry)
            self.canvas.Refresh()

    def typeL(self,event):
        if self.rad_L.GetValue()==True:
            self.canvas.geometry.joint_type = "L"
            Geometries.create_and_buffer_indicies(self.canvas.geometry)
            self.canvas.Refresh()

    def typeT(self,event):
        if self.rad_T.GetValue()==True:
            self.canvas.geometry.joint_type = "T"
            Geometries.create_and_buffer_indicies(self.canvas.geometry)
            self.canvas.Refresh()

    def typeX(self,event):
        if self.rad_X.GetValue()==True:
            self.canvas.geometry.joint_type = "X"
            Geometries.create_and_buffer_indicies(self.canvas.geometry)
            self.canvas.Refresh()

    def slide_rgt(self,event):
        self.canvas.geometry.sliding_direction = [2,0]
        Geometries.create_and_buffer_vertices(self.canvas.geometry)
        Geometries.voxel_matrix_from_height_field(self.canvas.geometry)
        Geometries.create_and_buffer_indicies(self.canvas.geometry)
        self.canvas.Refresh()

    def slide_up(self,event):
        self.canvas.geometry.sliding_direction = [1,0]
        Geometries.create_and_buffer_vertices(self.canvas.geometry)
        Geometries.voxel_matrix_from_height_field(self.canvas.geometry)
        Geometries.create_and_buffer_indicies(self.canvas.geometry)
        self.canvas.Refresh()

    def updateY(self,event):
        self.canvas.geometry.height_field[0][0] = (self.canvas.geometry.height_field[0][0]+1)%(self.canvas.geometry.dim+1)
        Geometries.voxel_matrix_from_height_field(self.canvas.geometry)
        Geometries.create_and_buffer_indicies(self.canvas.geometry)
        self.canvas.Refresh()

    def update(self,event):
        self.canvas.geometry.height_field[0][0] = (self.canvas.geometry.height_field[0][0]+1)%(self.canvas.geometry.dim+1)
        Geometries.voxel_matrix_from_height_field(self.canvas.geometry)
        Geometries.create_and_buffer_indicies(self.canvas.geometry)
        self.canvas.Refresh()


    def optimize(self,event):
        #self.canvas.optimize = True
        self.canvas.Refresh()

class MyFrame(wx.Frame):
    def __init__(self):
        self.size = (2000, 1600)
        wx.Frame.__init__(self, None, title="My wx frame", size=self.size,
                          style=wx.DEFAULT_FRAME_STYLE | wx.FULL_REPAINT_ON_RESIZE)
        #self.SetSize(1000,2000)
        self.SetMinSize(self.size)
        self.SetMaxSize(self.size)
        self.Bind(wx.EVT_CLOSE, self.on_close)
        self.panel = MyPanel(self)

    def on_close(self, event):
        self.Destroy()
        sys.exit(0)

class MyApp(wx.App):
    def OnInit(self):
        frame = MyFrame()
        frame.Show()
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)

        except:
            pass
        return True

if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
