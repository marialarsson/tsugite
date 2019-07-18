import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
import sys
from Geometries import Geometries
import ctypes

def create_texture_shaders():
    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec3 color;
    in layout(location = 2) vec2 inTexCoords;
    uniform mat4 transform;
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* vec4(position, 1.0f);
        newColor = color;
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
        outColor = texture(samplerTex, outTexCoords);
    }
    """
    # Compiling the shaders
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    return shader

def create_color_shaders():
    vertex_shader = """
    #version 330
    in layout(location = 0) vec3 position;
    in layout(location = 1) vec3 color;
    in layout(location = 2) vec2 inTexCoords;
    uniform mat4 transform;
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* vec4(position, 1.0f);
        newColor = color;
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

def keyCallback(window,key,scancode,action,mods):
    global HIDDEN_A, HIDDEN_B, SHOW_HIDDEN, OPEN, SHOW_MILLING_PATH
    if action==glfw.PRESS:
        # Joint geometry edit by height field
        if key==glfw.KEY_Y:   Geometries.update_height_field(MESH,0,2)
        elif key==glfw.KEY_U: Geometries.update_height_field(MESH,1,2)
        elif key==glfw.KEY_I: Geometries.update_height_field(MESH,2,2)
        elif key==glfw.KEY_H: Geometries.update_height_field(MESH,0,1)
        elif key==glfw.KEY_J: Geometries.update_height_field(MESH,1,1)
        elif key==glfw.KEY_K: Geometries.update_height_field(MESH,2,1)
        elif key==glfw.KEY_N: Geometries.update_height_field(MESH,0,0)
        elif key==glfw.KEY_M: Geometries.update_height_field(MESH,1,0)
        elif key==glfw.KEY_COMMA: Geometries.update_height_field(MESH,2,0)
        elif key==glfw.KEY_C: Geometries.clear_height_field(MESH)
        # Joint TYPE
        elif key==glfw.KEY_SEMICOLON and MESH.joint_type!="I":
            Geometries.update_joint_type(MESH,"I")
        elif key==glfw.KEY_L and MESH.joint_type!="L":
            Geometries.update_joint_type(MESH,"L")
        elif key==glfw.KEY_T and MESH.joint_type!="T":
            Geometries.update_joint_type(MESH,"T")
        elif key==glfw.KEY_X and MESH.joint_type!="X":
            Geometries.update_joint_type(MESH,"X")
        # Sliding direction
        elif key==glfw.KEY_UP and MESH.sliding_direction!=[2,0]:
            if MESH.joint_type!="X":
                Geometries.update_sliding_direction(MESH,[2,0])
        elif key==glfw.KEY_RIGHT and MESH.sliding_direction!=[1,0]:
            Geometries.update_sliding_direction(MESH,[1,0])
        # Preview options
        elif key==glfw.KEY_A: HIDDEN_A = not HIDDEN_A
        elif key==glfw.KEY_D: HIDDEN_B = not HIDDEN_B
        elif key==glfw.KEY_E: SHOW_HIDDEN = not SHOW_HIDDEN
        elif key==glfw.KEY_O:
            MESH.open_joint = not MESH.open_joint
            Geometries.create_and_buffer_vertices(MESH)
        elif key==glfw.KEY_S:
            print("Saving...")
            Geometries.save(MESH)
        elif key==glfw.KEY_W:
            print("Loading...")
            Geometries.load(MESH)
        elif key==glfw.KEY_G: SHOW_MILLING_PATH = not SHOW_MILLING_PATH
        elif key==glfw.KEY_2: Geometries.update_dimension(MESH,2)
        elif key==glfw.KEY_3: Geometries.update_dimension(MESH,3)
        elif key==glfw.KEY_4: Geometries.update_dimension(MESH,4)


def mouseCallback(window,button,action,mods):
    if button==glfw.MOUSE_BUTTON_LEFT:
        global DRAGGED, CLICK_TIME, DOUBLE_CLICKED, NEXT_CLICK_TIME
        if action==1: #pressed
            NEXT_CLICK_TIME = glfw.get_time()
            DRAGGED = True
            global XSTART, YSTART, XROT, YROT, XROT0, YROT0
            XSTART, YSTART = glfw.get_cursor_pos(window)
            XROT0, YROT0 = XROT, YROT
            if NEXT_CLICK_TIME-CLICK_TIME<0.3:
                DOUBLE_CLICKED = not DOUBLE_CLICKED
            else: DOUBLE_CLICKED=False
            CLICK_TIME = NEXT_CLICK_TIME
        elif action==0: #released
            DRAGGED = False

def updateRotation(window, DRAGGED, DOUBLE_CLICKED):
    global XROT, YROT, NEXT_CLICK_TIME
    # Rotate view by dragging
    if DRAGGED:
        xpos, ypos = glfw.get_cursor_pos(window)
        ratio = 0.001
        ydiff = ratio*(xpos-XSTART)
        xdiff = ratio*(ypos-YSTART)
        XROT = XROT0 + xdiff
        YROT = YROT0 + ydiff
    # Auto rotate view
    elif DOUBLE_CLICKED:
        XROT = XROT0 + 0.1 * (glfw.get_time()-NEXT_CLICK_TIME)
        YROT = YROT0 + 0.4 * (glfw.get_time()-NEXT_CLICK_TIME)

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

def initialize():
    # Initialize glfw
    if not glfw.init():
        return
    # Create window
    window = glfw.create_window(1600, 1600, "DISCO JOINT", None, None)
    if not window:
        glfw.terminate()
        return
    glfw.make_context_current(window)

    glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)

    # Enable and handle key events
    glfw.set_key_callback(window, keyCallback)
    glfw.set_input_mode(window, glfw.STICKY_KEYS,1)

    # Enable and hangle mouse events
    glfw.set_mouse_button_callback(window, mouseCallback);
    glfw.set_input_mode(window, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)
    global DOUBLE_CLICKED
    DOUBLE_CLICKED = False

    # Set properties
    glLineWidth(3)
    glEnable(GL_POLYGON_OFFSET_FILL)

    return window

def display(window, shader_tex, shader_col):
    global MESH

    iA = MESH.ifA + MESH.ifeA + MESH.ilA
    iB = MESH.ifB + MESH.ifeB + MESH.ilB

    glUseProgram(shader_tex)
    if MESH.connected:
        glClearColor(1.0, 1.0, 1.0, 1.0)
    else:
        glClearColor(1.0, 0.9, 0.9, 1.0)
    glEnable(GL_DEPTH_TEST)
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    rot_x = pyrr.Matrix44.from_x_rotation(XROT)
    rot_y = pyrr.Matrix44.from_y_rotation(YROT)
    transformLoc = glGetUniformLocation(shader_tex, "transform")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

    glPolygonOffset(1.0,1.0)

    ################# Textures faces

    ### Draw end grain faces (hidden in depth by full geometry) ###
    a0 = [GL_QUADS, MESH.ifeA, MESH.ifA]
    b0 = [GL_QUADS, MESH.ifeB, iA+MESH.ifB]
    a1 = [GL_QUADS, MESH.ifA, 0]
    b1 = [GL_QUADS, MESH.ifB, iA]
    G0 = []
    G1 = []
    if HIDDEN_A==False:
        G0.append(a0)
        G1.append(a1)
    if HIDDEN_B==False:
        G0.append(b0)
        G1.append(b1)
    if MESH.connected and (HIDDEN_A==False or HIDDEN_B==False):
        draw_geometry_with_excluded_area(G0,G1)

    ################## SWITCH SHADER FOR Lines
    glUseProgram(shader_col)
    transformLoc = glGetUniformLocation(shader_col, "transform")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

    ### Draw lines HIDDEN by other component ###
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) # dashed line
    glEnable(GL_LINE_STIPPLE)
    # Component A
    if HIDDEN_A==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES,  MESH.ilA,  MESH.ifA+ MESH.ifeA]]
        G1 = [[GL_QUADS,  MESH.ifA+ MESH.ifeA, 0]]
        draw_geometry_with_excluded_area(G0,G1)
    # Component B
    if HIDDEN_B==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES,  MESH.ilB, iA+MESH.ifB+ MESH.ifeB]]
        G1 = [[GL_QUADS,  MESH.ifB+ MESH.ifeB, iA]]
        draw_geometry_with_excluded_area(G0,G1)
    glPopAttrib()

    ### Draw visible lines ###
    glLineWidth(3)
    glClear(GL_DEPTH_BUFFER_BIT)
    a0 = [GL_LINES,  MESH.ilA, MESH.ifA+ MESH.ifeA]
    b0 = [GL_LINES,  MESH.ilB, iA+ MESH.ifB+ MESH.ifeB]
    a1 = [GL_QUADS,  MESH.ifA+ MESH.ifeA, 0]
    b1 = [GL_QUADS,  MESH.ifB+ MESH.ifeB, iA]
    G0 = []
    G1 = []
    if HIDDEN_A==False:
        G0.append(a0)
        G1.append(a1)
    if HIDDEN_B==False:
        G0.append(b0)
        G1.append(b1)
    if HIDDEN_A==False or HIDDEN_B==False:
        draw_geometry_with_excluded_area(G0,G1)

    ### Draw dashed lines when joint is open ###
    if MESH.open_joint==True and HIDDEN_A==False and HIDDEN_B==False:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        G0 = [[GL_LINES,  MESH.iopen, iA+iB]]
        a1 = [GL_QUADS,  MESH.ifA+ MESH.ifeA, 0]
        b1 = [GL_QUADS,  MESH.ifB+ MESH.ifeB, iA]
        G1 = [a1,b1]
        draw_geometry_with_excluded_area(G0,G1)
        glPopAttrib()

    ############# Milling paths
    ### Draw gpath lines ###
    glClear(GL_DEPTH_BUFFER_BIT)
    glLineWidth(1)
    if not HIDDEN_A and SHOW_MILLING_PATH:
        glDrawElements(GL_LINE_STRIP, MESH.imA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+MESH.iopen)))
    if not HIDDEN_B and SHOW_MILLING_PATH:
        glDrawElements(GL_LINE_STRIP, MESH.imB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+MESH.iopen+MESH.imA)))

    ## the end
    glfw.swap_buffers(window)

def main():
    global MESH

    # Initialize window
    window = initialize()

    # Create shaders
    shader_tex = create_texture_shaders()
    shader_col = create_color_shaders()

    MESH = Geometries()

    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        # Update Rotation
        updateRotation(window, DRAGGED, DOUBLE_CLICKED)

        # Display joint geometries
        display(window, shader_tex, shader_col)

    #glDeleteBuffers(2, [MESH.VBO,MESH.EBO])
    glfw.terminate()

if __name__ == "__main__":
    print("Hit ESC key to quit.")
    print("Rotate view with mouse / Autorotate with double click")
    print("Edit joint geometry:\nY U I\nH J K\nN M ;")
    print("Clear joint geometry with:\nC")
    print("Edit joint type: 1 L T X")
    print("Open joint: O")
    print("Hide components: A / D")
    print("Show milling path: G")
    print("Hide hidden lines: E")
    print("Press S to save joint geometry and W to open last saved geometry")

    # Declare global variables
    global XROT, YROT, XROT0, YROT0, XSTART, YSTART, CLICK_TIME, DRAGGED, DOUBLE_CLICKED
    global HIDDEN_A, HIDDEN_A, SHOW_HIDDEN
    global MESH, SHOW_MILLING_PATH

    HIDDEN_A = False
    HIDDEN_B = False
    SHOW_HIDDEN = True
    OPEN = False
    SHOW_MILLING_PATH = False

    # Variables for mouse callback and rotation
    XROT, YROT = 0.8, 0.4
    XROT0, YROT0 = XROT, YROT
    XSTART = YSTART = 0.0
    CLICK_TIME = 0
    DRAGGED = False
    DOUBLE_CLICKED = False

    main()
