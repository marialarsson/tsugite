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
    #extension GL_ARB_explicit_attrib_location : require
    #extension GL_ARB_explicit_uniform_location : require
    layout(location = 0) in vec3 position;
    layout(location = 1) in vec3 color;
    layout(location = 2) in vec2 inTexCoords;
    layout(location = 3) uniform mat4 transform;
    layout(location = 4) uniform mat4 translate;
    out vec3 newColor;
    out vec2 outTexCoords;
    void main()
    {
        gl_Position = transform* translate* vec4(position, 1.0f);
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

def keyCallback(window,key,scancode,action,mods):
    global HIDDEN_A, HIDDEN_B, SHOW_HIDDEN, OPEN, SHOW_MILLING_PATH
    if action==glfw.PRESS:
        # Joint geometry
        if key==glfw.KEY_C: Geometries.clear_height_field(MESH)
        # Joint type
        elif key==glfw.KEY_I and MESH.joint_type!="I":
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
        elif key==glfw.KEY_B: HIDDEN_B = not HIDDEN_B
        elif key==glfw.KEY_H: SHOW_HIDDEN = not SHOW_HIDDEN
        elif key==glfw.KEY_F:
            MESH.fab_geometry = not MESH.fab_geometry
            Geometries.create_and_buffer_indicies(MESH)
        elif key==glfw.KEY_O: OPEN = not OPEN
        elif key==glfw.KEY_S: print("Saving..."); Geometries.save(MESH)
        elif key==glfw.KEY_G: print("Loading..."); Geometries.load(MESH)
        elif key==glfw.KEY_P: MESH.show_milling_path = not MESH.show_milling_path
        elif key==glfw.KEY_2 and MESH.dim!=2: Geometries.update_dimension(MESH,2)
        elif key==glfw.KEY_3 and MESH.dim!=3: Geometries.update_dimension(MESH,3)
        elif key==glfw.KEY_4 and MESH.dim!=4: Geometries.update_dimension(MESH,4)
        elif key==glfw.KEY_5 and MESH.dim!=5: Geometries.update_dimension(MESH,5)
        elif key==glfw.KEY_R: Geometries.randomize_height_field(MESH)

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

def draw_geometry_with_excluded_area(show_geos,screen_geos,show_vecs,screen_vecs):
    glDisable(GL_DEPTH_TEST)
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
    glEnable(GL_STENCIL_TEST)
    glStencilFunc(GL_ALWAYS,1,1)
    glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
    for geo,move_vec in zip(show_geos,show_vecs):
        move = pyrr.matrix44.create_from_translation(move_vec)
        glUniformMatrix4fv(4, 1, GL_FALSE, move)
        glDrawElements(geo[0], geo[1], GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo[2]))
    glEnable(GL_DEPTH_TEST)
    glStencilFunc(GL_EQUAL,1,1)
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
    for geo,move_vec in zip(screen_geos,screen_vecs):
        move = pyrr.matrix44.create_from_translation(move_vec)
        glUniformMatrix4fv(4, 1, GL_FALSE, move)
        glDrawElements(geo[0], geo[1], GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo[2]))
    glDisable(GL_STENCIL_TEST)
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
    for geo,move_vec in zip(show_geos,show_vecs):
        move = pyrr.matrix44.create_from_translation(move_vec)
        glUniformMatrix4fv(4, 1, GL_FALSE, move)
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
    global MESH, OPEN_START_TIME, CLOSE_START_TIME, OPEN_START_DIST, CLOSE_START_DIST

    iA = MESH.ifA + MESH.ifeA + MESH.ifuA + MESH.ilA
    iB = MESH.ifB + MESH.ifeB + MESH.ifuB + MESH.ilB
    iffA = MESH.ifA + MESH.ifeA + MESH.ifuA
    iffB = MESH.ifB + MESH.ifeB + MESH.ifuB

    glUseProgram(shader_tex)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    rot_x = pyrr.Matrix44.from_x_rotation(XROT)
    rot_y = pyrr.Matrix44.from_y_rotation(YROT)
    if OPEN:
        if OPEN_START_TIME==None:
            OPEN_START_TIME = glfw.get_time()
            CLOSE_START_TIME = None
        dist = OPEN_START_DIST + 0.4 * (glfw.get_time()-OPEN_START_TIME)
        if dist>MESH.component_size: dist = MESH.component_size
        CLOSE_START_DIST = dist
    else:
        if CLOSE_START_TIME==None:
            CLOSE_START_TIME = glfw.get_time()
            OPEN_START_TIME = None
        dist = CLOSE_START_DIST - 0.4 * (glfw.get_time()-CLOSE_START_TIME)
        if dist<0: dist = 0
        OPEN_START_DIST = dist
    move_vec = [0,0,0]
    move_vec[MESH.sliding_direction[0]] = (2*MESH.sliding_direction[1]-1)*dist
    move_vec = np.array(move_vec)
    move = pyrr.matrix44.create_from_translation(move_vec)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)
    glUniformMatrix4fv(4, 1, GL_FALSE, move)
    glPolygonOffset(1.0,1.0)

    ################# Textures faces
    ### Draw end grain faces (hidden in depth by full geometry) ###
    a0 = [GL_QUADS, MESH.ifeA, MESH.ifA]
    b0 = [GL_QUADS, MESH.ifeB, iA+MESH.ifB]
    a1 = [GL_QUADS, MESH.ifA, 0]
    b1 = [GL_QUADS, MESH.ifB, iA]
    G0 = []
    G1 = []
    D0 = []
    D1 = []
    if HIDDEN_A==False:
        G0.append(a0)
        G1.append(a1)
        D0.append(move_vec)
        D1.append(move_vec)
    if HIDDEN_B==False:
        G0.append(b0)
        G1.append(b1)
        D0.append(np.negative(move_vec))
        D1.append(np.negative(move_vec))
    if HIDDEN_A==False or HIDDEN_B==False:
        draw_geometry_with_excluded_area(G0,G1,D0,D1)

    ######################## SWITCH TO COLOR SHADER ###########################
    glUseProgram(shader_col)
    move = pyrr.matrix44.create_from_translation(move_vec)
    sol_col = [[0.5,0.0,1.0]]
    sol_col = np.array(sol_col, dtype=np.uint32)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)
    glUniformMatrix4fv(4, 1, GL_FALSE, move)

    ######################## Draw hidden geometry of unconnected voxels ########
    glUniform3f(5, 1.0, 0.8, 0.7)
    glClear(GL_DEPTH_BUFFER_BIT)
    if not HIDDEN_A:
        move = pyrr.matrix44.create_from_translation(move_vec)
        glUniformMatrix4fv(4, 1, GL_FALSE, move)
        glDrawElements(GL_QUADS, MESH.ifuA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(MESH.ifA + MESH.ifeA)))
    if not HIDDEN_B:
        move = pyrr.matrix44.create_from_translation(np.negative(move_vec))
        glUniformMatrix4fv(4, 1, GL_FALSE, move)
        glDrawElements(GL_QUADS, MESH.ifuB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+MESH.ifB + MESH.ifeB)))

    ######################## Draw visible geometry of unconnected voxels ########
    glUniform3f(5, 1.0, 0.2, 0.0) # red orange
    glClear(GL_DEPTH_BUFFER_BIT)
    a0 = [GL_QUADS, MESH.ifuA, MESH.ifA + MESH.ifeA]
    b0 = [GL_QUADS, MESH.ifuB, iA + MESH.ifB + MESH.ifeB]
    a1 = [GL_QUADS, MESH.ifA+MESH.ifeA, 0]
    b1 = [GL_QUADS, MESH.ifB+MESH.ifeB, iA]
    G0 = []
    G1 = []
    D0 = []
    D1 = []
    if HIDDEN_A==False:
        G0.append(a0)
        G1.append(a1)
        D0.append(move_vec)
        D1.append(move_vec)
    if HIDDEN_B==False:
        G0.append(b0)
        G1.append(b1)
        D0.append(np.negative(move_vec))
        D1.append(np.negative(move_vec))
    if HIDDEN_A==False or HIDDEN_B==False:
        draw_geometry_with_excluded_area(G0,G1,D0,D1)


    ######################## Draw colors of unbridged components ########
    if HIDDEN_B==False and MESH.bridged==False:
        glUniform3f(5, 1.0, 0.6, 1.0) # light pink
        glClear(GL_DEPTH_BUFFER_BIT)
        b0 = [GL_QUADS, MESH.ifubB1, iA+iB]
        a1 = [GL_QUADS, iffA, 0]
        b1 = [GL_QUADS, MESH.ifubB2, iA+iB+MESH.ifubB1]
        c1 = [GL_QUADS, MESH.ifuA, MESH.ifA + MESH.ifeA]
        d1 = [GL_QUADS, MESH.ifuB, iA + MESH.ifB + MESH.ifeB]
        G0 = []
        G1 = []
        D0 = []
        D1 = []
        if HIDDEN_A==False:
            G1.append(a1)
            D1.append(move_vec)
        if MESH.connected_A==False:
            G1.append(c1)
            D1.append(move_vec)
        if MESH.connected_B==False:
           G1.append(d1)
           D1.append(np.negative(move_vec))
        G0.append(b0)
        G1.append(b1)
        D0.append(np.negative(move_vec))
        D1.append(np.negative(move_vec))
        draw_geometry_with_excluded_area(G0,G1,D0,D1)
        ### other side
        glUniform3f(5, 1.0, 0.8, 0.6) # light orange
        glClear(GL_DEPTH_BUFFER_BIT)
        b0 = [GL_QUADS, MESH.ifubB2, iA+iB+MESH.ifubB1]
        a1 = [GL_QUADS, iffA, 0]
        b1 = [GL_QUADS, MESH.ifubB1, iA+iB]
        G0 = []
        G1 = []
        D0 = []
        D1 = []
        if HIDDEN_A==False:
            G1.append(a1)
            D1.append(move_vec)
        if MESH.connected_A==False:
            G1.append(c1)
            D1.append(move_vec)
        if MESH.connected_B==False:
           G1.append(d1)
           D1.append(np.negative(move_vec))
        G0.append(b0)
        G1.append(b1)
        D0.append(np.negative(move_vec))
        D1.append(np.negative(move_vec))
        draw_geometry_with_excluded_area(G0,G1,D0,D1)

    glClear(GL_DEPTH_BUFFER_BIT)
    glUniform3f(5,0.0,0.0,0.0)
    ### Draw lines HIDDEN by other component ###
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) # dashed line
    glEnable(GL_LINE_STIPPLE)
    # Component A
    if HIDDEN_A==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES,  MESH.ilA,  iffA]]
        G1 = [[GL_QUADS,  iffA, 0]]
        D0 = [move_vec]
        D1 = [move_vec]
        draw_geometry_with_excluded_area(G0,G1,D0,D1)
    # Component B
    if HIDDEN_B==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES,  MESH.ilB, iA+iffB]]
        G1 = [[GL_QUADS,  iffB, iA]]
        D0 = [np.negative(move_vec)]
        D1 = [np.negative(move_vec)]
        draw_geometry_with_excluded_area(G0,G1,D0,D1)
    glPopAttrib()

    ### Draw visible lines ###
    glLineWidth(3)
    glClear(GL_DEPTH_BUFFER_BIT)
    a0 = [GL_LINES,  MESH.ilA, iffA]
    b0 = [GL_LINES,  MESH.ilB, iA+iffB]
    a1 = [GL_QUADS,  iffA, 0]
    b1 = [GL_QUADS,  iffB, iA]
    G0 = []
    G1 = []
    D0 = []
    D1 = []
    if HIDDEN_A==False:
        G0.append(a0)
        G1.append(a1)
        D0.append(move_vec)
        D1.append(move_vec)
    if HIDDEN_B==False:
        G0.append(b0)
        G1.append(b1)
        D0.append(np.negative(move_vec))
        D1.append(np.negative(move_vec))
    if HIDDEN_A==False or HIDDEN_B==False:
        draw_geometry_with_excluded_area(G0,G1,D0,D1)

    """
    ### Draw dashed lines when joint is open ###
    if OPEN==True and HIDDEN_A==False and HIDDEN_B==False and dist==MESH.component_size:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        ## A
        G0 = [[GL_LINES,  2*MESH.iopen, iA+iB]]
        D0 = [move_vec]
        a1 = [GL_QUADS,  MESH.ifA+ MESH.ifeA, 0]
        b1 = [GL_QUADS,  MESH.ifB+ MESH.ifeB, iA]
        G1 = [a1,b1]
        D1 = [move_vec,np.negative(move_vec)]
        draw_geometry_with_excluded_area(G0,G1,D0,D1)
        ## B
        G0 = [[GL_LINES,  MESH.iopen, iA+iB+MESH.iopen]]
        D0 = [np.negative(move_vec)]
        a1 = [GL_QUADS,  MESH.ifA+ MESH.ifeA, 0]
        b1 = [GL_QUADS,  MESH.ifB+ MESH.ifeB, iA]
        G1 = [a1,b1]
        D1 = [move_vec,np.negative(move_vec)]
        #draw_geometry_with_excluded_area(G0,G1,D0,D1)
        glPopAttrib()

    ############# Milling paths
    ### Draw gpath lines ###
    glClear(GL_DEPTH_BUFFER_BIT)
    glLineWidth(1)
    if not HIDDEN_A and MESH.show_milling_path:
        glDrawElements(GL_LINE_STRIP, MESH.imA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+2*MESH.iopen)))
    if not HIDDEN_B and MESH.show_milling_path:
        glDrawElements(GL_LINE_STRIP, MESH.imB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+2*MESH.iopen+MESH.imA)))
        """
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
    print("Edit joint geometry by pulling")
    print("Clear joint geometry with:\nC")
    print("Edit joint type: I L T X")
    print("Open joint: O")
    print("Hide components: A B")
    print("Show milling path: M")
    print("Hide hidden lines: H")
    print("Change voxel resolution: 2, 3, 4, 5")
    print("Press S to save joint geometry and G to open last saved geometry")

    # Declare global variables
    global XROT, YROT, XROT0, YROT0, XSTART, YSTART, CLICK_TIME, DRAGGED, DOUBLE_CLICKED
    global HIDDEN_A, HIDDEN_A, SHOW_HIDDEN, MESH
    global OPEN_START_TIME, CLOSE_START_TIME, OPEN_DIST

    HIDDEN_A = False
    HIDDEN_B = False
    SHOW_HIDDEN = True
    OPEN = False

    # Variables for mouse callback and rotation
    XROT, YROT = 0.8, 0.4
    XROT0, YROT0 = XROT, YROT
    XSTART = YSTART = 0.0
    CLICK_TIME = 0
    DRAGGED = False
    DOUBLE_CLICKED = False

    # Variables for animation of opening joint
    OPEN_START_TIME = None
    CLOSE_START_TIME = None
    OPEN_START_DIST = 0
    CLOSE_START_DIST = 0

    main()
