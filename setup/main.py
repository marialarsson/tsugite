import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from numpy import linalg
import pyrr
import sys
from Geometries import Geometries
from ViewSettings import ViewSettings
import ctypes
import math

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
    global mesh, view_opt
    if action==glfw.PRESS:
        # Joint geometry
        if key==glfw.KEY_C: Geometries.clear_height_field(mesh)
        # Joint type
        elif key==glfw.KEY_I and mesh.joint_type!="I":
            Geometries.update_joint_type(mesh,"I")
        elif key==glfw.KEY_L and mesh.joint_type!="L":
            Geometries.update_joint_type(mesh,"L")
        elif key==glfw.KEY_T and mesh.joint_type!="T":
            Geometries.update_joint_type(mesh,"T")
        elif key==glfw.KEY_X and mesh.joint_type!="X":
            Geometries.update_joint_type(mesh,"X")
        # Sliding direction
        elif key==glfw.KEY_UP and mesh.sliding_direction!=[2,0]:
            if mesh.joint_type!="X":
                Geometries.update_sliding_direction(mesh,[2,0])
        elif key==glfw.KEY_RIGHT and mesh.sliding_direction!=[1,0]:
            Geometries.update_sliding_direction(mesh,[1,0])
        # Preview options
        elif key==glfw.KEY_A: view_opt.hidden_a = not view_opt.hidden_a
        elif key==glfw.KEY_B: view_opt.hidden_b = not view_opt.hidden_b
        elif key==glfw.KEY_H: view_opt.show_hidden_lines = not view_opt.show_hidden_lines
        elif key==glfw.KEY_F:
            mesh.fab_geometry = not mesh.fab_geometry
            Geometries.create_and_buffer_indicies(mesh)
        elif key==glfw.KEY_O: view_opt.open_joint = not view_opt.open_joint
        elif key==glfw.KEY_S: print("Saving..."); Geometries.save(mesh)
        elif key==glfw.KEY_G: print("Loading..."); Geometries.load(mesh)
        elif key==glfw.KEY_P: mesh.show_milling_path = not mesh.show_milling_path
        elif key==glfw.KEY_2 and mesh.dim!=2: Geometries.update_dimension(mesh,2)
        elif key==glfw.KEY_3 and mesh.dim!=3: Geometries.update_dimension(mesh,3)
        elif key==glfw.KEY_4 and mesh.dim!=4: Geometries.update_dimension(mesh,4)
        elif key==glfw.KEY_5 and mesh.dim!=5: Geometries.update_dimension(mesh,5)
        elif key==glfw.KEY_R: Geometries.randomize_height_field(mesh)

def mouseCallback(window,button,action,mods):
    if button==glfw.MOUSE_BUTTON_LEFT:
        global mesh
        if mesh.Selected!=None:
            if action==1:
                mesh.Selected.activate(glfw.get_cursor_pos(window))
            elif action==0:
                Geometries.finalize_selection(mesh)
    elif button==glfw.MOUSE_BUTTON_RIGHT:
        global view_opt
        if action==1: ViewSettings.start_rotation(view_opt, window)
        elif action==0: ViewSettings.end_rotation(view_opt)

def draw_geometries(geos):
    for type,cnt,start,mat,on in geos:
        glUniformMatrix4fv(4, 1, GL_FALSE, mat)
        glDrawElements(type, cnt, GL_UNSIGNED_INT,  ctypes.c_void_p(4*start))

def draw_geometries_with_excluded_area(show_geos, screen_geos):
    glDisable(GL_DEPTH_TEST)
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
    glEnable(GL_STENCIL_TEST)
    glStencilFunc(GL_ALWAYS,1,1)
    glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
    for type,cnt,start,mat,on in show_geos:
        if on:
            glUniformMatrix4fv(4, 1, GL_FALSE, mat)
            glDrawElements(type, cnt, GL_UNSIGNED_INT,  ctypes.c_void_p(4*start))
    glEnable(GL_DEPTH_TEST)
    glStencilFunc(GL_EQUAL,1,1)
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
    for type,cnt,start,mat,on in screen_geos:
        if on:
            glUniformMatrix4fv(4, 1, GL_FALSE, mat)
            glDrawElements(type, cnt, GL_UNSIGNED_INT,  ctypes.c_void_p(4*start))
    glDisable(GL_STENCIL_TEST)
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
    for type,cnt,start,mat,on in show_geos:
        if on:
            glUniformMatrix4fv(4, 1, GL_FALSE, mat)
            glDrawElements(type, cnt, GL_UNSIGNED_INT,  ctypes.c_void_p(4*start))

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

    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(-1.0,1.0,-1.0,1.0,-1.0,1.0)
    glMatrixMode(GL_MODELVIEW)

    # Enable and handle key events
    glfw.set_key_callback(window, keyCallback)
    glfw.set_input_mode(window, glfw.STICKY_KEYS,1)

    # Enable and hangle mouse events
    glfw.set_mouse_button_callback(window, mouseCallback);
    glfw.set_input_mode(window, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)

    # Set properties
    glLineWidth(3)
    glEnable(GL_POLYGON_OFFSET_FILL)

    return window

def display(window, mesh, view_opt, shader_tex, shader_col):

    # Define translation matrices for opening of joint for components A and B
    move_vec = [0,0,0]
    move_vec[mesh.sliding_direction[0]] = (2*mesh.sliding_direction[1]-1)*view_opt.distance
    move_vec = np.array(move_vec)
    move_A = pyrr.matrix44.create_from_translation(move_vec)
    move_B = pyrr.matrix44.create_from_translation(np.negative(move_vec))

    # Define helper start / length indices
    iA = mesh.ifA + mesh.ifeA + mesh.ifuA + mesh.ilA
    iB = mesh.ifB + mesh.ifeB + mesh.ifuB + mesh.ilB
    iffA = mesh.ifA + mesh.ifeA + mesh.ifuA
    iffB = mesh.ifB + mesh.ifeB + mesh.ifuB
    ifubA = mesh.ifubA1+mesh.ifubA2
    ifubB = mesh.ifubB1+mesh.ifubB2
    iopen_start = iA+iB+ifubA+ifubB
    topA_start = iopen_start+2*mesh.iopen
    topB_start = topA_start+mesh.iftA+mesh.ifntA

    # Pre-define gemetries
    faces_ends_a = [GL_QUADS, mesh.ifeA, mesh.ifA, move_A, not view_opt.hidden_a]
    faces_not_ends_a =  [GL_QUADS, mesh.ifA, 0, move_A, not view_opt.hidden_a]
    faces_all_a =  [GL_QUADS,  iffA, 0, move_A, not view_opt.hidden_a]
    faces_connected_a = [GL_QUADS, mesh.ifA+mesh.ifeA, 0, move_A, not view_opt.hidden_a]
    faces_unconnected_a = [GL_QUADS, mesh.ifuA, mesh.ifA + mesh.ifeA, move_A, not mesh.connected_A and not view_opt.hidden_a]
    lines_a =  [GL_LINES,  mesh.ilA,  iffA, move_A, not view_opt.hidden_a]
    faces_unbridged_1_a = [GL_QUADS, mesh.ifubA1, iA+iB, move_A, not mesh.bridged_A and not view_opt.hidden_a]
    faces_unbridged_2_a = [GL_QUADS, mesh.ifubA2, iA+iB+mesh.ifubA1, move_A, not mesh.bridged_A and not view_opt.hidden_a]
    lines_open_a = [GL_LINES,  mesh.iopen, iopen_start, move_A, not view_opt.hidden_a]
    faces_not_tops_a = [GL_QUADS, mesh.ifntA, topA_start+mesh.iftA, move_A, not view_opt.hidden_a]

    faces_ends_b = [GL_QUADS, mesh.ifeB, iA+mesh.ifB, move_B, not view_opt.hidden_b]
    faces_not_ends_b = [GL_QUADS, mesh.ifB, iA, move_B, not view_opt.hidden_b]
    faces_all_b = [GL_QUADS,  iffB, iA, move_B, not view_opt.hidden_b]
    faces_connected_b = [GL_QUADS, mesh.ifB+mesh.ifeB, iA, move_B, not view_opt.hidden_b]
    faces_unconnected_b = [GL_QUADS, mesh.ifuB, iA + mesh.ifB + mesh.ifeB, move_B, not mesh.connected_B and not view_opt.hidden_b]
    lines_b = [GL_LINES,  mesh.ilB, iA+iffB, move_B, not view_opt.hidden_b]
    faces_unbridged_1_b = [GL_QUADS, mesh.ifubB1, iA+iB+ifubA, move_B, not mesh.bridged_B and not view_opt.hidden_b]
    faces_unbridged_2_b = [GL_QUADS, mesh.ifubB2, iA+iB+ifubA+mesh.ifubB1, move_B, not mesh.bridged_B and not view_opt.hidden_b]
    lines_open_b = [GL_LINES,  mesh.iopen, iopen_start+mesh.iopen, move_B, not view_opt.hidden_b]
    faces_not_tops_b = [GL_QUADS, mesh.ifntB, topB_start+mesh.iftB, move_B, not view_opt.hidden_b]

    ######################## START WITH TEXTURE SHADER ###########################
    glUseProgram(shader_tex)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)

    rot_x = pyrr.Matrix44.from_x_rotation(view_opt.xrot)
    rot_y = pyrr.Matrix44.from_y_rotation(view_opt.yrot)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)
    glPolygonOffset(1.0,1.0)

    ####################### Draw textures end grain faces #######################

    G0 = [faces_ends_a, faces_ends_b]
    G1 = [faces_not_ends_a, faces_not_ends_b]
    draw_geometries_with_excluded_area(G0,G1)

    ######################## SWITCH TO COLOR SHADER ###########################

    glUseProgram(shader_col)
    move = pyrr.matrix44.create_from_translation(move_vec)
    sol_col = [[0.5,0.0,1.0]]
    sol_col = np.array(sol_col, dtype=np.uint32)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)

    #################### Draw geometry of unconnected voxels ###################

    if not mesh.connected:

        # 1. Draw hidden geometry
        glUniform3f(5, 1.0, 0.8, 0.7) # light red orange
        glClear(GL_DEPTH_BUFFER_BIT)
        draw_geometries([faces_unconnected_a])
        draw_geometries([faces_unconnected_b])

        # 1. Draw visible geometry
        glUniform3f(5, 1.0, 0.2, 0.0) # red orange
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [faces_unconnected_a, faces_unconnected_b]
        G1 = [faces_connected_a,   faces_connected_b]
        draw_geometries_with_excluded_area(G0,G1)

    ##################### Draw colors unbridged components #####################

    # 1. Unbringed component A
    if not view_opt.hidden_a and not mesh.bridged_A:
        # a) Unbridge part 1
        glUniform3f(5, 1.0, 1.0, 0.6) # light yellow
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [faces_unbridged_1_a]
        G1 = [faces_unbridged_2_a, faces_all_b, faces_unconnected_a]
        draw_geometries_with_excluded_area(G0,G1)
        # b) Unbridge part 2
        glUniform3f(5, 0.6, 1.0, 1.0) # light turkoise
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [faces_unbridged_2_a]
        G1 = [faces_unbridged_1_a, faces_all_b, faces_unconnected_a]
        draw_geometries_with_excluded_area(G0,G1)

    # 1. Unbringed component B
    if not view_opt.hidden_b and not mesh.bridged_B:
        # a) Unbridge part 1
        glUniform3f(5, 1.0, 0.6, 1.0) # light pink
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [faces_unbridged_1_b]
        G1 = [faces_unbridged_2_b, faces_all_a, faces_unconnected_b]
        draw_geometries_with_excluded_area(G0,G1)
        # b) Unbridge part 2
        glUniform3f(5, 1.0, 0.8, 0.6) # light orange
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [faces_unbridged_2_b]
        G1 = [faces_unbridged_1_b, faces_all_a, faces_unconnected_b]
        draw_geometries_with_excluded_area(G0,G1)

    ################### Draw top face that is currently being hovered ##########
    if mesh.Selected!=None:
        # Draw base face (hovered)
        glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
        glUniform3f(5, 0.2, 0.2, 0.2) #dark grey
        index = mesh.dim*mesh.Selected.x+mesh.Selected.y
        if mesh.Selected.n==0:
            top_start=topA_start
            top_move_mat = move_A
        else:
            top_start = topB_start
            top_move_mat = move_B
        G0 = [[GL_QUADS, 4, top_start+4*index, top_move_mat, True]]
        G1 = [faces_not_tops_a,faces_not_tops_b]
        draw_geometries_with_excluded_area(G0,G1)
        # Outline
        glClear(GL_DEPTH_BUFFER_BIT)
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(1)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, ctypes.c_void_p(4*(top_start+4*index)))
        glPopAttrib()

    ################ Draw top face that is currently being pulled ###############
        if mesh.Selected.val!=0:
            for val in range(1,abs(mesh.Selected.val)+1):
                if mesh.Selected.val<0: val = -val
                pulled_vec = [0,0,0]
                ax = mesh.sliding_direction[0]
                pulled_vec[ax] = (2*mesh.Selected.n-1) * view_opt.distance + val*mesh.voxel_size
                move = pyrr.matrix44.create_from_translation(pulled_vec)
                glUniformMatrix4fv(4, 1, GL_FALSE, move)
                glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
                glPushAttrib(GL_ENABLE_BIT)
                glLineWidth(3)
                glEnable(GL_LINE_STIPPLE)
                glLineStipple(2, 0xAAAA)
                glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, ctypes.c_void_p(4*(top_start+4*index)))
                glPopAttrib()

    ############################# Draw hidden lines #############################
    glClear(GL_DEPTH_BUFFER_BIT)
    glUniform3f(5,0.0,0.0,0.0)
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) #dashed line
    glEnable(GL_LINE_STIPPLE)
    if view_opt.show_hidden_lines:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [lines_a]
        G1 = [faces_all_a]
        draw_geometries_with_excluded_area(G0,G1)
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [lines_b]
        G1 = [faces_all_b]
        draw_geometries_with_excluded_area(G0,G1)
    glPopAttrib()

    ############################ Draw visible lines #############################
    glLineWidth(3)
    glClear(GL_DEPTH_BUFFER_BIT)
    G0 = [lines_a, lines_b]
    G1 = [faces_all_a, faces_all_b]
    draw_geometries_with_excluded_area(G0,G1)

    ################ When joint is fully open, draw dahsed lines ################
    if not view_opt.hidden_a and not view_opt.hidden_b and view_opt.distance==mesh.component_size:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        G0 = [lines_open_a,lines_open_b]
        G1 = [faces_all_a, faces_all_b]
        draw_geometries_with_excluded_area(G0,G1)
        glPopAttrib()

    ############# Milling paths
    ### Draw gpath lines ###
    #glClear(GL_DEPTH_BUFFER_BIT)
    #glLineWidth(1)
    #if not view_opt.hidden_a and mesh.show_milling_path:
    #    glDrawElements(GL_LINE_STRIP, mesh.imA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+2*mesh.iopen)))
    #if not view_opt.hidden_b and mesh.show_milling_path:
    #    glDrawElements(GL_LINE_STRIP, mesh.imB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(iA+iB+2*mesh.iopen+mesh.imA)))

    ## the end
    glfw.swap_buffers(window)

def pick(window, mesh, view_opt, shader_col):

    iA = mesh.ifA + mesh.ifeA + mesh.ifuA + mesh.ilA
    iB = mesh.ifB + mesh.ifeB + mesh.ifuB + mesh.ilB
    topA_start = iA+iB+mesh.ifubA1+mesh.ifubB1+mesh.ifubB2+2*mesh.iopen
    topB_start = topA_start+mesh.iftA+mesh.ifntA

    move_vec = [0,0,0]
    move_vec[mesh.sliding_direction[0]] = (2*mesh.sliding_direction[1]-1)*view_opt.distance
    move_vec = np.array(move_vec)
    move_A = pyrr.matrix44.create_from_translation(move_vec)
    move_B = pyrr.matrix44.create_from_translation(np.negative(move_vec))

    ######################## COLOR SHADER ###########################
    glUseProgram(shader_col)
    glClearColor(1.0, 1.0, 1.0, 1.0) # white
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    rot_x = pyrr.Matrix44.from_x_rotation(view_opt.xrot)
    rot_y = pyrr.Matrix44.from_y_rotation(view_opt.yrot)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)
    glPolygonOffset(1.0,1.0)

    ########################## Draw colorful top faces ##########################
    if not view_opt.hidden_a:
        # Draw body geometry of A
        glUniform3f(5, 1.0, 0.0, 0.0)
        glUniformMatrix4fv(4, 1, GL_FALSE, move_A)
        glDrawElements(GL_QUADS, mesh.ifntA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(topA_start+mesh.iftA)))
        # Draw faces of A
        for i in range(mesh.dim*mesh.dim):
            i_ = int(i/mesh.dim)
            j = i%mesh.dim
            glUniform3f(5, 1.0, (i_+1)/(mesh.dim+2), (j+1)/(mesh.dim+2)) # color
            glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(topA_start+i*4)))
    if not view_opt.hidden_b:
        # Draw body geometry of B
        glUniform3f(5, 0.0, 0.0, 1.0)
        glUniformMatrix4fv(4, 1, GL_FALSE, move_B)
        glDrawElements(GL_QUADS, mesh.ifntB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(topB_start+mesh.iftB)))
        # Draw faces of B
        for i in range(mesh.dim*mesh.dim):
            i_ = int(i/mesh.dim)
            j = i%mesh.dim
            glUniform3f(5, (i_+1)/(mesh.dim+2), (j+1)/(mesh.dim+2), 1.0) # color
            glDrawElements(GL_QUADS, 4, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(topB_start+i*4)))

    ############### Read pixel color at mouse position ###############
    xpos,ypos = glfw.get_cursor_pos(window)
    mouse_pixel = glReadPixelsub(xpos, 1600-ypos, 1, 1, GL_RGB, outputType=None)[0][0]
    pick_n = pick_x = pick_y = None
    if mouse_pixel[0]==255 and mouse_pixel[1]!=255:
        pick_n=0
        if mouse_pixel[2]!=0:
            pick_x=int(mouse_pixel[1]*(mesh.dim+2)/255-1)
            pick_y=int(mouse_pixel[2]*(mesh.dim+2)/255-1)
    elif mouse_pixel[2]==255 and mouse_pixel[1]!=255:
        pick_n=1
        if mouse_pixel[0]!=0:
            pick_x=int(mouse_pixel[0]*(mesh.dim+2)/255-1)
            pick_y=int(mouse_pixel[1]*(mesh.dim+2)/255-1)
    if pick_x !=None and pick_y!=None:
        Geometries.init_selection(mesh,pick_x,pick_y,pick_n)
    else: mesh.Selected = None
    ## the end
    #glfw.swap_buffers(window) # for debugging
    glClearColor(1.0, 1.0, 1.0, 1.0)

def main():
    global mesh, view_opt

    # Initialize window
    window = initialize()

    # Create shaders
    shader_tex = create_texture_shaders()
    shader_col = create_color_shaders()

    mesh = Geometries()
    view_opt = ViewSettings()

    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):

        glfw.poll_events()

        # Update Rotation
        view_opt.update_rotation(window)

        # Update opening opening distance
        view_opt.set_joint_opening_distance(mesh)

        # Picking
        if mesh.Selected==None or not mesh.Selected.active:
            pick(window, mesh, view_opt, shader_col)
        elif mesh.Selected!=None and mesh.Selected.active:
            mesh.Selected.edit(glfw.get_cursor_pos(window), view_opt.xrot, view_opt.yrot)

        # Display joint geometries
        display(window, mesh, view_opt, shader_tex, shader_col)

    glfw.terminate()

if __name__ == "__main__":
    print("Hit ESC key to quit.")
    print("Rotate view with right mouse button")
    print("Edit joint geometry by pushing/pulling")
    print("Clear joint geometry with:\nC")
    print("Edit joint type: I L T X")
    print("Change voxel resolution: 2, 3, 4, 5")
    print("Edit joint sliding direction: arrow keys")
    print("Open joint: O")
    print("Hide components: A B")
    print("Hide hidden lines: H")
    print("Show milling path: M")
    print("Press S to save joint geometry and G to open last saved geometry")

    main()
