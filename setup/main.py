import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
from numpy import linalg
import pyrr
import sys
from Geometries import Geometries
from ViewSettings import ViewSettings
from Buffer import ElementProperties
import ctypes
import math
import cv2
import random

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
    mesh, view_opt = glfw.get_window_user_pointer(window)
    if action==glfw.PRESS:
        if key==glfw.KEY_LEFT_SHIFT or key==glfw.KEY_RIGHT_SHIFT:
            mesh.select.shift = True
            mesh.select.refresh = True
        # Joint geometry
        elif key==glfw.KEY_E: Geometries.clear_height_field(mesh)
        # Joint type
        elif key==glfw.KEY_I and mesh.joint_type!="I":
            Geometries.update_joint_type(mesh,"I",mesh.noc)
        elif key==glfw.KEY_L and mesh.joint_type!="L":
            Geometries.update_joint_type(mesh,"L",mesh.noc)
        elif key==glfw.KEY_T and mesh.joint_type!="T":
            Geometries.update_joint_type(mesh,"T",mesh.noc)
        elif key==glfw.KEY_X and mesh.joint_type!="X":
            Geometries.update_joint_type(mesh,"X",mesh.noc)
        elif key==glfw.KEY_Y:
            Geometries.update_joint_type(mesh,mesh.joint_type,5-mesh.noc)
        # Sliding direction
        elif key==glfw.KEY_UP and mesh.sliding_directions!=[[[2,0]],[[2,1]]]:
            if mesh.joint_type!="X":
                Geometries.update_sliding_direction(mesh,[[[2,0]],[[2,1]]])
        elif key==glfw.KEY_RIGHT and mesh.sliding_directions!=[[[1,0]],[[1,1]]]:
            Geometries.update_sliding_direction(mesh,[[[1,0]],[[1,1]]])
        # Preview options
        elif key==glfw.KEY_A: view_opt.hidden[0] = not view_opt.hidden[0]
        elif key==glfw.KEY_B: view_opt.hidden[1] = not view_opt.hidden[1]
        elif key==glfw.KEY_C: view_opt.hidden[2] = not view_opt.hidden[2]
        elif key==glfw.KEY_D: view_opt.show_arrows = not view_opt.show_arrows
        elif key==glfw.KEY_H: view_opt.show_hidden_lines = not view_opt.show_hidden_lines
        elif key==glfw.KEY_F:
            mesh.fab_geometry = not mesh.fab_geometry
            Geometries.create_and_buffer_indicies(mesh)
        elif key==glfw.KEY_O: view_opt.open_joint = (view_opt.open_joint+1)%mesh.noc
        elif key==glfw.KEY_S: print("Saving..."); Geometries.save(mesh)
        elif key==glfw.KEY_G: print("Loading..."); Geometries.load(mesh)
        elif key==glfw.KEY_M:
            view_opt.show_milling_path = not view_opt.show_milling_path
            if view_opt.show_milling_path:
                Geometries.create_vertices(mesh,True)
                Geometries.create_indices(mesh,True)
        elif key==glfw.KEY_2 and mesh.dim!=2: Geometries.update_dimension(mesh,2)
        elif key==glfw.KEY_3 and mesh.dim!=3: Geometries.update_dimension(mesh,3)
        elif key==glfw.KEY_4 and mesh.dim!=4: Geometries.update_dimension(mesh,4)
        elif key==glfw.KEY_5 and mesh.dim!=5: Geometries.update_dimension(mesh,5)
        elif key==glfw.KEY_R: Geometries.randomize_height_field(mesh)
        elif key==glfw.KEY_P: save_screenshot(window)
        elif key==glfw.KEY_K: mesh.fab.export_gcode("joint")
        elif key==glfw.KEY_Z: Geometries.undo(mesh)
    elif action==glfw.RELEASE:
        if key==glfw.KEY_LEFT_SHIFT or key==glfw.KEY_RIGHT_SHIFT:
            mesh.select.shift = False
            mesh.select.refresh = True

def mouseCallback(window,button,action,mods):
    mesh, view_opt = glfw.get_window_user_pointer(window)
    if button==glfw.MOUSE_BUTTON_LEFT:
        if action==1: #pressed
            mesh.select.start_pull(glfw.get_cursor_pos(window))
        elif action==0: #released
            mesh.select.end_pull()
    elif button==glfw.MOUSE_BUTTON_RIGHT:
        if action==1: ViewSettings.start_rotation(view_opt, window)
        elif action==0: ViewSettings.end_rotation(view_opt)

def save_screenshot(window):
    image_buffer = glReadPixels(0, 0, 1600, 1600, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(1600, 1600, 3)
    image = np.flip(image,axis=0)
    image = np.flip(image,axis=2)
    cv2.imwrite("screenshot.png", image)

def draw_geometries(window,geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
    mesh, view_opt = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening of joint for components A and B
    move_vec = [0,0,0]
    move_vec[mesh.sliding_directions[0][0][0]] = (2*mesh.sliding_directions[0][0][1]-1)*view_opt.distance
    move_vec = np.array(move_vec)
    move_vec = move_vec
    move_vec2 = [0,0,0]
    if mesh.noc>2: move_vec2[mesh.sliding_directions[2][0][0]] = (2*mesh.sliding_directions[2][0][1]-1)*view_opt.distance2
    move_vec2 = np.array(move_vec2)
    move_A = pyrr.matrix44.create_from_translation(move_vec-move_vec2+translation_vec)
    move_B = pyrr.matrix44.create_from_translation(np.negative(move_vec)-move_vec2+translation_vec)
    move_C = pyrr.matrix44.create_from_translation(np.negative(move_vec)+move_vec2+translation_vec)
    moves = [move_A,move_B,move_C]
    if clear_depth_buffer: glClear(GL_DEPTH_BUFFER_BIT)
    for geo in geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

def draw_geometries_with_excluded_area(window, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
    mesh, view_opt = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening of joint for components A and B
    move_vec = [0,0,0]
    move_vec[mesh.sliding_directions[0][0][0]] = (2*mesh.sliding_directions[0][0][1]-1)*view_opt.distance
    move_vec = np.array(move_vec)
    move_vec_show = move_vec + translation_vec
    move_vec2 = [0,0,0]
    if mesh.noc>2:
        move_vec2[mesh.sliding_directions[2][0][0]] = (2*mesh.sliding_directions[2][0][1]-1)*view_opt.distance2
    move_vec2 = np.array(move_vec2)
    move_A = pyrr.matrix44.create_from_translation(move_vec-move_vec2)
    move_B = pyrr.matrix44.create_from_translation(np.negative(move_vec)-move_vec2)
    move_C = pyrr.matrix44.create_from_translation(np.negative(move_vec)+move_vec2)
    moves = [move_A,move_B,move_C]
    move_A_show = pyrr.matrix44.create_from_translation(move_vec_show-move_vec2)
    move_B_show = pyrr.matrix44.create_from_translation(np.negative(move_vec_show)-move_vec2)
    move_C_show = pyrr.matrix44.create_from_translation(np.negative(move_vec_show)+move_vec2)
    moves_show = [move_A_show,move_B_show,move_C_show]
    #
    glClear(GL_DEPTH_BUFFER_BIT)
    glDisable(GL_DEPTH_TEST)
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
    glEnable(GL_STENCIL_TEST)
    glStencilFunc(GL_ALWAYS,1,1)
    glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
    for geo in show_geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
    glEnable(GL_DEPTH_TEST)
    glStencilFunc(GL_EQUAL,1,1)
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
    for geo in screen_geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
    glDisable(GL_STENCIL_TEST)
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
    for geo in show_geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

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

def init_display():
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glMatrixMode(GL_MODELVIEW)
    glPolygonOffset(1.0,1.0)

def init_shader(shader,view_opt):
    glUseProgram(shader)
    rot_x = pyrr.Matrix44.from_x_rotation(view_opt.xrot)
    rot_y = pyrr.Matrix44.from_y_rotation(view_opt.yrot)
    glUniformMatrix4fv(3, 1, GL_FALSE, rot_x * rot_y)

def display_end_grains(window,mesh):
    G0 = mesh.indices_fend
    G1 = mesh.indices_not_fend
    draw_geometries_with_excluded_area(window,G0,G1)

def display_unconnected(window,mesh):
    # 1. Draw hidden geometry
    glUniform3f(5, 1.0, 0.8, 0.7) # light red orange
    for n in range(mesh.noc):
        if not mesh.eval.connected[n]: draw_geometries(window,[mesh.indices_not_fcon[n]])

    # 1. Draw visible geometry
    glUniform3f(5, 1.0, 0.2, 0.0) # red orange
    G0 = []
    for n in range(mesh.noc):
        if not mesh.eval.connected[n]: G0.append(mesh.indices_not_fcon[n])
    G1 = mesh.indices_fcon
    draw_geometries_with_excluded_area(window,G0,G1)

def display_unbridged(window,mesh,view_opt):
    # Draw colored faces when unbridged
    for n in range(mesh.noc):
        if not mesh.eval.bridged[n]:
            for m in range(2): # browse the two parts
                # a) Unbridge part 1
                col = view_opt.unbridge_colors[n][m]
                glUniform3f(5, col[0], col[1], col[2])
                G0 = [mesh.indices_not_fbridge[n][m]]
                G1 = [mesh.indices_not_fbridge[n][1-m], mesh.indices_fall[1-n], mesh.indices_not_fcon[n]] # needs reformulation for 3 components
                draw_geometries_with_excluded_area(window,G0,G1)

def display_selected(window,mesh,view_opt):
    ################### Draw top face that is currently being hovered ##########
    # Draw base face (hovered)
    glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    glUniform3f(5, 0.2, 0.2, 0.2) #dark grey
    G1 = mesh.indices_not_ftop
    for face in mesh.select.faces:
        index = int(mesh.dim*face[0]+face[1])
        top = ElementProperties(GL_QUADS, 4, mesh.indices_ftop[mesh.select.n].start_index+4*index, mesh.select.n)
        draw_geometries_with_excluded_area(window,[top],G1)
    # Draw pulled face
    if mesh.select.state==2:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(3)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        ax = mesh.sliding_directions[mesh.select.n][0][0]
        for val in range(0,abs(mesh.select.val)+1):
            if mesh.select.val<0: val = -val
            pulled_vec = [0,0,0]
            pulled_vec[ax] = val*mesh.voxel_size
            draw_geometries(window,[mesh.outline_selected_faces],translation_vec=np.array(pulled_vec))
        glPopAttrib()

def display_joint_geometry(window,mesh,view_opt):
    ############################# Draw hidden lines #############################
    glClear(GL_DEPTH_BUFFER_BIT)
    glUniform3f(5,0.0,0.0,0.0) # black
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) #dashed line
    glEnable(GL_LINE_STIPPLE)
    if view_opt.show_hidden_lines:
        for n in range(mesh.noc):
            G0 = [mesh.indices_lns[n]]
            G1 = [mesh.indices_fall[n]]
            draw_geometries_with_excluded_area(window,G0,G1)
    glPopAttrib()
    ############################ Draw visible lines #############################
    glLineWidth(3)
    G0 = mesh.indices_lns
    G1 = mesh.indices_fall
    draw_geometries_with_excluded_area(window,G0,G1)
    ################ When joint is fully open, draw dahsed lines ################
    if not view_opt.hidden[0] and not view_opt.hidden[1] and view_opt.distance==mesh.component_size:
        #if view_opt.open_joint==1:
        #    for geo in mesh.indices_open_lines:
        #        geo.count = 8
        #    if len(mesh.indices_open_lines)>2: mesh.indices_open_lines.pop()
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        G0 = []
        for n in range(len(mesh.indices_open_lines)):
            if view_opt.distance2==mesh.component_size or n<2:
                G0.append(mesh.indices_open_lines[n])
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
        glPopAttrib()

def display_joint_geometry_lines(window,mesh,view_opt):
    glUniform3f(5,0.0,0.0,0.0) # black
    glLineWidth(3)
    G0 = [mesh.lines_a, mesh.lines_b]
    draw_geometries(window,G0)

def display_joint_faces(window,mesh,view_opt):
    glUniform3f(5,0.8,0.8,0.8) #grey
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )
    G0 = [mesh.faces_all_a, mesh.faces_all_b]
    draw_geometries(window,G0)
    glUniform3f(5,0.0,0.0,0.0) #black
    glLineWidth(1)
    glPolygonOffset(1.0, 1.0)
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE )
    G0 = [mesh.faces_all_a, mesh.faces_all_b]
    draw_geometries(window,G0,clear_depth_buffer=False)
    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL )

def display_arrows(window,mesh,view_opt):
    ############################## Direction arrows ################################
    if view_opt.show_arrows:
        for n in range(mesh.noc):
            glLineWidth(3)
            G1 = mesh.indices_fall
            G0 = mesh.indices_arrows[n]
            d0 = 3*mesh.component_length+0.55*mesh.component_size
            d1 = 2*mesh.component_length+0.55*mesh.component_size
            if len(mesh.fixed_sides[n])==2: d0 = d1
            if n>0: d0 = -d0 ###quick fix, not sure why???
            for ax,dir in mesh.fixed_sides[n]:
                vec = np.array([0,0,0],dtype=np.float)
                vec[ax] = (2*dir-1)*d0
                draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)

def display_milling_paths(window,mesh,view_opt):
    if len(mesh.indices_milling_path)==0: view_opt.show_milling_path = False
    if view_opt.show_milling_path:
        glLineWidth(1)
        glUniform3f(5,0.0,1.0,0.0)
        draw_geometries(window,[mesh.indices_milling_path[0]])
        glUniform3f(5,0.0,0.8,1.0)
        draw_geometries(window,[mesh.indices_milling_path[1]])

def pick(window, mesh, view_opt, shader_col):

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

    # Draw colorful geometries
    for n in range(mesh.noc):
        col = [0.0,0.0,0.0]
        col[n] = 1.0
        glUniform3f(5, col[0], col[1], col[2])
        draw_geometries(window,[mesh.indices_not_ftop[n]],clear_depth_buffer=False)
        # Draw top faces
        for i in range(mesh.dim*mesh.dim):
            i_ = int(i/mesh.dim)
            j = i%mesh.dim
            col = [(i_+1)/(mesh.dim+2), (j+1)/(mesh.dim+2)]
            col.insert(n,1.0)
            glUniform3f(5, col[0], col[1], col[2])
            top = ElementProperties(GL_QUADS, 4, mesh.indices_ftop[n].start_index+4*i, n)
            draw_geometries(window,[top],clear_depth_buffer=False)

    ############### Read pixel color at mouse position ###############

    xpos,ypos = glfw.get_cursor_pos(window)
    mouse_pixel = glReadPixelsub(xpos, 1600-ypos, 1, 1, GL_RGB, outputType=None)[0][0]
    mouse_pixel = np.array(mouse_pixel)
    pick_n = pick_x = pick_y = None
    if not np.all(mouse_pixel==255) and np.any(mouse_pixel==255):
        pick_n = np.where(mouse_pixel==255)[0][0]
        xyi = [0,1,2]
        xyi.pop(pick_n)
        pick_x=int(mouse_pixel[xyi[0]]*(mesh.dim+2)/255-1)
        pick_y=int(mouse_pixel[xyi[1]]*(mesh.dim+2)/255-1)

    ### Update selection
    if pick_x !=None and pick_y!=None:
        if pick_x!=-1 and pick_y!=-1:
            ### Initialize selection
            new_pos = False
            if pick_x!=mesh.select.x or pick_y!=mesh.select.y or pick_n!=mesh.select.n or mesh.select.refresh:
                mesh.select.update_pick(pick_x,pick_y,pick_n)
                mesh.select.refresh = False
            mesh.select.state = 0 # hovering
        else: mesh.select.state = -1
    glClearColor(1.0, 1.0, 1.0, 1.0)

def main():

    # Initialize window
    window = initialize()

    # Create shaders
    shader_tex = create_texture_shaders()
    shader_col = create_color_shaders()

    mesh = Geometries()
    view_opt = ViewSettings()

    glfw.set_window_user_pointer(window, [mesh, view_opt])

    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):

        glfw.poll_events()

        # Update view rotation
        view_opt.update_rotation(window)

        # Update joint opening distance
        view_opt.set_joint_opening_distance(mesh)

        # Pick faces
        if mesh.select.state!=2: pick(window, mesh, view_opt, shader_col)
        else: mesh.select.edit(glfw.get_cursor_pos(window), view_opt.xrot, view_opt.yrot)

        # Display joint geometries
        init_display()
        init_shader(shader_tex, view_opt)
        display_end_grains(window,mesh)
        init_shader(shader_col, view_opt)
        if not all(mesh.eval.connected): display_unconnected(window,mesh)
        if not all(mesh.eval.bridged): display_unbridged(window,mesh,view_opt)
        if mesh.select.state!=-1: display_selected(window,mesh,view_opt)
        display_joint_geometry(window,mesh,view_opt)
        if view_opt.show_arrows: display_arrows(window,mesh,view_opt)
        if view_opt.show_milling_path: display_milling_paths(window,mesh,view_opt)

        glfw.swap_buffers(window)

    glfw.terminate()

if __name__ == "__main__":
    print("Left mouse button - edit joint")
    print("Right mouse button - rotate view")
    print("R - randomize joint")
    print("I L T X - edit joint type")
    print("2 3 4 5 - change voxel resolution")
    print("Arrow keys - edit joint sliding direction")
    print("O - open joint")
    print("A B C - hide components")
    print("H - hide hidden lines")
    print("M - show milling path")
    print("K - export gcode for milling path")
    print("S - save")
    print("G - open")
    print("P - save screenshot")
    print("ESC - quit\n")
    main()
