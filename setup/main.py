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
import argparse
from itertools import product

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
    global glo_start_time
    global loc_start_time
    global click_cnt
    mesh, view_opt, args = glfw.get_window_user_pointer(window)
    if glo_start_time==None:
        glo_start_time = glfw.get_time()
        loc_start_time = glfw.get_time()

    if action==glfw.PRESS:
        if loc_start_time==None:
            loc_start_time = glfw.get_time()
            print("Starting next one...")

        # Active key commands, both during user study and not
        if key==glfw.KEY_LEFT_SHIFT or key==glfw.KEY_RIGHT_SHIFT:
            mesh.select.shift = True
            mesh.select.refresh = True
        # Joint geometry
        elif key==glfw.KEY_N: Geometries.clear_height_fields(mesh)
        elif key==glfw.KEY_R: Geometries.randomize_height_fields(mesh)
        elif key==glfw.KEY_Z: Geometries.undo(mesh)
        # Preview options
        elif key==glfw.KEY_A: view_opt.hidden[0] = not view_opt.hidden[0]
        elif key==glfw.KEY_B: view_opt.hidden[1] = not view_opt.hidden[1]
        elif key==glfw.KEY_C: view_opt.hidden[2] = not view_opt.hidden[2]
        elif key==glfw.KEY_D: view_opt.hidden[3] = not view_opt.hidden[3]
        elif key==glfw.KEY_E: view_opt.hidden[4] = not view_opt.hidden[4]
        elif key==glfw.KEY_F: view_opt.hidden[5] = not view_opt.hidden[5]
        elif key==glfw.KEY_X: view_opt.show_arrows = not view_opt.show_arrows
        elif key==glfw.KEY_H: view_opt.show_hidden_lines = not view_opt.show_hidden_lines
        elif key==glfw.KEY_SPACE: view_opt.open_joint = not view_opt.open_joint
        # Key commands only for user study
        if args.user_study:
            if key==glfw.KEY_ENTER:
                duration = glfw.get_time()-loc_start_time
                print("Saving "+args.username+"'s joint...")
                print("Design completed in", int(duration), "seconds and with", click_cnt, "clicks")
                Geometries.user_study_design_finished(mesh,args,duration,click_cnt)
                click_cnt = 0
                loc_start_time = None
        # Key commands locked during user study
        elif not args.user_study:
            # Sliding direction
            if (key==glfw.KEY_UP or key==glfw.KEY_DOWN) and mesh.sax!=2:
                Geometries.update_sliding_direction(mesh,2)
            elif key==glfw.KEY_RIGHT and mesh.sax!=1:
                Geometries.update_sliding_direction(mesh,1)
            elif key==glfw.KEY_LEFT and mesh.sax!=0:
                Geometries.update_sliding_direction(mesh,0)
            # Save / Open
            elif key==glfw.KEY_S: print("Saving joint..."); Geometries.save(mesh)
            elif key==glfw.KEY_O: print("Opening saved joint..."); Geometries.load(mesh)
            # Milling path
            elif key==glfw.KEY_M:
                view_opt.show_milling_path = not view_opt.show_milling_path
                if view_opt.show_milling_path:
                    Geometries.create_vertices(mesh,True)
                    Geometries.create_indices(mesh,True)
            elif key==glfw.KEY_G: mesh.fab.export_gcode("joint")
            # Change resolution
            elif key==glfw.KEY_RIGHT_BRACKET and mesh.dim<5: Geometries.update_dimension(mesh,mesh.dim+1)
            elif key==glfw.KEY_LEFT_BRACKET and mesh.dim>2: Geometries.update_dimension(mesh,mesh.dim-1)
            # Change number of components
            elif key==glfw.KEY_2 and mesh.noc!=2: Geometries.update_number_of_components(mesh,2)
            elif key==glfw.KEY_3 and mesh.noc!=3: Geometries.update_number_of_components(mesh,3)
            elif key==glfw.KEY_4 and mesh.noc!=4: Geometries.update_number_of_components(mesh,4)
            elif key==glfw.KEY_5 and mesh.noc!=5: Geometries.update_number_of_components(mesh,5)
            elif key==glfw.KEY_6 and mesh.noc!=6: Geometries.update_number_of_components(mesh,6)
            # Screenshot
            elif key==glfw.KEY_P: save_screenshot(window)
    elif action==glfw.RELEASE:
        # Face manipulation with shift button
        if key==glfw.KEY_LEFT_SHIFT or key==glfw.KEY_RIGHT_SHIFT:
            mesh.select.shift = False
            mesh.select.refresh = True

def mouseCallback(window,button,action,mods):
    mesh, view_opt, args = glfw.get_window_user_pointer(window)
    global glo_start_time
    global loc_start_time
    global click_cnt
    if glo_start_time==None:
        print("Starting...")
        glo_start_time = glfw.get_time()
    if loc_start_time==None:
        loc_start_time = glfw.get_time()
        print("Starting next one...")
    if button==glfw.MOUSE_BUTTON_LEFT:
        if action==1: # pressed
            if mesh.select.state==0: #face hovered
                mesh.select.start_pull(glfw.get_cursor_pos(window))
                click_cnt += 1
            elif mesh.select.state==10 and not args.user_study: #body hovered
                mesh.select.start_move(glfw.get_cursor_pos(window))
        elif action==0: #released
            if mesh.select.state==2: #face pulled
                mesh.select.end_pull()
            elif mesh.select.state==12 and not args.user_study: # body moved
                mesh.select.end_move()
    elif button==glfw.MOUSE_BUTTON_RIGHT:
        if action==1: ViewSettings.start_rotation(view_opt, window)
        elif action==0: ViewSettings.end_rotation(view_opt)

def save_screenshot(window):
    image_buffer = glReadPixels(0, 0, 1600, 1600, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(1600, 1600, 3)
    image = np.flip(image,axis=0)
    image = np.flip(image,axis=2)
    cv2.imwrite("screenshot.png", image)
    print("Saved screenshot.png.")

def draw_geometries(window,geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
    mesh, view_opt, args = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening
    move_vec = [0,0,0]
    move_vec[mesh.sax] = view_opt.open_ratio*mesh.component_size
    move_vec = np.array(move_vec)
    moves = []
    for n in range(mesh.noc):
        tot_move_vec = (2*n+1-mesh.noc)/(mesh.noc-1)*move_vec
        move_mat = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
        moves.append(move_mat)
    if clear_depth_buffer: glClear(GL_DEPTH_BUFFER_BIT)
    for geo in geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

def draw_geometries_with_excluded_area(window, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
    mesh, view_opt, args = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening
    move_vec = [0,0,0]
    move_vec[mesh.sax] = view_opt.open_ratio*mesh.component_size
    move_vec = np.array(move_vec)
    moves = []
    moves_show = []
    for n in range(mesh.noc):
        tot_move_vec = (2*n+1-mesh.noc)/(mesh.noc-1)*move_vec
        move_mat = pyrr.matrix44.create_from_translation(tot_move_vec)
        moves.append(move_mat)
        move_mat_show = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
        moves_show.append(move_mat_show)
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
    glfw.set_mouse_button_callback(window, mouseCallback)
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
    col = [1.0, 0.8, 0.7]  # light red orange
    glUniform3f(5, col[0], col[1], col[2])
    for n in range(mesh.noc):
        if not mesh.eval.connected[n]: draw_geometries(window,[mesh.indices_not_fcon[n]])

    # 1. Draw visible geometry
    col = [1.0, 0.2, 0.0] # red orange
    glUniform3f(5, col[0], col[1], col[2])
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
    G1 = mesh.indices_fpick_not_top
    for face in mesh.select.faces:
        if mesh.select.n==0 or mesh.select.n==mesh.noc-1: mos = 1
        else: mos = 2
        index = int(mesh.dim*face[0]+face[1])
        top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[mesh.select.n].start_index+mos*4*index+(mos-1)*4*mesh.select.dir, mesh.select.n)
        #top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[mesh.select.n].start_index+4*index, mesh.select.n)
        draw_geometries_with_excluded_area(window,[top],G1)
    # Draw pulled face
    if mesh.select.state==2:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(3)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        for val in range(0,abs(mesh.select.val)+1):
            if mesh.select.val<0: val = -val
            pulled_vec = [0,0,0]
            pulled_vec[mesh.sax] = val*mesh.voxel_size
            draw_geometries(window,[mesh.outline_selected_faces],translation_vec=np.array(pulled_vec))
        glPopAttrib()

def display_moving_rotating(window,mesh,view_opt):
    # Draw moved_rotated component before action is finalized
    if mesh.select.state==12 and mesh.outline_selected_component!=None:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(3)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        draw_geometries(window,[mesh.outline_selected_component])
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
    for n in range(mesh.noc):
        if mesh.eval.fab_direction_ok[n]: glUniform3f(5,0.0,0.0,0.0) # black
        else: glUniform3f(5,1.0,0.0,0.0) # red
        G0 = [mesh.indices_lns[n]]
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    ################ When joint is fully open, draw dahsed lines ################
    if not view_opt.hidden[0] and not view_opt.hidden[1] and view_opt.open_ratio==1+0.5*(mesh.noc-2):
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        G0 = mesh.indices_open_lines
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
        glPopAttrib()

def display_arrows(window,mesh,view_opt):
    glUniform3f(5, 0.0, 0.0, 0.0)
    ############################## Direction arrows ################################
    if view_opt.show_arrows:
        for n in range(mesh.noc):
            glLineWidth(3)
            G1 = mesh.indices_fall
            G0 = mesh.indices_arrows[n]
            d0 = 2.55*mesh.component_size
            d1 = 1.55*mesh.component_size
            if len(mesh.fixed_sides[n])==2: d0 = d1
            for ax,dir in mesh.fixed_sides[n]:
                vec = np.array([0,0,0],dtype=np.float)
                vec[ax] = (2*dir-1)*d0
                draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)

def display_checker(window,mesh,view_opt):
    # 1. Draw hidden geometry
    glUniform3f(5, 1.0, 0.2, 0.0) # red orange
    glLineWidth(8)
    for n in range(mesh.noc):
        if mesh.eval.checker[n]:
            draw_geometries(window,[mesh.indices_chess_lines[n]])
    glUniform3f(5, 0.0, 0.0, 0.0) # back to black

def display_breakable(window,mesh,view_opt):
    # 1. Draw hidden geometry
    glPushAttrib(GL_ENABLE_BIT)
    glUniform3f(5, 1.0, 0.9, 0.0) # yellow
    glLineWidth(3)
    glEnable(GL_LINE_STIPPLE)
    glLineStipple(4, 0xAAAA)
    for n in range(mesh.noc):
        G0 = [mesh.indices_breakable_lines[n]]
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    glPopAttrib()

def display_milling_paths(window,mesh,view_opt):
    if len(mesh.indices_milling_path)==0: view_opt.show_milling_path = False
    if view_opt.show_milling_path:
        cols = [[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0],[0.0,1.0,1.0],[1.0,0,1.0]]
        glLineWidth(3)
        for n in range(mesh.noc):
            if mesh.eval.fab_direction_ok[n]:
                glUniform3f(5,cols[n][0],cols[n][1],cols[n][2])
                draw_geometries(window,[mesh.indices_milling_path[n]])

def pick(window, mesh, view_opt, shader_col, show_col=False):

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
    col_step = 1.0/(2+2*mesh.dim*mesh.dim)
    for n in range(mesh.noc):
        col = np.zeros(3, dtype=np.float64)
        col[n%mesh.dim] = 1.0
        if n>2: col[(n+1)%mesh.dim] = 1.0
        glUniform3f(5, col[0], col[1], col[2])
        draw_geometries(window,[mesh.indices_fpick_not_top[n]],clear_depth_buffer=False)
        if n==0 or n==mesh.noc-1: mos = 1
        else: mos = 2
        # mos is "number of sides"
        for m in range(mos):
            # Draw top faces
            for i in range(mesh.dim*mesh.dim):
                col -= col_step
                glUniform3f(5, col[0], col[1], col[2])
                top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[n].start_index+mos*4*i+4*m, n)
                draw_geometries(window,[top],clear_depth_buffer=False)

    ############### Read pixel color at mouse position ###############
    xpos,ypos = glfw.get_cursor_pos(window)
    mouse_pixel = glReadPixelsub(xpos, 1600-ypos, 1, 1, GL_RGB, outputType=None)[0][0]
    mouse_pixel = np.array(mouse_pixel)
    pick_n = pick_d = pick_x = pick_y = None
    if not np.all(mouse_pixel==255): # not white / background
        non_zeros = np.where(mouse_pixel!=0)
        if len(non_zeros)>0:
            if len(non_zeros[0]>0):
                pick_n = non_zeros[0][0]
                if len(non_zeros[0])>1:
                    pick_n = pick_n+mesh.dim
                    if mouse_pixel[0]==mouse_pixel[2]: pick_n = 5
                val = 255-mouse_pixel[non_zeros[0][0]]
                i = int(0.5+val*(2+2*mesh.dim*mesh.dim)/255)-1
                if i>=0:
                    pick_x = (int(i/mesh.dim))%mesh.dim
                    pick_y = i%mesh.dim
                pick_d = 0
                if pick_n==mesh.noc-1: pick_d = 1
                elif int(i/mesh.dim)>=mesh.dim: pick_d = 1
                #print(pick_n,pick_d,pick_x,pick_y)

    ### Update selection
    if pick_x !=None and pick_d!=None and pick_y!=None and pick_n!=None:
        ### Initialize selection
        new_pos = False
        if pick_x!=mesh.select.x or pick_y!=mesh.select.y or pick_n!=mesh.select.n or pick_d!=mesh.select.dir or mesh.select.refresh:
            mesh.select.update_pick(pick_x,pick_y,pick_n,pick_d)
            mesh.select.refresh = False
            mesh.select.state = 0 # hovering
    elif pick_n!=None:
        mesh.select.state = 10 # hovering component body
        mesh.select.update_pick(pick_x,pick_y,pick_n,pick_d)
    else: mesh.select.state = -1
    if not show_col: glClearColor(1.0, 1.0, 1.0, 1.0)

def main():

    # Arguments for user study
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_study', default="F", dest='user_study', type=str)
    parser.add_argument('--username', default="test", type=str, dest='username')
    parser.add_argument('--feedback', default="T", dest='feedback', type=str)
    parser.add_argument('--type', default=0, type=int, dest='type')
    parser.add_argument('--stats', default="F", dest='stats', type=str)
    args = parser.parse_args()
    if args.user_study=="T" or args.user_study=="True": args.user_study=True
    else: args.user_study=False
    if args.feedback=="T" or args.feedback=="True": args.feedback=True
    else: args.feedback=False
    if args.stats=="T" or args.stats=="True": args.stats=True
    else: args.stats=False

    global glo_start_time
    global loc_start_time
    global click_cnt
    glo_start_time = None
    loc_start_time = None
    click_cnt = 0
    time_passed = 0
    max_time = 5*60 #seconds
    last_printed = 0

    # Initialize window
    window = initialize()

    # Create shaders
    shader_tex = create_texture_shaders()
    shader_col = create_color_shaders()

    if args.type==0:
        fixed_sides = [[[2,0]],[[2,1]]]
        sax = 2
    elif args.type==1:
        fixed_sides = [[[2,0]],[[0,0]]]
        sax = 2
    else:
        fixed_sides = [[[2,0]],[[0,0],[0,1]]]
        sax = 2

    mesh = Geometries(fixed_sides,sax)
    view_opt = ViewSettings()

    glfw.set_window_user_pointer(window, [mesh, view_opt, args])

    if args.stats:
        # browse all possibilities, reinitiate mesh as such... count failure modes
        uncon_cnt = unbri_cnt = unfab_cnt = check_cnt = mslid_cnt = frag_cnt = valid_cnt = total_cnt = 0
        # Get all combinations of 0-dim and length dim*dim
        hlist = []
        #for h in range(mesh.dim+1): hlist.append(h)
        for h in range(2): hlist.append(h)
        roll = product(hlist, repeat=mesh.dim*mesh.dim)
        fixed_sides = [[[2,0]],[[2,1]]]
        sax = 2
        for i,flat_hf in enumerate(list(roll)):
            # translate flat field to 2d matrix
            hf = np.zeros((mesh.dim,mesh.dim))
            for i_ in range(mesh.dim):
                for j_ in range(mesh.dim):
                    hf[i_][j_] = list(flat_hf)[i_*mesh.dim+j_]
            # reinitiate mesh with new height field
            mesh = Geometries(fixed_sides,sax,hfs=[hf],draw=False)
            # evaluate
            if not all(mesh.eval.connected): uncon_cnt+=1
            elif not all(mesh.eval.bridged): unbri_cnt+=1
            elif not all(mesh.eval.fab_direction_ok): unfab_cnt+=1
            elif any(mesh.eval.checker): check_cnt+=1
            elif sorted(mesh.eval.number_of_slides)[-1]>1: mslid_cnt+=1
            elif any(mesh.eval.breakable): frag_cnt+=1
            else: valid_cnt+=1
            total_cnt +=1
        print("Totoal:\t\t",total_cnt)
        print("Unconnected:\t",uncon_cnt)
        print("Unbridged:\t",unbri_cnt)
        print("Unfabricatable:\t",unfab_cnt)
        print("Checkerboard:\t",check_cnt)
        print("Mult slides:\t",mslid_cnt)
        print("Fragile parts:\t",frag_cnt)
        print("Valid:\t\t",valid_cnt)

    while not args.stats and glfw.get_key(window,glfw.KEY_ESCAPE)!=glfw.PRESS and not glfw.window_should_close(window) and not time_passed>max_time:

        glfw.poll_events()

        if glo_start_time!=None and args.user_study:
            time_passed = glfw.get_time()-glo_start_time

        if time_passed>(last_printed*60+1) and int(time_passed)%60==0:
            print(int(int(time_passed)/60),"min")
            last_printed = int(int(time_passed)/60)

        if time_passed>max_time:
            print("Time has run up. Closing...")

        # Update view rotation
        view_opt.update_rotation(window)

        # Update joint opening distance
        if (view_opt.open_joint and view_opt.open_ratio<mesh.noc-1) or (not view_opt.open_joint and view_opt.open_ratio>0):
            view_opt.set_joint_opening_distance(mesh.noc)

        # Pick faces -1: nothing, 0: hovered, 1: adding, 2: pulling
        if not mesh.select.state==2 and not mesh.select.state==12:
            pick(window, mesh, view_opt, shader_col, show_col=False) # active pulling
        elif mesh.select.state==2:
            mesh.select.edit(glfw.get_cursor_pos(window), view_opt.xrot, view_opt.yrot)
        elif mesh.select.state==12:
            mesh.select.move(glfw.get_cursor_pos(window), view_opt.xrot, view_opt.yrot)

        # Display joint geometries
        init_display()
        init_shader(shader_tex, view_opt)
        display_end_grains(window,mesh)
        init_shader(shader_col, view_opt)
        if args.feedback:
            if not all(mesh.eval.connected): display_unconnected(window,mesh)
            if not all(mesh.eval.bridged): display_unbridged(window,mesh,view_opt)
            if any(mesh.eval.checker): display_checker(window,mesh,view_opt)
            if view_opt.show_arrows: display_arrows(window,mesh,view_opt)
        if mesh.select.state!=-1:
            display_selected(window,mesh,view_opt)
            display_moving_rotating(window,mesh,view_opt)
        display_joint_geometry(window,mesh,view_opt)
        #if args.feedback and mesh.eval.breakable: display_breakable(window,mesh,view_opt)
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
    print("SPACE - open joint")
    print("A B C - hide components")
    print("H - hide hidden lines")
    print("M - show milling path")
    print("G - export gcode for milling path")
    print("S - save")
    print("O - open")
    print("P - save screenshot")
    print("ESC - quit\n")
    main()
