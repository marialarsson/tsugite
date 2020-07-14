import wx
from wx import glcanvas
from OpenGL.GL import *
import OpenGL.GL.shaders
from pyrr import Matrix44, matrix44, Vector3
import time, sys
import ctypes
import numpy as np
from numpy import linalg
import pyrr
from Types import Types
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

def save_screenshot(window):
    image_buffer = glReadPixels(0, 0, 2000, 1600, OpenGL.GL.GL_RGB, OpenGL.GL.GL_UNSIGNED_BYTE)
    image = np.frombuffer(image_buffer, dtype=np.uint8).reshape(1600, 2000, 3)
    image = np.flip(image,axis=0)
    image = np.flip(image,axis=2)
    cv2.imwrite("screenshot.png", image)
    print("Saved screenshot.png.")

def draw_geometries(window,geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
    type, view_opt, args = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening
    move_vec = [0,0,0]
    move_vec[type.sax] = view_opt.open_ratio*type.component_size
    move_vec = np.array(move_vec)
    moves = []
    for n in range(type.noc):
        tot_move_vec = (2*n+1-type.noc)/(type.noc-1)*move_vec
        move_mat = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
        moves.append(move_mat)
    if clear_depth_buffer: glClear(GL_DEPTH_BUFFER_BIT)
    for geo in geos:
        if geo==None: continue
        if view_opt.hidden[geo.n]: continue
        glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
        glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

def draw_geometries_with_excluded_area(window, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
    type, view_opt, args = glfw.get_window_user_pointer(window)
    # Define translation matrices for opening
    move_vec = [0,0,0]
    move_vec[type.sax] = view_opt.open_ratio*type.component_size
    move_vec = np.array(move_vec)
    moves = []
    moves_show = []
    for n in range(type.noc):
        tot_move_vec = (2*n+1-type.noc)/(type.noc-1)*move_vec
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
    for n in range(mesh.parent.noc):
        if not mesh.eval.connected[n]: draw_geometries(window,[mesh.indices_not_fcon[n]])

    # 1. Draw visible geometry
    col = [1.0, 0.2, 0.0] # red orange
    glUniform3f(5, col[0], col[1], col[2])
    G0 = mesh.indices_not_fcon
    G1 = mesh.indices_fcon
    draw_geometries_with_excluded_area(window,G0,G1)

def display_area(window,mesh,view_opt):
    # 1. Draw hidden geometry
    if view_opt.show_friction:
        #tin = mesh.eval.friction_nums[0]/50
        #col = [0.9-tin, 1.0, 0.9-tin]  # green
        G0 = mesh.indices_ffric
        G1 = mesh.indices_not_ffric
    else:
        #tin = mesh.eval.contact_nums[0]/50
        #col = [0.9-tin, 0.9-tin, 1.0]  # blue
        G0 = mesh.indices_fcont
        G1 = mesh.indices_not_fcont
    #glUniform3f(5, col[0], col[1], col[2])
    draw_geometries_with_excluded_area(window,G0,G1)

def display_unbridged(window,mesh,view_opt):
    # Draw colored faces when unbridged
    for n in range(mesh.parent.noc):
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
        if mesh.select.n==0 or mesh.select.n==mesh.parent.noc-1: mos = 1
        else: mos = 2
        index = int(mesh.parent.dim*face[0]+face[1])
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
            pulled_vec[mesh.parent.sax] = val*mesh.parent.voxel_size
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

def display_joint_geometry(window,mesh,view_opt,lw=3,hidden=True,zoom=False):

    ############################# Draw hidden lines #############################
    glClear(GL_DEPTH_BUFFER_BIT)
    glUniform3f(5,0.0,0.0,0.0) # black
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) #dashed line
    glEnable(GL_LINE_STIPPLE)
    if hidden and view_opt.show_hidden_lines:
        for n in range(mesh.parent.noc):
            G0 = [mesh.indices_lns[n]]
            G1 = [mesh.indices_fall[n]]
            draw_geometries_with_excluded_area(window,G0,G1)
    glPopAttrib()

    ############################ Draw visible lines #############################
    for n in range(mesh.parent.noc):
        if not mesh.mainmesh or (mesh.eval.interlocks[n] and view_opt.show_feedback) or not view_opt.show_feedback:
            glUniform3f(5,0.0,0.0,0.0) # black
            glLineWidth(lw)
        else:
            glUniform3f(5,1.0,0.0,0.0) # red
            glLineWidth(lw+1)
        G0 = [mesh.indices_lns[n]]
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)


    if mesh.mainmesh:
        ################ When joint is fully open, draw dahsed lines ################
        if hidden and not view_opt.hidden[0] and not view_opt.hidden[1] and view_opt.open_ratio==1+0.5*(mesh.parent.noc-2):
            glUniform3f(5,0.0,0.0,0.0) # black
            glPushAttrib(GL_ENABLE_BIT)
            glLineWidth(2)
            glLineStipple(1, 0x00FF)
            glEnable(GL_LINE_STIPPLE)
            G0 = mesh.indices_open_lines
            G1 = mesh.indices_fall
            draw_geometries_with_excluded_area(window,G0,G1)
            glPopAttrib()

def display_arrows(window,mesh,view_opt):
    #glClear(GL_DEPTH_BUFFER_BIT)
    glUniform3f(5, 0.0, 0.0, 0.0)
    ############################## Direction arrows ################################
    for n in range(mesh.parent.noc):
        if (mesh.eval.interlocks[n]): glUniform3f(5,0.0,0.0,0.0) # black
        else: glUniform3f(5,1.0,0.0,0.0) # red
        glLineWidth(3)
        G1 = mesh.indices_fall
        G0 = mesh.indices_arrows[n]
        d0 = 2.55*mesh.parent.component_size
        d1 = 1.55*mesh.parent.component_size
        if len(mesh.parent.fixed_sides[n])==2: d0 = d1
        for ax,dir in mesh.parent.fixed_sides[n]:
            vec = d0*(2*dir-1)*mesh.parent.pos_vecs[ax]/np.linalg.norm(mesh.parent.pos_vecs[ax])
            #draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)
            draw_geometries(window,G0,translation_vec=vec)

def display_checker(window,mesh,view_opt):
    # 1. Draw hidden geometry
    glUniform3f(5, 1.0, 0.2, 0.0) # red orange
    glLineWidth(8)
    for n in range(mesh.parent.noc):
        if mesh.eval.checker[n]:
            draw_geometries(window,[mesh.indices_chess_lines[n]])
    glUniform3f(5, 0.0, 0.0, 0.0) # back to black

def display_breakable_faces(window,mesh,view_opt):
    # 1. Draw hidden geometry
    col = [1.0, 1.0, 0.8] # super light yellow
    glUniform3f(5, col[0], col[1], col[2])
    for n in range(mesh.parent.noc):
        draw_geometries_with_excluded_area(window,[mesh.indices_fbrk[n]],[mesh.indices_not_fbrk[n]])

    # Draw visible geometry
    col = [1.0, 1.0, 0.4] # light yellow
    glUniform3f(5, col[0], col[1], col[2])
    draw_geometries_with_excluded_area(window,mesh.indices_fbrk,mesh.indices_not_fbrk)

def display_unfabricatable(window,mesh,view_opt):
    col = [1.0, 0.8, 0.5] # orange
    glUniform3f(5, col[0], col[1], col[2])
    for n in range(mesh.parent.noc):
        if not mesh.eval.fab_direction_ok[n]:
            G0 = [mesh.indices_fall[n]]
            G1 = []
            for n2 in range(mesh.parent.noc):
                if n2!=n: G1.append(mesh.indices_fall[n2])
            draw_geometries_with_excluded_area(window,G0,G1)

def display_breakable_lines(window,mesh,view_opt):
    # 1. Draw hidden geometry
    glPushAttrib(GL_ENABLE_BIT)
    glUniform3f(5, 1.0, 0.4, 0.0) # red
    glLineWidth(5)
    glEnable(GL_LINE_STIPPLE)
    glLineStipple(4, 0xAAAA)
    for n in range(mesh.parent.noc):
        G0 = [mesh.indices_breakable_lines[n]]
        G1 = mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    glPopAttrib()

def display_milling_paths(window,mesh,view_opt):
    if len(mesh.indices_milling_path)==0: view_opt.show_milling_path = False
    if view_opt.show_milling_path:
        cols = [[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0],[0.0,1.0,1.0],[1.0,0,1.0]]
        glLineWidth(3)
        for n in range(mesh.parent.noc):
            if mesh.eval.fab_direction_ok[n]:
                glUniform3f(5,cols[n][0],cols[n][1],cols[n][2])
                draw_geometries(window,[mesh.indices_milling_path[n]])

def display_diff_voxel_from_suggestion(window,type,view_opt):
    index = type.mesh.select.suggstate
    glPushAttrib(GL_ENABLE_BIT)
    # draw faces of additional part
    glUniform3f(5, 1.0, 1.0, 1.0) # white
    for n in range(type.noc):
        G0 = [type.sugs[index].indices_fall[n]]
        G1 = type.mesh.indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    # draw faces of subtracted part
    glUniform3f(5, 1.0, 0.5, 0.5) # pink/red
    for n in range(type.noc):
        G0 = [type.mesh.indices_fall[n]]
        G1 = type.sugs[index].indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    # draw outlines
    glUniform3f(5, 0.0, 0.0, 0.0) # black
    glLineWidth(3)
    glEnable(GL_LINE_STIPPLE)
    glLineStipple(2, 0xAAAA)
    for n in range(type.noc):
        G0 = [type.sugs[index].indices_lns[n]]
        G1 = type.sugs[index].indices_fall
        draw_geometries_with_excluded_area(window,G0,G1)
    glPopAttrib()

def pick(window, mesh, view_opt, shader_col, show_col=False):

    if not view_opt.gallery:
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
        col_step = 1.0/(2+2*mesh.parent.dim*mesh.parent.dim)
        for n in range(mesh.parent.noc):
            col = np.zeros(3, dtype=np.float64)
            col[n%3] = 1.0
            if n>2: col[(n+1)%mesh.parent.dim] = 1.0
            glUniform3f(5, col[0], col[1], col[2])
            draw_geometries(window,[mesh.indices_fpick_not_top[n]],clear_depth_buffer=False)
            if n==0 or n==mesh.parent.noc-1: mos = 1
            else: mos = 2
            # mos is "number of sides"
            for m in range(mos):
                # Draw top faces
                for i in range(mesh.parent.dim*mesh.parent.dim):
                    col -= col_step
                    glUniform3f(5, col[0], col[1], col[2])
                    top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[n].start_index+mos*4*i+4*m, n)
                    draw_geometries(window,[top],clear_depth_buffer=False)

    ############### Read pixel color at mouse position ###############
    xpos,ypos = glfw.get_cursor_pos(window)
    mouse_pixel = glReadPixelsub(xpos, 1600-ypos, 1, 1, GL_RGB, outputType=None)[0][0]
    mouse_pixel = np.array(mouse_pixel)
    pick_n = pick_d = pick_x = pick_y = None
    mesh.select.suggstate = -1
    mesh.select.gallstate = -1
    if not view_opt.gallery:
        if xpos>1600: # suggestion side
            if ypos>0 and ypos<1600:
                index = int(ypos/400)
                if mesh.select.suggstate!=index:
                    mesh.select.suggstate=index
        elif not np.all(mouse_pixel==255): # not white / background
                non_zeros = np.where(mouse_pixel!=0)
                if len(non_zeros)>0:
                    if len(non_zeros[0]>0):
                        pick_n = non_zeros[0][0]
                        if len(non_zeros[0])>1:
                            pick_n = pick_n+mesh.parent.dim
                            if mouse_pixel[0]==mouse_pixel[2]: pick_n = 5
                        val = 255-mouse_pixel[non_zeros[0][0]]
                        i = int(0.5+val*(2+2*mesh.parent.dim*mesh.parent.dim)/255)-1
                        if i>=0:
                            pick_x = (int(i/mesh.parent.dim))%mesh.parent.dim
                            pick_y = i%mesh.parent.dim
                        pick_d = 0
                        if pick_n==mesh.parent.noc-1: pick_d = 1
                        elif int(i/mesh.parent.dim)>=mesh.parent.dim: pick_d = 1
                        #print(pick_n,pick_d,pick_x,pick_y)
    else: #gallerymode
        if xpos>0 and xpos<2000 and ypos>0 and ypos<1600:
            i = int(xpos/400)
            j = int(ypos/400)
            index = i*4+j
            mesh.select.gallstate=index
            mesh.select.state = -1
            mesh.select.suggstate = -1

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
    parser.add_argument('--user_study', action='store_true')
    parser.add_argument('--username', default="test", type=str, dest='username')
    parser.add_argument('--nofeedback', action='store_true')
    parser.add_argument('--ang', default=90.0, type=float)
    parser.add_argument('--dim', default=3, type=int)
    parser.add_argument('--sax', default=2, type=int)
    parser.add_argument('--w', default=32, type=int)
    parser.add_argument('--d', default=-1, type=int)
    args = parser.parse_args()

    if args.d<0: args.d=args.w
    print("Timber dimensions",args.w,"x",args.d,"mm")

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

    #fs=[ [[2,0]] , [[1,0]] , [[0,1]] ]
    fs=[ [[2,0]] , [[0,0],[0,1]] ]

    # Initiate
    type = Types(fs=fs,sax=args.sax,dim=args.dim,ang=args.ang, wd=[args.w,args.d])
    view_opt = ViewSettings()

    glfw.set_window_user_pointer(window, [type, view_opt, args])

    while glfw.get_key(window,glfw.KEY_ESCAPE)!=glfw.PRESS and not glfw.window_should_close(window) and not time_passed>max_time:
        glfw.poll_events()

        glViewport(0,0,1600,1600)
        glLoadIdentity()

        # Check time passed for user study
        if glo_start_time!=None and args.user_study:
            time_passed = glfw.get_time()-glo_start_time
        if time_passed>(last_printed*60+1) and int(time_passed)%60==0:
            print(int(int(time_passed)/60),"min")
            last_printed = int(int(time_passed)/60)
        if time_passed>max_time:
            print("Time has run up. Closing...")

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

        type = Types()
        view_opt = ViewSettings()

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

        # Display joint geometries (main window)
        init_shader(shader_tex, view_opt)
        display_end_grains(window,type.mesh)
        init_shader(shader_col, view_opt)
        display_joint_geometry(window,type.mesh,view_opt)


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
