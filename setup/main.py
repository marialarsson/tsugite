import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr
import sys
import random

def create_shaders():

    vertex_shader = """
    #version 330
    in vec3 position;
    in vec3 color;
    uniform mat4 transform;
    out vec3 newColor;
    void main()
    {
        gl_Position = transform* vec4(position, 1.0f);
        newColor = color;
    }
    """

    fragment_shader = """
    #version 330
    in vec3 newColor;
    out vec4 outColor;
    void main()
    {
        outColor = vec4(newColor, 1.0f);
    }
    """
    # Compiling the shaders
    shader = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                              OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))
    return shader

def keyCallback(window,key,scancode,action,mods):
    global TYPE, HIDDEN_A, HIDDEN_B, VOX_MAT, UPDATE_JOINT_INDICIES, SHOW_HIDDEN, OPEN
    UPDATE_JOINT_INDICIES = True
    if action==glfw.PRESS:
        # Joint geometry edit by height field
        if key==glfw.KEY_Y:
            HF[0][0] = (HF[0][0]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_U:
            HF[0][1] = (HF[0][1]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_I:
            HF[0][2] = (HF[0][2]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_H:
            HF[1][0] = (HF[1][0]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_J:
            HF[1][1] = (HF[1][1]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_K:
            HF[1][2] = (HF[1][2]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_B:
            HF[2][0] = (HF[2][0]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_N:
            HF[2][1] = (HF[2][1]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        elif key==glfw.KEY_M:
            HF[2][2] = (HF[2][2]+1)%4
            VOX_MAT = voxel_matrix_from_height_field(HF)
        # Joint TYPE
        elif key==glfw.KEY_1:
            TYPE = "I"
            print("Joint TYPE:",TYPE)
        elif key==glfw.KEY_L:
            TYPE = "L"
            print("Joint TYPE:",TYPE)
        elif key==glfw.KEY_T:
            TYPE = "T"
            print("Joint TYPE:",TYPE)
        elif key==glfw.KEY_X:
            TYPE = "X"
            print("Joint TYPE:",TYPE)
        # Preview options
        elif key==glfw.KEY_A:
            HIDDEN_A = not HIDDEN_A
        elif key==glfw.KEY_D:
            HIDDEN_B = not HIDDEN_B
        elif key==glfw.KEY_E:
            SHOW_HIDDEN = not SHOW_HIDDEN
        elif key==glfw.KEY_C:
            OPEN = not OPEN
            vn_fA, vn_lA, vn_fB, vn_lB = create_buffer_vertices()

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

def joint_vertices(comp,r,g,b):
    global OPEN, DIM, VOXEL_SIZE, COMP_LENGTH
    vertices = []
    # Add all vertices of the DIM*DIM*DIM voxel cube
    for i in range(DIM+1):
        for j in range(DIM+1):
            for k in range(DIM+1):
                x = i*VOXEL_SIZE-0.5*DIM*VOXEL_SIZE
                y = j*VOXEL_SIZE-0.5*DIM*VOXEL_SIZE
                z = k*VOXEL_SIZE-0.5*DIM*VOXEL_SIZE
                vertices.extend([x,y,z,r,g,b])
    # Add component base vertices
    component_vertices = []
    for ax in range(3):
        for n in range(2):
            corners = get_corner_indices(ax,n)
            for step in range(2,4):
                for corner in corners:
                    new_vertex = []
                    for i in range(6):
                        new_vertex_param = vertices[6*corner+i]
                        if i==ax: new_vertex_param = new_vertex_param + (2*n-1)*step*COMP_LENGTH
                        new_vertex.append(new_vertex_param)
                    vertices.extend(new_vertex)
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    # Open joint by moving vertices in sliding direction
    if OPEN==True:
        f = 2.5
        if comp=="A": f=-f
        for i in range(0,len(vertices),6):
            for j in range(3): vertices[i+j] = vertices[i+j] + f*SLIDE[j]*VOXEL_SIZE
    return vertices

def get_same_neighbors(ind,fixed_sides):
    global VOX_MAT, DIM
    neighbors = []
    val = VOX_MAT[tuple(ind)]
    for ax in range(3):
        for n in range(2):
            add = [0,0]
            add.insert(ax,2*n-1)
            add = np.array(add)
            ind2 = ind+add
            if (ind2[ax]<0 or ind2[ax]>=DIM) and [ax,n] in fixed_sides: val2 = val
            elif np.all(ind2>=0) and np.all(ind2<DIM):
                val2 = VOX_MAT[tuple(ind2)]
            else: val2=None
            if val==val2:
                neighbors.append([ax,n])
    return neighbors

def joint_face_indicies(comp,offset,fixed_sides):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    indices_ends = []
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                ind = np.array([i,j,k])
                val = VOX_MAT[tuple(ind)]
                # Open faces on fixed side
                if (comp=="A" and val==1) or (comp=="B" and val==0):
                    for ax,n in fixed_sides:
                        if ind[ax]!=n*(DIM-1): continue
                        for a in range(2):
                            for b in range(2):
                                add = [a,abs(a-b)]
                                add.insert(ax,n)
                                indices_ends.append(get_index(ind,add))
                    continue
                # Exterior faces of voxels
                same_neighbors = get_same_neighbors(ind,fixed_sides)
                for ax in range(3):
                    for n in range(2):
                        if [ax,n] in same_neighbors: continue # skip interior faces
                        face_inds = []
                        for a in range(2):
                            for b in range(2):
                                add = [a,abs(a-b)]
                                add.insert(ax,n)
                                face_inds.append(get_index(ind,add))
                        if ax==fixed_sides[0][0]: indices_ends.extend(face_inds)
                        else: indices.extend(face_inds)
        # 2. Faces of component base
        d = DIM+1
        start = d*d*d
        for ax,n in fixed_sides:
            a1,b1,c1,d1 = get_corner_indices(ax,n)
            step = 1
            if TYPE=="X" or (TYPE=="T" and comp=="B"): step = 0
            off = 16*ax+8*n+4*step
            a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
            # Add component side to indices
            indices_ends.extend([a0,b0,d0,c0]) #bottom face
            indices.extend([a0,b0,b1,a1]) #side face 1
            indices.extend([b0,d0,d1,b1]) #side face 2
            indices.extend([d0,c0,c1,d1]) #side face 3
            indices.extend([c0,a0,a1,c1]) ##side face 4
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    indices_ends = np.array(indices_ends, dtype=np.uint32)
    indices_ends = indices_ends + offset
    return indices, indices_ends

def get_count(ind,neighbors,fixed_sides):
    global VOX_MAT
    cnt = 0
    val = int(VOX_MAT[ind])
    for item in neighbors:
        i = ind[0]+item[0]
        j = ind[1]+item[1]
        k = ind[2]+item[2]
        ###
        val2 = None
        # Check fixed sides
        if (i<0 or i>=DIM) and j>=0 and j<DIM and k>=0 and k<DIM:
            if i<0 and [0,0] in fixed_sides: val2 = val
            elif i>=DIM and [0,1] in fixed_sides: val2 = val
        elif (j<0 or j>=DIM) and i>=0 and i<DIM and k>=0 and k<DIM:
            if j<0 and [1,0] in fixed_sides: val2 = val
            elif j>=DIM and [1,1] in fixed_sides: val2 = val
        elif (k<0 or k>=DIM) and i>=0 and i<DIM and j>=0 and j<DIM:
            if k<0 and [2,0] in fixed_sides: val2 = val
            elif k>=DIM and [2,1] in fixed_sides: val2 = val
        # Check neighbours
        elif np.all(np.array([i,j,k])>=0) and np.all(np.array([i,j,k])<DIM):
            val2 = int(VOX_MAT[i,j,k])
        if val==val2: cnt = cnt+1
        dia = val2
    #print("c",cnt,"d",dia,"v",val,"v2",val2,"ind",ind,"ind2",i,j,k)

    return cnt,dia

def get_fixed_sides():
    # Component A
    fixed_sides_A = []
    fixed_sides_A.append([2,0]) # [axis (x,y, or z), side (0 or 1)]
    if TYPE=="X": fixed_sides_A.append([2,1])
    # Component B
    fixed_sides_B = []
    if TYPE=="I": fixed_sides_B.append([2,1])
    else: fixed_sides_B.append([0,0])
    if TYPE=="T" or TYPE=="X": fixed_sides_B.append([0,1])
    #
    return [fixed_sides_A,fixed_sides_B]

def get_index(ind,add):
    d = DIM+1
    (i,j,k) = ind
    index = (i+add[0])*d*d + (j+add[1])*d + k+add[2]
    return index

def joint_line_indicies(comp,offset,fixed_sides):
    # Make indices for draw elements method GL_LINES
    d = DIM+1
    indices = []
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                ind = np.array([i,j,k])
                val = VOX_MAT[tuple(ind)]
                if (comp=="A" and val==1) or (comp=="B" and val==0):
                    # Base lines of fixed sides
                    for ax,n in fixed_sides:
                        if ind[ax]!=n*(DIM-1): continue
                        other_axes = np.array([0,1,2])
                        other_axes = np.delete(other_axes,np.where(other_axes==ax))
                        for ax2 in other_axes:
                            for n2 in range(2):
                                if ind[ax2]!=n2*(DIM-1): continue
                                temp = np.copy(other_axes)
                                ax3 = np.delete(temp,np.where(temp==ax2))[0]
                                add = np.array([0,0,0])
                                add[ax] = n
                                add[ax2] = n2
                                add2 = np.copy(add)
                                add2[ax3] =+1
                                a = get_index(ind,add)
                                b = get_index(ind,add2)
                                indices.extend([a,b])
                    continue
                # Side lines conditionally / i aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[0,2*x-1,0],[0,0,2*y-1],[0,2*x-1,2*y-1]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[0,x,y]),get_index(ind,[1,x,y])])
                # Side lines conditionally / j aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,0,2*y-1],[2*x-1,0,2*y-1]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[x,0,y]),get_index(ind,[x,1,y])])
                # Side lines conditionally / k aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,2*y-1,0],[2*x-1,2*y-1,0]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[x,y,0]),get_index(ind,[x,y,1])])
    #Outline of component base
    start = d*d*d
    for ax,n in fixed_sides:
        a1,b1,c1,d1 = get_corner_indices(ax,n)
        step = 1
        if TYPE=="X" or (TYPE=="T" and comp=="B"): step = 0
        off = 16*ax+8*n+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
        indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    return indices

def get_corner_indices(ax,n):
    global DIM
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    ind[ax] = n*DIM
    corner_indices = []
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*DIM
            add[other_axes[1]] = y*DIM
            corner_indices.append(get_index(ind,add))
    return corner_indices

def open_line_indicies(offset1,offset2):
    global DIM, HF
    indices = []
    for i in range(2):
        for j in range(2):
            h = HF[i*(DIM-1)][j*(DIM-1)]
            i0 = i*DIM
            j0 = j*DIM
            ind = [i0,j0,h]
            indices.extend([get_index(ind,[0,0,0])+offset2,get_index(ind,[0,0,0])+offset1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    #print('Number of open line indices', len(indices))
    return indices

def get_random_height_field():
    hf = np.zeros((DIM,DIM))
    for i in range(DIM):
        for j in range(DIM): hf[i,j]=random.randint(0,DIM)
    return hf

def voxel_matrix_from_height_field(hf_mat):
    vox_mat = np.zeros(shape=(DIM,DIM,DIM))
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                h = hf_mat[i][j]
                if k>=h: vox_mat[i,j,k]=1
    vox_mat = np.array(vox_mat)
    return vox_mat

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

def create_buffer_vertices():
    global DIM, VOXEL_SIZE, COMP_LENGTH
    # Vertices of component A
    v_faces_A = joint_vertices("A",0.95,0.95,0.95)
    v_lines_A = joint_vertices("A",0.0,0.0,0.0)

    # Vertices of component B
    v_faces_B = joint_vertices("B",0.95,0.95,0.95)
    v_lines_B = joint_vertices("B",0.0,0.0,0.0)

    # Join all vertices into one list
    vertices_all = np.concatenate([v_faces_A,v_lines_A,v_faces_B,v_lines_B])

    # Vertex buffer
    VBO = glGenBuffers(1) # vertex buffer object - the vertices
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*len(vertices_all), vertices_all, GL_DYNAMIC_DRAW) #uploadning data to the buffer. Specifying size / bites of data (x4)

    return len(v_faces_A), len(v_lines_B), len(v_faces_A), len(v_lines_B)

def create_buffer_indicies(vfA, vlA, vfB, vlB):
    fixed_sides_AB = get_fixed_sides()

    vA = vfA+vlA

    # Indices of component A
    faces_A,faces_end_A = joint_face_indicies("A",0,fixed_sides_AB[0])
    lines_A = joint_line_indicies("A",int(vfA/6),fixed_sides_AB[0])

    # Indices of component B
    faces_B,faces_end_B = joint_face_indicies("B",int(vA/6),fixed_sides_AB[1])
    lines_B = joint_line_indicies("B",int((vA+vfB)/6),fixed_sides_AB[1])

    # Indicies of sliding lines
    lines_open = open_line_indicies(int(vfA/6),int((vA+vfB)/6))

    # Join all indices into one list
    all_indices = np.concatenate([faces_A,faces_end_A,lines_A,faces_B,faces_end_B,lines_B,lines_open])

    # Elements Buffer
    EBO = glGenBuffers(1) # element buffer object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(all_indices), all_indices, GL_DYNAMIC_DRAW)

    return len(faces_A), len(faces_end_A), len(lines_A), len(faces_B), len(faces_end_B), len(lines_B), len(lines_open)

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

def display(window, shader, ifA, ifeA, ilA, ifB, ifeB, ilB, iopen):
    # Set up
    iA = ifA + ifeA + ilA
    iB = ifB + ifeB + ilB
    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)
    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)
    glUseProgram(shader)
    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST)
    glfw.poll_events()
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
    rot_x = pyrr.Matrix44.from_x_rotation(XROT)
    rot_y = pyrr.Matrix44.from_y_rotation(YROT)
    transformLoc = glGetUniformLocation(shader, "transform")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

    # Draw geometries
    glPolygonOffset(1.0,1.0)

    ### Draw end grain faces (hidden in depth by full geometry) ###
    a0 = [GL_QUADS,ifeA,ifA]
    b0 = [GL_QUADS,ifeB,iA+ifB]
    a1 = [GL_QUADS,ifA,0]
    b1 = [GL_QUADS,ifB,iA]
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

    ### Draw lines HIDDEN by other component ###
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA) # dashed line
    glEnable(GL_LINE_STIPPLE)
    # Component A
    if HIDDEN_A==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES, ilA, ifA+ifeA]]
        G1 = [[GL_QUADS, ifA+ifeA, 0]]
        draw_geometry_with_excluded_area(G0,G1)
    # Component B
    if HIDDEN_B==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        G0 = [[GL_LINES, ilB, iA+ifB+ifeB]]
        G1 = [[GL_QUADS, ifB+ifeB, iA]]
        draw_geometry_with_excluded_area(G0,G1)
    glPopAttrib()

    ### Draw visible lines ###
    glLineWidth(3)
    glClear(GL_DEPTH_BUFFER_BIT)
    a0 = [GL_LINES, ilA, ifA+ifeA]
    b0 = [GL_LINES, ilB, iA+ifB+ifeB]
    a1 = [GL_QUADS, ifA+ifeA, 0]
    b1 = [GL_QUADS, ifB+ifeB, iA]
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
    if OPEN==True and HIDDEN_A==False and HIDDEN_B==False:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        G0 = [[GL_LINES, iopen, iA+iB]]
        a1 = [GL_QUADS, ifA+ifeA, 0]
        b1 = [GL_QUADS, ifB+ifeB, iA]
        G1 = [a1,b1]
        draw_geometry_with_excluded_area(G0,G1)
        glPopAttrib()

    glfw.swap_buffers(window)

def main():
    global UPDATE_JOINT_INDICIES
    # Initialize window
    window = initialize()
    # Create shaders
    shader = create_shaders()
    # Create and buffer joint vertices
    vn_fA, vn_lA, vn_fB, vn_lB = create_buffer_vertices()

    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):
        # Update Rotation
        updateRotation(window, DRAGGED, DOUBLE_CLICKED)
        # Create and buffer joint indicies
        if UPDATE_JOINT_INDICIES:
            in_fA, in_feA, in_lA, in_fB, in_feB, in_lB, in_open = create_buffer_indicies(vn_fA, vn_lA, vn_fB, vn_lB)
            UPDATE_JOINT_INDICIES=False
        # Display joint geometries
        display(window, shader, in_fA, in_feA, in_lA, in_fB, in_feB, in_lB, in_open)
    glfw.terminate()

if __name__ == "__main__":
    print("Hit ESC key to quit.")
    print("Rotate view with mouse / Autorotate with double click")
    print("Edit joint geometry with:\nY U I\nH J K\nB N M")
    print("Edit joint type with: 1 L T X")
    print("Press S to save joint geometry and O to open last saved geometry (not yet implemented)")

    # Declare global variables
    global HF, TYPE, XROT, YROT, XROT0, YROT0, XSTART, YSTART, CLICK_TIME
    global DRAGGED, DOUBLE_CLICKED, DIM, VOXEL_SIZE, COMP_LENGTH, VOX_MAT
    global HIDDEN_B, UPDATE_JOINT_INDICIES, SHOW_HIDDEN, SLIDE, OPEN
    TYPE = "I"
    # Variables for mouse callback and rotation
    XROT, YROT = 0.8, 0.4
    XROT0, YROT0 = XROT, YROT
    XSTART = YSTART = 0.0
    CLICK_TIME = 0
    DRAGGED = False
    DOUBLE_CLICKED = False
    HIDDEN_A = False
    HIDDEN_B = False
    UPDATE_JOINT_INDICIES = True
    SHOW_HIDDEN = True
    SLIDE = [0,0,1]
    OPEN = False
    # Set geometric variables for joint geometry
    DIM = 3
    VOXEL_SIZE = 0.075
    COMP_LENGTH = VOXEL_SIZE*2
    #
    HF = get_random_height_field()
    VOX_MAT = voxel_matrix_from_height_field(HF)
    main()
