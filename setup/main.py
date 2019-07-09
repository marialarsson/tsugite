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
    global TYPE, HIDDEN_A, HIDDEN_B, VOX_MAT, UPDATE_JOINT_INDICIES, SHOW_HIDDEN
    global OPEN
    UPDATE_JOINT_INDICIES = True
    if action==glfw.PRESS:
        # Joint geometry
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
    global OPEN, VOXEL_SIZE
    vertices = []
    # Add all vertices of the DIM*DIM*DIM voxel cube
    for i in range(DIM+1):
        for j in range(DIM+1):
            for k in range(DIM+1):
                x = i*VOXEL_SIZE-0.5*DIM*VOXEL_SIZE
                y = j*VOXEL_SIZE-0.5*DIM*VOXEL_SIZE
                z = k*VOXEL_SIZE
                vertices.extend([x,y,z,r,g,b])
    # Add component base vertices on bottom and top
    for k in range(2):
        corners = []
        for i in range(2):
            for j in range(2):
                i0 = i*DIM
                j0 = j*DIM
                k0 = k*DIM
                vertex = []
                for n in range(6):
                    vertex.append(vertices[6*(i0*(DIM+1)*(DIM+1)+j0*(DIM+1)+k0)+n])
                corners.append(vertex)
        vertices.extend(component_vertices(corners,[0,0,2*k-1],COMP_LENGTH))
        vertices.extend(component_vertices(corners,[0,0,2*k-1],COMP_LENGTH))
    # Add component base vertices on left and right
    for i in range(2):
        corners = []
        for j in range(2):
            for k in range(2):
                i0 = i*DIM
                j0 = j*DIM
                k0 = k*DIM
                vertex = []
                for n in range(6):
                    vertex.append(vertices[6*(i0*(DIM+1)*(DIM+1)+j0*(DIM+1)+k0)+n])
                corners.append(vertex)
        vertices.extend(component_vertices(corners,[2*i-1,0,0],COMP_LENGTH))
        vertices.extend(component_vertices(corners,[2*i-1,0,0],COMP_LENGTH))
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    #print('Number of vertices',int(len(vertices)/6))
    # Mover vertices for opening component
    if OPEN==True and comp=="B":
        for i in range(0,len(vertices),6):
            vertices[i]   = vertices[i]   + 5*SLIDE[0]*VOXEL_SIZE
            vertices[i+1] = vertices[i+1] + 5*SLIDE[1]*VOXEL_SIZE
            vertices[i+2] = vertices[i+2] + 5*SLIDE[2]*VOXEL_SIZE
    return vertices

def component_vertices(corners,vec,dist):
    comp_corners = []
    for corner in corners:
        corner[0] += dist*vec[0]
        corner[1] += dist*vec[1]
        corner[2] += dist*vec[2]
        comp_corners.extend(corner)
    return comp_corners

def voxel_corners(i,j,k):
    a = i*(DIM+1)*(DIM+1) + j*(DIM+1) + k
    b = i*(DIM+1)*(DIM+1) + (j+1)*(DIM+1) + k
    d = (i+1)*(DIM+1)*(DIM+1) + j*(DIM+1) + k
    c = (i+1)*(DIM+1)*(DIM+1) + (j+1)*(DIM+1) + k
    e = i*(DIM+1)*(DIM+1) + j*(DIM+1) + k+1
    f = i*(DIM+1)*(DIM+1) + (j+1)*(DIM+1) + k+1
    h = (i+1)*(DIM+1)*(DIM+1) + j*(DIM+1) + k+1
    g = (i+1)*(DIM+1)*(DIM+1) + (j+1)*(DIM+1) + k+1
    return a,b,c,d,e,f,g,h

def joint_face_indicies(comp,offset):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                val = VOX_MAT[i,j,k]
                if (comp=="A" and val==1) or (comp=="B" and val==0): continue
                # Define the 8 corners of the voxel
                a,b,c,d,e,f,g,h = voxel_corners(i,j,k)
                # Append the 6 faces of the voxel
                indices.extend([a,b,c,d])
                indices.extend([e,f,g,h])
                indices.extend([a,b,f,e])
                indices.extend([b,c,g,f])
                indices.extend([c,d,h,g])
                indices.extend([d,a,e,h])
        # Define indicies of component according to joint type
        start = (DIM+1)*(DIM+1)*(DIM+1)
        if comp=="A":
            a1,b1,c1,d1 = 0,(DIM+1)*(DIM+1)-DIM-1,start-(DIM+1)*(DIM+1),start-DIM-1
            if TYPE!="X":
                a0,b0,c0,d0 = start+4,start+5,start+6,start+7
            else:
                a0,b0,c0,d0 = start,start+1,start+2,start+3
            if TYPE=="X":
                a2,b2,c2,d2 = start+8,start+9,start+10,start+11
                a3,b3,c3,d3 = DIM,(DIM+1)*(DIM+1)-1,start-(DIM+1)*(DIM+1)+DIM,start-1
        elif comp=='B':
            if TYPE=="I":
                a1,b1,c1,d1 = DIM,(DIM+1)*(DIM+1)-1,start-(DIM+1)*(DIM+1)+DIM,start-1
            else:
                a1,b1,c1,d1 = 0,DIM,(DIM+1)*(DIM+1)-DIM-1,(DIM+1)*(DIM+1)-1
            if TYPE=="I":
                a0,b0,c0,d0 = start+12,start+13,start+14,start+15
            elif TYPE=="L":
                a0,b0,c0,d0 = start+20,start+21,start+22,start+23
            else:
                a0,b0,c0,d0 = start+16,start+17,start+18,start+19
            if TYPE=="T" or TYPE=="X":
                a2,b2,c2,d2 = start+24,start+25,start+26,start+27
                a3,b3,c3,d3 = start-(DIM+1)*(DIM+1),start-(DIM+1)*(DIM+1)+DIM,start-DIM-1,start-1
    # Add component side to indices
    indices.extend([a0,b0,d0,c0]) #bottom face
    indices.extend([a1,b1,d1,c1]) #top face
    indices.extend([a0,b0,b1,a1]) #side face 1
    indices.extend([b0,d0,d1,b1]) #side face 2
    indices.extend([d0,c0,c1,d1]) #side face 3
    indices.extend([c0,a0,a1,c1]) ##side face 4
    # When applicable, add second component side
    if (comp=="A" and TYPE=="X") or (comp=="B" and (TYPE=="X" or TYPE=="T")):
        indices.extend([a2,b2,d2,c2]) #bottom face
        indices.extend([a3,b3,d3,c3]) #top face
        indices.extend([a2,b2,b3,a3]) #side face 1
        indices.extend([b2,d2,d3,b3]) #side face 2
        indices.extend([d2,c2,c3,d3]) #side face 3
        indices.extend([c2,a2,a3,c3]) ##side face 4
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    #print('Number of face indices', len(indices))
    return indices

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
            if i<0 and [-1,0,0] in fixed_sides: val2 = val
            elif i>=DIM and [1,0,0] in fixed_sides: val2 = val
        elif (k<0 or k>=DIM) and i>=0 and i<DIM and j>=0 and j<DIM:
            if k<0 and [0,0,-1] in fixed_sides: val2 = val
            elif k>=DIM and [0,0,1] in fixed_sides: val2 = val
        elif (j<0 or j>=DIM) and i>=0 and i<DIM and k>=0 and k<DIM:
            if j<0 and [0,-1,0] in fixed_sides: val2 = val
            elif j>=DIM and [0,1,0] in fixed_sides: val2 = val
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
    fixed_sides_A.append([0,0,-1])
    if TYPE=="X": fixed_sides_A.append([0,0,1])
    # Component B
    fixed_sides_B = []
    if TYPE=="I": fixed_sides_B.append([0,0,1])
    else: fixed_sides_B.append([-1,0,0])
    if TYPE=="T" or TYPE=="X": fixed_sides_B.append([1,0,0])
    return [fixed_sides_A,fixed_sides_B]

def get_index(ind,a,b,c):
    d = DIM+1
    (i,j,k) = ind
    index = (i+a)*d*d + (j+b)*d + k+c
    return index

def joint_line_indicies(comp,offset,fixed_sides):
    # Make indices of lines (bases on heightfiled)
    # For draw elements method GL_LINES
    indices = []
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                ind = (i,j,k)
                val = VOX_MAT[i,j,k]
                d = DIM+1
                if (comp=="A" and val==1) or (comp=="B" and val==0):
                    # Base lines of fixed sides
                    if i==0 and [-1,0,0] in fixed_sides:
                        if k==0: indices.extend([get_index(ind,0,0,0), get_index(ind,0,1,0)])
                        if k==DIM-1: indices.extend([get_index(ind,0,0,1), get_index(ind,0,1,1)])
                        if j==0: indices.extend([get_index(ind,0,0,0), get_index(ind,0,0,1)])
                        if j==DIM-1: indices.extend([get_index(ind,0,1,0), get_index(ind,0,1,1)])
                    if i==DIM-1 and [1,0,0] in fixed_sides:
                        if k==0: indices.extend([get_index(ind,1,0,0), get_index(ind,1,1,0)])
                        if k==DIM-1: indices.extend([get_index(ind,1,0,1), get_index(ind,1,1,1)])
                        if j==0: indices.extend([get_index(ind,1,0,0), get_index(ind,1,0,1)])
                        if j==DIM-1: indices.extend([get_index(ind,1,1,0), get_index(ind,1,1,1)])
                    if k==0 and [0,0,-1] in fixed_sides:
                        if i==0: indices.extend([get_index(ind,0,0,0), get_index(ind,0,1,0)])
                        if i==DIM-1: indices.extend([get_index(ind,1,0,0), get_index(ind,1,1,0)])
                        if j==0: indices.extend([get_index(ind,0,0,0), get_index(ind,1,0,0)])
                        if j==DIM-1: indices.extend([get_index(ind,0,1,0), get_index(ind,1,1,0)])
                    if k==DIM-1 and [0,0,1] in fixed_sides:
                        if i==0: indices.extend([get_index(ind,0,0,1), get_index(ind,0,1,1)])
                        if i==DIM-1: indices.extend([get_index(ind,1,0,1), get_index(ind,1,1,1)])
                        if j==0: indices.extend([get_index(ind,0,0,1), get_index(ind,1,0,1)])
                        if j==DIM-1: indices.extend([get_index(ind,0,1,1), get_index(ind,1,1,1)])
                    continue
                # Side lines conditionally / i aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[0,2*x-1,0],[0,0,2*y-1],[0,2*x-1,2*y-1]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([i*d*d+(j+x)*d+k+y, (i+1)*d*d+(j+x)*d+k+y])
                # Side lines conditionally / j aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,0,2*y-1],[2*x-1,0,2*y-1]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([(i+x)*d*d+j*d+k+y, (i+x)*d*d+(j+1)*d+k+y])
                # Side lines conditionally / k aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,2*y-1,0],[2*x-1,2*y-1,0]],fixed_sides)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([(i+x)*d*d+(j+y)*d+k, (i+x)*d*d+(j+y)*d+k+1])

    #Outline of component base
    start = (DIM+1)*(DIM+1)*(DIM+1)
    if comp=="A":
        a1,b1,c1,d1 = 0,(DIM+1)*(DIM+1)-DIM-1,start-(DIM+1)*(DIM+1),start-DIM-1
        if TYPE!="X":
            a0,b0,c0,d0 = start+4,start+5,start+6,start+7
        else:
            a0,b0,c0,d0 = start,start+1,start+2,start+3
        if TYPE=="X":
            a2,b2,c2,d2 = start+8,start+9,start+10,start+11
            a3,b3,c3,d3 = DIM,(DIM+1)*(DIM+1)-1,start-(DIM+1)*(DIM+1)+DIM,start-1
    elif comp=='B':
        if TYPE=="I":
            a1,b1,c1,d1 = DIM,(DIM+1)*(DIM+1)-1,start-(DIM+1)*(DIM+1)+DIM,start-1
        else:
            a1,b1,c1,d1 = 0,DIM,(DIM+1)*(DIM+1)-DIM-1,(DIM+1)*(DIM+1)-1
        if TYPE=="I":
            a0,b0,c0,d0 = start+12,start+13,start+14,start+15
        elif TYPE=="L":
            a0,b0,c0,d0 = start+20,start+21,start+22,start+23
        else:
            a0,b0,c0,d0 = start+16,start+17,start+18,start+19
        if TYPE=="T" or TYPE=="X":
            a2,b2,c2,d2 = start+24,start+25,start+26,start+27
            a3,b3,c3,d3 = start-(DIM+1)*(DIM+1),start-(DIM+1)*(DIM+1)+DIM,start-DIM-1,start-1
    # Add component side to indices
    indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
    #indices.extend([a1,b1, b1,d1, d1,c1, c1,a1])
    indices.extend([a0,a1, b0,b1,c0,c1, d0,d1])
    # When applicable, add second component side
    if (comp=="A" and TYPE=="X") or (comp=="B" and (TYPE=="X" or TYPE=="T")):
        indices.extend([a2,b2, b2,d2, d2,c2, c2,a2])
        #indices.extend([a3,b3, b3,d3, d3,c3, c3,a3])
        indices.extend([a2,a3, b2,b3,c2,c3, d2,d3])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    #print('Number of line indices', len(indices))
    return indices

def open_line_indicies(offset1,offset2):
    global DIM, HF
    indices = []
    for i in range(2):
        for j in range(2):
            h = HF[j*(DIM-1)][i*(DIM-1)]
            i0 = i*DIM
            j0 = j*DIM
            ind = [i0,j0,h]
            indices.extend([get_index(ind,0,0,0)+offset2,get_index(ind,0,0,0)+offset1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    #print('Number of open line indices', len(indices))
    return indices

def get_random_height_field():
    a = random.randint(0,DIM)
    b = random.randint(0,DIM)
    c = random.randint(0,DIM)
    d = random.randint(0,DIM)
    e = random.randint(0,DIM)
    f = random.randint(0,DIM)
    g = random.randint(0,DIM)
    h = random.randint(0,DIM)
    i = random.randint(0,DIM)
    hf = [[a,b,c],[d,e,f],[g,h,i]]
    return hf

def voxel_matrix_from_height_field(hf_mat):
    vox_mat = np.zeros(shape=(DIM,DIM,DIM))
    for i in range(DIM):
        for j in range(DIM):
            for k in range(DIM):
                h = hf_mat[i][j]
                if k>=h: vox_mat[i,j,k]=1
    vox_mat = np.array(vox_mat)
    vox_mat = np.swapaxes(vox_mat,0,1)
    #vox_mat = np.swapaxes(vox_mat,1,2)
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
    v_faces_A = joint_vertices("A",0.8,0.8,0.8)
    v_lines_A = joint_vertices("A",0.0,0.0,0.0)

    # Vertices of component B
    v_faces_B = joint_vertices("B",0.8,0.8,0.8)
    v_lines_B = joint_vertices("B",0.0,0.0,0.0)

    # Join all vertices into one list
    vertices_all = np.concatenate([v_faces_A,v_lines_A,v_faces_B,v_lines_B])

    # Vertex buffer
    VBO = glGenBuffers(1) # vertex buffer object - the vertices
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*len(vertices_all), vertices_all, GL_DYNAMIC_DRAW) #uploadning data to the buffer. Specifying size / bites of data (x4)

    return len(v_faces_A), len(v_lines_B), len(v_faces_A), len(v_lines_B)

def create_buffer_indicies(vn_fA, vn_lA, vn_fB, vn_lB):
    fixed_sides_AB = get_fixed_sides()

    # Indices of component A
    i_faces_A = joint_face_indicies("A",0)
    i_lines_A = joint_line_indicies("A",int(vn_fA/6),fixed_sides_AB[0])

    # Indices of component B
    i_faces_B = joint_face_indicies("B",int((vn_fA+vn_lA)/6))
    i_lines_B = joint_line_indicies("B",int((vn_fA+vn_lA+vn_fB)/6),fixed_sides_AB[1])

    # Indicies of sliding lines
    i_open = open_line_indicies(int(vn_fA/6),int((vn_fA+vn_lA+vn_fB)/6))

    # Join all indices into one list
    i_all = np.concatenate([i_faces_A,i_lines_A,i_faces_B,i_lines_B,i_open])

    # Elements Buffer
    EBO = glGenBuffers(1) # element buffer object
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(i_all), i_all, GL_DYNAMIC_DRAW)

    return len(i_faces_A), len(i_lines_A), len(i_faces_B), len(i_lines_B), len(i_open)

def display(window, shader, in_fA, in_lA, in_fB, in_lB, in_open):

    position = glGetAttribLocation(shader, "position")
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    color = glGetAttribLocation(shader, "color")
    glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(color)

    glUseProgram(shader)

    glClearColor(1.0, 1.0, 1.0, 1.0)
    glEnable(GL_DEPTH_TEST) #enable the depth testing

    glfw.poll_events()

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

    # Rotation
    rot_x = pyrr.Matrix44.from_x_rotation(XROT)
    rot_y = pyrr.Matrix44.from_y_rotation(YROT)
    transformLoc = glGetUniformLocation(shader, "transform")
    glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

    # Draw the geometries
    glPolygonOffset(1.0,1.0)

    ### DRAW ALL LINES INCLUDING HIDDEN ###
    glPushAttrib(GL_ENABLE_BIT)
    glLineWidth(1)
    glLineStipple(3, 0xAAAA)
    glEnable(GL_LINE_STIPPLE)
    # Component A
    if HIDDEN_A==False and SHOW_HIDDEN==True:
        glDisable(GL_DEPTH_TEST)
        glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_ALWAYS,1,1)
        glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
        glDrawElements(GL_LINES, in_lA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*in_fA))
        glEnable(GL_DEPTH_TEST)
        glStencilFunc(GL_EQUAL,1,1)
        glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
        glDrawElements(GL_QUADS, in_fA, GL_UNSIGNED_INT,  ctypes.c_void_p(0))
        glDisable(GL_STENCIL_TEST)
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
        glDrawElements(GL_LINES, in_lA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*in_fA))
    # Component B
    if HIDDEN_B==False and SHOW_HIDDEN==True:
        glClear(GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_ALWAYS,1,1)
        glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
        glDrawElements(GL_LINES, in_lB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB)))
        glEnable(GL_DEPTH_TEST)
        glStencilFunc(GL_EQUAL,1,1)
        glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
        glDrawElements(GL_QUADS, in_fB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA)))
        glDisable(GL_STENCIL_TEST)
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
        glDrawElements(GL_LINES, in_lB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB)))
    glPopAttrib()

    glClear(GL_STENCIL_BUFFER_BIT)
    ### DRAW ONLY VISIBLE LINES ###
    glLineWidth(3)
    glDisable(GL_DEPTH_TEST)
    glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
    glEnable(GL_STENCIL_TEST)
    glStencilFunc(GL_ALWAYS,1,1)
    glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
    glDrawElements(GL_LINES, in_lA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*in_fA))
    glDrawElements(GL_LINES, in_lB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB)))
    glDrawElements(GL_LINES, in_open, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB+in_lB)))
    glEnable(GL_DEPTH_TEST)
    glStencilFunc(GL_EQUAL,1,1)
    glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
    if HIDDEN_A==False: glDrawElements(GL_QUADS, in_fA, GL_UNSIGNED_INT,  ctypes.c_void_p(0))
    if HIDDEN_B==False: glDrawElements(GL_QUADS, in_fB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA)))
    glDisable(GL_STENCIL_TEST)
    glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
    if HIDDEN_A==False: glDrawElements(GL_LINES, in_lA, GL_UNSIGNED_INT,  ctypes.c_void_p(4*in_fA))
    if HIDDEN_B==False: glDrawElements(GL_LINES, in_lB, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB)))

    if OPEN==True:
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(2)
        glLineStipple(1, 0x00FF)
        glEnable(GL_LINE_STIPPLE)
        glDrawElements(GL_LINES, in_open, GL_UNSIGNED_INT,  ctypes.c_void_p(4*(in_fA+in_lA+in_fB+in_lB)))
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
            in_fA, in_lA, in_fB, in_lB, in_open = create_buffer_indicies(vn_fA, vn_lA, vn_fB, vn_lB)
            UPDATE_JOINT_INDICIES=False
        # Display joint geometries
        display(window, shader, in_fA, in_lA, in_fB, in_lB, in_open)
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
    print(np.array(HF))
    VOX_MAT = voxel_matrix_from_height_field(HF)
    print(VOX_MAT)
    main()
