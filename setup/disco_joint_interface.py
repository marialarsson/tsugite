import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import pyrr # for doing vector and matrix multiplications etc.
import sys
import random

def draw_rect(x, y, width, height):
    glBegin(GL_QUADS)                                  # start drawing a rectangle
    glVertex2f(x, y)                                   # bottom left point
    glVertex2f(x + width, y)                           # bottom right point
    glVertex2f(x + width, y + height)                  # top right point
    glVertex2f(x, y + height)                          # top left point
    glEnd()

def joint_verticies(r,g,b):

    # Set dimentions of joint
    vox_size = 0.1
    component_length = vox_size*7

    # Make vertexes
    verticies = []
    # Add 4 compnent base verticies
    for i in range(2):
        i = 2*i-1
        for j in range(2):
            j = 2*j-1
            x = 1.5*vox_size*i
            y = -component_length
            z = 1.5*vox_size*j
            verticies.extend([x,y,z,r,g,b]) # indluding vertex color, remove later?
    # Add all vertices of the 3*3*3 cube (redundant)
    for i in range(4):
        for j in range(4):
            for k in range(4):
                x = (i-1.5)*vox_size
                y = vox_size*k
                z = (j-1.5)*vox_size
                verticies.extend([x,y,z,r,g,b])
    verticies = np.array(verticies, dtype = np.float32) #converts to correct format
    #print('Number of verticies',int(len(verticies)/6))
    return verticies

def joint_faces(offset):
    # Outline of base
    indices = [0,1,3,2,
               0,4,16,1,
               0,2,52,4,
               2,3,64,52,
               1,16,64,3]
    # Outline of joint
    for i in range(3):
        for j in range(3):
            # Current height value
            val = HF[i][j]
            # Neighboring height values
            val_rt = val_lf = val_up = val_dn = val_dn_lf = None
            if(i<2): val_up = HF[i+1][j]
            if(i>0): val_dn = HF[i-1][j]
            if(j<2): val_rt = HF[i][j+1]
            if(j>0): val_lf = HF[i][j-1]
            if(i>0 and j>0): val_dn_lf = HF[i-1][j-1]
            # Current index
            ind0 = 4+16*j+4*i
            ind = ind0+val
            # Current verticies
            v00 = ind
            v01 = ind+4
            v10 = ind+16
            v11 = ind+20
            # Make horisontal faces
            indices.extend([v00,v01,v11,v10])
            # Make sides on edges
            if val!=0:
                if (i==0): indices.extend([ind0,ind0+16,v10,v00])
                elif (i==2): indices.extend([v11,v01,ind0+4,ind0+20])
                if (j==0): indices.extend([v00,ind0,ind0+4,v01])
                elif (j==2): indices.extend([ind0+16,v10,v11,ind0+20])
            # Internal sides alinged with j direction
            if j!=0:
                diff = val-val_lf
                if abs(diff)>0:
                    indices.extend([v01, v00, ind0+val_lf, ind0+4+val_lf])
            if i!=0:
                diff = val-val_dn
                if abs(diff)>0:
                    indices.extend([v10, v00, ind0+val_dn, ind0+16+val_dn])

    indices = np.array(indices, dtype=np.uint32)

    indices = indices + offset

    #print('Number of face indices', len(indices))
    return indices

def joint_lines(offset):
    # Make indices of lines that connect the correct vertices (base on heightfiled)
    # For draw elements method GL_LINES (define start and end point of each line segment)
    # Outline of base
    indices = [0,1,1,3,3,2,2,0,0,4,1,16,2,52,3,64]
    # Outline of joint
    for i in range(3):
        for j in range(3):
            # Current height value
            val = HF[i][j]
            # Neighboring height values
            val_rt = val_lf = val_up = val_dn = val_dn_lf = None
            if(i<2): val_up = HF[i+1][j]
            if(i>0): val_dn = HF[i-1][j]
            if(j<2): val_rt = HF[i][j+1]
            if(j>0): val_lf = HF[i][j-1]
            if(i>0 and j>0): val_dn_lf = HF[i-1][j-1]
            # Current index
            ind0 = 4+16*j+4*i
            ind = ind0+val
            # Current verticies
            v00 = ind
            v01 = ind+4
            v10 = ind+16
            v11 = ind+20
            # Make horizontal lines conditionally
            if(j==0 or val!=val_lf):
                indices.extend([v00,v01])
            if(i==0 or val!=val_dn):
                indices.extend([v00,v10])
            if(j==2 or val!=val_rt):
                indices.extend([v10,v11])
            if(i==2 or val!=val_up):
                indices.extend([v01,v11])
            # Make vertical lines conditionally
            # a) Corners
            if val!=0:
                if (i==0 and j==0):
                    indices.extend([v00,v00-val])
                elif (i==0 and j==2):
                    indices.extend([v10,v10-val])
                elif (i==2 and j==0):
                    indices.extend([v01,v01-val])
                elif (i==2 and j==2):
                    indices.extend([v11,v11-val])
            # b) Sides
            if val_rt!=None:
                if (i==0 and val!=val_rt):
                    indices.extend([v10,v10-(val-val_rt)])
                elif (i==2 and val!=val_rt):
                    indices.extend([v11,v11-(val-val_rt)])
            if val_up!=None:
                if (j==0 and val!=val_up):
                    indices.extend([v01,v01-(val-val_up)])
                elif (j==2 and val!=val_up):
                    indices.extend([v11,v11-(val-val_up)])
            # Inner lines
            if (i!=0 and j!=0):
                heights = np.array([val,val_lf,val_dn,val_dn_lf])
                for k in range(3):
                    n = 3-k
                    count = np.count_nonzero(heights >= n)
                    if(count==1 or count==3):
                        indices.extend([ind0+n,ind0+n-1])
                    elif(count==2):
                        if((val==val_dn_lf and val<=n) or (val_lf==val_dn and val_lf<=n)):
                            indices.extend([ind0+n,ind0+n-1])

    indices = np.array(indices, dtype=np.uint32)

    indices = indices + offset

    #print('Number of line indices', len(indices))

    return indices

def get_random_height_field(n):
    a = random.randint(0,3)
    b = random.randint(0,3)
    c = random.randint(0,3)
    d = random.randint(0,3)
    e = random.randint(0,3)
    f = random.randint(0,3)
    g = random.randint(0,3)
    h = random.randint(0,3)
    i = random.randint(0,3)
    hf = [[a,b,c],[d,e,f],[g,h,i]]
    return hf

def keyCallback(window,key,scancode,action,mods):
    if action==glfw.PRESS:
        # Joint geometry
        if key==glfw.KEY_Y:
            HF[0][0] = (HF[0][0]+1)%4
        elif key==glfw.KEY_U:
            HF[0][1] = (HF[0][1]+1)%4
        elif key==glfw.KEY_I:
            HF[0][2] = (HF[0][2]+1)%4
        elif key==glfw.KEY_H:
            HF[1][0] = (HF[1][0]+1)%4
        elif key==glfw.KEY_J:
            HF[1][1] = (HF[1][1]+1)%4
        elif key==glfw.KEY_K:
            HF[1][2] = (HF[1][2]+1)%4
        elif key==glfw.KEY_B:
            HF[2][0] = (HF[2][0]+1)%4
        elif key==glfw.KEY_N:
            HF[2][1] = (HF[2][1]+1)%4
        elif key==glfw.KEY_M:
            HF[2][2] = (HF[2][2]+1)%4
        # Joint type
        elif key==glfw.KEY_1:
            type = "I"
            print("Joint type:",type)
        elif key==glfw.KEY_L:
            type = "L"
            print("Joint type:",type)
        elif key==glfw.KEY_T:
            type = "T"
            print("Joint type:",type)
        elif key==glfw.KEY_X:
            type = "X"
            print("Joint type:",type)

def mouseCallback(window,button,action,mods):
    # action 1 = press, action 0 = release
    if button==glfw.MOUSE_BUTTON_LEFT:
        global dragged
        if action==1:
            dragged = True
            global xstart, ystart
            xstart, ystart = glfw.get_cursor_pos(window)
        elif action==0: dragged = False

def updateRotation():
    global xrot, yrot, xstart, ystart
    xpos, ypos = glfw.get_cursor_pos(window)

def main():
    # initialize glfw
    if not glfw.init():
        return

    window = glfw.create_window(2000, 1400, "My OpenGL window", None, None)

    if not window:
        glfw.terminate()
        return

    glfw.make_context_current(window)

    # Enable and handle key events
    glfw.set_key_callback(window, keyCallback)
    glfw.set_input_mode(window, glfw.STICKY_KEYS,1)
    # Enable and hangle mouse events
    glfw.set_mouse_button_callback(window, mouseCallback);
    #glfw.set_input_mode(window, glfw.STICKY_MOUSE_BUTTONS, glfw.TRUE)

    glLineWidth(1)
    glEnable(GL_POLYGON_OFFSET_FILL)

    #glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    #glEnable( GL_BLEND );

    ## Create Shaders

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

    # Verticies
    verticies_faces = joint_verticies(1.0,1.0,0.0)
    verticies_lines = joint_verticies(0.0,0.0,1.0)
    verticies_all = np.concatenate([verticies_faces,verticies_lines])
    # Vertex buffer
    VBO = glGenBuffers(1) # vertex buffer object - the verticies
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, 4*len(verticies_all), verticies_all, GL_DYNAMIC_DRAW) #uploadning data to the buffer. Specifying size / bites of data (x4)


    while glfw.get_key(window,glfw.KEY_ESCAPE) != glfw.PRESS and not glfw.window_should_close(window):

        # Mouse rotate
        if dragged==True: updateRotation()

        # Incicies
        indices_faces = joint_faces(0)
        indices_lines = joint_lines(int(len(verticies_faces)/6))
        indices_all = np.concatenate([indices_faces,indices_lines])
        # Elements Buffer
        EBO = glGenBuffers(1) # element buffer object - the indices
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(indices_all), indices_all, GL_DYNAMIC_DRAW)

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

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Rotation
        rot_x = pyrr.Matrix44.from_x_rotation(xrot) #0.5
        rot_y = pyrr.Matrix44.from_y_rotation(yrot) #0.8
        transformLoc = glGetUniformLocation(shader, "transform")
        glUniformMatrix4fv(transformLoc, 1, GL_FALSE, rot_x * rot_y)

        # Draw the geometries
        glPolygonOffset(1.0,1.0)
        glDrawElements(GL_QUADS, int(len(indices_faces)), GL_UNSIGNED_INT,  ctypes.c_void_p(0))
        glDrawElements(GL_LINES, int(len(indices_lines)), GL_UNSIGNED_INT,  ctypes.c_void_p(4*len(indices_faces))) #

        #2D?
        #draw_rect(0.5,0.5,-0.5,-0.5)

        glfw.swap_buffers(window)

if __name__ == "__main__":
    print("Hit ESC key to quit.")
    print("Rotate view with mouse (not yet implemented)")
    print("Edit joint geometry with:\nY U I\nH J K\nB N M")
    print("Edit joint type with: 1 L T X (not yet implemented)")
    print("Press S to save joint geometry and O to open last saved geometry (not yet implemented)")
    #HF = [[3,0,0],[0,1,0],[0,0,2]]
    HF = get_random_height_field(3)
    type = "I"
    xrot = 0.4
    yrot = 0.2
    xstart = ystart = 0.0
    dragged = False
    main()
