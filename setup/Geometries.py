from OpenGL.GL import *
import numpy as np
import random

def get_random_height_field(dim):
    hf = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(dim): hf[i,j]=random.randint(0,dim)
    return hf

def get_index(ind,add,dim):
    d = dim+1
    (i,j,k) = ind
    index = (i+add[0])*d*d + (j+add[1])*d + k+add[2]
    return index

def get_corner_indices(ax,n,dim):
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    ind[ax] = n*dim
    corner_indices = []
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*dim
            add[other_axes[1]] = y*dim
            corner_indices.append(get_index(ind,add,dim))
    return corner_indices

def joint_vertices(self,comp,r,g,b):
    vertices = []
    # Add all vertices of the dim*dim*dim voxel cube
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                x = (i-0.5*self.dim)*self.voxel_size
                y = (j-0.5*self.dim)*self.voxel_size
                z = (k-0.5*self.dim)*self.voxel_size
                vertices.extend([x,y,z,r,g,b])
    # Add component base vertices
    component_vertices = []
    for ax in range(3):
        for n in range(2):
            corners = get_corner_indices(ax,n,self.dim)
            for step in range(2,4):
                for corner in corners:
                    new_vertex = []
                    for i in range(6):
                        new_vertex_param = vertices[6*corner+i]
                        if i==ax: new_vertex_param = new_vertex_param + (2*n-1)*step*self.component_length
                        new_vertex.append(new_vertex_param)
                    vertices.extend(new_vertex)
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    # Open joint by moving vertices in sliding direction
    if self.open_joint==True:
        f = 2.5
        if comp=="B": f=-f
        ax,n = self.sliding_direction
        dir = (2*n-1)
        for i in range(0,len(vertices),6):
            vertices[i+ax] += f*dir*self.voxel_size
    return vertices

def get_fixed_sides(joint_type):
    # Component A
    fixed_sides_A = []
    fixed_sides_A.append([2,0]) # [axis (x,y, or z), side (0 or 1)]
    if joint_type=="X": fixed_sides_A.append([2,1])
    # Component B
    fixed_sides_B = []
    if joint_type=="I": fixed_sides_B.append([2,1])
    else: fixed_sides_B.append([0,0])
    if joint_type=="T" or joint_type=="X": fixed_sides_B.append([0,1])
    return [fixed_sides_A,fixed_sides_B]

def get_same_neighbors(ind,fixed_sides,voxel_matrix,dim):
    neighbors = []
    val = voxel_matrix[tuple(ind)]
    for ax in range(3):
        for n in range(2):
            add = [0,0]
            add.insert(ax,2*n-1)
            add = np.array(add)
            ind2 = ind+add
            if (ind2[ax]<0 or ind2[ax]>=dim) and [ax,n] in fixed_sides:
                val2 = val
            elif np.all(ind2>=0) and np.all(ind2<dim):
                val2 = voxel_matrix[tuple(ind2)]
            else: val2=None
            if val==val2:
                neighbors.append([ax,n])
    return neighbors

def joint_face_indicies(self,comp,offset,fixed_sides):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    indices_ends = []
    for i in range(self.dim):
        for j in range(self.dim):
            for k in range(self.dim):
                ind = np.array([i,j,k])
                val = self.voxel_matrix[tuple(ind)]
                # Open faces on fixed side
                if (comp=="A" and val==1) or (comp=="B" and val==0):
                    for ax,n in fixed_sides:
                        if ind[ax]!=n*(self.dim-1): continue
                        for a in range(2):
                            for b in range(2):
                                add = [a,abs(a-b)]
                                add.insert(ax,n)
                                indices_ends.append(get_index(ind,add,self.dim))
                    continue
                # Exterior faces of voxels
                same_neighbors = get_same_neighbors(ind,fixed_sides,self.voxel_matrix,self.dim) ################
                for ax in range(3):
                    for n in range(2):
                        if [ax,n] in same_neighbors: continue # skip interior faces
                        face_inds = []
                        for a in range(2):
                            for b in range(2):
                                add = [a,abs(a-b)]
                                add.insert(ax,n)
                                face_inds.append(get_index(ind,add,self.dim))
                        if ax==fixed_sides[0][0]: indices_ends.extend(face_inds)
                        else: indices.extend(face_inds)
        # 2. Faces of component base
        d = self.dim+1
        start = d*d*d
        for ax,n in fixed_sides:
            a1,b1,c1,d1 = get_corner_indices(ax,n,self.dim)
            step = 1
            if self.joint_type=="X" or (self.joint_type=="T" and comp=="B"): step = 0
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

def get_count(ind,neighbors,fixed_sides,voxel_matrix,dim):
    cnt = 0
    val = int(voxel_matrix[ind])
    for item in neighbors:
        i = ind[0]+item[0]
        j = ind[1]+item[1]
        k = ind[2]+item[2]
        ###
        val2 = None
        # Check fixed sides
        if (i<0 or i>=dim) and j>=0 and j<dim and k>=0 and k<dim:
            if i<0 and [0,0] in fixed_sides: val2 = val
            elif i>=dim and [0,1] in fixed_sides: val2 = val
        elif (j<0 or j>=dim) and i>=0 and i<dim and k>=0 and k<dim:
            if j<0 and [1,0] in fixed_sides: val2 = val
            elif j>=dim and [1,1] in fixed_sides: val2 = val
        elif (k<0 or k>=dim) and i>=0 and i<dim and j>=0 and j<dim:
            if k<0 and [2,0] in fixed_sides: val2 = val
            elif k>=dim and [2,1] in fixed_sides: val2 = val
        # Check neighbours
        elif np.all(np.array([i,j,k])>=0) and np.all(np.array([i,j,k])<dim):
            val2 = int(voxel_matrix[i,j,k])
        if val==val2: cnt = cnt+1
        dia = val2
    return cnt,dia

def joint_line_indicies(self,comp,offset,fixed_sides):
    # Make indices for draw elements method GL_LINES
    d = self.dim+1
    indices = []
    for i in range(self.dim):
        for j in range(self.dim):
            for k in range(self.dim):
                ind = np.array([i,j,k])
                val = self.voxel_matrix[tuple(ind)]
                if (comp=="A" and val==1) or (comp=="B" and val==0):
                    # Base lines of fixed sides
                    for ax,n in fixed_sides:
                        if ind[ax]!=n*(self.dim-1): continue
                        other_axes = np.array([0,1,2])
                        other_axes = np.delete(other_axes,np.where(other_axes==ax))
                        for ax2 in other_axes:
                            for n2 in range(2):
                                if ind[ax2]!=n2*(self.dim-1): continue
                                temp = np.copy(other_axes)
                                ax3 = np.delete(temp,np.where(temp==ax2))[0]
                                add = np.array([0,0,0])
                                add[ax] = n
                                add[ax2] = n2
                                add2 = np.copy(add)
                                add2[ax3] =+1
                                a = get_index(ind,add,self.dim)
                                b = get_index(ind,add2,self.dim)
                                indices.extend([a,b])
                    continue
                # Side lines conditionally / i aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[0,2*x-1,0],[0,0,2*y-1],[0,2*x-1,2*y-1]],fixed_sides,self.voxel_matrix,self.dim)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[0,x,y],self.dim),get_index(ind,[1,x,y],self.dim)])
                # Side lines conditionally / j aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,0,2*y-1],[2*x-1,0,2*y-1]],fixed_sides,self.voxel_matrix,self.dim)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[x,0,y],self.dim),get_index(ind,[x,1,y],self.dim)])
                # Side lines conditionally / k aligned
                for x in range(2):
                    for y in range(2):
                        cnt,dia = get_count((i,j,k),[[2*x-1,0,0],[0,2*y-1,0],[2*x-1,2*y-1,0]],fixed_sides,self.voxel_matrix,self.dim)
                        if cnt==0 or cnt==2 or (cnt==1 and dia==val):
                            indices.extend([get_index(ind,[x,y,0],self.dim),get_index(ind,[x,y,1],self.dim)])
    #Outline of component base
    start = d*d*d
    for ax,n in fixed_sides:
        a1,b1,c1,d1 = get_corner_indices(ax,n,self.dim)
        step = 1
        if self.joint_type=="X" or ( self.joint_type=="T" and comp=="B"): step = 0
        off = 16*ax+8*n+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
        indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    return indices

def open_line_indicies(self,offset0,offset1):
    ax,n = self.sliding_direction
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    ind[ax] = n*self.dim
    d = self.dim-1
    heights = [[self.height_field[0][0], self.height_field[0][d]],
              [self.height_field[d][0],  self.height_field[d][d]]]
    indices = []
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*self.dim
            add[other_axes[1]] = y*self.dim
            add[ax] = heights[x][y]
            indices.append(get_index(ind,add,self.dim)+offset0)
            indices.append(get_index(ind,add,self.dim)+offset1)
    # Format
    indices = np.array(indices, dtype=np.uint32)
    return indices

class Geometries:
    def __init__(self):
        self.open_joint = False
        self.joint_type = "I"
        self.sliding_direction = [2,0]
        self.dim = 3
        self.voxel_size = 0.075
        self.component_length = 2*self.voxel_size
        self.height_field = get_random_height_field(self.dim)
        self.voxel_matrix = None
        self.ifA = None
        self.ifeA = None
        self.ilA = None
        self.ifB = None
        self.ifeB = None
        self.ilB = None
        self.iopen = None

        self.voxel_matrix_from_height_field()

        # Create verticies
        self.vn = self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def save(self):
        np.savetxt("saved_joint_geometry.txt",self.height_field)

    def load(self):
        self.height_field = np.loadtxt("saved_joint_geometry.txt")
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def voxel_matrix_from_height_field(self):
        vox_mat = np.zeros(shape=(self.dim,self.dim,self.dim))
        ax = self.sliding_direction[0]
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    h = self.height_field[i][j]
                    if k>=h: vox_mat[i,j,k]=1
        vox_mat = np.array(vox_mat)
        vox_mat = np.swapaxes(vox_mat,2,ax)
        self.voxel_matrix = vox_mat

    def update_sliding_direction(self,sliding_direction_):
        self.sliding_direction = sliding_direction_
        self.voxel_matrix_from_height_field()
        self.vn = self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def update_height_field(self,i,j):
        self.height_field[i][j] = (self.height_field[i][j]+1)%4
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def clear_height_field(self):
        self.height_field = np.zeros((self.dim,self.dim))
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def create_and_buffer_vertices(self):
        #print("Updating vertices...")

        # Vertices of component A
        v_faces_A = joint_vertices(self,"A",0.95,0.95,0.95)
        v_lines_A = joint_vertices(self,"A",0.0,0.0,0.0)

        # Vertices of component B
        v_faces_B = joint_vertices(self,"B",0.95,0.95,0.95)
        v_lines_B = joint_vertices(self,"B",0.0,0.0,0.0)

        # Join all vertices into one list
        vertices_all = np.concatenate([v_faces_A,v_lines_A,v_faces_B,v_lines_B])
        # Vertex buffer
        VBO = glGenBuffers(1) # vertex buffer object - the vertices
        glBindBuffer(GL_ARRAY_BUFFER, VBO)
        glBufferData(GL_ARRAY_BUFFER, 4*len(vertices_all), vertices_all, GL_DYNAMIC_DRAW) #uploadning data to the buffer. Specifying size / bites of data (x4)

        # vertex attribute pointers
        position = 0
        glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
        glEnableVertexAttribArray(position)
        color = 1
        glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
        glEnableVertexAttribArray(color)

        return int(len(v_faces_A)/6)

    def create_and_buffer_indicies(self):
        #print("Updating indices...")

        fixed_sides_AB = get_fixed_sides(self.joint_type)

        # Indices of component A
        faces_A,faces_end_A = joint_face_indicies(self,"A",0,fixed_sides_AB[0])
        lines_A = joint_line_indicies(self,"A",self.vn,fixed_sides_AB[0])

        # Indices of component B
        faces_B,faces_end_B = joint_face_indicies(self,"B",2*self.vn,fixed_sides_AB[1])
        lines_B = joint_line_indicies(self,"B",3*self.vn,fixed_sides_AB[1])

        # Indicies of sliding lines
        lines_open = open_line_indicies(self,self.vn,3*self.vn)

        # Join all indices into one list
        all_indices = np.concatenate([faces_A, faces_end_A, lines_A,
                                      faces_B, faces_end_B, lines_B,
                                      lines_open])
        # Elements Buffer
        EBO = glGenBuffers(1) # element buffer object
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(all_indices), all_indices, GL_DYNAMIC_DRAW)

        self.ifA = len(faces_A)
        self.ifeA = len(faces_end_A)
        self.ilA = len(lines_A)
        self.ifB = len(faces_B)
        self.ifeB = len(faces_end_B)
        self.ilB = len(lines_B)
        self.iopen = len(lines_open)
