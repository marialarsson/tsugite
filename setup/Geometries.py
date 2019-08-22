from OpenGL.GL import *
import numpy as np
from numpy import linalg
import random
import math
import pyrr
from Selection import Selection
from Fabrication import Fabrication
from Evaluation import Evaluation
from Buffer import Buffer
from Buffer import ElementProperties

# Supporting functions

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

def face_neighbors(mat,ind,ax,n,fixed_sides):
    values = []
    dim = len(mat)
    for i in range(2):
        val = None
        ind2 = ind.copy()
        ind2[ax] = ind2[ax]-i
        ind2 = np.array(ind2)
        if np.all(ind2>=0) and np.all(ind2<dim):
            val = mat[tuple(ind2)]
        elif len(fixed_sides)>0:
            for fixed_side in fixed_sides:
                ind3 = np.delete(ind2,fixed_side[0])
                if np.all(ind3>=0) and np.all(ind3<dim):
                    if ind2[fixed_side[0]]<0 and fixed_side[1]==0: val = n
                    elif ind2[fixed_side[0]]>=dim and fixed_side[1]==1: val = n
        values.append(val)
    values = np.array(values)
    count = np.count_nonzero(values==n)
    return count,values

def get_count(ind,neighbors,fixed_sides,voxel_matrix,dim):
    cnt = 0
    val = int(voxel_matrix[ind])
    vals2 = []
    for item in neighbors:
        i = ind[0]+item[0]
        j = ind[1]+item[1]
        k = ind[2]+item[2]
        ###
        val2 = None
        # Check fixed sides
        if (i<0 or i>=dim) and j>=0 and j<dim and k>=0 and k<dim:
            if i<0 and [0,0] in fixed_sides:
                val2 = val
            elif i>=dim and [0,1] in fixed_sides:
                val2 = val
        elif (j<0 or j>=dim) and i>=0 and i<dim and k>=0 and k<dim:
            if j<0 and [1,0] in fixed_sides:
                val2 = val
            elif j>=dim and [1,1] in fixed_sides:
                val2 = val
        elif (k<0 or k>=dim) and i>=0 and i<dim and j>=0 and j<dim:
            if k<0 and [2,0] in fixed_sides:
                val2 = val
            elif k>=dim and [2,1] in fixed_sides:
                val2 = val
        # Check neighbours
        elif np.all(np.array([i,j,k])>=0) and np.all(np.array([i,j,k])<dim):
            val2 = int(voxel_matrix[i,j,k])
        if val==val2: cnt = cnt+1
        vals2.append(val2)
    return cnt,vals2[2],vals2[0],vals2[1]

def line_neighbors(self,ind,ax,n):
    values = []
    for i in range(-1,1):
        for j in range(-1,1):
            val = None
            add = [i,j]
            add.insert(ax,0)
            ind2 = np.array(ind)+np.array(add)
            if np.all(ind2>=0) and np.all(ind2<self.dim):
                val = self.voxel_matrix[tuple(ind2)]
            else:
                for n2 in range(self.noc):
                    for fixed_side in self.fixed_sides[n2]:
                        ind3 = np.delete(ind2,fixed_side[0])
                        if np.all(ind3>=0) and np.all(ind3<self.dim):
                            if ind2[fixed_side[0]]<0 and fixed_side[1]==0: val = n2
                            elif ind2[fixed_side[0]]>=self.dim and fixed_side[1]==1: val = n2
            values.append(val)
    values = np.array(values)
    count = np.count_nonzero(values==n)
    return count,values

def get_fixed_sides(joint_type,noc):
    fixed_sides = []
    if noc==2:
        if joint_type=="I":
            fixed_sides = [[[2,0]], [[2,1]]]
        elif joint_type=="L":
            fixed_sides = [[[2,0]], [[0,0]]]
        elif joint_type=="T":
            fixed_sides = [[[2,0]], [[0,0],[0,1]]]
        elif joint_type=="X":
            fixed_sides = [[[2,0],[2,1]], [[0,0],[0,1]]]
    if noc==3:
        if joint_type=="L":
            fixed_sides = [[[2,0]], [[0,0]], [[1,0]]]
        elif joint_type=="T":
            fixed_sides = [[[2,0]], [[0,0],[0,1]], [[1,0]]]
        elif joint_type=="X":
            fixed_sides = [[[2,0]], [[0,0],[0,1]], [[1,0],[1,1]]]
    return fixed_sides

def get_vertex(index,verts,n):
    x = verts[n*index]
    y = verts[n*index+1]
    z = verts[n*index+2]
    return np.array([x,y,z])

def get_next_same_axial_index(ind,ax,mat,dim):
    if ind[ax]<dim-1:
        val = mat[tuple(ind)]
        ind_next = ind.copy()
        ind_next[ax] += 1
        val_next = mat[tuple(ind_next)]
        if val==val_next:
            ind_next_next = get_next_same_axial_index(ind_next,ax,mat,dim)
            return ind_next_next
        else: return ind
    else: return ind

def mat_from_field(hf,ax):
    dim = len(hf)
    mat = np.zeros(shape=(dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                h = hf[i][j]
                if k>=h: mat[i,j,k]=1
    mat = np.array(mat)
    mat = np.swapaxes(mat,2,ax)
    return mat

def get_top_corner_heights(mat,n,ax,dir):
    heights = []
    dim = len(mat)
    for i in range(2):
        i = i*(dim-1)
        temp = []
        for j in range(2):
            j = j*(dim-1)
            top_cor = 0
            for k in range(dim):
                ind = [i,j]
                ind.insert(ax,k)
                val = mat[tuple(ind)]
                if val==n: top_cor=k
            if dir==1: top_cor = dim-top_cor
            temp.append(top_cor)
        heights.append(temp)
    return heights

# Create vertex lists functions

def joint_vertices(self,n):
    vertices = []
    r = g = b = 0.0
    # Add all vertices of the dim*dim*dim voxel cube
    ax = self.fixed_sides[n][0][0]
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                x = (i-0.5*self.dim)*self.voxel_size
                y = (j-0.5*self.dim)*self.voxel_size
                z = (k-0.5*self.dim)*self.voxel_size
                tex_coords = [i,j,k]
                tex_coords.pop(ax)
                tx = tex_coords[0]/self.dim
                ty = tex_coords[1]/self.dim
                vertices.extend([x,y,z,r,g,b,tx,ty])
    # Add component base vertices
    component_vertices = []
    for ax in range(3):
        for n in range(2):
            corners = get_corner_indices(ax,n,self.dim)
            for step in range(1,4):
                for corner in corners:
                    new_vertex = []
                    for i in range(8):
                        new_vertex_param = vertices[8*corner+i]
                        if i==ax: new_vertex_param = new_vertex_param + (2*n-1)*step*self.component_length
                        new_vertex.append(new_vertex_param)
                    vertices.extend(new_vertex)
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    return vertices

def joint_extra_vertices(self,n,r,g,b):
    vertices = []
    # Add all vertices of the dim*dim*dim voxel cube
    fab_ax = self.sliding_directions[n][0][0]
    comp_ax = self.fixed_sides[n][0][0]
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                x = (i-0.5*self.dim)*self.voxel_size
                y = (j-0.5*self.dim)*self.voxel_size
                z = (k-0.5*self.dim)*self.voxel_size
                tex_coords = [i,j,k]
                tex_coords.pop(comp_ax)
                tx = tex_coords[0]/self.dim
                ty = tex_coords[1]/self.dim
                for u in range(-1,2,2):
                    add = [self.rad*u,0]
                    add.insert(fab_ax,0)
                    x_ = x+add[0]
                    y_ = y+add[1]
                    z_ = z+add[2]
                    tex_coords = [x_,y_,z_]
                    tex_coords.pop(comp_ax)
                    tx = (tex_coords[0]+0.5*self.component_size)/self.component_size
                    ty = (tex_coords[1]+0.5*self.component_size)/self.component_size
                    vertices.extend([x_, y_, z_, r,g,b, tx,ty])
                for v in range(-1,2,2):
                    add = [0,self.rad*v]
                    add.insert(fab_ax,0)
                    x_ = x+add[0]
                    y_ = y+add[1]
                    z_ = z+add[2]
                    tex_coords = [x_,y_,z_]
                    tex_coords.pop(comp_ax)
                    tx = (tex_coords[0]+0.5*self.component_size)/self.component_size
                    ty = (tex_coords[1]+0.5*self.component_size)/self.component_size
                    vertices.extend([x_, y_, z_, r,g,b, tx,ty])

    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    return vertices

def arrow_vertices(self):
    vertices = []
    r=g=b=0.0
    tx=ty=0.0
    vertices.extend([0,0,0, r,g,b, tx,ty]) # origin
    for ax in range(3):
        for n in range(-1,2,2):
            xyz = [0,0,0]
            xyz[ax] = 0.4*n*self.component_size
            vertices.extend([xyz[0],xyz[1],xyz[2], r,g,b, tx,ty]) # end of line
            for i in range(-1,2,2):
                for j in range(-1,2,2):
                    w = 0.03*n*self.component_size
                    xyz = [w*i,w*j]
                    xyz.insert(ax,0.3*n*self.component_size)
                    vertices.extend([xyz[0],xyz[1],xyz[2], r,g,b, tx,ty]) # arrow head indices
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    return vertices

def milling_path_vertices(self,n):
    #if n==1: self.coords[2] = -self.coords[2]
    vertices = []
    # Parameters
    r = g = b = tx = ty = 0.0
    n_ = 1-n
    # Define number of steps
    no_lanes = math.ceil(2+(self.voxel_size-4*self.fab.rad)/(2*self.fab.rad))
    w = (self.voxel_size-2*self.fab.rad)/(no_lanes-1)
    no_z = int(self.voxel_size/self.fab.dep)
    dep = self.voxel_size/no_z
    # Defines axes
    ax = self.sliding_directions[n][0][0] # mill bit axis
    axes = [0,1,2]
    axes.pop(ax)
    mill_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis
    # create offset direction vectors
    vec = [0,0,0]
    vec[off_ax]=1
    vec = np.array(vec)
    vec2 = [0,0,0]
    vec2[mill_ax]=1
    vec2 = np.array(vec2)
    # get top ones to cut out
    for i in range(self.dim):
        for j in range(self.dim):
            for k in range(self.dim):
                ind = [j,k]
                ind.insert(ax,n_*(self.dim-n_)+(2*n-1)*i) # 0 when n is 1, dim-1 when n is 0
                val = self.voxel_matrix[tuple(ind)]
                if val==n: continue # dont cut out if material is there, continue
                # if previous was mill axis neighbor, its already been cut, continue
                if ind[mill_ax]>0:
                    ind_pre = ind.copy()
                    ind_pre[mill_ax] = ind_pre[mill_ax]-1
                    val_pre = self.voxel_matrix[tuple(ind_pre)]
                    if val_pre==val: continue
                # if next is mill axis neighbor, redefine pt_b
                ind_next = get_next_same_axial_index(ind,mill_ax,self.voxel_matrix,self.dim)
                # get 4 corners of the top of the voxel
                add = [0,0,0]
                add[ax]=n_
                i_a = get_index(ind,add,self.dim)
                add[mill_ax]=1
                i_b = get_index(ind_next,add,self.dim)
                # get x y z vertices corresponding to the indexes
                pt_a = get_vertex(i_a,self.jverts[n],self.vertex_no_info)
                pt_b = get_vertex(i_b,self.jverts[n],self.vertex_no_info)
                # define offsetted verticies
                layer_vertices = []
                for num in range(no_lanes):
                    p0 = pt_a+self.fab.rad*vec+num*w*vec+self.fab.rad*vec2
                    p1 = pt_b+self.fab.rad*vec+num*w*vec-self.fab.rad*vec2
                    if num%2==1: p0, p1 = p1, p0
                    layer_vertices.append([p0[0],p0[1],p0[2],r,g,b,tx,ty])
                    layer_vertices.append([p1[0],p1[1],p1[2],r,g,b,tx,ty])
                # add startpoint
                start_vert = layer_vertices[0].copy()
                safe_height = layer_vertices[0][ax]-(2*n-1)*i*self.voxel_size-0.2*(2*n-1)*self.voxel_size
                start_vert[ax] = safe_height
                vertices.extend(start_vert)
                for num in range(no_z):
                    if num%2==1: layer_vertices.reverse()
                    for vert in layer_vertices:
                        vert[ax] = vert[ax]+(2*n-1)*dep
                        vertices.extend(vert)
                # add enpoint
                end_vert = layer_vertices[-1].copy()
                end_vert[ax] = safe_height
                vertices.extend(end_vert)
    vertices = np.array(vertices, dtype = np.float32)
    return vertices

# Create index lists functions

def joint_face_indices(self,all_indices,mat,fixed_sides,n,offset):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    indices_ends = []
    d = self.dim+1
    indices = []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                ind = [i,j,k]
                for ax in range(3):
                    test_ind = np.array([i,j,k])
                    test_ind = np.delete(test_ind,ax)
                    if np.any(test_ind==self.dim): continue
                    cnt,vals = face_neighbors(mat,ind,ax,n,fixed_sides)
                    if cnt==1:
                        for x in range(2):
                            for y in range(2):
                                add = [x,abs(y-x)]
                                add.insert(ax,0)
                                index = get_index(ind,add,self.dim)
                                if len(fixed_sides)>0:
                                    if fixed_sides[0][0]==ax: indices_ends.append(index)
                                    else: indices.append(index)
                                else: indices.append(index)

        # 2. Faces of component base
        d = self.dim+1
        start = d*d*d
        if len(fixed_sides)>0:
            for ax,dir in fixed_sides:
                a1,b1,c1,d1 = get_corner_indices(ax,dir,self.dim)
                step = 2
                if self.joint_type=="X" or (self.joint_type=="T" and n==1): step = 1
                off = 24*ax+12*dir+4*step
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
    # Store
    indices_prop = ElementProperties(GL_QUADS, len(indices), len(all_indices), n)
    if len(all_indices)>0: all_indices = np.concatenate([all_indices, indices])
    else: all_indices = indices
    indices_ends_prop = ElementProperties(GL_QUADS, len(indices_ends), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices_ends])
    indices_all_prop = ElementProperties(GL_QUADS, len(indices)+len(indices_ends), indices_prop.start_index, n)
    # Return
    return indices_prop, indices_ends_prop, indices_all_prop, all_indices

def joint_top_face_indices(self,all_indices,n,offset):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    indices_tops = []
    d = self.dim+1
    indices = []
    sax = self.sliding_directions[n][0][0]
    sdir = self.sliding_directions[n][0][1]
    for i in range(d):
        for j in range(d):
            one_face_added = False
            for k in range(d):
                ind = [i,j]
                if sdir==0: k = d-k-1
                ind.insert(sax,k)
                for ax in range(3):
                    # make sure that the index is withing range
                    test_ind = np.array(ind)
                    test_ind = np.delete(test_ind,ax)
                    if np.any(test_ind==self.dim): continue
                    # count number of neigbors (0, 1, or 2)
                    cnt,vals = face_neighbors(self.voxel_matrix,ind,ax,n,self.fixed_sides[n])
                    on_free_base = False
                    if ax==sax and ax!=self.fixed_sides[n][0][0]:
                        base = sdir*self.dim
                        if ind[ax]==base: on_free_base=True
                    if cnt==1 or (cnt!=1 and on_free_base):
                        for x in range(2):
                            for y in range(2):
                                add = [x,abs(y-x)]
                                add.insert(ax,0)
                                index = get_index(ind,add,self.dim)
                                if ax==sax and not one_face_added:
                                    if ax==sax and not on_free_base:
                                        indices_tops.append(index)
                                    elif cnt!=1 and on_free_base:
                                        indices_tops.append(index)
                                    else: indices.append(index)
                                else: indices.append(index)
                        if ax==sax: one_face_added = True
    # 2. Faces of component base
    d = self.dim+1
    start = d*d*d
    for ax,dir in self.fixed_sides[n]:
        a1,b1,c1,d1 = get_corner_indices(ax,dir,self.dim)
        step = 2
        if self.joint_type=="X" or (self.joint_type=="T" and n==1): step = 1
        off = 24*ax+12*dir+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        # Add component side to indices
        indices.extend([a0,b0,d0,c0]) #bottom face
        indices.extend([a0,b0,b1,a1]) #side face 1
        indices.extend([b0,d0,d1,b1]) #side face 2
        indices.extend([d0,c0,c1,d1]) #side face 3
        indices.extend([c0,a0,a1,c1]) ##side face 4
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    indices_tops = np.array(indices_tops, dtype=np.uint32)
    indices_tops = indices_tops + offset
    # Store
    indices_prop = ElementProperties(GL_QUADS, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    indices_tops_prop = ElementProperties(GL_QUADS, len(indices_tops), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices_tops])
    # Return
    return indices_prop, indices_tops_prop, all_indices

def joint_selected_top_line_indices(self,select,all_indices):
    # Make indices of lines for drawing method GL_LINES
    n = select.n
    offset = n*self.vn
    sax = self.sliding_directions[n][0][0]
    h = self.height_field[tuple(select.faces[0])]
    # 1. Outline of selected top faces of joint
    indices = []
    for face in select.faces:
        ind = [int(face[0]),int(face[1])]
        ind.insert(sax,h)
        other_axes = [0,1,2]
        other_axes.pop(sax)
        for i in range(2):
            ax = other_axes[i]
            for j in range(2):
                # Check neighboring faces
                nface = face.copy()
                nface[i] += 2*j-1
                nface = np.array(nface, dtype=np.uint32)
                if np.all(nface>=0) and np.all(nface<self.dim):
                    unique = True
                    for face2 in select.faces:
                        if nface[0]==face2[0] and nface[1]==face2[1]:
                            unique = False
                            break
                    if not unique: continue
                for k in range(2):
                    add = [k,k,k]
                    add[ax] = j
                    add[sax] = 0
                    index = get_index(ind,add,self.dim)
                    indices.append(index)
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    # Store
    indices_prop = ElementProperties(GL_LINES, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def joint_line_indices(self,all_indices,n,offset):
    fixed_sides = self.fixed_sides[n]
    d = self.dim+1
    indices = []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                ind = [i,j,k]
                for ax in range(3):
                    if ind[ax]==self.dim: continue
                    cnt,vals = line_neighbors(self,ind,ax,n)
                    diagonal = False
                    if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                    if cnt==1 or cnt==3 or (cnt==2 and diagonal):
                        add = [0,0,0]
                        add[ax] = 1
                        start_i = get_index(ind,[0,0,0],self.dim)
                        end_i = get_index(ind,add,self.dim)
                        indices.extend([start_i,end_i])
    #Outline of component base
    start = d*d*d
    for ax,dir in fixed_sides:
        a1,b1,c1,d1 = get_corner_indices(ax,dir,self.dim)
        step = 2
        if self.joint_type=="X" or ( self.joint_type=="T" and n==1): step = 1
        off = 24*ax+12*dir+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
        indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    # Store
    indices_prop = ElementProperties(GL_LINES, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def open_line_indices(self,all_indices,n,slide_dir,offset):
    indices = []
    ax,dir = slide_dir
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    #ind[ax] = dir*self.dim
    d = self.dim-1
    heights = get_top_corner_heights(self.voxel_matrix,n,ax,dir)
    #heights = [[self.height_field2[0][0], self.height_field2[0][d]],
    #          [self.height_field2[d][0],  self.height_field2[d][d]]]
    d = self.dim+1
    #if n==0: dir = 1-dir
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*self.dim
            add[other_axes[1]] = y*self.dim
            #add[ax] = heights[x][y]
            add[ax] = dir*self.dim
            start = get_index(ind,add,self.dim)
            end = d*d*d+24*ax+12*(1-dir)+2*x+y
            #off = 24*ax+12*dir+4*step
            indices.extend([start,end])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    # Store
    indices_prop = ElementProperties(GL_LINES, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def arrow_indices(self,all_indices,slide_dirs,n,offset):
    line_indices = []
    face_indices = []
    for ax,dir in slide_dirs:
        if n==1: dir = 1-dir
        #arrow line
        start = 1+10*ax+5*dir
        line_indices.extend([0, start])
        #arrow head (triangles)
        face_indices.extend([start+1, start+2, start+4])
        face_indices.extend([start+1, start+4, start+3])
        face_indices.extend([start+1, start+2, start])
        face_indices.extend([start+2, start+4, start])
        face_indices.extend([start+3, start+4, start])
        face_indices.extend([start+1, start+3, start])
    # Format
    line_indices = np.array(line_indices, dtype=np.uint32)
    line_indices = line_indices + offset
    face_indices = np.array(face_indices, dtype=np.uint32)
    face_indices = face_indices + offset
    # Store
    line_indices_prop = ElementProperties(GL_LINES, len(line_indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, line_indices])
    face_indices_prop = ElementProperties(GL_TRIANGLES, len(face_indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, face_indices])
    # Return
    return line_indices_prop, face_indices_prop, all_indices

def milling_path_indices(self,all_indices,no,start,n):
    indices = []
    for i in range(start,start+no):
        indices.append(int(i))
    # Format
    indices = np.array(indices, dtype=np.uint32)
    # Store
    indices_prop = ElementProperties(GL_LINE_LOOP, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

class Geometries:
    def __init__(self):
        self.joint_type = "I"
        self.noc = 2 #number of components
        self.fixed_sides = get_fixed_sides(self.joint_type,self.noc)
        self.sliding_directions = [[[2,0]],[[2,1]]]
        self.dim = 3
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.5*self.component_size
        self.vertex_no_info = 8
        self.height_field = get_random_height_field(self.dim)
        self.height_field2 =get_random_height_field(self.dim)
        self.pre_height_field = self.height_field
        self.pre_height_field2 = self.height_field2
        self.select = Selection(self)
        self.fab = Fabrication(self)
        self.voxel_matrix_from_height_fields()
        self.buff = Buffer(self)
        self.vertices = self.create_vertices()
        self.indices = self.create_indices()

    def voxel_matrix_from_height_fields(self):
        ax = self.sliding_directions[0][0][0]
        vox_mat = mat_from_field(self.height_field,ax)
        if self.noc==3:
            ax = self.sliding_directions[2][0][0]
            vox_mat2 = mat_from_field(self.height_field2,ax)
            # combine matrices
            for i in range(self.dim):
                for j in range(self.dim):
                    for k in range(self.dim):
                        ind = tuple([i,j,k])
                        if vox_mat[ind]==1 and vox_mat2[ind]==0: vox_mat[ind]=2
        self.voxel_matrix = vox_mat
        self.eval = Evaluation(self)

    def update_sliding_direction(self,sliding_directions):
        self.sliding_directions = sliding_directions
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def randomize_height_field(self):
        self.pre_height_field = self.height_field.copy()
        self.pre_height_field2 = self.height_field2.copy()
        self.height_field = get_random_height_field(self.dim)
        self.height_field2 = get_random_height_field(self.dim)
        self.voxel_matrix_from_height_fields()
        self.create_indices()

    def clear_height_field(self):
        self.pre_height_field = self.height_field.copy()
        self.pre_height_field2 = self.height_field2.copy()
        self.height_field = np.zeros((self.dim,self.dim))
        self.height_field2 = np.zeros((self.dim,self.dim))
        self.voxel_matrix_from_height_fields()
        self.create_indices()

    def edit_height_field(self,faces,h,n):
        self.pre_height_field = self.height_field.copy()
        self.pre_height_field2 = self.height_field2.copy()
        for ind in faces:
            if n!=2: self.height_field[tuple(ind)] = h
            else: self.height_field2[tuple(ind)] = h
            self.voxel_matrix_from_height_fields()
            self.create_indices()

    def update_joint_type(self,joint_type_):
        self.joint_type = joint_type_
        if self.joint_type=="X" and self.sliding_directions[0][0]==[2,0]:
            self.update_sliding_direction([[[1,0]],[[1,1]]])
        self.fixed_sides = get_fixed_sides(self.joint_type,self.noc)
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def update_dimension(self,dim_):
        pdim = self.dim
        self.dim = dim_
        self.voxel_size = self.component_size/self.dim
        if self.dim>pdim:
            f = self.dim/pdim
            a = int(0.5*(self.dim-pdim)+0.5)
            b = self.dim-pdim-a
            self.height_field = f*np.pad(self.height_field, ((a, b),(a, b)), 'edge')
        elif self.dim<pdim:
            f = self.dim/pdim
            f2 = pdim/self.dim
            hf = np.zeros((self.dim,self.dim))
            for i in range(self.dim):
                for j in range(self.dim):
                    sum = 0
                    cnt = 0
                    for x in range(self.dim):
                        for y in range(self.dim):
                            a = int(f2*i+x)
                            b = int(f2*j+y)
                            if a>=pdim or b>=pdim: continue
                            sum += self.height_field[a][b]
                            cnt +=1
                    hf[i][j] = int(f*sum/cnt+0.5)
            self.height_field = hf
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def update_number_of_components(self,num):
        self.noc = num
        self.fixed_sides = get_fixed_sides(self.joint_type,self.noc)
        self.sliding_directions = [ [[2,0]], [[2,1],[1,1]], [[1,0]] ]
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def create_vertices(self):

        ### VERTICIES FOR JOINT ###
        self.jverts = []
        for n in range(self.noc):
            self.jverts.append(joint_vertices(self,n))

        ### EXTRA VERTICIES FOR ROUNDED CORNERS OF FABRICATED JOINT ###
        #ve_A = joint_extra_vertices(self,"A",0.0,0.0,0.0)
        #ve_B = joint_extra_vertices(self,"B",0.0,0.0,0.0)

        ### VERTICIES FOR ARROWS DISPLAYING SLIDING DIRECTIONS ###
        va = arrow_vertices(self)

        ### VERTICIES FOR MILLING PATHS ###
        #self.mverts = []
        #for n in range(self.noc):
        #    self.mverts.append(milling_path_vertices(self,n))

        jverts = np.concatenate(self.jverts)
        #mverts = np.concatenate(self.mverts)
        self.vertices = np.concatenate([jverts, va])

        self.vn =  int(len(self.jverts[0])/8)
        self.ven = 0
        van = int(len(va)/8)
        #self.m_start = []
        #mst = 2*self.vn+2*self.ven+van
        #for n in range(self.noc):
        #    self.m_start.append(mst)
        #    mst += int(len(self.mverts[n])/8)
        Buffer.buffer_vertices(self.buff)

    def create_indices(self):
        all_inds = []

        self.indices_fend = []
        self.indices_not_fend = []
        self.indices_fcon = []
        self.indices_not_fcon = []
        self.indices_fall = []
        self.indices_lns = []
        self.indices_open_lines = []
        self.indices_not_fbridge = []
        self.indices_arrows = []
        self.indices_ftop = []
        self.indices_not_ftop = []
        self.outline_selected_faces = None
        self.indices_milling_path = []

        for n in range(self.noc):
            #Faces
            nend,end,con,all_inds = joint_face_indices(self, all_inds,
                    self.eval.voxel_matrix_connected,self.fixed_sides[n],n,n*self.vn)
            if not self.eval.connected[n]:
                fne,fe,uncon,all_inds = joint_face_indices(self,all_inds,self.eval.voxel_matrix_unconnected,[],n,n*self.vn)
                self.indices_not_fcon.append(uncon)
                all = ElementProperties(GL_QUADS, con.count+uncon.count, con.start_index, n)
            else:
                self.indices_not_fcon.append(None)
                all = con
            if not self.eval.bridged[n]:
                unbris = []
                for m in range(2):
                    fne,fe,unbri,all_inds = joint_face_indices(self, all_inds,self.eval.voxel_matrices_unbridged[n][m],[self.fixed_sides[n][m]],n,n*self.vn)
                    unbris.append(unbri)
            else: unbris = None
            #mill,all_inds = milling_path_indices(self,all_inds,len(self.mverts[n]),self.m_start[n],n)

            #picking faces
            faces_not_tops, faces_tops, all_inds = joint_top_face_indices(self,all_inds,n,n*self.vn)

            #Lines
            lns,all_inds = joint_line_indices(self,all_inds,n,n*self.vn)

            for m in range(len(self.sliding_directions[n])):
                open,all_inds = open_line_indices(self,all_inds,n,self.sliding_directions[n][m],n*self.vn)
                self.indices_open_lines.append(open)

            #arrows
            #larr, farr, all_inds = arrow_indices(self, all_inds,[self.sliding_directions[0]],n,self.noc*self.vn+2*self.ven)
            larr, farr, all_inds = arrow_indices(self, all_inds,self.eval.slides[n],n,self.noc*self.vn+2*self.ven)
            arrows = [larr,farr]

            # Append lists
            self.indices_fend.append(end)
            self.indices_not_fend.append(nend)
            self.indices_fcon.append(con)
            self.indices_fall.append(all)
            self.indices_lns.append(lns)
            self.indices_not_fbridge.append(unbris)
            self.indices_arrows.append(arrows)
            self.indices_ftop.append(faces_tops)
            self.indices_not_ftop.append(faces_not_tops)
            #self.indices_milling_path.append(mill)

        #outline of selected faces
        if self.select.state==2:
            self.outline_selected_faces, all_inds = joint_selected_top_line_indices(self,self.select,all_inds)

        self.indices = all_inds
        Buffer.buffer_indices(self.buff)

    def save(self):
        np.save("data/saved_height_field.npy",self.height_field)
        np.save("data/saved_height_field2.npy",self.height_field2)
        #np.save("data/saved_voxel_matrix.npy",self.voxel_matrix)
        #np.save("data/saved_voxel_matrix_with_fixed_sides.npy",self.voxel_matrix_with_sides)
        #np.save("data/saved_fixed_sides.npy",self.fixed_sides)

    def load(self):
        self.height_field = np.load("data/saved_height_field.npy")
        self.height_field2 = np.load("data/saved_height_field2.npy")
        self.voxel_matrix_from_height_fields()
        self.create_indices()

    def undo(self):
        print("undoing...")
        temp = self.height_field.copy()
        self.height_field = self.pre_height_field.copy()
        self.pre_height_field = temp
        temp = self.height_field2.copy()
        self.height_field2 = self.pre_height_field2.copy()
        self.pre_height_field2 = temp
        self.voxel_matrix_from_height_fields()
        self.create_indices()
