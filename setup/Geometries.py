from OpenGL.GL import *
import numpy as np
from numpy import linalg
import random
from PIL import Image
import math
import pyrr
from Fabrication import Fabrication

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
    return count

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
                for n2 in range(2):
                    for fixed_side in self.fixed_sides[n2]:
                        ind3 = np.delete(ind2,fixed_side[0])
                        if np.all(ind3>=0) and np.all(ind3<self.dim):
                            if ind2[fixed_side[0]]<0 and fixed_side[1]==0: val = n2
                            elif ind2[fixed_side[0]]>=self.dim and fixed_side[1]==1: val = n2
            values.append(val)
    values = np.array(values)
    count = np.count_nonzero(values==n)
    return count,values

def get_fixed_sides(joint_type):
    fixed_sides = []
    if joint_type=="I":   fixed_sides = [[[2,0]], [[2,1]]]
    elif joint_type=="L": fixed_sides = [[[2,0]], [[0,0]]]
    elif joint_type=="T": fixed_sides = [[[2,0]], [[0,0],[0,1]]]
    elif joint_type=="X": fixed_sides = [[[2,0],[2,1]], [[0,0],[0,1]]]
    return fixed_sides

def add_fixed_sides(mat,fixed_sides):
    dim = len(mat)
    pad_loc = [[0,0],[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1],[-1,-1]]
    for n in range(2):
        for ax,dir in fixed_sides[n]:
            pad_loc[ax][dir] = 1
            pad_val[ax][dir] = n
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)
    # Take care of corners
    for ax,dir in fixed_sides[0]:
        for ax2,dir2 in fixed_sides[1]:
            if ax==ax2: continue
            for i in range(dim):
                ind = [i,i,i]
                ind[ax] =  dir*(mat.shape[ax]-1)
                ind[ax2] = dir2*(mat.shape[ax2]-1)
                mat[tuple(ind)] = -1
    return mat

def get_axial_neighbors(mat,ind,ax):
    indices = []
    values = []
    m = ax
    for n in range(2):      # go up and down one step
        n=2*n-1             # -1,1
        ind0 = list(ind)
        ind0[m] = ind[m]+n
        ind0 = tuple(ind0)
        if ind0[m]>=0 and ind0[m]<mat.shape[m]:
            indices.append(ind0)
            try: values.append(int(mat[ind0]))
            except: values.append(mat[ind0])
    return indices,values

def get_neighbors(mat,ind):
    indices = []
    values = []
    for m in range(len(ind)):   # For each direction (x,y)
        for n in range(2):      # go up and down one step
            n=2*n-1             # -1,1
            ind0 = list(ind)
            ind0[m] = ind[m]+n
            ind0 = tuple(ind0)
            if ind0[m]>=0 and ind0[m]<mat.shape[m]:
                indices.append(ind0)
                values.append(int(mat[ind0]))
    return indices, np.array(values)

def get_all_same_connected(mat,indices):
    start_n = len(indices)
    val = int(mat[indices[0]])
    all_same_neighbors = []
    for ind in indices:
        n_indices,n_values = get_neighbors(mat,ind)
        for n_ind,n_val in zip(n_indices,n_values):
            if n_val==val: all_same_neighbors.append(n_ind)
    indices.extend(all_same_neighbors)
    if len(indices)>0:
        indices = np.unique(indices, axis=0)
        indices = [tuple(ind) for ind in indices]
        if len(indices)>start_n: indices = get_all_same_connected(mat,indices)
    return indices

def reverse_columns(cols):
    new_cols = []
    for i in range(len(cols)):
        temp = []
        for j in range(len(cols[i])):
            temp.append(cols[i][len(cols[i])-j-1].astype(int))
        new_cols.append(temp)
    return new_cols

def get_columns(mat,ax):
    columns = []
    if ax==0:
        for j in range(len(mat[0])):
            for k in range(len(mat[0][0])):
                col = []
                for i in range(len(mat)): col.append(mat[i][j][k])
                columns.append(col)
    elif ax==1:
        for i in range(len(mat)):
            for k in range(len(mat[0][0])):
                col = []
                for j in range(len(mat[0])): col.append(mat[i][j][k])
                columns.append(col)
    elif ax==2:
        for layer in mat:
            for col in layer: columns.append(col)
    columns2 = []
    for col in columns:
        col = np.array(col)
        col = col[np.logical_not(np.isnan(col))] #remove nans
        if len(col)==0: continue
        col = col.astype(int)
        columns2.append(col)
    return columns2

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

def get_indices_of_same_neighbors(indices,mat):
    d = len(mat)
    val = mat[tuple(indices[0])]
    neighbors = []
    for ind in indices:
        for ax in range(3):
            for dir in range(2):
                dir = 2*dir-1
                ind2 = ind.copy()
                ind2[ax] = ind2[ax]+dir
                if ind2[ax]>=0 and ind2[ax]<d:
                    val2 = mat[tuple(ind2)]
                    if val==val2:
                        neighbors.append(ind2)
    if len(neighbors)>0:
        neighbors = np.array(neighbors)
        neighbors = np.unique(neighbors, axis=0)
    return neighbors

def is_connected_to_fixed_side(indices,mat,fixed_sides):
    connected = False
    val = mat[tuple(indices[0])]
    d = len(mat)
    for ind in indices:
        for ax,dir in fixed_sides:
            if ind[ax]==0 and dir==0:
                connected=True
                break
            elif ind[ax]==d-1 and dir==1:
                connected=True
                break
        if connected: break
    if not connected:
        neighbors = get_indices_of_same_neighbors(indices,mat)
        if len(neighbors)>0:
            new_indices = np.concatenate([indices,neighbors])
            new_indices = np.unique(new_indices, axis=0)
            if len(new_indices)>len(indices):
                connected = is_connected_to_fixed_side(new_indices,mat,fixed_sides)
    return connected

# Create vertex lists functions

def joint_vertices(self,comp,r,g,b):
    vertices = []
    # Add all vertices of the dim*dim*dim voxel cube
    ax = 2
    if comp=="B" and self.joint_type!="I": ax = 0
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

def joint_extra_vertices(self,comp,r,g,b):
    vertices = []
    # Add all vertices of the dim*dim*dim voxel cube
    fab_ax = self.sliding_direction[0]
    comp_ax = 2
    if comp=="B" and self.joint_type!="I": comp_ax = 0
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
    vertices = []
    # Parameters
    r = g = b = tx = ty = 0.0
    n_ = 1-n
    # Define number of steps
    no_lanes = math.ceil(2+(self.voxel_size-4*self.rad)/(2*self.rad))
    w = (self.voxel_size-2*self.rad)/(no_lanes-1)
    no_z = int(self.voxel_size/self.dep)
    self.dep = self.voxel_size/no_z
    # Get reference vertices
    if n==0: verts = self.v_A
    else: verts = self.v_B
    # Defines axes
    ax = self.sliding_direction[0] # mill bit axis
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
                pt_a = get_vertex(i_a,verts,self.vertex_no_info)
                pt_b = get_vertex(i_b,verts,self.vertex_no_info)
                # define offsetted verticies
                layer_vertices = []
                for num in range(no_lanes):
                    p0 = pt_a+self.rad*vec+num*w*vec+self.rad*vec2
                    p1 = pt_b+self.rad*vec+num*w*vec-self.rad*vec2
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
                        vert[ax] = vert[ax]+(2*n-1)*self.dep
                        vertices.extend(vert)
                # add enpoint
                end_vert = layer_vertices[-1].copy()
                end_vert[ax] = safe_height
                vertices.extend(end_vert)

    vertices = np.array(vertices, dtype = np.float32)
    return vertices

# Create index lists functions

def joint_face_indicies(self,all_indices,mat,fixed_sides,n,offset):
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
                    cnt = face_neighbors(mat,ind,ax,n,fixed_sides)
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

def joint_face_fab_indicies(self,all_indices,mat,fixed_sides,n,offset,offset_extra):
    offset_extra = offset_extra-offset
    fab_ax = self.sliding_direction[0]
    other_axes = [0,1,2]
    other_axes.pop(fab_ax)
    # Matrix of rounded corners
    mat_shape = [self.dim+1,self.dim+1,self.dim+1,4]
    mat_shape[fab_ax] = self.dim
    rounded_corners = np.full(tuple(mat_shape), False, dtype=bool)
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                ind = [i,j,k]
                if ind[fab_ax]==self.dim: continue
                cnt,vals = line_neighbors(self,ind,fab_ax,n)
                diagonal = False
                if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                if cnt==3 or (cnt==2 and diagonal):
                    rounded_corners[i][j][k] = np.array(vals!=n)
                cnt_other,vals_other = line_neighbors(self,ind,fab_ax,1-n)
                if cnt_other==3 and cnt>0:
                    rounded_corners_other = np.array(vals_other!=(1-n))
                    for index in range(4):
                        if rounded_corners_other[index]:
                            rounded_corners[i][j][k][index] = True
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
                    cnt = face_neighbors(mat,ind,ax,n,fixed_sides)
                    if cnt==1: # if a face is to be drawn at all
                        # Normal case, no rounded corner
                        face_indexes = []
                        for x in range(2):
                            for y in range(2):
                                add = [x,y]
                                add.insert(ax,0)
                                index = get_index(ind,add,self.dim)
                                face_indexes.append(index)
                        # Axis not perpendicular to fabrication direciton, check for rounded corners
                        if ax!=fab_ax:
                            # start-side
                            a,b,add = 2,3,1
                            if ax==other_axes[0]: a,b,add = 1,3,3
                            if rounded_corners[i][j][k][a] or rounded_corners[i][j][k][b]:
                                face_indexes[0] = 4*face_indexes[0]+offset_extra+add
                                face_indexes[1] = 4*face_indexes[1]+offset_extra+add
                            # end side
                            a,b,add = 0,1,0
                            if ax==other_axes[0]: a,b,add = 0,2,2
                            add1 = [1,1,1]
                            add1[fab_ax] = 0
                            add1[ax] = 0
                            inda = [i+add1[0],j+add1[1],k+add1[2],a]
                            indb = [i+add1[0],j+add1[1],k+add1[2],b]
                            if rounded_corners[tuple(inda)] or rounded_corners[tuple(indb)]:
                                face_indexes[2] = 4*face_indexes[2]+offset_extra+add
                                face_indexes[3] = 4*face_indexes[3]+offset_extra+add
                        # Make triangles
                        tris = [face_indexes[0],face_indexes[1],face_indexes[3]]
                        tri2 = [face_indexes[0],face_indexes[3],face_indexes[2]]
                        tris.extend(tri2)
                        if ax==fab_ax and ind[fab_ax]>-n and ind[fab_ax]<self.dim+1-n: #top face, might have one or more rounded corners
                            rcorners =  [False,False,False,False]
                            box_inds = [3,2,1,0]
                            for x in range(2):
                                for y in range(2):
                                    add = [x,y]
                                    add.insert(fab_ax,-1+n)
                                    box_i = box_inds[2*x+y]
                                    cor_i = [i+add[0],j+add[1],k+add[2],box_i]
                                    if rounded_corners[tuple(cor_i)]:
                                        rcorners[2*x+y]=True
                            if np.any(np.array(rcorners)):
                                face_indexes.append(offset_extra+4*face_indexes[0]+1) #wrong
                                face_indexes.append(offset_extra+4*face_indexes[0]+3)
                                face_indexes.append(offset_extra+4*face_indexes[1]+2)
                                face_indexes.append(offset_extra+4*face_indexes[1]+1)
                                face_indexes.append(offset_extra+4*face_indexes[3]+0)
                                face_indexes.append(offset_extra+4*face_indexes[3]+2)
                                face_indexes.append(offset_extra+4*face_indexes[2]+3)
                                face_indexes.append(offset_extra+4*face_indexes[2]+0)
                                tris = [face_indexes[9],face_indexes[6],face_indexes[5]]
                                tri2 = [face_indexes[5],face_indexes[9],face_indexes[10]]
                                tri3 = [face_indexes[9],face_indexes[8],face_indexes[6]]
                                tri4 = [face_indexes[8],face_indexes[7],face_indexes[6]]
                                tri5 = [face_indexes[10],face_indexes[11],face_indexes[4]]
                                tri6 = [face_indexes[10],face_indexes[5],face_indexes[4]]
                                tris.extend(tri2)
                                tris.extend(tri3)
                                tris.extend(tri4)
                                tris.extend(tri5)
                                tris.extend(tri6)
                                if not rcorners[0]: tris.extend([face_indexes[0],face_indexes[4],face_indexes[5]])
                                if not rcorners[1]: tris.extend([face_indexes[6],face_indexes[1],face_indexes[7]])
                                if not rcorners[2]: tris.extend([face_indexes[2],face_indexes[11],face_indexes[10]])
                                if not rcorners[3]: tris.extend([face_indexes[9],face_indexes[8],face_indexes[3]])
                        ### Seperate end grains...
                        if len(fixed_sides)>0 and fixed_sides[0][0]==ax:
                            indices_ends.extend(tris)
                        else: indices.extend(tris)
    # Make corner long edge
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                ind = [i,j,k]
                if ind[fab_ax]==self.dim: continue
                cnt,vals = line_neighbors(self,ind,fab_ax,n)
                diagonal = False
                if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                if cnt==1 or cnt==3 or (cnt==2 and diagonal): # If a face is to be drawn at all
                    add = [0,0,0]
                    add[fab_ax] = 1
                    start_i = get_index(ind,[0,0,0],self.dim)
                    end_i = get_index(ind,add,self.dim)
                    if np.any(rounded_corners[i][j][k]):
                        for index in range(4):
                            rounded_i = [i,j,k,index]
                            if rounded_corners[tuple(rounded_i)]==True:
                                u,v = int(index/2), 2+index%2
                                # Vertical lines
                                su_i = offset_extra+4*start_i+u
                                eu_i = offset_extra+4*end_i+u
                                sv_i = offset_extra+4*start_i+v
                                ev_i = offset_extra+4*end_i+v
                                tris = [su_i,eu_i,sv_i,sv_i,ev_i,eu_i]
                                indices.extend(tris)
                                below_i = rounded_i.copy()
                                above_i = rounded_i.copy()
                                below_i[fab_ax] -= 1
                                above_i[fab_ax] += 1
                                if n==0 or cnt!=1:
                                    if below_i[fab_ax]<0 or rounded_corners[tuple(below_i)]==False:
                                        if len(fixed_sides)>0 and fixed_sides[0][0]==fab_ax:
                                            indices_ends.extend([start_i,su_i,sv_i])
                                        else: indices.extend([start_i,su_i,sv_i])
                                if n==1 or cnt!=1:
                                    if above_i[fab_ax]>=self.dim or rounded_corners[tuple(above_i)]==False:
                                        if len(fixed_sides)>0 and fixed_sides[0][0]==fab_ax:
                                            indices_ends.extend([end_i,eu_i,ev_i])
                                        else: indices.extend([end_i,eu_i,ev_i])

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
                indices_ends.extend([a0,b0,d0]) #bottom face
                indices_ends.extend([a0,d0,c0]) # bottom face
                indices.extend([a0,b0,b1]) #side face 1
                indices.extend([a0,b1,a1]) #side face 1
                indices.extend([b0,d0,d1]) #side face 2
                indices.extend([b0,d1,b1]) #side face 2
                indices.extend([d0,c0,c1]) #side face 3
                indices.extend([d0,c1,d1]) #side face 3
                indices.extend([c0,a0,a1]) ##side face 4
                indices.extend([c0,a1,c1]) ##side face 4
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    indices_ends = np.array(indices_ends, dtype=np.uint32)
    indices_ends = indices_ends + offset
    # Store
    indices_prop = ElementProperties(GL_TRIANGLES, len(indices), len(all_indices), n)
    if len(all_indices)>0: all_indices = np.concatenate([all_indices, indices])
    else: all_indices = indices
    indices_ends_prop = ElementProperties(GL_TRIANGLES, len(indices_ends), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices_ends])
    indices_all_prop = ElementProperties(GL_TRIANGLES, len(indices)+len(indices_ends), indices_prop.start_index, n)
    # Return
    return indices_prop, indices_ends_prop, indices_all_prop, all_indices

def joint_top_face_indices(self,all_indices,n,offset):
    # Make indices of faces for drawing method GL_QUADS
    # 1. Faces of joint
    indices = []
    indices_tops = []
    d = self.dim+1
    indices = []
    for i in range(d):
        for j in range(d):
            for k in range(d):
                ind = [i,j]
                ind.insert(self.sliding_direction[0],k)
                for ax in range(3):
                    test_ind = np.array(ind)
                    test_ind = np.delete(test_ind,ax)
                    if np.any(test_ind==self.dim): continue
                    cnt = face_neighbors(self.voxel_matrix,ind,ax,n,self.fixed_sides[n])
                    on_free_base = False
                    if ax==self.sliding_direction[0] and ax!=self.fixed_sides[n][0][0]:
                        dir = abs(self.sliding_direction[1]-n)
                        base = dir*self.dim
                        if ind[ax]==base: on_free_base=True
                    if cnt==1 or (cnt!=1 and on_free_base==True):
                        for x in range(2):
                            for y in range(2):
                                add = [x,abs(y-x)]
                                add.insert(ax,0)
                                index = get_index(ind,add,self.dim)
                                if ax==self.sliding_direction[0] and on_free_base==False:
                                    indices_tops.append(index)
                                elif cnt!=1 and on_free_base==True:
                                    indices_tops.append(index)
                                else: indices.append(index)
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

def joint_selected_top_line_indices(self,selected,all_indices):
    # Make indices of lines for drawing method GL_LINES
    n = selected.n
    offset = n*self.vn
    sax=self.sliding_direction[0]
    h = self.height_field[tuple(selected.selected_faces[0])]
    # 1. Outline of selected top faces of joint
    indices = []
    for face in selected.selected_faces:
        ind = [int(face[0]),int(face[1])]
        ind.insert(sax,h)
        #print(ind)
        other_axes = [0,1,2]
        other_axes.pop(sax)
        for i in range(2):
            ax = other_axes[i]
            for j in range(2):
                nface = face.copy()
                nface[ax] += 2*j-1
                nface = np.array(nface, dtype=np.uint32)
                if np.all(nface>=0) and np.all(nface<self.dim):
                    unique = True
                    for face2 in selected.selected_faces:
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

def joint_line_indicies(self,all_indices,n,offset):
    fixed_sides = self.fixed_sides[n]
    fab_ax = self.sliding_direction[0]
    # Make indices for draw elements method GL_LINES
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

def joint_line_fab_indicies(self,all_indices,n,offset,offset_extra):
    offset_extra = offset_extra-offset
    fixed_sides = self.fixed_sides[n]
    fab_ax = self.sliding_direction[0]
    other_axes = [0,1,2]
    other_axes.pop(fab_ax)
    d = self.dim+1
    indices = []
    # Matrix of rounded corners
    mat_shape = [self.dim+1,self.dim+1,self.dim+1,4]
    mat_shape[fab_ax] = self.dim
    rounded_corners = np.full(tuple(mat_shape), False, dtype=bool)
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                ind = [i,j,k]
                if ind[fab_ax]==self.dim: continue
                cnt,vals = line_neighbors(self,ind,fab_ax,n)
                diagonal = False
                if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                if cnt==3 or (cnt==2 and diagonal):
                    rounded_corners[i][j][k] = np.array(vals!=n)
                cnt_other,vals_other = line_neighbors(self,ind,fab_ax,1-n)
                if cnt_other==3 and cnt>0:
                    rounded_corners_other = np.array(vals_other!=(1-n))
                    for index in range(4):
                        if rounded_corners_other[index]:
                            rounded_corners[i][j][k][index] = True
    # Make line indices
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
                ind = [i,j,k]
                for ax in range(3):
                    if ind[ax]==self.dim: continue
                    cnt,vals = line_neighbors(self,ind,ax,n)
                    diagonal = False
                    if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                    if cnt==1 or cnt==3 or (cnt==2 and diagonal): # If a line is to be drawn at all
                        add = [0,0,0]
                        add[ax] = 1
                        start_i = get_index(ind,[0,0,0],self.dim)
                        end_i = get_index(ind,add,self.dim)
                        if ax!=fab_ax:
                            ind_above = ind.copy()
                            ind_below = ind.copy()
                            ind_below[fab_ax] -= 1
                            inds = []
                            if ind_above[fab_ax]>=0 and ind_above[fab_ax]<self.dim:
                                inds.append(ind_above)
                            if ind_below[fab_ax]>=0 and ind_below[fab_ax]<self.dim:
                                inds.append(ind_below)
                            for ind0 in inds:
                                a,b,add = 2,3,1
                                if ax==other_axes[1]: a,b,add = 1,3,3
                                ind1 = ind0.copy()
                                ind1.append(a)
                                ind1b = ind1.copy()
                                ind1b[3] = b
                                if rounded_corners[tuple(ind1)]==True or rounded_corners[tuple(ind1b)]==True:
                                    start_i = offset_extra+4*start_i+add
                                ind2 = ind0.copy()
                                ind2[ax] += 1
                                a,b,add = 0,1,0
                                if ax==other_axes[1]: a,b,add = 0,2,2
                                ind2.append(a)
                                ind2b = ind2.copy()
                                ind2b[3] = b
                                if rounded_corners[tuple(ind2)]==True or rounded_corners[tuple(ind2b)]==True:
                                    end_i = offset_extra+4*end_i+add
                            indices.extend([start_i,end_i])
                        elif np.all(rounded_corners[i][j][k]==False):
                            indices.extend([start_i,end_i])
                        else:
                            for index in range(4):
                                rounded_i = [i,j,k,index]
                                if rounded_corners[tuple(rounded_i)]==True:
                                    u,v = int(index/2), 2+index%2
                                    # Vertical lines
                                    indices.extend([offset_extra+4*start_i+u, offset_extra+4*end_i+u])
                                    indices.extend([offset_extra+4*start_i+v, offset_extra+4*end_i+v])
                                    # Horizontal lines
                                    below_i = rounded_i.copy()
                                    above_i = rounded_i.copy()
                                    below_i[fab_ax] -= 1
                                    above_i[fab_ax] += 1
                                    if below_i[fab_ax]<0 or rounded_corners[tuple(below_i)]==False:
                                        indices.extend([offset_extra+4*start_i+u, offset_extra+4*start_i+v])
                                        if below_i[fab_ax]>=0:
                                            below_i.pop()
                                            cnt,vals = line_neighbors(self,below_i,ax,n)
                                            diagonal = False
                                            if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                                            if cnt==2 and not diagonal:
                                                if vals[0]==vals[2]:
                                                    indices.extend([offset_extra+4*start_i+u,start_i])
                                                else:
                                                    indices.extend([offset_extra+4*start_i+v,start_i])

                                    if above_i[fab_ax]>=self.dim or rounded_corners[tuple(above_i)]==False:
                                        indices.extend([offset_extra+4*end_i+u, offset_extra+4*end_i+v]) #bot a, top b
                                        if above_i[fab_ax]>=0:
                                            above_i.pop()
                                            cnt,vals = line_neighbors(self,above_i,ax,n)
                                            diagonal = False
                                            if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                                            if cnt==2 and not diagonal:
                                                if vals[0]==vals[2]:
                                                    indices.extend([offset_extra+4*end_i+u,end_i])
                                                else:
                                                    indices.extend([offset_extra+4*end_i+v,end_i])
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

def open_line_indicies(self,all_indices,n,offset):
    ax,dir = self.sliding_direction
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==ax))
    ind = np.array([0,0,0])
    #ind[ax] = dir*self.dim
    d = self.dim-1
    heights = [[self.height_field[0][0], self.height_field[0][d]],
              [self.height_field[d][0],  self.height_field[d][d]]]
    d = self.dim+1
    indices = []
    if n==0: dir = 1-dir
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*self.dim
            add[other_axes[1]] = y*self.dim
            add[ax] = heights[x][y]
            start = get_index(ind,add,self.dim)
            end = d*d*d+24*ax+12*dir+2*x+y
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

# Analyze joint functions

def get_friction(mat,slides):
    friction = 0
    # Define which axes are acting in friction
    axes = [0,1,2]
    bad_axes = []
    for n in range(2): #for each material
        for item in slides[n]: #for each sliding direction
            bad_axes.append(item[0])
    axes = [x for x in axes if x not in bad_axes]
    # Check neighbors in relevant axes. If neighbor is other, friction is acting!
    indices = np.argwhere(mat==0)
    for ind in indices:
        for ax in axes:
            n_indices,n_values = get_axial_neighbors(mat,ind,ax)
            for n_val in n_values:
                if n_val==1: friction += 1
    return friction

def is_connected(mat,n):
    connected = False
    all_same = np.count_nonzero(mat==n) # Count number of ones in matrix
    if all_same>0:
        ind = tuple(np.argwhere(mat==n)[0]) # Pick a random one
        inds = get_all_same_connected(mat,[ind]) # Get all its neighbors (recursively)
        connected_same = len(inds)
        if connected_same==all_same: connected = True
    return connected

def get_sliding_directions(mat):
    sliding_directions = []
    for n in range(2): # Browse the components (0, 1 / two materials)
        mat_sliding = []
        for ax in range(3): # Browse the three possible sliding axes
            for dir in range(2): # Browse the two possible directions of the axis
                slides_in_this_direction = True
                cols = get_columns(mat,ax) # Columns corresponding to this direction
                if dir==0: cols = reverse_columns(cols)
                for col in cols:
                    first_same = False
                    for i in range(len(col)):
                        if col[i]==n: first_same = True; continue
                        elif first_same==True and (col[i]==1-n):
                            slides_in_this_direction=False; break
                    if slides_in_this_direction==False: break #stop checking further columns if one was blocking the slide
                if slides_in_this_direction==True:
                    mat_sliding.append([ax,dir])
        sliding_directions.append(mat_sliding)
    return sliding_directions

def get_milling_path_length(self,path):
    return "x"

def get_same_height_neighbors(hfield,inds):
    dim = len(hfield)
    val = hfield[tuple(inds[0])]
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if np.all(ind2>=0) and np.all(ind2<dim):
                    val2 = hfield[tuple(ind2)]
                    if val2==val:
                        unique = True
                        for ind3 in new_inds:
                            if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                                unique = False
                                break
                        if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_same_height_neighbors(hfield,new_inds)
    return new_inds

class Selection:
    def __init__(self,parent,x,y,n):
        self.parent = parent
        self.x = x
        self.y = y
        self.x2 = x
        self.y2 = y
        self.n = n
        self.pos = np.array([self.x,self.y])
        self.active = False
        self.start_pos = self.current_pos = None
        self.start_height = self.current_height = None
        self.val = 0
        self.clicked_mouse_pos = [0,0]
        self.refresh = False
        self.mouse_pressed = False
        self.selection_mode = False
        self.nc = 2
        if parent.shift:
            self.selected_faces = [self.pos]
        else:
            self.selected_faces = get_same_height_neighbors(parent.height_field,[self.pos])

    def activate(self,mouse_pos):
        self.active=True
        self.selection_mode = False
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.start_height = self.parent.height_field[self.x][self.y]
        Geometries.create_and_buffer_indicies(self.parent)

    def add(self,mouse_pos):
        self.clicked_mouse_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        unique = True
        for face in self.selected_faces:
            if face[0]==self.x2 and face[1]==self.y2:
                unique = False
                break
        if unique:
            self.selected_faces.append(np.array([self.x2,self.y2]))
            Geometries.create_and_buffer_indicies(self.parent)

    def set_current(self,x,y):
        self.x2 = x
        self.y2 = y

    def edit(self,mouse_pos,screen_xrot,screen_yrot):
        self.current_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.current_height = self.start_height
        ## Mouse vector
        mouse_vec = self.current_pos-self.start_pos
        mouse_vec[0] = mouse_vec[0]/800
        mouse_vec[1] = mouse_vec[1]/800
        ## Sliding direction vector
        sdir_vec = [0,0,0]
        ax = self.parent.sliding_direction[0]
        dir = self.parent.sliding_direction[1]
        sdir_vec[ax] = (2*dir-1)*self.parent.voxel_size
        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
        sdir_vec = np.dot(sdir_vec,rot_x*rot_y)
        sdir_vec = np.delete(sdir_vec,2) # delete Z-value
        ## Calculate angle between mouse vector and sliding direction vector
        cosang = np.dot(mouse_vec, sdir_vec) # Negative / positive depending on direction
        #sinang = linalg.norm(np.cross(mouse_vec, joint_vec))
        #ang = math.degrees(np.arctan2(sinang, cosang))
        val = int(linalg.norm(mouse_vec)/linalg.norm(sdir_vec)+0.5)
        if cosang!=None and cosang>0: val = -val
        if self.start_height + val>self.parent.dim: val = self.parent.dim-self.start_height
        elif self.start_height+val<0: val = -self.start_height
        self.current_height = self.start_height + val
        self.val = int(val)

class ElementProperties:
    def __init__(self, draw_type, count, start_index, n):
        self.draw_type = draw_type
        self.count = count
        self.start_index = start_index
        self.n = n

class Geometries:
    def __init__(self):
        self.shift = False
        self.outline_selected_faces = None
        self.joint_type = "I"
        self.fixed_sides = get_fixed_sides(self.joint_type)
        self.sliding_direction = [2,0]
        self.dim = 3
        self.fab_geometry = False
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.5*self.component_size
        self.rad = 0.015 #milling bit radius
        self.dep = 0.015 #milling depth
        self.height_field = get_random_height_field(self.dim)
        self.pre_height_field = self.height_field
        self.slides = []
        self.connected = True
        self.Selected = None
        self.fab_a = self.fab_b = None
        self.voxel_matrix = None
        self.voxel_matrix_with_sides = None
        self.voxel_matrix_connected = None
        self.voxel_matrix_unconnected = None
        self.voxel_matrix_unbridged_1 = None
        self.voxel_matrix_unbridged_2 = None
        self.ifA = self.ifeA = self.ilA = None
        self.ifB = self.ifeB = self.ilB = None
        self.ifuA = self.ifuB = None
        self.iopen = None
        self.iftA = self.ifntA = self.iftB = self.ifntB = None
        self.v_faces_A = self.v_faces_B = None
        self.vertex_no_info = 8
        self.connected_A = self.connected_B = True
        self.bridged_A = self.bridged_B = True
        self.voxel_matrix_from_height_field()
        self.vn = self.ven = self.vmA = self.vmB = None
        image = Image.open("textures/end_grain.jpg")
        self.img_data = np.array(list(image.getdata()), np.uint8)
        self.milling_path_A = self.milling_path_B = None

        #Element property class variables
        self.f_not_ends_a = self.f_ends_a = None
        self.f_connected_a = self.f_unconnected_a = None
        self.faces_all_a = None
        self.lines_a = None
        self.f_not_ends_b = self.f_ends_b =None
        self.f_connected_b = self.f_unconnected_b = None
        self.faces_all_b = None
        self.lines_b = None
        self.faces_unbridged_1_a = self.faces_unbridged_2_a = None
        self.faces_unbridged_1_b = self.faces_unbridged_2_b = None
        self.lines_open_a = self.lines_open_b = None
        self.faces_not_tops_a = self.faces_tops_a = None
        self.faces_not_tops_b = self.faces_tops_b = None
        self.arrow_lines_a = self.arrow_faces_a = None
        self.arrow_lines_b = self.arrow_faces_b = None
        self.arrow_other_lines_a = self.arrow_other_faces_a = None
        self.arrow_other_lines_b = self.arrow_other_faces_b = None
        self.lines_mill_a = self.lines_mill_b = None

        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def save(self):
        np.save("data/saved_height_field.npy",self.height_field)
        np.save("data/saved_voxel_matrix.npy",self.voxel_matrix)
        np.save("data/saved_voxel_matrix_with_fixed_sides.npy",self.voxel_matrix_with_sides)
        np.save("data/saved_fixed_sides.npy",self.fixed_sides)

    def load(self):
        self.height_field = np.load("data/saved_height_field.npy")
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
        self.voxel_matrix_with_sides = add_fixed_sides(self.voxel_matrix, self.fixed_sides)
        # check connections
        self.voxel_matrix_connected = self.voxel_matrix.copy()
        self.voxel_matrix_unconnected = None
        self.connected_A = is_connected(self.voxel_matrix_with_sides,0)
        self.connected_B = is_connected(self.voxel_matrix_with_sides,1)
        self.bridged_A=True
        self.bridged_B=True
        if not self.connected_A or not self.connected_B:
            self.seperate_unconnected()
            if self.joint_type=="T" or self.joint_type=="X":
                voxel_matrix_connected_with_sides = add_fixed_sides(self.voxel_matrix_connected, self.fixed_sides)
                self.bridged_A = is_connected(voxel_matrix_connected_with_sides,0)
                if self.bridged_A==False:
                    self.voxel_matrix_unbridged_A_1, self.voxel_matrix_unbridged_A_2 = self.seperate_unbridged(0)
                self.bridged_B = is_connected(voxel_matrix_connected_with_sides,1)
                if self.bridged_B==False:
                    self.voxel_matrix_unbridged_B_1, self.voxel_matrix_unbridged_B_2 = self.seperate_unbridged(1)
        self.evaluate_joint()

    def seperate_unconnected(self):
        connected_mat = np.zeros((self.dim,self.dim,self.dim))-1
        unconnected_mat = np.zeros((self.dim,self.dim,self.dim))-1
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    connected = False
                    ind = [i,j,k]
                    val = self.voxel_matrix[tuple(ind)]
                    connected = is_connected_to_fixed_side(np.array([ind]),self.voxel_matrix,self.fixed_sides[int(val)])
                    if connected: connected_mat[tuple(ind)] = val
                    else: unconnected_mat[tuple(ind)] = val
        self.voxel_matrix_connected = connected_mat
        self.voxel_matrix_unconnected = unconnected_mat

    def seperate_unbridged(self, n):
        unbridged_1 = np.zeros((self.dim,self.dim,self.dim))-1
        unbridged_2 = np.zeros((self.dim,self.dim,self.dim))-1
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    ind = [i,j,k]
                    val = self.voxel_matrix[tuple(ind)]
                    if val!=n: continue
                    conn_1 = is_connected_to_fixed_side(np.array([ind]),self.voxel_matrix,[self.fixed_sides[n][0]])
                    conn_2 = is_connected_to_fixed_side(np.array([ind]),self.voxel_matrix,[self.fixed_sides[n][1]])
                    if conn_1: unbridged_1[tuple(ind)] = val
                    if conn_2: unbridged_2[tuple(ind)] = val
        return unbridged_1, unbridged_2

    def update_sliding_direction(self,sliding_direction_):
        self.sliding_direction = sliding_direction_
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def init_selection(self,x,y,n):
        self.Selected = Selection(self,x,y,n)

    def finalize_selection(self):
        self.pre_height_field = self.height_field.copy()
        if self.Selected.val!=0:
            for ind in self.Selected.selected_faces:
                self.height_field[tuple(ind)] = self.Selected.current_height
            self.voxel_matrix_from_height_field()
            self.voxel_matrix_with_sides = add_fixed_sides(self.voxel_matrix, self.fixed_sides)
            self.create_and_buffer_indicies()
        self.Selected = None

    def undo(self):
        print("undoing...")
        temp = self.height_field.copy()
        self.height_field = self.pre_height_field.copy()
        self.pre_height_field = temp
        self.voxel_matrix_from_height_field()
        self.voxel_matrix_with_sides = add_fixed_sides(self.voxel_matrix, self.fixed_sides)
        self.create_and_buffer_indicies()

    def randomize_height_field(self):
        self.pre_height_field = self.height_field.copy()
        self.height_field = get_random_height_field(self.dim)
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def clear_height_field(self):
        self.pre_height_field = self.height_field.copy()
        self.height_field = np.zeros((self.dim,self.dim))
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def update_joint_type(self,joint_type_):
        self.joint_type = joint_type_
        if self.joint_type=="X" and self.sliding_direction==[2,0]:
            self.update_sliding_direction([1,0])
        self.fixed_sides = get_fixed_sides(self.joint_type)
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()
        self.evaluate_joint()

    def update_dimension(self,dim_): # not always working OS error
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
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()
        self.evaluate_joint()

    def update_number_of_components(self,num):
        self.nc = num
        self.create_and_buffer_indicies()        

    def create_and_buffer_vertices(self):

        ### VERTICIES FOR JOINT ###
        self.v_A = joint_vertices(self,"A",0.0,0.0,0.0)
        self.v_B = joint_vertices(self,"B",0.0,0.0,0.0)

        ### EXTRA VERTICIES FOR ROUNDED CORNERS OF FABRICATED JOINT ###
        ve_A = joint_extra_vertices(self,"A",0.0,0.0,0.0)
        ve_B = joint_extra_vertices(self,"B",0.0,0.0,0.0)

        ### VERTICIES FOR ARROWS DISPLAYING SLIDING DIRECTIONS ###
        va = arrow_vertices(self)

        ### VERTICIES FOR MILLING PATHS ###
        vm_A = milling_path_vertices(self,0)
        self.fab_a = Fabrication(vm_A,self.component_size,self.sliding_direction,self.joint_type,0)
        vm_B = milling_path_vertices(self,1)
        self.fab_b = Fabrication(vm_B,self.component_size,self.sliding_direction,self.joint_type,1)

        vertices_all = np.concatenate([self.v_A,  self.v_B, ve_A, ve_B, va, vm_A, vm_B])

        try:
            glBufferData(GL_ARRAY_BUFFER, 6*len(vertices_all), vertices_all, GL_DYNAMIC_DRAW)
        except:
            print("--------------------------ERROR IN ARRAY BUFFER WRAPPER -------------------------------------")
            print("All vertices:",vertices_all)

        # vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0)) #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12)) #color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24)) #texture
        glEnableVertexAttribArray(2)
        texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 400, 400, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data)

        self.vn =  int(len(self.v_A)/8)
        self.ven = int(len(ve_A)/8)
        self.var = int(len(va)/8)
        self.vmA = int(len(vm_A)/8)
        self.vmB = int(len(vm_B)/8)
        self.vmA_start = 2*self.vn+2*self.ven+self.var
        self.vmB_start = self.vmA_start+self.vmA

    def create_and_buffer_indicies(self):
        all_indices = []

        ### INDICES FOR COMPONENT A ###

        # Faces of not ends grains and end grains
        if not self.fab_geometry:
            self.f_not_ends_a,self.f_ends_a,self.f_connected_a,all_indices = joint_face_indicies(self,
                                    all_indices,self.voxel_matrix_connected, self.fixed_sides[0],0,0)
        else:
            self.f_not_ends_a,self.f_ends_a,self.f_connected_a,all_indices = joint_face_fab_indicies(self,
                                    all_indices,self.voxel_matrix_connected, self.fixed_sides[0],0,0,2*self.vn)
        count_all = self.f_connected_a.count
        # Faces of unconnected
        if not self.connected_A:
            fnend,fend,self.f_unconnected_a,all_indices = joint_face_indicies(self,
                                all_indices,self.voxel_matrix_unconnected,[],0,0)
            count_all += self.f_unconnected_a.count
        # All faces
        self.faces_all_a = ElementProperties(self.f_ends_a.draw_type, count_all, self.f_not_ends_a.start_index, 0)
        # Lines
        if not self.fab_geometry:
            self.lines_a, all_indices = joint_line_indicies(self,all_indices,0,0)
        else:
            self.lines_a, all_indices = joint_line_fab_indicies(self,all_indices,0,0,2*self.vn)

        ### INDICES FOR COMPONENT B ###

        # Faces of not ends grains and end grain
        if not self.fab_geometry:
            self.f_not_ends_b,self.f_ends_b,self.f_connected_b,all_indices = joint_face_indicies(self,
                                    all_indices, self.voxel_matrix_connected,self.fixed_sides[1],1,self.vn)
        else:
            self.f_not_ends_b,self.f_ends_b,self.f_connected_b,all_indices = joint_face_fab_indicies(self,
                                    all_indices, self.voxel_matrix_connected,self.fixed_sides[1],1,self.vn,2*self.vn+self.ven)
        count_all = self.f_connected_b.count
        # Faces of unconnected
        if not self.connected_B:
            fnend,fend,self.f_unconnected_b,all_indices = joint_face_indicies(self,
                                all_indices, self.voxel_matrix_unconnected,[],1,self.vn)
            count_all += self.f_unconnected_b.count
        # All faces
        self.faces_all_b = ElementProperties(self.f_ends_b.draw_type, count_all, self.f_not_ends_b.start_index, 1)
        # Lines
        if not self.fab_geometry:
            self.lines_b,all_indices = joint_line_indicies(self,all_indices,1,self.vn)
        else:
            self.lines_b, all_indices = joint_line_fab_indicies(self,all_indices,1,self.vn,2*self.vn+self.ven)

        ### INDICES FOR UNBRIDGED COMPONENTS ###
        if not self.bridged_A:
            fnend,fend,self.faces_unbridged_1_a,all_indices = joint_face_indicies(self,
                                all_indices,self.voxel_matrix_unbridged_A_1,[self.fixed_sides[0][0]],0,self.vn)
            fnend,fend,self.faces_unbridged_2_a,all_indices = joint_face_indicies(self,
                                all_indices, self.voxel_matrix_unbridged_A_2,[self.fixed_sides[0][1]],0,self.vn)
        if not self.bridged_B:
            fnend,fend,self.faces_unbridged_1_b,all_indices = joint_face_indicies(self,
                                all_indices,self.voxel_matrix_unbridged_B_1,[self.fixed_sides[1][0]],1,self.vn)
            fnend,fend,self.faces_unbridged_2_b,all_indices = joint_face_indicies(self,
                                all_indices,self.voxel_matrix_unbridged_B_2,[self.fixed_sides[1][1]],1,self.vn)

        ### INDICES OPENING LINES ###
        self.lines_open_a, all_indices = open_line_indicies(self,all_indices,0,0)
        self.lines_open_b, all_indices = open_line_indicies(self,all_indices,1,self.vn)

        ### INDICES OF TOP FACES FOR PICKING ###
        self.faces_not_tops_a, self.faces_tops_a, all_indices = joint_top_face_indices(self,all_indices,0,0)
        self.faces_not_tops_b, self.faces_tops_b, all_indices = joint_top_face_indices(self,all_indices,1,self.vn)

        ### INDICES OUTLINE OF ACTIVE SELECTED FACE OF MANY FACES ###
        if self.Selected!=None and self.Selected.active:
            self.outline_selected_faces, all_indices = joint_selected_top_line_indices(self,self.Selected,all_indices)

        ### Indicies for arrows ###
        self.arrow_lines_a, self.arrow_faces_a, all_indices = arrow_indices(self,
                                all_indices,[self.sliding_direction],0,2*self.vn+2*self.ven)
        self.arrow_lines_b, self.arrow_faces_b, all_indices = arrow_indices(self,
                                all_indices,[self.sliding_direction],1,2*self.vn+2*self.ven)

        self.arrow_other_lines_a, self.arrow_other_faces_a, all_indices = arrow_indices(self,
                                all_indices,self.slides[0],0,2*self.vn+2*self.ven)
        self.arrow_other_lines_b, self.arrow_other_faces_b, all_indices = arrow_indices(self,
                                all_indices,self.slides[0],1,2*self.vn+2*self.ven)

        ### INDICES FOR MILLING PATHS ###
        self.lines_mill_a, all_indices = milling_path_indices(self, all_indices, self.vmA, self.vmA_start, 0)
        self.lines_mill_b, all_indices = milling_path_indices(self, all_indices, self.vmB, self.vmB_start, 1)

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(all_indices), all_indices, GL_DYNAMIC_DRAW)

    def evaluate_joint(self):
        connected_A = is_connected(self.voxel_matrix_with_sides,0)
        connected_B = is_connected(self.voxel_matrix_with_sides,1)
        if connected_A and connected_B: self.connected = True
        else: self.connected=False
        self.slides = get_sliding_directions(self.voxel_matrix_with_sides)
        if len(self.slides[0])!=len(self.slides[1]): print("Sliding calculation error")
        friciton = get_friction(self.voxel_matrix_with_sides,self.slides)
        #path_length = get_milling_path_length(milling_path_A,milling_path_B)
        #print("\n---JOINT EVALUATION---")
        #print("Connected:",self.connected)
        #print("Slidings: ", len(slides[0]))
        #print("Friction: ", friciton)
        #print("Milling path:", path_length, "meters")
