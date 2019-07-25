from OpenGL.GL import *
import numpy as np
from numpy import linalg
import random
from PIL import Image
import math

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
            for step in range(2,4):
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
                    vertices.extend([x+add[0], y+add[1], z+add[2], r,g,b, tx,ty])
                for v in range(-1,2,2):
                    add = [0,self.rad*v]
                    add.insert(fab_ax,0)
                    vertices.extend([x+add[0], y+add[1], z+add[2], r,g,b, tx,ty])
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    return vertices

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

def joint_face_indicies(self,mat,fixed_sides,n,offset):
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
                step = 1
                if self.joint_type=="X" or (self.joint_type=="T" and n==1): step = 0
                off = 16*ax+8*dir+4*step
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

def joint_line_indicies(self,n,offset):
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
                        add0 = [0,0,0]
                        add1 = [0,0,0]
                        add1[ax] = 1
                        start_i = get_index(ind,add0,self.dim)
                        end_i = get_index(ind,add1,self.dim)
                        indices.extend([start_i,end_i])
    #Outline of component base
    start = d*d*d
    for ax,dir in fixed_sides:
        a1,b1,c1,d1 = get_corner_indices(ax,dir,self.dim)
        step = 1
        if self.joint_type=="X" or ( self.joint_type=="T" and n==1): step = 0
        off = 16*ax+8*dir+4*step
        a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
        indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
        indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    return indices

def joint_line_fab_indicies(self,n,offset,offset_extra):
    offset_extra = offset_extra-offset
    if comp=="A": n=0
    elif comp=="B": n=1
    fixed_sides = self.fixed_sides[n]
    fab_ax = self.sliding_direction[0]
    d = self.dim+1
    indices = []
    ## Matrix of rounded corners
    rounded_corners = []
    for i in range(self.dim+1):
        temp = []
        for j in range(self.dim+1):
            temp2 = []
            for k in range(self.dim+1):
                ind = [i,j,k]
                if ind[fab_ax]==self.dim: continue
                corner = []
                cnt,vals = line_neighbors(self,ind,fab_ax,n)
                diagonal = False
                if vals[0]==vals[3] or vals[1]==vals[2]: diagonal = True
                if cnt==1 or cnt==3 or (cnt==2 and diagonal):
                    if cnt!=1:
                        if cnt==2 and vals[0]==n: corner = [1,2]
                        elif cnt==2 and vals[1]==n: corner = [0,3]
                        elif cnt==3 and vals[0]!=n: corner = [0]
                        elif cnt==3 and vals[1]!=n: corner = [1]
                        elif cnt==3 and vals[2]!=n: corner = [2]
                        elif cnt==3 and vals[3]!=n: corner = [3]
                temp2.append(corner)
            temp.append(temp2)
        rounded_corners.append(temp)
    print(rounded_corners)
    corner_indices = [[0,2],[0,3],[1,2],[1,3]]
    ##
    for i in range(self.dim+1):
        for j in range(self.dim+1):
            for k in range(self.dim+1):
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
                        if ax==fab_ax:
                            if cnt==1: indices.extend([start_i,end_i])
                            else:
                                for corner in rounded_corners[i][j][k]:
                                    u,v = corner_indices[corner]
                                    indices.extend([offset_extra+4*start_i+u, offset_extra+4*end_i+u])
                                    indices.extend([offset_extra+4*start_i+v, offset_extra+4*end_i+v])
                                    indices.extend([offset_extra+4*start_i+u, offset_extra+4*start_i+v])
                                    indices.extend([offset_extra+4*end_i+u, offset_extra+4*end_i+v])
                        else:
                            # is the one above start a rounded corner?
                            # need more intelligent way to index the rounded corners....############
                            #if ind[fab_ax]<self.dim:
                            #    if len(rounded_corners[i][j][k])>0:
                            #        start_i = offset_extra+4*start_i+corner_indices[rounded_corners[i][j][k][0]][0]
                            indices.extend([start_i,end_i])

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

def open_line_indicies(self,n,offset):
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
    if n==1: dir = 1-dir
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*self.dim
            add[other_axes[1]] = y*self.dim
            add[ax] = heights[x][y]
            start = get_index(ind,add,self.dim)
            end = d*d*d+16*ax+8*dir+2*x+y
            indices.extend([start,end])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    return indices

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

def is_connected(mat,n):
    connected = False
    all_same = np.count_nonzero(mat==n) # Count number of ones in matrix
    if all_same>0:
        ind = tuple(np.argwhere(mat==n)[0]) # Pick a random one
        inds = get_all_same_connected(mat,[ind]) # Get all its neighbors (recursively)
        connected_same = len(inds)
        if connected_same==all_same: connected = True
    return connected

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

def milling_path_vertices(self,n,r,g,b):
    vertices = []
    # Parameters
    n_ = 1-n
    # Define number of steps
    no_lanes = math.ceil(2+(self.voxel_size-4*self.rad)/(2*self.rad))
    w = (self.voxel_size-2*self.rad)/(no_lanes-1)
    no_z = int(self.voxel_size/self.dep)
    self.dep = self.voxel_size/no_z
    # Texture coordinates (not used, just for conistent format)
    tx = 0.0
    ty = 0.0
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

def milling_path_indices(self,no,start):
    indices = []
    for i in range(start,start+no):
        indices.append(int(i))
    # Format
    indices = np.array(indices, dtype=np.uint32)
    return indices

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

class Geometries:
    def __init__(self):
        self.joint_type = "I"
        self.fixed_sides = get_fixed_sides(self.joint_type)
        self.sliding_direction = [2,0]
        self.dim = 3
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.55*self.component_size
        self.rad = 0.015 #milling bit radius
        self.dep = 0.015 #milling depth
        self.height_field = get_random_height_field(self.dim)
        self.fab_geometry = False
        self.show_milling_path = False
        self.connected = True
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
        self.v_faces_A = self.v_faces_B = None
        self.vertex_no_info = 8
        self.connected_A = self.connected_B = True
        self.bridged = True
        self.voxel_matrix_from_height_field()
        self.vn = self.ven = self.vmA = self.vmB = None
        image = Image.open("textures/end_grain.jpg")
        self.img_data = np.array(list(image.getdata()), np.uint8)
        self.milling_path_A = self.milling_path_B = None
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def save(self):
        np.save("saved_height_field.npy",self.height_field)
        np.save("saved_voxel_matrix.npy",self.voxel_matrix)
        np.save("saved_voxel_matrix_with_fixed_sides.npy",self.voxel_matrix_with_sides)
        np.save("saved_fixed_sides.npy",self.fixed_sides)

    def load(self):
        self.height_field = np.load("saved_height_field.npy")
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
        self.voxel_matrix_connected = self.voxel_matrix.copy()
        self.voxel_matrix_unconnected = None
        self.connected_A = is_connected(self.voxel_matrix_with_sides,0)
        self.connected_B = is_connected(self.voxel_matrix_with_sides,1)
        self.bridged=True
        if not self.connected_A or not self.connected_B:
            self.seperate_unconnected()
            if self.joint_type=="T" or self.joint_type=="X":
                voxel_matrix_connected_with_sides = add_fixed_sides(self.voxel_matrix_connected, self.fixed_sides)
                self.bridged = is_connected(voxel_matrix_connected_with_sides,1)
                if self.bridged==False: self.seperate_unbridged()
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

    def seperate_unbridged(self):
        unbridged_1 = np.zeros((self.dim,self.dim,self.dim))-1
        unbridged_2 = np.zeros((self.dim,self.dim,self.dim))-1
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    ind = [i,j,k]
                    val = self.voxel_matrix[tuple(ind)]
                    if val!=1: continue
                    conn_1 = is_connected_to_fixed_side(np.array([ind]),self.voxel_matrix,[self.fixed_sides[1][0]])
                    conn_2 = is_connected_to_fixed_side(np.array([ind]),self.voxel_matrix,[self.fixed_sides[1][1]])
                    if conn_1: unbridged_1[tuple(ind)] = val
                    if conn_2: unbridged_2[tuple(ind)] = val
        self.voxel_matrix_unbridged_1 = unbridged_1
        self.voxel_matrix_unbridged_2 = unbridged_2

    def update_sliding_direction(self,sliding_direction_):
        self.sliding_direction = sliding_direction_
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def update_height_field(self,i,j):
        self.height_field[i][j] = (self.height_field[i][j]+1)%4
        self.voxel_matrix_from_height_field()
        self.voxel_matrix_with_sides = add_fixed_sides(self.voxel_matrix, self.fixed_sides)
        #self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()

    def randomize_height_field(self):
        self.height_field = get_random_height_field(self.dim)
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_indicies()

    def clear_height_field(self):
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
        self.dim = dim_
        self.voxel_size = self.component_size/self.dim
        self.height_field = get_random_height_field(self.dim)
        self.voxel_matrix_from_height_field()
        self.create_and_buffer_vertices()
        self.create_and_buffer_indicies()
        self.evaluate_joint()

    def create_and_buffer_vertices(self):

        ### VERTICIES FOR JOINT ###
        self.v_A = joint_vertices(self,"A",0.0,0.0,0.0)
        self.v_B = joint_vertices(self,"B",0.0,0.0,0.0)

        ### EXTRA VERTICIES FOR ROUNDED CORNERS OF FABRICATED JOINT ###
        ve_A = joint_extra_vertices(self,"A",0.0,0.0,0.0)
        ve_B = joint_extra_vertices(self,"B",0.0,0.0,0.0)

        ### VERTICIES FOR MILLING PATHS ###
        self.milling_path_A = milling_path_vertices(self,0, 0.0,1.0,0.0)
        self.milling_path_B = milling_path_vertices(self,1, 0.0,0.8,1.0)

        vertices_all = np.concatenate([self.v_A,  self.v_B, ve_A, ve_B,
                                       self.milling_path_A, self.milling_path_B])

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
        self.vmA = int(len(self.milling_path_A)/8)
        self.vmB = int(len(self.milling_path_B)/8)

    def create_and_buffer_indicies(self):

        ### INDICES FOR JOINT ###
        faces_A,faces_end_A = joint_face_indicies(self,self.voxel_matrix_connected,self.fixed_sides[0],0,0)
        faces_ucA = []
        if not self.connected_A:
            faces_ucA,faces_end_ucA = joint_face_indicies(self,self.voxel_matrix_unconnected,[],0,0)
            faces_ucA = np.concatenate([faces_ucA,faces_end_ucA])
        lines_A = joint_line_indicies(self,0,0)

        faces_B,faces_end_B = joint_face_indicies(self,self.voxel_matrix_connected,self.fixed_sides[1],1,self.vn)
        faces_ucB = []
        if not self.connected_B:
            faces_ucB,faces_end_ucB = joint_face_indicies(self,self.voxel_matrix_unconnected,[],1,self.vn)
            faces_ucB = np.concatenate([faces_ucB,faces_end_ucB])
        lines_B = joint_line_indicies(self,1,self.vn)

        faces_ubB1 = []
        faces_ubB2 = []
        if not self.bridged:
            faces_ubB1, faces_end_ubB1 = joint_face_indicies(self,self.voxel_matrix_unbridged_1,[self.fixed_sides[1][0]],1,self.vn)
            if len(faces_ubB1)>0: faces_ubB1 = np.concatenate([faces_ubB1,faces_end_ubB1])
            faces_ubB2, faces_end_ubB2 = joint_face_indicies(self,self.voxel_matrix_unbridged_2,[self.fixed_sides[1][1]],1,self.vn)
            if len(faces_ubB2)>0: faces_ubB2 = np.concatenate([faces_ubB2,faces_end_ubB2])

        #lines_open_A = open_line_indicies(self,1,0)
        #lines_open_B = open_line_indicies(self,0,self.vn)

        ### INDICES FOR MILLING PATHS ###
        #lines_gpath_A = milling_path_indices(self, self.vmA, 2*self.vn+2*self.ven)
        #lines_gpath_B = milling_path_indices(self, self.vmB, 2*self.vn+2*self.ven+self.vmA)

        all_indices = np.concatenate([faces_A, faces_end_A])
        if len(faces_ucA)>0: all_indices = np.concatenate([all_indices, faces_ucA])
        all_indices = np.concatenate([all_indices, lines_A, faces_B, faces_end_B])
        if len(faces_ucB)>0: all_indices = np.concatenate([all_indices, faces_ucB])
        all_indices = np.concatenate([all_indices, lines_B])
        if len(faces_ubB1)>0: all_indices = np.concatenate([all_indices, faces_ubB1])
        if len(faces_ubB2)>0: all_indices = np.concatenate([all_indices, faces_ubB2])
                                      #lines_open_A, lines_open_B,
                                      #lines_gpath_A, lines_gpath_B])

        glBufferData(GL_ELEMENT_ARRAY_BUFFER, 4*len(all_indices), all_indices, GL_DYNAMIC_DRAW)

        self.ifA = len(faces_A)
        self.ifeA = len(faces_end_A)
        self.ilA = len(lines_A)
        self.ifB = len(faces_B)
        self.ifeB = len(faces_end_B)
        self.ilB = len(lines_B)
        self.ifuA = len(faces_ucA)
        self.ifuB = len(faces_ucB)
        self.ifubB1 = len(faces_ubB1)
        self.ifubB2 = len(faces_ubB2)
        #self.iopen = len(lines_open_A)
        #self.imA = len(lines_gpath_A)
        #self.imB = len(lines_gpath_B)

    def evaluate_joint(self):
        connected_A = is_connected(self.voxel_matrix_with_sides,0)
        connected_B = is_connected(self.voxel_matrix_with_sides,1)
        if connected_A and connected_B: self.connected = True
        else: self.connected=False
        slides = get_sliding_directions(self.voxel_matrix_with_sides)
        if len(slides[0])!=len(slides[1]): print("Sliding calculation error")
        friciton = get_friction(self.voxel_matrix_with_sides,slides)
        #path_length = get_milling_path_length(milling_path_A,milling_path_B)
        #print("\n---JOINT EVALUATION---")
        #print("Connected:",self.connected)
        #print("Slidings: ", len(slides[0]))
        #print("Friction: ", friciton)
        #print("Milling path:", path_length, "meters")
