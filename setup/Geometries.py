from OpenGL.GL import *
import numpy as np
from numpy import linalg
import random
import math
import pyrr
from Selection import Selection
from Fabrication import Fabrication
from Fabrication import MillVertex
from Fabrication import RegionVertex
from Fabrication import RoughPixel
from Evaluation import Evaluation
from Buffer import Buffer
from Buffer import ElementProperties
import copy

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

def check_free_sides(self,ind,dir_ax,off_ax,n):
    dir_sides = [1,1] #free
    off_sides = [1,1]
    #
    # dir
    if ind[dir_ax]>0:
        pind = ind.copy()
        pind[dir_ax] += -1
        pval = self.voxel_matrix[tuple(pind)]
        if int(pval)==int(n): dir_sides[0]=0
    elif [dir_ax,0] in self.fixed_sides[n]:
        dir_sides[0]=0
    #
    if ind[dir_ax]<self.dim-1:
        pind = ind.copy()
        pind[dir_ax] += 1
        pval = self.voxel_matrix[tuple(pind)]
        if int(pval)==int(n): dir_sides[1]=0
    elif [dir_ax,1] in self.fixed_sides[n]:
        dir_sides[1]=0
    #
    # off
    if ind[off_ax]>0:
        pind = ind.copy()
        pind[off_ax] += -1
        pval = self.voxel_matrix[tuple(pind)]
        if int(pval)==int(n): off_sides[0]=0
    elif [off_ax,0] in self.fixed_sides[n]:
        off_sides[0]=0
    #
    if ind[off_ax]<self.dim-1:
        pind = ind.copy()
        pind[off_ax] += 1
        pval = self.voxel_matrix[tuple(pind)]
        if int(pval)==int(n): off_sides[1]=0
    elif [off_ax,1] in self.fixed_sides[n]:
        off_sides[1]=0
    #
    dir_sides = np.array(dir_sides)
    off_sides = np.array(off_sides)
    return dir_sides,off_sides

def get_layered_vertices(self,outline,n,i,ax,no_z,dep):
    verts = []
    mverts = []

    r = g = b = tx = ty = 0.0

    # add startpoint
    start_vert = [outline[0].x,outline[0].y,outline[0].z]
    safe_height = outline[0].pt[ax]-(2*n-1)*i*self.voxel_size-0.2*(2*n-1)*self.voxel_size
    start_vert[ax] = safe_height
    mverts.append(MillVertex(start_vert))
    verts.extend([start_vert[0],start_vert[1],start_vert[2],r,g,b,tx,ty])

    # add layers with Z-height
    for num in range(no_z):
        for mv in outline:
            pt = [mv.x,mv.y,mv.z]
            pt[ax] += num*(2*n-1)*dep
            if mv.is_arc:
                ctr = [mv.arc_ctr[0],mv.arc_ctr[1],mv.arc_ctr[2]]
                ctr[ax] += num*(2*n-1)*dep
                mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
            else:
                mverts.append(MillVertex(pt))
            verts.extend([pt[0],pt[1],pt[2],r,g,b,tx,ty])
        outline.reverse()

    # add enpoint
    end_vert = [outline[0].x,outline[0].y,outline[0].z]
    end_vert[ax] = safe_height
    mverts.append(MillVertex(end_vert))
    verts.extend([end_vert[0],end_vert[1],end_vert[2],r,g,b,tx,ty])

    return verts,mverts

def get_diff_neighbors(mat2,inds,val):
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if ind2[ax]>=0 and ind2[ax]<mat2.shape[ax]:
                    val2 = mat2[tuple(ind2)]
                    if val2==val or val2==-1: continue
                    unique = True
                    for ind3 in new_inds:
                        if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                            unique = False
                            break
                    if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_diff_neighbors(mat2,new_inds,val)
    return new_inds

def get_neighbors_in_out(ind,reg_inds,lay_mat,n):
    # 0 = in region
    # 1 = outside region, block
    # 2 = outside region, free
    in_out = []
    for add0 in range(-1,1,1):
        temp = []
        for add1 in range(-1,1,1):
            type = -1

            # Define neighbor index to test
            nind = [ind[0]+add0,ind[1]+add1]

            # Check if this index is in the list of region-included indices
            for rind in reg_inds:
                if rind[0]==nind[0] and rind[1]==nind[1]:
                    type = 0 # in region
                    break

            if type!=0:
                # If there are out of bound indices they are free
                if np.any(np.array(nind)<0) or nind[0]>=lay_mat.shape[0] or nind[1]>=lay_mat.shape[1]:
                    type = 2 # free
                elif lay_mat[tuple(nind)]<0:
                    type = 2 # free
                else: type = 1 # blocked
            temp.append(type)
        in_out.append(temp)
    return in_out

def filleted_points(pt,one_voxel,off_dist,ax,n):
    ##
    addx = (one_voxel[0]*2-1)*off_dist
    addy = (one_voxel[1]*2-1)*off_dist
    ###
    pt1 = pt.copy()
    add = [addx,-addy]
    add.insert(ax,0)
    pt1[0] += add[0]
    pt1[1] += add[1]
    pt1[2] += add[2]
    #
    pt2 = pt.copy()
    add = [-addx,addy]
    add.insert(ax,0)
    pt2[0] += add[0]
    pt2[1] += add[1]
    pt2[2] += add[2]
    #
    if n%2==1: pt1,pt2 = pt2,pt1
    return [pt1,pt2]

def is_additional_outer_corner(self,rv,ind,ax,n):
    outer_corner = False
    if rv.region_count==1 and rv.block_count==1:
        other_fixed_sides = self.fixed_sides.copy()
        other_fixed_sides.pop(n)
        for sides in other_fixed_sides:
            for oax,odir in sides:
                if oax==ax: continue
                axes = [0,0,0]
                axes[oax] = 1
                axes.pop(ax)
                oax = axes.index(1)
                not_oax = axes.index(0)
                if rv.ind[oax]==odir*self.dim:
                    if rv.ind[not_oax]!=0 and rv.ind[not_oax]!=self.dim:
                        outer_corner = True
                        break
            if outer_corner: break
    return outer_corner

def rough_milling_path(self,rough_pixs,lay_num,n):
    mvertices = []

    no_lanes = 2+math.ceil((self.fab.real_voxel_size-2*self.fab.dia)/self.fab.dia)
    lane_width = (self.voxel_size-self.fab.vdia)/(no_lanes-1)

    # Defines axes
    ax = self.sliding_directions[n][0][0] # mill bit axis
    dir = self.sliding_directions[n][0][1]
    axes = [0,1,2]
    axes.pop(ax)
    dir_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis

    # create offset direction vectors
    off_vec = [0,0,0]
    off_vec[off_ax]=1
    off_vec = np.array(off_vec)
    dir_vec = [0,0,0]
    dir_vec[dir_ax]=1
    dir_vec = np.array(dir_vec)

    # get top ones to cut out
    for pix in rough_pixs:
        mverts = []
        if pix.outside: continue
        if no_lanes<=2:
            if pix.neighbors[0][0]==1 and pix.neighbors[0][1]==1: continue
            elif pix.neighbors[1][0]==1 and pix.neighbors[1][1]==1: continue
        pix_end = pix

        # check that there is no previous same
        nind = pix.ind_abs.copy()
        nind[dir_ax] -=1
        found = False
        for pix2 in rough_pixs:
            if pix2.outside: continue
            if pix2.ind_abs[0]==nind[0] and pix2.ind_abs[1]==nind[1]:
                if pix.neighbors[1][0]==pix2.neighbors[1][0]:
                    if pix.neighbors[1][1]==pix2.neighbors[1][1]:
                        found = True
                        break
        if found: continue

        # find next same
        for i in range(self.dim):
            nind = pix.ind_abs.copy()
            nind[0] +=i
            found = False
            for pix2 in rough_pixs:
                if pix2.outside: continue
                if pix2.ind_abs[0]==nind[0] and pix2.ind_abs[1]==nind[1]:
                    if pix.neighbors[1][0]==pix2.neighbors[1][0]:
                        if pix.neighbors[1][1]==pix2.neighbors[1][1]:
                            found = True
                            pix_end = pix2
                            break
            if found==False: break

        # start
        ind = list(pix.ind_abs)
        ind.insert(ax,(self.dim-1)*(1-dir)+(2*n-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
        add = [0,0,0]
        add[ax] = 1-dir
        i_pt = get_index(ind,add,self.dim)
        pt1 = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
        #end
        ind = list(pix_end.ind_abs)
        ind.insert(ax,(self.dim-1)*(1-dir)+(2*n-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
        add = [0,0,0]
        add[ax] = 1-dir
        add[dir_ax] = 1
        i_pt = get_index(ind,add,self.dim)
        pt2 = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)

        ### REFINE THIS FUNCTION
        dir_add1 = pix.neighbors[dir_ax][0]*3*self.fab.vrad*dir_vec
        dir_add2 = -pix_end.neighbors[dir_ax][1]*3*self.fab.vrad*dir_vec

        pt1 = pt1+self.fab.vrad*off_vec+dir_add1
        pt2 = pt2+self.fab.vrad*off_vec+dir_add2
        for i in range(no_lanes):
            # skip lane if on bloked side in off direction
            if pix.neighbors[1][0]==1 and i==0: continue
            elif pix.neighbors[1][1]==1 and i==no_lanes-1: continue

            ptA = pt1+lane_width*off_vec*i
            ptB = pt2+lane_width*off_vec*i
            pts = [ptA,ptB]
            if i%2==1: pts.reverse()
            for pt in pts: mverts.append(MillVertex(pt))
        mvertices.append(mverts)
    return mvertices

def layer_mat_from_cube(self,ax,dir,lay_num,n):
    mat = np.ndarray(shape=(self.dim,self.dim), dtype=int)
    for i in range(self.dim):
        for j in range(self.dim):
            ind = [i,j]
            zval = (self.dim-1)*(1-dir)+(2*n-1)*lay_num
            ind.insert(ax,zval)
            mat[i][j]=int(self.voxel_matrix[tuple(ind)])
    return mat

def pad_layer_mat_with_fixed_sides(self,mat,ax):
    pad_loc = [[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1]]
    for n2 in range(len(self.fixed_sides)):
        for oax,odir in self.fixed_sides[n2]:
            if oax==ax: continue
            axes = [0,0,0]
            axes[oax] = 1
            axes.pop(ax)
            oax = axes.index(1)
            pad_loc[oax][odir] = 1
            pad_val[oax][odir] = n2
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)
    # take care of -1 corners
    for fixed_sides_1 in self.fixed_sides:
        for fixed_sides_2 in self.fixed_sides:
            for ax1,dir1 in fixed_sides_1:
                if ax1==ax: continue
                axes = [0,0,0]
                axes[ax1] = 1
                axes.pop(ax)
                ax1 = axes.index(1)
                for ax2,dir2 in fixed_sides_2:
                    if ax2==ax: continue
                    axes = [0,0,0]
                    axes[ax2] = 1
                    axes.pop(ax)
                    ax2 = axes.index(1)
                    if ax1==ax2: continue
                    ind = [0,0]
                    ind[ax1] = dir1*(mat.shape[ax1]-1)
                    ind[ax2] = dir2*(mat.shape[ax2]-1)
                    mat[tuple(ind)] = -1
    return mat,pad_loc

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
                tx = (math.cos(self.grain_rotation)*tex_coords[0]-math.sin(self.grain_rotation)*tex_coords[1])/self.dim
                ty = (math.sin(self.grain_rotation)*tex_coords[0]+math.cos(self.grain_rotation)*tex_coords[1])/self.dim
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

def joint_extra_vertices(self,n):
    vertices = []
    r = g = b = 0.0
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
                    add = [self.fab.vrad*u,0]
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
                    add = [0,self.fab.vrad*v]
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
    milling_vertices = []

    # Check that the milling bit is not too large for the voxel size
    if self.fab.real_voxel_size<self.fab.dia: print("Could not generate milling path. The milling bit is too large")

    # Calculate depth constants
    no_z = int(self.fab.real_voxel_size/self.fab.dep)
    dep = self.voxel_size/no_z

    # Defines axes
    ax = self.sliding_directions[n][0][0] # mill bit axis
    dir = self.sliding_directions[n][0][1]
    axes = [0,1,2]
    axes.pop(ax)
    dir_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis

    # Browse layers
    for lay_num in range(self.dim):

        # Create a 2D matrix of current layer
        lay_mat = layer_mat_from_cube(self,ax,dir,lay_num,n)

        # Pad 2d matrix with fixed sides
        lay_mat,pad_loc = pad_layer_mat_with_fixed_sides(self,lay_mat,ax)

        # Get/browse regions
        for reg_num in range(self.dim*self.dim):

            # Get indices of a region
            inds = np.argwhere((lay_mat!=-1) & (lay_mat!=n))
            if len(inds)==0: break
            reg_inds = get_diff_neighbors(lay_mat,[inds[0]],n)

            # Anaylize which voxels needs to be roguhly cut initially
            # 1. Add all open voxels in the region
            rough_inds = []
            for ind in reg_inds:
                rough_inds.append(RoughPixel(ind, lay_mat, pad_loc,self.dim,n))
            # 2. Produce rough milling path
            rough_paths = rough_milling_path(self,rough_inds,lay_num,n)
            for rough_path in rough_paths:
                if len(rough_path)>0:
                    verts,mverts = get_layered_vertices(self,rough_path,n,lay_num,ax,no_z,dep)
                    vertices.extend(verts)
                    milling_vertices.extend(mverts)

            # Overwrite detedted regin in original matrix
            for reg_ind in reg_inds: lay_mat[tuple(reg_ind)]=n

            # Make a list of all edge vertices of the outline of the region
            reg_verts = []
            for i in range(lay_mat.shape[0]+1):
                for j in range(lay_mat.shape[1]+1):
                    ind = [i,j]
                    neigbors = get_neighbors_in_out(ind,reg_inds,lay_mat,n)
                    flat_neigbors = [x for sublist in neigbors for x in sublist]
                    neigbors = np.array(neigbors)
                    flat_neigbors = np.array(flat_neigbors)
                    ind[0] -= pad_loc[0][0]
                    ind[1] -= pad_loc[1][0]
                    if np.any(flat_neigbors==0) and not np.all(flat_neigbors==0):
                        rv = RegionVertex(ind,neigbors,flat_neigbors)
                        reg_verts.append(rv)
                        if np.sum(flat_neigbors==0)==2 and np.sum(flat_neigbors==1)==2 and neigbors[0][1]==neigbors[1][0]: # diagonal
                            rv2 = RegionVertex(ind,neigbors,flat_neigbors)
                            reg_verts.append(rv2) # add extra vertex

            # Order the vertices to create an outline
            for isl_num in range(10):
                reg_ord_verts = []
                if len(reg_verts)==0: break

                #Make sure first item in region vertices is on blocked/free corner, or blocked
                closed = False
                first_i = None
                second_i = None
                for i,rv in enumerate(reg_verts):
                    if rv.block_count>0:
                        if rv.free_count>0: first_i=i
                        else: second_i = i
                if first_i==None:
                    first_i=second_i
                    closed = True
                if first_i==None: first_i=0
                reg_verts.insert(0,reg_verts[first_i])
                reg_verts.pop(first_i+1)

                # Start ordered vertices with the first item
                reg_ord_verts.append(reg_verts[0])
                reg_verts.remove(reg_verts[0])
                for i in range(len(reg_verts)):
                    found_next = False
                    #try all directions to look for next vertex
                    for vax in range(2):
                        for vdir in range(-1,2,2):
                            # check if there is an avaliable vertex
                            next_ind = reg_ord_verts[-1].ind.copy()
                            next_ind[vax]+=vdir
                            next_rv = None
                            for rv in reg_verts:
                                if rv.ind==next_ind:
                                    if len(reg_ord_verts)>2 and rv.ind==reg_ord_verts[-2].ind: break
                                    p_neig = reg_ord_verts[-1].neighbors.copy()
                                    # exception for diagonal layouts
                                    if reg_ord_verts[-1].dia:
                                        ppdiff = np.array(reg_ord_verts[-2].ind)-np.array(reg_ord_verts[-1].ind)
                                        pax = np.argwhere(ppdiff!=0)[0][0]
                                        pdir = ppdiff[pax]
                                        if pdir==-1: pdir=0
                                        pind0 = [pdir,pdir]
                                        pind0[pax] = 0
                                        p_neig[tuple(pind0)]=1
                                        pind0[pax] = 1
                                        p_neig[tuple(pind0)]=1
                                    # check so that it is not crossing a blocked region etc
                                    vaxval = int(0.5*(vdir+1))
                                    nind0 = [0,0]
                                    nind0[vax] = vaxval
                                    nind1 = [1,1]
                                    nind1[vax] = vaxval
                                    ne0 = p_neig[nind0[0]][nind0[1]]
                                    ne1 = p_neig[nind1[0]][nind1[1]]
                                    if ne0==1 or ne1==1: # at least one should be blocked
                                        if int(0.5*(ne0+1))!=int(0.5*(ne1+1)): # not crossing blocked material
                                            found_next=True
                                            if reg_ord_verts[-1].dia:
                                                reg_ord_verts[-1].neighbors = p_neig
                                                reg_ord_verts[-1].region_count = 1
                                                reg_ord_verts[-1].block_count = 3
                                                reg_ord_verts[-1].flat_neighbors = [x for sublist in p_neig for x in sublist]
                                            reg_ord_verts.append(rv)
                                            reg_verts.remove(rv)
                                            break
                            if found_next: break
                        if found_next: break
                    if found_next: continue

                # Offset vertices according to boundary condition (and remove if redundant)
                outline = []
                for i,rv in enumerate(list(reg_ord_verts)):

                    # remove vertices with neighbor count 2
                    if rv.region_count==2 and rv.free_count==2: continue
                    if rv.region_count==2 and rv.block_count==2: continue
                    if rv.block_count==0: continue
                    if rv.ind[0]<0 or rv.ind[0]>self.dim: continue
                    if rv.ind[1]<0 or rv.ind[1]>self.dim: continue

                    # add vertex information
                    ind = rv.ind.copy()
                    ind.insert(ax,(self.dim-1)*(1-dir)+(2*n-1)*lay_num)
                    add = [0,0,0]
                    add[ax] = 1-dir
                    i_pt = get_index(ind,add,self.dim)
                    pt = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)

                    # move vertex according to boundry condition
                    one_voxels = []
                    off_dists = []
                    if rv.block_count==1:
                        off_dists.append(-self.fab.vrad)
                        one_voxels.append(np.argwhere(rv.neighbors==1)[0])
                    if rv.region_count==1 and rv.free_count!=3:
                        off_dists.append(self.fab.vrad)
                        one_voxels.append(np.argwhere(rv.neighbors==0)[0])
                    addx = 0
                    addy = 0
                    for one_voxel,off_dist in zip(one_voxels,off_dists):
                        addx += (one_voxel[0]*2-1)*off_dist
                        addy += (one_voxel[1]*2-1)*off_dist
                    if len(one_voxels)!=0:
                        addx = addx/len(one_voxels)
                        addy = addy/len(one_voxels)
                    if rv.region_count==3: # or additional_outer_corner: #outer corner - add fillet
                        pt1 = pt.copy()
                        add = [addx,-addy]
                        add.insert(ax,0)
                        pt1[0] += add[0]
                        pt1[1] += add[1]
                        pt1[2] += add[2]
                        #
                        pt2 = pt.copy()
                        add = [-addx,addy]
                        add.insert(ax,0)
                        pt2[0] += add[0]
                        pt2[1] += add[1]
                        pt2[2] += add[2]
                        #
                        pts = [pt1,pt2]

                        # arc center
                        ctr = pt.copy()
                        add = [-addx,-addy]
                        add.insert(ax,0)
                        ctr[0] += add[0]
                        ctr[1] += add[1]
                        ctr[2] += add[2]

                        # Reorder pt1 and pt2 so that the first one is closer to the last item in the outline
                        if len(outline)>0:
                            ppt = outline[-1].pt
                            dist1 = np.linalg.norm(np.array(pt1)-np.array(ppt))
                            dist2 = np.linalg.norm(np.array(pt2)-np.array(ppt))
                            if dist1>dist2: pts.reverse()
                        elif len(reg_ord_verts)>i:
                            next_ind = reg_ord_verts[i+1].ind.copy()
                            next_ind.insert(ax,(self.dim-1)*(1-dir)+(2*n-1)*lay_num)
                            add = [0,0,0]
                            add[ax] = 1-dir
                            i_pt = get_index(next_ind,add,self.dim)
                            npt = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
                            dist1 = np.linalg.norm(np.array(pt1)-np.array(npt))
                            dist2 = np.linalg.norm(np.array(pt2)-np.array(npt))
                            if dist1<dist2: pts.reverse()
                        outline.append(MillVertex(pts[0],is_arc=True,arc_ctr=ctr))
                        outline.append(MillVertex(pts[1],is_arc=True,arc_ctr=ctr))
                    else: # other corner
                        add = [addx,addy]
                        add.insert(ax,0)
                        pt[0] += add[0]
                        pt[1] += add[1]
                        pt[2] += add[2]
                        outline.append(MillVertex(pt))

                # Get z height and extend vertices to global list
                if len(outline)>0:
                    if closed: outline.append(MillVertex(outline[0].pt))
                    verts,mverts = get_layered_vertices(self,outline,n,lay_num,ax,no_z,dep)
                    vertices.extend(verts)
                    milling_vertices.extend(mverts)

    # Format and return
    vertices = np.array(vertices, dtype = np.float32)
    return vertices, milling_vertices

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

def milling_path_indices(self,all_indices,count,start,n):
    indices = []
    for i in range(count):
        indices.append(int(start+i))
    # Format
    indices = np.array(indices, dtype=np.uint32)
    # Store
    #indices_prop = ElementProperties(GL_POINTS, len(indices), len(all_indices), n)
    indices_prop = ElementProperties(GL_LINE_STRIP, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def joint_face_fab_indicies(self,all_indices,mat,fixed_sides,n,offset,offset_extra):
    offset_extra = offset+offset_extra
    fab_ax = self.sliding_directions[n][0][0]
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

def joint_line_fab_indices(self,all_indices,n,offset,offset_extra):
    offset_extra = offset+offset_extra
    fixed_sides = self.fixed_sides[n]
    fab_ax = self.sliding_directions[n][0][0]
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

class Geometries:
    def __init__(self):
        self.joint_type = "T"
        self.grain_rotation = 0
        self.fab_geometry = False
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

    def update_joint_type(self,joint_type,noc):
        self.joint_type = joint_type
        if self.noc!=noc:
            if noc==3: self.sliding_directions = [ [[2,0]], [[2,1],[1,1]], [[1,0]] ]
            else: self.sliding_directions = [ [[2,0]], [[2,1]] ]
        self.noc = noc
        if self.joint_type=="X" and self.sliding_directions[0][0]==[2,0] and self.noc==2:
            self.update_sliding_direction([[[1,0]],[[1,1]]])
        if self.joint_type=="I" and self.noc==3:
            self.joint_type="L"

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

    def create_vertices(self, milling_path=False):

        self.jverts = []
        self.everts = []
        self.mverts = []
        self.gcodeverts = []

        for n in range(self.noc):
            self.jverts.append(joint_vertices(self,n))


        if self.fab_geometry:
            for n in range(self.noc):
                self.everts.append(joint_extra_vertices(self,n))

        if milling_path:
            for n in range(self.noc):
                mvs, gvs = milling_path_vertices(self,n)
                self.mverts.append(mvs)
                self.gcodeverts.append(gvs)

        va = arrow_vertices(self)

        # Combine
        jverts = np.concatenate(self.jverts)
        if self.fab_geometry:
            everts = np.concatenate(self.everts)
            jverts = np.concatenate([jverts,everts])


        if milling_path and len(self.mverts[0])>0:
            mverts = np.concatenate(self.mverts)
            self.vertices = np.concatenate([jverts, va, mverts])
        else:
            self.vertices = np.concatenate([jverts, va])

        self.vn =  int(len(self.jverts[0])/8)
        if self.fab_geometry: self.ven = int(len(self.everts[0])/8)
        else: self.ven = 0
        self.van = int(len(va)/8)
        if milling_path and len(self.mverts[0])>0:
            self.m_start = []
            mst = 2*self.vn +2*self.ven+self.van
            for n in range(self.noc):
                self.m_start.append(mst)
                mst += int(len(self.mverts[n])/8)
        Buffer.buffer_vertices(self.buff)

    def create_indices(self, milling_path=False):
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
            if self.fab_geometry:
                nend,end,con,all_inds = joint_face_fab_indicies(self,all_inds,
                        self.eval.voxel_matrix_connected,self.fixed_sides[n],n,n*self.vn,self.ven)
                all = con
                unbris = None
            else:
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


            #picking faces
            faces_not_tops, faces_tops, all_inds = joint_top_face_indices(self,all_inds,n,n*self.vn)

            #Lines
            if self.fab_geometry:
                lns,all_inds = joint_line_fab_indices(self,all_inds,n,n*self.vn,self.ven)
            else:
                lns,all_inds = joint_line_indices(self,all_inds,n,n*self.vn)

            for m in range(len(self.sliding_directions[n])):
                open,all_inds = open_line_indices(self,all_inds,n,self.sliding_directions[n][m],n*self.vn)
                self.indices_open_lines.append(open)

            #arrows
            larr, farr, all_inds = arrow_indices(self, all_inds,self.eval.slides[n],n,self.noc*self.vn+2*self.ven)
            arrows = [larr,farr]

            if milling_path and len(self.mverts[0])>0:
                mill,all_inds = milling_path_indices(self,all_inds,int(len(self.mverts[n])/8),self.m_start[n],n)

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
            if milling_path and len(self.mverts[0])>0:
                self.indices_milling_path.append(mill)


        #outline of selected faces
        if self.select.state==2:
            self.outline_selected_faces, all_inds = joint_selected_top_line_indices(self,self.select,all_inds)

        self.indices = all_inds
        Buffer.buffer_indices(self.buff)

    def save(self):
        np.save("data/saved_height_field.npy",self.height_field)
        np.save("data/saved_height_field2.npy",self.height_field2)
        np.save("data/saved_voxel_matrix.npy",self.voxel_matrix)
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
