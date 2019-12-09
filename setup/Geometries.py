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
        for i,mv in enumerate(outline):
            pt = [mv.x,mv.y,mv.z]
            pt[ax] += num*(2*n-1)*dep
            if mv.is_arc:
                ctr = [mv.arc_ctr[0],mv.arc_ctr[1],mv.arc_ctr[2]]
                ctr[ax] += num*(2*n-1)*dep
                mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
            else:
                mverts.append(MillVertex(pt))
            if i>0:
                pmv = outline[i-1]
            if i>0 and mv.is_arc and pmv.is_arc and np.array_equal(mv.arc_ctr,pmv.arc_ctr):
                ppt = [pmv.x,pmv.y,pmv.z]
                ppt[ax] += num*(2*n-1)*dep
                arc_pts = arc_points(ppt,pt,ctr,ax,10)
                for arc_pt in arc_pts: verts.extend([arc_pt[0],arc_pt[1],arc_pt[2],r,g,b,tx,ty])
            else:
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
    ax = self.sax # mill bit axis
    dir = self.fab_directions[n]
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
        dir_add1 = pix.neighbors[dir_ax][0]*2.5*self.fab.vrad*dir_vec
        dir_add2 = -pix_end.neighbors[dir_ax][1]*2.5*self.fab.vrad*dir_vec

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

def arc_points(st,en,ctr,ax,cnt):
    pts = []
    # numpy arrays
    st = np.array(st)
    en = np.array(en)
    ctr = np.array(ctr)
    # cw or ccw?
    clockwise = False
    vec1 = en-ctr
    vec1 = vec1/np.linalg.norm(vec1)
    zvec = np.array([0,0,0])
    zvec[ax] = 1
    xvec = np.cross(vec1,zvec)
    vec2 = st-ctr
    vec2 = vec2/np.linalg.norm(vec2)
    dist = np.linalg.norm(xvec-vec2)
    if dist>0: clockwise = True
    #
    vec = st-ctr
    vec = [vec[0],vec[1],vec[2]]
    vec.pop(ax)
    rad_x = vec[0]
    rad_y = vec[1]
    astep = 0.5*math.pi/cnt
    if (clockwise and ax==2) or (not clockwise and ax!=2): astep = -astep
    for i in range(1,cnt+1):
        x = rad_x*math.cos(astep*i)-rad_y*math.sin(astep*i)
        y = rad_y*math.cos(astep*i)+rad_x*math.sin(astep*i)
        rad = [x,y]
        rad.insert(ax,0)
        rad = np.array(rad)
        pt = ctr+rad
        pts.append(pt)
    return pts

def next_fixed_sides(fixed_sides):
    new_fixed_sides = []
    for ax in range(3):
        ax = 2-ax
        for dir in range(2):
            new_fixed_side = [ax,dir]
            unique = True
            for other_fixed_sides in fixed_sides:
                for oax,odir in other_fixed_sides:
                    if oax==ax and odir==dir:
                        unique=False
                        break
                if not unique: break
            if unique:
                new_fixed_sides.append(new_fixed_side)
                break
        if len(new_fixed_sides)>0:
            break
    return new_fixed_sides

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
    ax = self.sax # mill bit axis
    dir = self.fab_directions[n]
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
                if len(self.fixed_sides[n])==2: step = 1
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
    sax = self.sax
    sdir = self.fab_directions[n]
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
        if len(self.fixed_sides[n])==2: step = 1
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
    sax = self.sax
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
        if len(self.fixed_sides[n])==2: step = 1
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

def open_line_indices(self,all_indices,n,offset):
    indices = []
    dir = self.fab_directions[n]
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==self.sax))
    ind = np.array([0,0,0])
    d = self.dim-1
    heights = get_top_corner_heights(self.voxel_matrix,n,self.sax,dir)
    d = self.dim+1
    for x in range(2):
        for y in range(2):
            add = np.array([0,0,0])
            add[other_axes[0]] = x*self.dim
            add[other_axes[1]] = y*self.dim
            add[self.sax] = dir*self.dim
            start = get_index(ind,add,self.dim)
            end = d*d*d+24*self.sax+12*(1-dir)+2*x+y
            indices.extend([start,end])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    # Store
    indices_prop = ElementProperties(GL_LINES, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def chess_line_indices(self,all_indices,chess_verts,n,offset):
    indices = []
    for vert in chess_verts:
        add = [0,0,0]
        st = get_index(vert,add,self.dim)
        add[self.sax]=1
        en = get_index(vert,add,self.dim)
        indices.extend([st,en])
    # Format
    indices = np.array(indices, dtype=np.uint32)
    indices = indices + offset
    # Store
    indices_prop = ElementProperties(GL_LINES, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

def break_line_indices(self,all_indices,break_inds,n,offset):
    indices = []
    for ind3d in break_inds:
        add = [0,0,0]
        ind = get_index(ind3d,add,self.dim)
        indices.append(ind)
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

class Geometries:
    def __init__(self):
        self.grain_rotation = random.uniform(0,math.pi)
        self.noc = 2 #number of components
        self.fixed_sides = [[[2,0]], [[2,1]]]
        self.sax = 2 # sliding axis
        self.fab_directions = []
        for i in range(self.noc):
            if i==0: self.fab_directions.append(0)
            else: self.fab_directions.append(1)
        self.dim = 3
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.5*self.component_size
        self.vertex_no_info = 8
        self.height_fields = [get_random_height_field(self.dim)]
        self.pre_height_fields = self.height_fields
        self.select = Selection(self)
        self.fab = Fabrication(self)
        self.voxel_matrix_from_height_fields()
        self.buff = Buffer(self)
        self.vertices = self.create_vertices()
        self.indices = self.create_indices()

    def voxel_matrix_from_height_fields(self):
        vox_mat = mat_from_field(self.height_fields[0],self.sax)
        # ombine multiple hf:s somehow
        self.voxel_matrix = vox_mat
        self.eval = Evaluation(self)

    def update_sliding_direction(self,sax):
        self.sax = sax
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
        self.pre_height_fields = copy.deepcopy(self.height_fields)
        for ind in faces:
            if n!=2: self.height_field[tuple(ind)] = h
            else: self.height_field2[tuple(ind)] = h
            self.voxel_matrix_from_height_fields()
            self.create_indices()

    def update_number_of_components(self,new_noc):
        # Fab direction
        new_fab_directions = []
        for i in range(new_noc):
            if i<self.noc and i!=new_noc-1:
                new_fab_directions.append(self.fab_directions[i])
            else: new_fab_directions.append(1)
        self.fab_directions = new_fab_directions

        # Fixed sides
        if new_noc<self.noc:
            for i in range(self.noc-new_noc):
                self.fixed_sides.pop()
        else:
            for i in range(new_noc-self.noc):
                self.fixed_sides.insert(-1,next_fixed_sides(self.fixed_sides))
        # Update noc
        self.noc = new_noc

        # Rebuffer
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def update_dimension(self,dim_):
        pdim = self.dim
        self.dim = dim_
        print(pdim,dim_)
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

        if milling_path:
            for n in range(self.noc):
                mvs, gvs = milling_path_vertices(self,n)
                self.mverts.append(mvs)
                self.gcodeverts.append(gvs)

        va = arrow_vertices(self)

        # Combine
        jverts = np.concatenate(self.jverts)

        if milling_path and len(self.mverts[0])>0:
            mverts = np.concatenate(self.mverts)
            self.vertices = np.concatenate([jverts, va, mverts])
        else:
            self.vertices = np.concatenate([jverts, va])

        self.vn =  int(len(self.jverts[0])/8)
        self.van = int(len(va)/8)
        if milling_path and len(self.mverts[0])>0:
            self.m_start = []
            mst = 2*self.vn+self.van
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
        self.indices_chess_lines = []
        self.indices_breakable_lines = []
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

            #picking faces
            faces_not_tops, faces_tops, all_inds = joint_top_face_indices(self,all_inds,n,n*self.vn)

            #Lines
            lns,all_inds = joint_line_indices(self,all_inds,n,n*self.vn)

            # Chessboard feedback lines
            if self.eval.chess:
                chess,all_inds = chess_line_indices(self,all_inds,self.eval.chess_vertices,n,n*self.vn)

            # Breakable lines
            if self.eval.breakable:
                break_lns, all_inds = break_line_indices(self,all_inds,self.eval.breakable_voxel_inds[n],n,n*self.vn)

            open,all_inds = open_line_indices(self,all_inds,n,n*self.vn)
            self.indices_open_lines.append(open)


            #arrows
            larr, farr, all_inds = arrow_indices(self, all_inds,self.eval.slides[n],n,self.noc*self.vn)
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
            if self.eval.chess: self.indices_chess_lines.append(chess)
            if self.eval.breakable: self.indices_breakable_lines.append(break_lns)
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
