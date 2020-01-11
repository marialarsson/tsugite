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
import os

# Supporting functions

def get_random_height_fields(dim,noc):
    hfs = []
    phf = np.zeros((dim,dim))
    for n in range(noc-1):
        hf = np.zeros((dim,dim))
        for i in range(dim):
            for j in range(dim): hf[i,j]=random.randint(phf[i,j],dim)
        hfs.append(hf)
        phf = copy.deepcopy(hf)
    return hfs

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

def mat_from_fields(hfs,ax):
    dim = len(hfs[0])
    mat = np.zeros(shape=(dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind = [i,j]
                ind3d = ind.copy()
                ind3d.insert(ax,k)
                ind3d = tuple(ind3d)
                ind2d = tuple(ind)
                h = 0
                for n,hf in enumerate(hfs):
                    if k<hf[ind2d]: mat[ind3d]=n; break
                    else: mat[ind3d]=n+1
    mat = np.array(mat)
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

def get_layered_vertices(self,outline,n,i,no_z,dep):
    verts = []
    mverts = []

    r = g = b = tx = ty = 0.0

    fdir = self.fab_directions[n]
    # add startpoint
    start_vert = [outline[0].x,outline[0].y,outline[0].z]
    safe_height = outline[0].pt[self.sax]-(2*fdir-1)*i*self.voxel_size-0.2*(2*fdir-1)*self.voxel_size
    start_vert[self.sax] = safe_height
    mverts.append(MillVertex(start_vert))
    verts.extend([start_vert[0],start_vert[1],start_vert[2],r,g,b,tx,ty])

    # add layers with Z-height
    for num in range(1,no_z+1):
        for i,mv in enumerate(outline):
            pt = [mv.x,mv.y,mv.z]
            pt[self.sax] += num*(2*fdir-1)*dep
            if mv.is_arc:
                ctr = [mv.arc_ctr[0],mv.arc_ctr[1],mv.arc_ctr[2]]
                ctr[self.sax] += num*(2*fdir-1)*dep
                mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
            else:
                mverts.append(MillVertex(pt))
            if i>0:
                pmv = outline[i-1]
            if i>0 and mv.is_arc and pmv.is_arc and np.array_equal(mv.arc_ctr,pmv.arc_ctr):
                ppt = [pmv.x,pmv.y,pmv.z]
                ppt[self.sax] += num*(2*fdir-1)*dep
                arc_pts = arc_points(ppt,pt,ctr,self.sax,10)
                for arc_pt in arc_pts: verts.extend([arc_pt[0],arc_pt[1],arc_pt[2],r,g,b,tx,ty])
            else:
                verts.extend([pt[0],pt[1],pt[2],r,g,b,tx,ty])
        outline.reverse()

    # add enpoint
    end_vert = [outline[0].x,outline[0].y,outline[0].z]
    end_vert[self.sax] = safe_height
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

def get_neighbors_in_out(ind,reg_inds,lay_mat,org_lay_mat,n):
    # 0 = in region
    # 1 = outside region, block
    # 2 = outside region, free
    in_out = []
    values = []
    for add0 in range(-1,1,1):
        temp = []
        temp2 = []
        for add1 in range(-1,1,1):

            # Define neighbor index to test
            nind = [ind[0]+add0,ind[1]+add1]

            # FIND TYPE
            type = -1
            val = None
            # Check if this index is in the list of region-included indices
            for rind in reg_inds:
                if rind[0]==nind[0] and rind[1]==nind[1]:
                    type = 0 # in region
                    break
            if type!=0:
                # If there are out of bound indices they are free
                if np.any(np.array(nind)<0) or nind[0]>=lay_mat.shape[0] or nind[1]>=lay_mat.shape[1]:
                    type = 2 # free
                    val =-1
                elif lay_mat[tuple(nind)]<0:
                    type = 2 # free
                else: type = 1 # blocked

            if val==None:
                val=org_lay_mat[tuple(nind)]

            temp.append(type)
            temp2.append(val)
        in_out.append(temp)
        values.append(temp2)
    return in_out, values

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
        ind.insert(ax,(self.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
        add = [0,0,0]
        add[ax] = 1-dir
        i_pt = get_index(ind,add,self.dim)
        pt1 = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
        #end
        ind = list(pix_end.ind_abs)
        ind.insert(ax,(self.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
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

def layer_mat_from_cube(self,lay_num,n):
    mat = np.ndarray(shape=(self.dim,self.dim), dtype=int)
    fdir = self.fab_directions[n]
    for i in range(self.dim):
        for j in range(self.dim):
            ind = [i,j]
            zval = (self.dim-1)*(1-fdir)+(2*fdir-1)*lay_num
            ind.insert(self.sax,zval)
            mat[i][j]=int(self.voxel_matrix[tuple(ind)])
    return mat

def pad_layer_mat_with_fixed_sides(self,mat):
    pad_loc = [[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1]]
    for n2 in range(len(self.fixed_sides)):
        for oax,odir in self.fixed_sides[n2]:
            if oax==self.sax: continue
            axes = [0,0,0]
            axes[oax] = 1
            axes.pop(self.sax)
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
                if ax1==self.sax: continue
                axes = [0,0,0]
                axes[ax1] = 1
                axes.pop(self.sax)
                ax1 = axes.index(1)
                for ax2,dir2 in fixed_sides_2:
                    if ax2==self.sax: continue
                    axes = [0,0,0]
                    axes[ax2] = 1
                    axes.pop(self.sax)
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

def get_region_outline_vertices(reg_inds,lay_mat,org_lay_mat,pad_loc,n):
    # also duplicate vertices on diagonal
    reg_verts = []
    for i in range(lay_mat.shape[0]+1):
        for j in range(lay_mat.shape[1]+1):
            ind = [i,j]
            neigbors,neighbor_values = get_neighbors_in_out(ind,reg_inds,lay_mat,org_lay_mat,n)
            neigbors = np.array(neigbors)
            abs_ind = ind.copy()
            ind[0] -= pad_loc[0][0]
            ind[1] -= pad_loc[1][0]
            if np.any(neigbors.flatten()==0) and not np.all(neigbors.flatten()==0): # some but not all region neighbors
                dia1 = neigbors[0][1]==neigbors[1][0]
                dia2 = neigbors[0][0]==neigbors[1][1]
                if np.sum(neigbors.flatten()==0)==2 and  np.sum(neigbors.flatten()==1)==2 and dia1 and dia2: # diagonal detected
                    other_indices = np.argwhere(neigbors==0)
                    for oind in other_indices:
                        oneigbors = copy.deepcopy(neigbors)
                        oneigbors[tuple(oind)] = 1
                        oneigbors = np.array(oneigbors)
                        reg_verts.append(RegionVertex(ind,abs_ind,oneigbors,neighbor_values,dia=True))
                else: # normal situation
                    reg_verts.append(RegionVertex(ind,abs_ind,neigbors,neighbor_values))
    return reg_verts

def set_starting_vert(verts):
    first_i = None
    second_i = None
    for i,rv in enumerate(verts):
        if rv.block_count>0:
            if rv.free_count>0: first_i=i
            else: second_i = i
    if first_i==None:
        first_i=second_i
    if first_i==None: first_i=0
    verts.insert(0,verts[first_i])
    verts.pop(first_i+1)
    return verts

def get_sublist_of_ordered_verts(verts):
    ord_verts = []

    # Start ordered vertices with the first item (simultaneously remove from main list)
    ord_verts.append(verts[0])
    verts.remove(verts[0])

    browse_num = len(verts)
    for i in range(browse_num):
        found_next = False
        #try all directions to look for next vertex
        for vax in range(2):
            for vdir in range(-1,2,2):
                # check if there is an avaliable vertex
                next_ind = ord_verts[-1].ind.copy()
                next_ind[vax]+=vdir
                next_rv = None
                for rv in verts:
                    if rv.ind==next_ind:
                        if len(ord_verts)>1 and rv.ind==ord_verts[-2].ind: break # prevent going back
                        # check so that it is not crossing a blocked region etc
                        # 1) from point of view of previous point
                        p_neig = ord_verts[-1].neighbors
                        vaxval = int(0.5*(vdir+1))
                        nind0 = [0,0]
                        nind0[vax] = vaxval
                        nind1 = [1,1]
                        nind1[vax] = vaxval
                        ne0 = p_neig[nind0[0]][nind0[1]]
                        ne1 = p_neig[nind1[0]][nind1[1]]
                        if ne0!=1 and ne1!=1: continue # no block
                        if int(0.5*(ne0+1))==int(0.5*(ne1+1)): continue # trying to cross blocked material
                        # 2) from point of view of point currently tested
                        nind0 = [0,0]
                        nind0[vax] = 1-vaxval
                        nind1 = [1,1]
                        nind1[vax] = 1-vaxval
                        ne0 = rv.neighbors[nind0[0]][nind0[1]]
                        ne1 = rv.neighbors[nind1[0]][nind1[1]]
                        if ne0!=1 and ne1!=1: continue # no block
                        if int(0.5*(ne0+1))==int(0.5*(ne1+1)): continue # trying to cross blocked material
                        # If you made it here, you found the next vertex!
                        found_next=True
                        ord_verts.append(rv)
                        verts.remove(rv)
                        break
                if found_next: break
            if found_next: break
        if found_next: continue

    # check if outline is closed by ckecing if endpoint finds startpoint

    closed = False
    if len(ord_verts)>3: # needs to be at least 4 vertices to be able to close
        start_ind = np.array(ord_verts[0].ind.copy())
        end_ind = np.array(ord_verts[-1].ind.copy())
        diff_ind = start_ind-end_ind ###reverse?
        if len(np.argwhere(diff_ind==0))==1: #difference only in one axis
            vax = np.argwhere(diff_ind!=0)[0][0]
            if abs(diff_ind[vax])==1: #difference is only one step
                vdir = diff_ind[vax]
               # check so that it is not crossing a blocked region etc
                p_neig = ord_verts[-1].neighbors
                vaxval = int(0.5*(vdir+1))
                nind0 = [0,0]
                nind0[vax] = vaxval
                nind1 = [1,1]
                nind1[vax] = vaxval
                ne0 = p_neig[nind0[0]][nind0[1]]
                ne1 = p_neig[nind1[0]][nind1[1]]
                if ne0==1 or ne1==1:
                    if int(0.5*(ne0+1))!=int(0.5*(ne1+1)):
                        # If you made it here, you found the next vertex!
                        closed=True


    return ord_verts, verts, closed

def offset_verts(self,verts,lay_num,n):
    outline = []

    fdir = self.fab_directions[n]

    for i,rv in enumerate(list(verts)):

        # remove vertices with neighbor count 2
        if rv.region_count==2 and rv.block_count==2: continue # redundant
        if rv.block_count==0: continue # redundant
        if rv.ind[0]<0 or rv.ind[0]>self.dim: continue #out of bounds
        if rv.ind[1]<0 or rv.ind[1]>self.dim: continue #out of bounds

        # add vertex information
        ind = rv.ind.copy()
        ind.insert(self.sax,(self.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
        add = [0,0,0]
        add[self.sax] = 1-fdir
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
        rounded = False
        if rv.region_count==3: # outer corner, check if it sould be rounded or not
            # check if this outer corner correspond to an inner corner of another mateiral
            for n2 in range(self.noc):
                if n2==n: continue
                cnt = np.sum(rv.flat_neighbor_values==n2)
                if cnt==3: rounded = True
                elif cnt==2:
                    # Check if it is a diagonal
                    dia1 = rv.neighbor_values[0][0]==rv.neighbor_values[1][1]
                    dia2 = rv.neighbor_values[0][1]==rv.neighbor_values[1][0]
                    if dia1 or dia2:
                        rounded = True
        if rounded:
            # tolerances # add later.....
            extra_off_dist=-self.fab.vtol
            one_voxel = np.argwhere(rv.neighbors==1)[0]
            extra_addx = (one_voxel[0]*2-1)*extra_off_dist
            extra_addy = (one_voxel[1]*2-1)*extra_off_dist

            pt1 = pt.copy()
            add = [addx,-addy-extra_addy]
            add.insert(self.sax,0)
            pt1[0] += add[0]
            pt1[1] += add[1]
            pt1[2] += add[2]
            #
            pt2 = pt.copy()
            add = [-addx-extra_addx,addy]
            add.insert(self.sax,0)
            pt2[0] += add[0]
            pt2[1] += add[1]
            pt2[2] += add[2]
            #
            pts = [pt1,pt2]

            # arc center
            ctr = pt.copy()
            add = [-addx-extra_addx,-addy-extra_addy]
            add.insert(self.sax,0)
            ctr[0] += add[0]
            ctr[1] += add[1]
            ctr[2] += add[2]

            # Reorder pt1 and pt2 so that the first one is closer to the last item in the outline
            if len(outline)>0:
                ppt = outline[-1].pt
                dist1 = np.linalg.norm(np.array(pt1)-np.array(ppt))
                dist2 = np.linalg.norm(np.array(pt2)-np.array(ppt))
                if dist1>dist2: pts.reverse()
            elif len(verts)>i:
                next_ind = verts[i+1].ind.copy()
                next_ind.insert(self.sax,(self.dim-1)*(1-fdir)+(2*n-1)*lay_num)
                add = [0,0,0]
                add[self.sax] = 1-fdir
                i_pt = get_index(next_ind,add,self.dim)
                npt = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
                dist1 = np.linalg.norm(np.array(pt1)-np.array(npt))
                dist2 = np.linalg.norm(np.array(pt2)-np.array(npt))
                if dist1<dist2: pts.reverse() #??? Is this ok???
            outline.append(MillVertex(pts[0],is_arc=True,arc_ctr=ctr))
            outline.append(MillVertex(pts[1],is_arc=True,arc_ctr=ctr))
        else: # other corner
            add = [addx,addy]
            add.insert(self.sax,0)
            pt[0] += add[0]
            pt[1] += add[1]
            pt[2] += add[2]
            outline.append(MillVertex(pt))
    return outline

def get_outline(self,verts,lay_num,n):
    fdir = self.fab_directions[n]
    outline = []
    for rv in verts:
        ind = rv.ind.copy()
        ind.insert(self.sax,(self.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
        add = [0,0,0]
        add[self.sax] = 1-fdir
        i_pt = get_index(ind,add,self.dim)
        pt = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
        outline.append(MillVertex(pt))
    return outline

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
        for dir in range(2):
            corners = get_corner_indices(ax,dir,self.dim)
            for step in range(3):
                if step==0: step=0.5
                for corner in corners:
                    new_vertex = []
                    for i in range(8):
                        new_vertex_param = vertices[8*corner+i]
                        if i==ax: new_vertex_param = new_vertex_param + (2*dir-1)*step*self.component_size
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
    fdir = self.fab_directions[n]
    axes = [0,1,2]
    axes.pop(self.sax)
    dir_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis

    # Browse layers
    for lay_num in range(self.dim):

        # Create a 2D matrix of current layer
        lay_mat = layer_mat_from_cube(self,lay_num,n)

        # Pad 2d matrix with fixed sides
        lay_mat,pad_loc = pad_layer_mat_with_fixed_sides(self,lay_mat)
        org_lay_mat = copy.deepcopy(lay_mat)

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

            # 2. Produce rough milling paths
            rough_paths = rough_milling_path(self,rough_inds,lay_num,n)
            for rough_path in rough_paths:
                if len(rough_path)>0:
                    verts,mverts = get_layered_vertices(self,rough_path,n,lay_num,no_z,dep)
                    vertices.extend(verts)
                    milling_vertices.extend(mverts)


            # Overwrite detected regin in original matrix
            for reg_ind in reg_inds: lay_mat[tuple(reg_ind)]=n

            # Make a list of all edge vertices of the outline of the region
            reg_verts = get_region_outline_vertices(reg_inds,lay_mat,org_lay_mat,pad_loc,n)

            # Order the vertices to create an outline
            for isl_num in range(10):
                reg_ord_verts = []
                if len(reg_verts)==0: break

                #Make sure first item in region vertices is on blocked/free corner, or blocked
                reg_verts = set_starting_vert(reg_verts)

                #Get a sequence of ordered vertices
                reg_ord_verts,reg_verts,closed = get_sublist_of_ordered_verts(reg_verts)

                # Make outline of ordered vertices (for dedugging olny!!!!!!!)
                #if len(reg_ord_verts)>1: outline = get_outline(self,reg_ord_verts,lay_num,n)

                # Offset vertices according to boundary condition (and remove if redundant)
                outline = offset_verts(self,reg_ord_verts,lay_num,n)

                # Get z height and extend vertices to global list
                if len(reg_ord_verts)>1 and len(outline)>0:
                    if closed: outline.append(MillVertex(outline[0].pt))
                    verts,mverts = get_layered_vertices(self,outline,n,lay_num,no_z,dep)
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

def joint_top_face_indices(self,all_indices,n,noc,offset):
    # Make indices of faces for drawing method GL_QUADS
    # Set face direction
    if n==0: sdirs = [0]
    elif n==noc-1: sdirs = [1]
    else: sdirs = [0,1]
    # 1. Faces of joint
    indices = []
    indices_tops = []
    indices = []
    sax = self.sax
    for ax in range(3):
        for i in range(self.dim):
            for j in range(self.dim):
                top_face_indices_cnt=0
                for k in range(self.dim+1):
                    if sdirs[0]==0: k = self.dim-k
                    ind = [i,j]
                    ind.insert(ax,k)
                    # count number of neigbors (0, 1, or 2)
                    cnt,vals = face_neighbors(self.voxel_matrix,ind,ax,n,self.fixed_sides[n])
                    on_free_base = False
                    # add base if edge component
                    if ax==sax and ax!=self.fixed_sides[n][0][0] and len(sdirs)==1:
                        base = sdirs[0]*self.dim
                        if ind[ax]==base: on_free_base=True
                    if cnt==1 or on_free_base:
                        for x in range(2):
                            for y in range(2):
                                add = [x,abs(y-x)]
                                add.insert(ax,0)
                                index = get_index(ind,add,self.dim)
                                if ax==sax and top_face_indices_cnt<4*len(sdirs):
                                    indices_tops.append(index)
                                    top_face_indices_cnt+=1
                                #elif not on_free_base: indices.append(index)
                                else: indices.append(index)
                if top_face_indices_cnt<4*len(sdirs) and ax==sax:
                    neg_i = -offset-1
                    for k in range(4*len(sdirs)-top_face_indices_cnt):
                        indices_tops.append(neg_i)
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
    dir = select.dir
    offset = n*self.vn
    sax = self.sax
    h = self.height_fields[n-dir][tuple(select.faces[0])]
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

def component_outline_indices(self,all_indices,fixed_sides,n,offset):
    d = self.dim+1
    indices = []
    start = d*d*d
    #Outline of component base
    #1) Base of first fixed side
    ax = fixed_sides[0][0]
    dir = fixed_sides[0][1]
    step = 2
    if len(fixed_sides)==2: step = 1
    off = 24*ax+12*dir+4*step
    a0,b0,c0,d0 = start+off,start+off+1,start+off+2,start+off+3
    #2) Base of first fixed side OR top of component
    if len(fixed_sides)==2:
        ax = fixed_sides[1][0]
        dir = fixed_sides[1][1]
        off = 24*ax+12*dir+4*step
        a1,b1,c1,d1 = start+off,start+off+1,start+off+2,start+off+3
    else:
        a1,b1,c1,d1 = get_corner_indices(ax,1-dir,self.dim)
    # append list of indices
    indices.extend([a0,b0, b0,d0, d0,c0, c0,a0])
    indices.extend([a0,a1, b0,b1, c0,c1, d0,d1])
    indices.extend([a1,b1, b1,d1, d1,c1, c1,a1])
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
    dirs = [0,1]
    if n==0: dirs=[0]
    elif n==self.noc-1: dirs=[1]
    other_axes = np.array([0,1,2])
    other_axes = np.delete(other_axes,np.where(other_axes==self.sax))
    ind = np.array([0,0,0])
    d = self.dim-1
    for dir in dirs:
        heights = get_top_corner_heights(self.voxel_matrix,n,self.sax,dir)
        d = self.dim+1
        for x in range(2):
            for y in range(2):
                add = np.array([0,0,0])
                add[other_axes[0]] = x*self.dim
                add[other_axes[1]] = y*self.dim
                add[self.sax] = dir*self.dim
                start = get_index(ind,add,self.dim)
                step=0
                end = d*d*d+24*self.sax+12*(1-dir)+2*x+y+4*step
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
    indices_prop = ElementProperties(GL_LINE_STRIP, len(indices), len(all_indices), n)
    all_indices = np.concatenate([all_indices, indices])
    # Return
    return indices_prop, all_indices

class Geometries:
    def __init__(self,fs,sax,noc=2,hfs=None,draw=True):
        self.sax = sax
        self.fixed_sides = fs
        self.grain_rotation = random.uniform(0,math.pi)
        self.noc = noc #number of components
        self.fab_directions = []
        for i in range(self.noc):
            if i==0: self.fab_directions.append(0)
            else: self.fab_directions.append(1)
        self.dim = 3
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.5*self.component_size
        self.vertex_no_info = 8
        if hfs==None: self.height_fields = get_random_height_fields(self.dim,self.noc)
        else: self.height_fields = hfs
        if draw: self.pre_height_fields = self.height_fields
        if draw: self.select = Selection(self)
        if draw: self.fab = Fabrication(self)
        self.voxel_matrix_from_height_fields()
        if draw: self.buff = Buffer(self)
        if draw: self.vertices = self.create_vertices()
        if draw: self.indices = self.create_indices()
        if draw: self.update_unblocked_fixed_sides()

    def voxel_matrix_from_height_fields(self):
        vox_mat = mat_from_fields(self.height_fields,self.sax)
        self.voxel_matrix = vox_mat
        self.eval = Evaluation(self)

    def update_sliding_direction(self,sax):
        blocked = False
        for i,sides in enumerate(self.fixed_sides):
            for ax,dir in sides:
                if ax==sax:
                    if dir==0 and i==0: continue
                    if dir==1 and i==self.noc-1: continue
                    blocked = True
        if not blocked:
            self.sax = sax
            self.voxel_matrix_from_height_fields()
            self.update_unblocked_fixed_sides()
            self.create_vertices()
            self.create_indices()
        else: print("Blocked sliding direction")

    def randomize_height_fields(self):
        self.height_fields = get_random_height_fields(self.dim,self.noc)
        self.voxel_matrix_from_height_fields()
        self.create_indices()

    def clear_height_fields(self):
        self.height_fields = []
        for n in range(self.noc-1):
            hf = np.zeros((self.dim,self.dim))
            self.height_fields.append(hf)
        self.voxel_matrix_from_height_fields()
        self.create_indices()

    def edit_height_fields(self,faces,h,n,dir):
        for ind in faces:
            self.height_fields[n-dir][tuple(ind)] = h
            if dir==0: # If editiing top
                # If new height is higher than following hf, update to same height
                for i in range(n-dir+1,self.noc-1):
                    h2 = self.height_fields[i][tuple(ind)]
                    if h>h2: self.height_fields[i][tuple(ind)]=h
            if dir==1: # If editiing bottom
                # If new height is lower than previous hf, update to same height
                for i in range(0,n-dir):
                    h2 = self.height_fields[i][tuple(ind)]
                    if h<h2: self.height_fields[i][tuple(ind)]=h
            self.voxel_matrix_from_height_fields()
            self.create_indices()

    def update_component_position(self,new_fixed_sides,n):
        self.fixed_sides[n] = new_fixed_sides
        self.voxel_matrix_from_height_fields()
        self.update_unblocked_fixed_sides()
        self.create_vertices()
        self.create_indices()

    def update_number_of_components(self,new_noc):
        # Increasing number of components
        if new_noc>self.noc:
            if len(self.unblocked_fixed_sides)>=(new_noc-self.noc):
                for i in range(new_noc-self.noc):
                    nfs = next_fixed_sides(self.fixed_sides)
                    if self.fixed_sides[-1][0][0]==self.sax: # last component is aligned with the sliding axis
                        self.fixed_sides.insert(-1,nfs)
                        self.fab_directions.insert(-1,0)
                    else:
                        self.fixed_sides.append(nfs)
                        self.fab_directions.append(1)
                    #also consider if it is aligned and should be the first one in line... rare though...
                self.height_fields = get_random_height_fields(self.dim,new_noc)
                self.noc = new_noc
        # Decreasing number of components
        elif new_noc<self.noc:
            for i in range(self.noc-new_noc):
                self.fixed_sides.pop()
                self.fab_directions.pop()
                self.height_fields.pop()
            self.noc = new_noc


        # Rebuffer
        self.voxel_matrix_from_height_fields()
        self.update_unblocked_fixed_sides()
        self.create_vertices()
        self.create_indices()

    def update_dimension(self,dim):
        self.dim = dim
        self.voxel_size = self.component_size/self.dim
        self.fab.real_voxel_size = self.fab.real_component_size/self.dim
        self.height_fields = get_random_height_fields(self.dim,self.noc)
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
            mst = self.noc*self.vn+self.van
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
        self.indices_fpick_top = []
        self.indices_fpick_not_top = []
        self.outline_selected_faces = None
        self.outline_selected_component = None
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
            faces_pick_not_tops, faces_pick_tops, all_inds = joint_top_face_indices(self,all_inds,n,self.noc,n*self.vn)

            #Lines
            lns,all_inds = joint_line_indices(self,all_inds,n,n*self.vn)

            # Chessboard feedback lines
            if self.eval.checker[n]:
                chess,all_inds = chess_line_indices(self,all_inds,self.eval.checker_vertices[n],n,n*self.vn)
            else: chess = []
            # Breakable lines
            if self.eval.breakable:
                break_lns, all_inds = break_line_indices(self,all_inds,self.eval.breakable_voxel_inds[n],n,n*self.vn)

            # Opening lines
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
            self.indices_fpick_top.append(faces_pick_tops)
            self.indices_fpick_not_top.append(faces_pick_not_tops)
            self.indices_chess_lines.append(chess)
            if self.eval.breakable: self.indices_breakable_lines.append(break_lns)
            if milling_path and len(self.mverts[0])>0:
                self.indices_milling_path.append(mill)


        #outline of selected faces
        if self.select.state==2:
            self.outline_selected_faces, all_inds = joint_selected_top_line_indices(self,self.select,all_inds)

        if self.select.n!=None and self.select.new_fixed_sides_for_display!=None:
            self.outline_selected_component, all_inds = component_outline_indices(self,all_inds,self.select.new_fixed_sides_for_display,self.select.n,self.select.n*self.vn)

        self.indices = all_inds
        Buffer.buffer_indices(self.buff)

    def save(self):
        np.save("data/saved_height_fields.npy",self.height_fields)
        np.save("data/saved_fixed_sides.npy",self.fixed_sides)

    def user_study_design_finished(self,args,duration,click_cnt):
        dir = os.getcwd()
        # remove "/setup"
        dir = dir.split("\\")
        dir.pop()
        loc = ""
        for item in dir: loc+=item+"/"
        loc += "/user_study"
        if not os.path.exists(loc): os.mkdir(loc)
        if args.feedback: loc += "/with_feedback"
        else: loc += "/without_feedback"
        if not os.path.exists(loc): os.mkdir(loc)
        loc += "/stage"+str(args.type)
        if not os.path.exists(loc): os.mkdir(loc)
        loc += "/"+args.username
        if not os.path.exists(loc): os.mkdir(loc)
        path = loc+"/height_fields_%s.npy"
        i = 0
        while os.path.exists(path % i): i=i+1
        # save geometry
        np.save(path % i,self.height_fields)
        # save data
        path = loc+"/data_%s.txt"
        np.savetxt(path % i, [duration,click_cnt])   # x,y,z equal sized 1D arrays
        # save joint performance evaluation
        m1 = 0 # connectivity
        m3 = 0 # checker board
        m5 = 0 # multiple sliding directions
        m6 = 0 # fragile parts
        for n in range(self.noc):
            if not self.eval.connected[n]: m1 = 1
            if self.eval.checker[n]: m3 = 1
            if len(self.eval.slides[n])>1: m5 = 1
            if self.eval.breakable[n]: m6 = 1
        path = loc+"/eval_%s.txt"
        np.savetxt(path % i, [m1,m3,m5,m6])   # x,y,z equal sized 1D arrays

    def load(self):
        self.height_fields = np.load("data/saved_height_fields.npy")
        self.fixed_sides = np.load("data/saved_fixed_sides.npy")
        self.noc = len(self.fixed_sides)
        self.fab_directions = []
        for i in range(self.noc):
            if i==0: self.fab_directions.append(0)
            else: self.fab_directions.append(1)
        self.voxel_matrix_from_height_fields()
        self.create_vertices()
        self.create_indices()

    def update_unblocked_fixed_sides(self):
        unblocked_sides = []
        for ax in range(3):
            for dir in range(2):
                side = [ax,dir]
                blocked = False
                for sides in self.fixed_sides:
                    if side in sides:
                        blocked=True
                        break
                if not blocked:
                    unblocked_sides.append(side)
        self.unblocked_fixed_sides = unblocked_sides


    """
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
    """
