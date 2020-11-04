from OpenGL.GL import *
from PyQt5.QtWidgets import *
import numpy as np
from numpy import linalg
import math
import copy
import os
import random
from Buffer import Buffer
from Evaluation import Evaluation
from Fabrication import *
from Geometries import Geometries
from Geometries import get_index
from Misc import FixedSides

def normalize(v):
    norm = linalg.norm(v)
    if norm == 0: return v
    else: return v / norm

def mat_from_fields(hfs,ax): ### duplicated function - also exists in Geometries
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

def angle_between(vector_1, vector_2):
    unit_vector_1 = vector_1 / np.linalg.norm(vector_1)
    unit_vector_2 = vector_2 / np.linalg.norm(vector_2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return angle

def rotate_vector_around_axis(vec=[3,5,0], axis=[4,4,1], theta=1.2): #example values
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    mat = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])
    rotated_vec = np.dot(mat, vec)
    return rotated_vec

def arrow_vertices(self):
    vertices = []
    r=g=b=0.0
    tx=ty=0.0
    vertices.extend([0,0,0, r,g,b, tx,ty]) # origin
    for ax in range(3):
        for dir in range(-1,2,2):
            #arrow base
            xyz = dir*self.pos_vecs[ax]*self.dim*0.4
            vertices.extend([xyz[0],xyz[1],xyz[2], r,g,b, tx,ty]) # end of line
            #arrow head
            for i in range(-1,2,2):
                for j in range(-1,2,2):
                    other_axes = [0,1,2]
                    other_axes.pop(ax)
                    pos = dir*self.pos_vecs[ax]*self.dim*0.3
                    pos+= i*self.pos_vecs[other_axes[0]]*self.dim*0.025
                    pos+= j*self.pos_vecs[other_axes[1]]*self.dim*0.025
                    vertices.extend([pos[0],pos[1],pos[2], r,g,b, tx,ty]) # arrow head indices
    # Format
    vertices = np.array(vertices, dtype = np.float32) #converts to correct format
    return vertices

def produce_suggestions(type,hfs):
    valid_suggestions = []
    for i in range(len(hfs)):
        for j in range(type.dim):
            for k in range(type.dim):
                for add in range(-1,2,2):
                    sugg_hfs = copy.deepcopy(hfs)
                    sugg_hfs[i][j][k]+=add
                    val = sugg_hfs[i][j][k]
                    if val>=0 and val<=type.dim:
                        sugg_voxmat = mat_from_fields(sugg_hfs,type.sax)
                        sugg_eval = Evaluation(sugg_voxmat,type,mainmesh=False)
                        if sugg_eval.valid:
                            valid_suggestions.append(sugg_hfs)
                            if len(valid_suggestions)==4: break
    return valid_suggestions

def layer_mat_from_cube(type,lay_num,n):
    mat = np.ndarray(shape=(type.dim,type.dim), dtype=int)
    fdir = type.mesh.fab_directions[n]
    for i in range(type.dim):
        for j in range(type.dim):
            ind = [i,j]
            zval = (type.dim-1)*(1-fdir)+(2*fdir-1)*lay_num
            ind.insert(type.sax,zval)
            mat[i][j]=int(type.mesh.voxel_matrix[tuple(ind)])
    return mat

def pad_layer_mat_with_fixed_sides(mat,type,n):
    pad_loc = [[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1]]
    for n2 in range(len(type.fixed.sides)):
        for oside in type.fixed.sides[n2]:
            if oside.ax==type.sax: continue
            axes = [0,0,0]
            axes[oside.ax] = 1
            axes.pop(type.sax)
            oax = axes.index(1)
            pad_loc[oax][oside.dir] = 1
            pad_val[oax][oside.dir] = n2
    # If it is an angled joint, pad so that the edge of a joint located on an edge will be trimmed well
    #if abs(type.ang-90)>1 and len(type.fixed.sides[n])==1 and type.fixed.sides[n][0].ax!=type.sax:
    #    print("get here")
    #    ax = type.fixed.sides[n][0].ax
    #    dir = type.fixed.sides[n][0].dir
    #    odir = 1-dir
    #    axes = [0,0,0]
    #    axes[ax] = 1
    #    axes.pop(type.sax)
    #    oax = axes.index(1)
    #    pad_loc[oax][odir] = 1
    #    pad_val[oax][odir] = 9
    # Perform the padding
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)
    # take care of -1 corners # does this still work after adding former step??????????????
    # This could be shorter for sure...
    for fixed_sides_1 in type.fixed.sides:
        for fixed_sides_2 in type.fixed.sides:
            for side1 in fixed_sides_1:
                if side1.ax==type.sax: continue
                axes = [0,0,0]
                axes[side1.ax] = 1
                axes.pop(type.sax)
                ax1 = axes.index(1)
                for side2 in fixed_sides_2:
                    if side2.ax==type.sax: continue
                    axes = [0,0,0]
                    axes[side2.ax] = 1
                    axes.pop(type.sax)
                    ax2 = axes.index(1)
                    if ax1==ax2: continue
                    ind = [0,0]
                    ind[ax1] = side1.dir*(mat.shape[ax1]-1)
                    ind[ax2] = side2.dir*(mat.shape[ax2]-1)
                    mat[tuple(ind)] = -1
    return mat,pad_loc

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
                    if any_minus_one_neighbor(ind,lay_mat): mon = True
                    else: mon = False
                    reg_verts.append(RegionVertex(ind,abs_ind,neigbors,neighbor_values,minus_one_neighbor=mon))
    return reg_verts

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

def rough_milling_path(type,rough_pixs,lay_num,n):
    mvertices = []

    # Defines axes
    ax = type.sax # mill bit axis
    dir = type.mesh.fab_directions[n]
    axes = [0,1,2]
    axes.pop(ax)
    dir_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis

    # Define fabrication parameters

    no_lanes = 2+math.ceil(((type.real_tim_dims[axes[1]]/type.dim)-2*type.fab.dia)/type.fab.dia)
    lane_width = (type.voxel_sizes[axes[1]]-type.fab.vdia)/(no_lanes-1)
    ratio = np.linalg.norm(type.pos_vecs[axes[1]])/type.voxel_sizes[axes[1]]
    v_vrad = type.fab.vrad*ratio
    lane_width = lane_width*ratio


    # create offset direction vectors
    dir_vec = normalize(type.pos_vecs[axes[0]])
    off_vec = normalize(type.pos_vecs[axes[1]])

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
        for i in range(type.dim):
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
        ind.insert(ax,(type.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
        add = [0,0,0]
        add[ax] = 1-dir
        i_pt = get_index(ind,add,type.dim)
        pt1 = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)
        #end
        ind = list(pix_end.ind_abs)
        ind.insert(ax,(type.dim-1)*(1-dir)+(2*dir-1)*lay_num) # 0 when n is 1, dim-1 when n is 0
        add = [0,0,0]
        add[ax] = 1-dir
        add[dir_ax] = 1
        i_pt = get_index(ind,add,type.dim)
        pt2 = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)

        ### REFINE THIS FUNCTION
        dir_add1 = pix.neighbors[dir_ax][0]*2.5*type.fab.vrad*dir_vec
        dir_add2 = -pix_end.neighbors[dir_ax][1]*2.5*type.fab.vrad*dir_vec

        pt1 = pt1+v_vrad*off_vec+dir_add1
        pt2 = pt2+v_vrad*off_vec+dir_add2
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

def edge_milling_path(type,lay_num,n):
    mverts = []

    if len(type.fixed.sides[n])==1 and type.fixed.sides[n][0].ax!=type.sax:

        # ax dir of current fixed side
        ax = type.fixed.sides[n][0].ax
        dir = type.fixed.sides[n][0].dir
        # oax - axis perp. to component axis
        oax = [0,1,2]
        oax.remove(type.sax)
        oax.remove(ax)
        oax=oax[0]
        # fabrication direction
        fdir = type.mesh.fab_directions[n]

        # check so that that part is not removed anyways...
        # i.e. if the whole bottom row in that direction is of other material
        ind = [0,0,0]
        ind[ax] = (1-dir)*(type.dim-1)
        ind[type.sax] = fdir*(type.dim-1)
        free=True
        for i in range(type.dim):
            ind[oax]=i
            val = type.mesh.voxel_matrix[tuple(ind)]
            if int(val)==n:
                free=False
                break

        if not free:
            # define start (pt0) and end (pt1) points of edge
            ind = [0,0,0]
            add = [0,0,0]
            ind[ax] = (1-dir)*type.dim
            ind[type.sax] = type.dim*(1-fdir)+(2*fdir-1)*lay_num
            i_pt = get_index(ind,add,type.dim)
            pt0 = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)
            ind[oax] = type.dim
            i_pt = get_index(ind,add,type.dim)
            pt1 = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)

            # offset edge line by radius of millingbit
            dir_vec = normalize(pt0-pt1)
            sax_vec = [0,0,0]
            sax_vec[type.sax] = 2*fdir-1
            off_vec = rotate_vector_around_axis(dir_vec, sax_vec, math.radians(90))
            off_vec = (2*dir-1)*type.fab.vrad*off_vec
            pt0 = pt0+off_vec
            pt1 = pt1+off_vec

            # Write to mverts
            mverts=[MillVertex(pt0),MillVertex(pt1)]


    return mverts

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

def set_vector_length(vec,new_norm):
    norm = np.linalg.norm(vec)
    vec = vec/norm
    vec = new_norm*vec
    return vec

def offset_verts(type,neighbor_vectors,neighbor_vectors_a,neighbor_vectors_b,verts,lay_num,n):
    outline = []
    corner_artifacts = []

    fdir = type.mesh.fab_directions[n]

    test_first=True
    for i,rv in enumerate(list(verts)): # browse each vertex in the outline

        # remove vertices with neighbor count 2 #OK
        if rv.region_count==2 and rv.block_count==2: continue # redundant
        if rv.block_count==0: continue                        # redundant
        if rv.ind[0]<0 or rv.ind[0]>type.dim: continue        # out of bounds
        if rv.ind[1]<0 or rv.ind[1]>type.dim: continue        # out of bounds

        # add vertex information #OK
        ind = rv.ind.copy()
        ind.insert(type.sax,(type.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
        add = [0,0,0]
        add[type.sax] = 1-fdir
        i_pt = get_index(ind,add,type.dim)
        pt = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)

        # move vertex according to boundry condition <---needs to be updated
        off_vecs = []
        if rv.block_count==1:
            nind = tuple(np.argwhere(rv.neighbors==1)[0])
            off_vecs.append(-neighbor_vectors[nind])
        if rv.region_count==1 and rv.free_count!=3:
            nind = tuple(np.argwhere(rv.neighbors==0)[0])
            off_vecs.append(neighbor_vectors[nind])
            if np.any(rv.flat_neighbor_values==-2):
                nind = tuple(np.argwhere(rv.neighbor_values==-2)[0])
                off_vecs.append(neighbor_vectors[nind])

        off_vec = np.average(off_vecs,axis=0)
        # check if it is an outer corner that should be rounded
        rounded = False
        if rv.region_count==3: # outer corner, check if it sould be rounded or not
            # check if this outer corner correspond to an inner corner of another mateiral
            for n2 in range(type.noc):
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
            nind = tuple(np.argwhere(rv.neighbors==1)[0])
            off_vec_a = -neighbor_vectors_a[nind]
            off_vec_b = -neighbor_vectors_b[nind]
            le2 = math.sqrt(math.pow(2*np.linalg.norm(off_vec_a+off_vec_b),2)-math.pow(2*type.fab.vrad,2))-np.linalg.norm(off_vec_a)
            off_vec_a2 = set_vector_length(off_vec_a,le2)
            off_vec_b2 = set_vector_length(off_vec_b,le2)

            # define end points and the center point of the arc
            pt1 = pt+off_vec_a-off_vec_b2
            pt2 = pt+off_vec_b-off_vec_a2
            pts = [pt1,pt2]
            ctr = pt-off_vec_a-off_vec_b # arc center


            # Reorder pt1 and pt2
            if len(outline)>0: # if it is not the first point in the outline
                ppt = outline[-1].pt
                v1 = pt1-ppt
                v2 = pt2-ppt
                ang1 = angle_between(v1,off_vec_b) #should be 0 if order is already good
                ang2 = angle_between(v2,off_vec_b) #should be more than 0
                if ang1>ang2: pts.reverse()
            outline.append(MillVertex(pts[0],is_arc=True,arc_ctr=ctr))
            outline.append(MillVertex(pts[1],is_arc=True,arc_ctr=ctr))

            # Extreme case where corner is very rounded and everything is not cut
            dist = linalg.norm(pt-ctr)
            if dist>type.fab.vdia and lay_num<type.dim-1:
                artifact = []
                v0 = type.fab.vdia*normalize(pt+off_vec-pts[0])
                v1 = type.fab.vdia*normalize(pt+off_vec-pts[1])
                vp = type.fab.vrad*normalize(pts[1]-pts[0])
                pts3 =  [pts[0]-vp+v0,pt+2*off_vec,pts[1]+vp+v1]
                while linalg.norm(pts3[2]-pts3[0])>type.fab.vdia:
                    pts3[0] += vp
                    pts3[1] += -off_vec
                    pts3[2] += -vp
                    for i in range(3): artifact.append(MillVertex(pts3[i]))
                    pts3.reverse()
                    vp = -vp
                if len(artifact)>0:
                    corner_artifacts.append(artifact)

        else: # other corner
            pt = pt+off_vec
            outline.append(MillVertex(pt))
        if len(outline)>2 and outline[0].is_arc and test_first:
            # if the previous one was an arc but it was the first point of the outline,
            # so we couldnt verify the order of the points
            # we might need to retrospectively switch order of the arc points
            npt = outline[2].pt
            d1 = np.linalg.norm(outline[0].pt-npt)
            d2 = np.linalg.norm(outline[1].pt-npt)
            if d1<d2: outline[0],outline[1] = outline[1],outline[0]
            test_first=False

    return outline, corner_artifacts

def get_outline(type,verts,lay_num,n):
    fdir = type.mesh.fab_directions[n]
    outline = []
    for rv in verts:
        ind = rv.ind.copy()
        ind.insert(type.sax,(type.dim-1)*(1-fdir)+(2*fdir-1)*lay_num)
        add = [0,0,0]
        add[type.sax] = 1-fdir
        i_pt = get_index(ind,add,type.dim)
        pt = get_vertex(i_pt,type.jverts[n],type.vertex_no_info)
        outline.append(MillVertex(pt))
    return outline

def get_vertex(index,verts,n):
    x = verts[n*index]
    y = verts[n*index+1]
    z = verts[n*index+2]
    return np.array([x,y,z])

def get_milling_end_points(type,n,last_z):
    verts = []
    mverts = []

    r = g = b = tx = ty = 0.0

    fdir = type.mesh.fab_directions[n]

    origin_vert = [0,0,0]
    origin_vert[type.sax] = last_z

    extra_zheight = 15/type.ratio
    above_origin_vert = [0,0,0]
    above_origin_vert[type.sax] = last_z-(2*fdir-1)*extra_zheight

    mverts.append(MillVertex(origin_vert, is_tra=True))
    mverts.append(MillVertex(above_origin_vert, is_tra=True))
    verts.extend([origin_vert[0],origin_vert[1],origin_vert[2],r,g,b,tx,ty])
    verts.extend([above_origin_vert[0],above_origin_vert[1],above_origin_vert[2],r,g,b,tx,ty])

    return verts,mverts

def get_segment_proportions(outline):
    olen = 0
    slens = []
    sprops = []

    for i in range(1,len(outline)):
        ppt = outline[i-1].pt
        pt = outline[i].pt
        dist = linalg.norm(pt-ppt)
        slens.append(dist)
        olen+=dist

    olen2=0
    sprops.append(0.0)
    for slen in slens:
        olen2+=slen
        sprop = olen2/olen
        sprops.append(sprop)

    return sprops

def get_layered_vertices(type,outline,n,lay_num,no_z,dep):
    verts = []
    mverts = []

    r = g = b = tx = ty = 0.0

    fdir = type.mesh.fab_directions[n]
    # add startpoint
    start_vert = [outline[0].x,outline[0].y,outline[0].z]
    safe_height = outline[0].pt[type.sax]-(2*fdir-1)*(lay_num*type.voxel_sizes[type.sax]+2*dep)
    start_vert[type.sax] = safe_height
    mverts.append(MillVertex(start_vert,is_tra=True))
    verts.extend([start_vert[0],start_vert[1],start_vert[2],r,g,b,tx,ty])
    if lay_num!=0:
        start_vert2 = [outline[0].x,outline[0].y,outline[0].z]
        safe_height2 = outline[0].pt[type.sax]-(2*fdir-1)*dep
        start_vert2[type.sax] = safe_height2
        mverts.append(MillVertex(start_vert2,is_tra=True))
        verts.extend([start_vert2[0],start_vert2[1],start_vert2[2],r,g,b,tx,ty])

    # add layers with Z-height
    # set start number (one layer earlier if first layer)
    if lay_num==0: stn=0
    else: stn=1
    # set end number (one layer more if last layer and not sliding direction aligned component)
    if lay_num==type.dim-1 and type.sax!=type.fixed.sides[n][0].ax: enn=no_z+2
    else: enn=no_z+1
    if type.incremental:
        enn+=1
        seg_props = get_segment_proportions(outline)
    else: seg_props = [1.0]*len(outline)
    #calculate depth for incremental setting

    for num in range(stn,enn):
        if type.incremental and num==enn-1: seg_props = [0.0]*len(outline)
        for i, (mv, sp) in enumerate(zip(outline,seg_props)):
            pt = [mv.x,mv.y,mv.z]
            pt[type.sax] += (2*fdir-1)*(num-1+sp)*dep
            if mv.is_arc:
                ctr = [mv.arc_ctr[0],mv.arc_ctr[1],mv.arc_ctr[2]]
                ctr[type.sax] += (2*fdir-1)*(num-1+sp)*dep
                mverts.append(MillVertex(pt, is_arc=True, arc_ctr=ctr))
            else:
                mverts.append(MillVertex(pt))
            if i>0:
                pmv = outline[i-1]
            if i>0 and connected_arc(mv,pmv):
                ppt = [pmv.x,pmv.y,pmv.z]
                ppt[type.sax] += (2*fdir-1)*(num-1+sp)*dep
                pctr = [pmv.arc_ctr[0],pmv.arc_ctr[1],pmv.arc_ctr[2]]
                pctr[type.sax] += (2*fdir-1)*(num-1+sp)*dep
                arc_pts = arc_points(ppt,pt,pctr,ctr,type.sax,math.radians(5))
                for arc_pt in arc_pts: verts.extend([arc_pt[0],arc_pt[1],arc_pt[2],r,g,b,tx,ty])
            else:
                verts.extend([pt[0],pt[1],pt[2],r,g,b,tx,ty])
        outline.reverse()

    # add enpoint
    end_vert = [outline[0].x,outline[0].y,outline[0].z]
    end_vert[type.sax] = safe_height
    mverts.append(MillVertex(end_vert, is_tra=True))
    verts.extend([end_vert[0],end_vert[1],end_vert[2],r,g,b,tx,ty])

    return verts,mverts

def any_minus_one_neighbor(ind,lay_mat):
    bool = False
    for add0 in range(-1,1,1):
        temp = []
        temp2 = []
        for add1 in range(-1,1,1):
            # Define neighbor index to test
            nind = [ind[0]+add0,ind[1]+add1]
            # If test index is within bounds
            if np.all(np.array(nind)>=0) and nind[0]<lay_mat.shape[0] and nind[1]<lay_mat.shape[1]:
                # If the value is -1
                if lay_mat[tuple(nind)]==-1:
                    bool = True
                    break
    return bool

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
                    val = -2
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

def is_additional_outer_corner(type,rv,ind,ax,n):
    outer_corner = False
    if rv.region_count==1 and rv.block_count==1:
        other_fixed_sides = type.fixed.sides.copy()
        other_fixed_sides.pop(n)
        for sides in other_fixed_sides:
            for side in sides:
                if side.ax==ax: continue
                axes = [0,0,0]
                axes[side.ax] = 1
                axes.pop(ax)
                oax = axes.index(1)
                not_oax = axes.index(0)
                if rv.ind[oax]==odir*type.dim:
                    if rv.ind[not_oax]!=0 and rv.ind[not_oax]!=type.dim:
                        outer_corner = True
                        break
            if outer_corner: break
    return outer_corner

def milling_path_vertices(type,n):

    vertices = []
    milling_vertices = []


    min_vox_size = np.min(type.voxel_sizes)
    # Check that the milling bit is not too large for the voxel size
    if np.min(type.voxel_sizes)<type.fab.vdia: print("Could not generate milling path. The milling bit is too large.")

    # Calculate depth constants
    no_z = int(type.ratio*type.voxel_sizes[type.sax]/type.fab.dep)
    dep = type.voxel_sizes[type.sax]/no_z

    # Defines axes and vectors
    fdir = type.mesh.fab_directions[n]
    axes = [0,1,2]
    axes.pop(type.sax)
    dir_ax = axes[0] # primary milling direction axis
    off_ax = axes[1] # milling offset axis
    ### new for oblique angles ### neighbor vectors
    le = type.fab.vrad/math.cos(abs(math.radians(-type.ang)))
    dir_vec = le*type.pos_vecs[axes[0]]/np.linalg.norm(type.pos_vecs[axes[0]])
    off_vec = le*type.pos_vecs[axes[1]]/np.linalg.norm(type.pos_vecs[axes[1]])
    neighbor_vectors = []
    neighbor_vectors_a = []
    neighbor_vectors_b = []
    for x in range(-1,2,2):
        temp = []
        tempa = []
        tempb = []
        for y in range(-1,2,2):
            temp.append(x*dir_vec+y*off_vec)
            tempa.append(x*dir_vec)
            tempb.append(y*off_vec)
        neighbor_vectors.append(temp)
        neighbor_vectors_a.append(tempa)
        neighbor_vectors_b.append(tempb)
    neighbor_vectors = np.array(neighbor_vectors)
    neighbor_vectors_a = np.array(neighbor_vectors_a)
    neighbor_vectors_b = np.array(neighbor_vectors_b)

    # Browse layers
    for lay_num in range(type.dim):

        # Create a 2D matrix of current layer
        lay_mat = layer_mat_from_cube(type,lay_num,n) #OK

        # Pad 2d matrix with fixed sides
        lay_mat,pad_loc = pad_layer_mat_with_fixed_sides(lay_mat,type,n) #OK
        org_lay_mat = copy.deepcopy(lay_mat) #OK

        # Get/browse regions
        for reg_num in range(type.dim*type.dim):

            # Get indices of a region
            inds = np.argwhere((lay_mat!=-1) & (lay_mat!=n)) #OK
            if len(inds)==0: break #OK
            reg_inds = get_diff_neighbors(lay_mat,[inds[0]],n) #OK

            #If oblique joint, create path to trim edge
            edge_path = []
            if abs(type.ang)>1: edge_path = edge_milling_path(type,lay_num,n)
            if len(edge_path)>0:
                verts,mverts = get_layered_vertices(type,edge_path,n,lay_num,no_z,dep)
                vertices.extend(verts)
                milling_vertices.extend(mverts)

            # Anaylize which voxels needs to be roguhly cut initially
            # 1. Add all open voxels in the region
            rough_inds = []
            for ind in reg_inds:
                rough_inds.append(RoughPixel(ind, lay_mat, pad_loc,type.dim,n)) #should be same...
            # 2. Produce rough milling paths
            rough_paths = rough_milling_path(type,rough_inds,lay_num,n)
            for rough_path in rough_paths:
                if len(rough_path)>0:
                    verts,mverts = get_layered_vertices(type,rough_path,n,lay_num,no_z,dep)
                    vertices.extend(verts)
                    milling_vertices.extend(mverts)

            # Overwrite detected regin in original matrix
            for reg_ind in reg_inds: lay_mat[tuple(reg_ind)]=n #OK

            # Make a list of all edge vertices of the outline of the region
            reg_verts = get_region_outline_vertices(reg_inds,lay_mat,org_lay_mat,pad_loc,n) #OK

            # Order the vertices to create an outline
            for isl_num in range(10):
                reg_ord_verts = []
                if len(reg_verts)==0: break

                #Make sure first item in region vertices is on blocked/free corner, or blocked
                reg_verts = set_starting_vert(reg_verts) #OK

                #Get a sequence of ordered vertices
                reg_ord_verts,reg_verts,closed = get_sublist_of_ordered_verts(reg_verts) #OK

                # Make outline of ordered vertices (for dedugging only!!!!!!!)
                #if len(reg_ord_verts)>1: outline = get_outline(type,reg_ord_verts,lay_num,n)

                # Offset vertices according to boundary condition (and remove if redundant)
                outline,corner_artifacts = offset_verts(type,neighbor_vectors,neighbor_vectors_a,neighbor_vectors_b,reg_ord_verts,lay_num,n) #<----needs to be updated for oblique angles!!!!!<---

                # Get z height and extend vertices to global list
                if len(reg_ord_verts)>1 and len(outline)>0:
                    if closed: outline.append(MillVertex(outline[0].pt))
                    verts,mverts = get_layered_vertices(type,outline,n,lay_num,no_z,dep)
                    vertices.extend(verts)
                    milling_vertices.extend(mverts)

                if len(corner_artifacts)>0:
                    for artifact in corner_artifacts:
                        verts,mverts = get_layered_vertices(type,artifact,n,lay_num,no_z,dep)
                        vertices.extend(verts)
                        milling_vertices.extend(mverts)

    # Add end point
    end_verts, end_mverts = get_milling_end_points(type,n,milling_vertices[-1].pt[type.sax])
    vertices.extend(end_verts)
    milling_vertices.extend(end_mverts)

    # Format and return
    vertices = np.array(vertices, dtype = np.float32)

    return vertices, milling_vertices

class Types:
    def __init__(self,parent,fs=[],sax=2,dim=3,ang=0.0, td=[44.0,44.0,44.0], fspe=400, fspi=6000, fabtol=0.15, fabdia=6.00, align_ax=0, fabext="gcode", incremental=False, hfs=[], finterp=True):
        self.parent=parent
        self.sax = sax
        self.fixed = FixedSides(self)
        self.noc = len(self.fixed.sides) #number of components
        self.dim = dim
        self.suggestions_on = True
        self.component_size = 0.275
        self.real_tim_dims = np.array(td)
        self.component_length = 0.5*self.component_size
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab = Fabrication(self, tol=fabtol, dia=fabdia, ext=fabext, align_ax=align_ax, interp=finterp, spi=fspi, spe=fspe)
        self.vertex_no_info = 8
        self.ang = ang
        self.buff = Buffer(self) #initiating the buffer
        self.fixed.update_unblocked()
        self.vertices = self.create_and_buffer_vertices(milling_path=False) # create and buffer vertices
        self.mesh = Geometries(self, hfs=hfs)
        self.sugs = []
        self.gals = []
        self.update_suggestions()
        self.combine_and_buffer_indices()
        self.gallary_start_index = -20
        self.incremental = incremental

    def create_and_buffer_vertices(self, milling_path=False):
        self.jverts = []
        self.everts = []
        self.mverts = []
        self.gcodeverts = []

        for ax in range(3):
            self.jverts.append(self.create_joint_vertices(ax))

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
            mst = 3*self.vn+self.van
            for n in range(self.noc):
                self.m_start.append(mst)
                mst += int(len(self.mverts[n])/8)

        self.buff.buffer_vertices()

    def create_joint_vertices(self, ax):
        vertices = []
        r = g = b = 0.0
        # Create vectors - one for each of the 3 axis
        vx = np.array([1.0,0,0])*self.voxel_sizes[0]
        vy = np.array([0,1.0,0])*self.voxel_sizes[1]
        vz = np.array([0,0,1.0])*self.voxel_sizes[2]
        self.pos_vecs = [vx,vy,vz]
        # If it is possible to rotate the geometry, rotate position vectors
        if self.rot:
            non_sax = [0,1,2]
            non_sax.remove(self.sax)
            for i,ax in enumerate(non_sax):
                theta = math.radians(0.5*self.ang)
                if i%2==1: theta = -theta
                self.pos_vecs[ax] = rotate_vector_around_axis(self.pos_vecs[ax],self.pos_vecs[self.sax],theta)
                self.pos_vecs[ax] = self.pos_vecs[ax]/math.cos(math.radians(abs(self.ang)))
        # Add all vertices of the dim*dim*dim voxel cube
        for i in range(self.dim+1):
            for j in range(self.dim+1):
                for k in range(self.dim+1):
                    # position coordinates
                    ivec = (i-0.5*self.dim)*self.pos_vecs[0]
                    jvec = (j-0.5*self.dim)*self.pos_vecs[1]
                    kvec = (k-0.5*self.dim)*self.pos_vecs[2]
                    pos = ivec+jvec+kvec
                    x,y,z = pos
                    # texture coordinates
                    tex_coords = [i,j,k] ##################################################################
                    tex_coords.pop(ax)
                    tx = tex_coords[0]/self.dim
                    ty = tex_coords[1]/self.dim
                    # extend list of vertices
                    vertices.extend([x,y,z,r,g,b,tx,ty])
        # Calculate extra length for angled components
        extra_len=0
        if self.ang!=0.0 and self.rot:
            extra_len = 0.1*self.component_size*math.tan(math.radians(abs(self.ang)))
        # Add component base vertices
        for ax in range(3):
            if ax==self.sax: extra_l=0
            else: extra_l=extra_len
            for dir in range(-1,2,2):
                for step in range(3):
                    if step==0: step=1
                    else: step+=0.5+extra_len
                    axvec = dir*step*(self.component_size+extra_l)*self.pos_vecs[ax]/np.linalg.norm(self.pos_vecs[ax])
                    for x in range(2):
                        for y in range(2):
                            other_vecs = copy.deepcopy(self.pos_vecs)
                            other_vecs.pop(ax)
                            if ax!=self.sax and self.rot and step!=0.5:
                                #cvec = copy.deep(self.pos_vecs[ax])
                                xvec = (x-0.5)*self.dim*other_vecs[0]#+cvec
                                yvec = (y-0.5)*self.dim*other_vecs[1]#-cvec
                            else:
                                xvec = (x-0.5)*self.dim*other_vecs[0]
                                yvec = (y-0.5)*self.dim*other_vecs[1]
                            pos = axvec+xvec+yvec
                            # texture coordinates
                            tex_coords = [x,y]
                            tx = tex_coords[0]
                            ty = tex_coords[1]
                            # extend list of vertices
                            vertices.extend([pos[0],pos[1],pos[2],r,g,b,tx,ty])
        # Format
        vertices = np.array(vertices, dtype = np.float32) #converts to correct format
        return vertices

    def combine_and_buffer_indices(self, milling_path=False):
        self.update_suggestions()
        self.mesh.create_indices(milling_path=milling_path)
        glo_off = len(self.mesh.indices) # global offset
        for i in range(len(self.sugs)):
            self.sugs[i].create_indices(glo_off=glo_off,milling_path=False)
            glo_off+=len(self.sugs[i].indices)
        for i in range(len(self.gals)):
            self.gals[i].create_indices(glo_off=glo_off,milling_path=False)
            glo_off+=len(self.gals[i].indices)
        indices = []
        indices.extend(self.mesh.indices)
        for mesh in self.sugs: indices.extend(mesh.indices)
        for mesh in self.gals: indices.extend(mesh.indices)
        self.indices = np.array(indices, dtype=np.uint32)
        Buffer.buffer_indices(self.buff)

    def update_sliding_direction(self,sax):
        blocked = False
        for i,sides in enumerate(self.fixed.sides):
            for side in sides:
                if side.ax==sax:
                    if side.dir==0 and i==0: continue
                    if side.dir==1 and i==self.noc-1: continue
                    blocked = True
        if blocked:
            return False, "This sliding direction is blocked"
        else:
            self.sax = sax
            self.fixed.update_unblocked()
            self.create_and_buffer_vertices(milling_path=False)
            self.mesh.voxel_matrix_from_height_fields()
            for mesh in self.sugs: mesh.voxel_matrix_from_height_fields()
            self.combine_and_buffer_indices()
            return True, ''

    def update_dimension(self,add):
        self.dim+=add
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.create_and_buffer_vertices(milling_path=False)
        self.mesh.randomize_height_fields()

    def update_angle(self,ang):
        self.ang = ang
        self.create_and_buffer_vertices(milling_path=False)

    def update_timber_width_and_height(self,inds,val,milling_path=False):
        for i in inds: self.real_tim_dims[i]=val
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab.vdia = self.fab.dia/self.ratio
        self.fab.vrad = self.fab.rad/self.ratio
        self.fab.vtol = self.fab.tol/self.ratio
        self.create_and_buffer_vertices(milling_path)

    def update_number_of_components(self,new_noc):
        if new_noc!=self.noc:
            # Increasing number of components
            if new_noc>self.noc:
                if len(self.fixed.unblocked)>=(new_noc-self.noc):
                    for i in range(new_noc-self.noc):
                        random_i = random.randint(0,len(self.fixed.unblocked)-1)
                        if self.fixed.sides[-1][0].ax==self.sax: # last component is aligned with the sliding axis
                            self.fixed.sides.insert(-1,[self.fixed.unblocked[random_i]])
                        else:
                            self.fixed.sides.append([self.fixed.unblocked[random_i]])
                        #also consider if it is aligned and should be the first one in line... rare though...
                        self.fixed.update_unblocked()
                    self.noc = new_noc
            # Decreasing number of components
            elif new_noc<self.noc:
                for i in range(self.noc-new_noc):
                    self.fixed.sides.pop()
                self.noc = new_noc
            # Rebuffer
            self.fixed.update_unblocked()
            self.create_and_buffer_vertices(milling_path=False)
            self.mesh.randomize_height_fields()

    def update_component_position(self,new_sides,n):
        self.fixed.sides[n] = new_sides
        self.fixed.update_unblocked()
        self.create_and_buffer_vertices(milling_path=False)
        self.mesh.voxel_matrix_from_height_fields()
        self.combine_and_buffer_indices()

    def reset(self, fs=None, sax=2, dim=3, ang=90., td=[44.0,44.0,44.0], incremental=False, align_ax=0, fabdia=6.0, fabtol=0.15, finterp=True, fabrot=0.0, fabext="gcode", hfs=[], fspe=400, fspi=600):
        self.fixed = FixedSides(self,fs=fs)
        self.noc=len(self.fixed.sides)
        self.sax=sax
        self.dim=dim
        self.ang=ang
        self.real_tim_dims = np.array(td)
        self.ratio = np.average(self.real_tim_dims)/self.component_size
        self.voxel_sizes = np.copy(self.real_tim_dims)/(self.ratio*self.dim)
        self.fab.tol=fabtol
        self.fab.real_dia = fabdia
        self.fab.rad = 0.5*self.fab.real_dia-self.fab.tol
        self.fab.dia = 2*self.fab.rad
        self.fab.vdia = self.fab.dia/self.ratio
        self.fab.vrad = self.fab.rad/self.ratio
        self.fab.vtol = self.fab.tol/self.ratio
        self.fab.speed = fspe
        self.fab.spindlespeed = fspi
        self.fab.extra_rot_deg=fabrot
        self.fab.ext=fabext
        self.fab.align_ax=align_ax
        self.fab.interp = finterp
        self.incremental=incremental
        self.mesh = Geometries(self, hfs=hfs)
        self.fixed.update_unblocked()
        self.create_and_buffer_vertices(milling_path=False)
        self.combine_and_buffer_indices()

    def update_suggestions(self):
        self.sugs = [] # clear list of suggestions
        if self.suggestions_on:
            sugg_hfs = []
            if not self.mesh.eval.valid:
                sugg_hfs = produce_suggestions(self,self.mesh.height_fields)
                for i in range(len(sugg_hfs)): self.sugs.append(Geometries(self,mainmesh=False,hfs=sugg_hfs[i]))

    def init_gallery(self,start_index):
        self.gallary_start_index = start_index
        self.gals = []
        self.sugs = []
        # Folder
        location = os.path.abspath(os.getcwd())
        location = location.split(os.sep)
        location.pop()
        location = os.sep.join(location)
        location += os.sep+"search_results"+os.sep+"noc_"+str(self.noc)+os.sep+"dim_"+str(self.dim)+os.sep+"fs_"
        for i in range(len(self.fixed.sides)):
            for fs in self.fixed.sides[i]:
                location+=str(fs.ax)+str(fs.dir)
            if i!=len(self.fixed.sides)-1: location+=("_")
        location+=os.sep+"allvalid"
        maxi = len(os.listdir(location))-1
        for i in range(20):
            if (i+start_index)>maxi: break
            try:
                hfs = np.load(location+os.sep+"height_fields_"+str(start_index+i)+".npy")
                self.gals.append(Geometries(self,mainmesh=False,hfs=hfs))
            except:
                abc = 0

    def save(self,filename="joint.tsu"):

        #Inititate
        file = open(filename,"w")

        # Joint properties
        file.write("SAX "+str(self.sax)+"\n")
        file.write("NOT "+str(self.noc)+"\n")
        file.write("RES "+str(self.dim)+"\n")
        file.write("ANG "+str(self.ang)+"\n")
        file.write("TDX "+str(self.real_tim_dims[0])+"\n")
        file.write("TDY "+str(self.real_tim_dims[1])+"\n")
        file.write("TDZ "+str(self.real_tim_dims[2])+"\n")
        file.write("DIA "+str(self.fab.real_dia)+"\n")
        file.write("TOL "+str(self.fab.tol)+"\n")
        file.write("SPE "+str(self.fab.speed)+"\n")
        file.write("SPI "+str(self.fab.spindlespeed)+"\n")
        file.write("INC "+str(self.incremental)+"\n")
        file.write("FIN "+str(self.fab.interp)+"\n")
        file.write("ALN "+str(self.fab.align_ax)+"\n")
        file.write("EXT "+self.fab.ext+"\n")

        # Fixed sides
        file.write("FSS ")
        for n in range(len(self.fixed.sides)):
            for i in range(len(self.fixed.sides[n])):
                file.write(str(int(self.fixed.sides[n][i].ax))+",")
                file.write(str(int(self.fixed.sides[n][i].dir)))
                if i!=len(self.fixed.sides[n])-1: file.write(".")
            if n!=len(self.fixed.sides)-1: file.write(":")

        # Joint geometry
        file.write("\nHFS \n")
        for n in range(len(self.mesh.height_fields)):
            for i in range(len(self.mesh.height_fields[n])):
                for j in range(len(self.mesh.height_fields[n][i])):
                    file.write(str(int(self.mesh.height_fields[n][i][j])))
                    if j!=len(self.mesh.height_fields[n][i])-1: file.write(",")
                if i!=len(self.mesh.height_fields[n])-1: file.write(":")
            if n!=len(self.mesh.height_fields)-1: file.write("\n")

        #Finalize
        print("Saved",filename)
        file.close()

    def open(self,filename="joint.tsu"):

        # Open
        file = open(filename,"r")

        # Default values
        sax = self.sax
        noc = self.noc
        dim = self.dim
        ang = self.ang
        dx, dy, dz = self.real_tim_dims
        dia = self.fab.real_dia
        tol = self.fab.tol
        spe = self.fab.speed
        spi = self.fab.spindlespeed
        inc = self.incremental
        aln = self.fab.align_ax
        ext = self.fab.ext
        fs = self.fixed.sides
        fin = self.fab.interp

        # Read
        hfs = []
        hfi = 999
        for i,line in enumerate(file.readlines()):
            items = line.split( )
            if items[0]=="SAX": sax = int(items[1])
            elif items[0]=="NOT": noc = int(items[1])
            elif items[0]=="RES": dim = int(items[1])
            elif items[0]=="ANG": ang = float(items[1])
            elif items[0]=="TDX": dx = float(items[1])
            elif items[0]=="TDY": dy = float(items[1])
            elif items[0]=="TDZ": dz = float(items[1])
            elif items[0]=="DIA": dia = float(items[1])
            elif items[0]=="TOL": tol = float(items[1])
            elif items[0]=="SPE": spe = float(items[1])
            elif items[0]=="SPI": spi = float(items[1])
            elif items[0]=="INC":
                if items[1]=="True": inc = True
                else: inc = False
            elif items[0]=="FIN":
                if items[1]=="True": fin = True
                else: fin = False
            elif items[0]=="ALN": aln = float(items[1])
            elif items[0]=="EXT": ext = items[1]
            elif items[0]=="FSS": fs = FixedSides(self,side_str=items[1]).sides
            elif items[0]=="HFS": hfi = i
            elif i>hfi:
                hf = []
                for row in line.split(":"):
                    temp = []
                    for item in row.split(","): temp.append(int(float(item)))
                    hf.append(temp)
                hfs.append(hf)
        hfs = np.array(hfs)

        # Reinitiate
        self.reset(fs=fs, sax=sax, dim=dim, ang=ang, td=[dx,dy,dz], fabdia=dia, fabtol=tol, align_ax=aln, finterp=fin, incremental=inc, fabext=ext, hfs=hfs, fspe=spe, fspi=spi)
