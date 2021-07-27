import numpy as np
import copy
from Fabrication import RegionVertex
from Misc import FixedSide

def get_ordered_outline(verts):
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
                # check if there is an available vertex
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
    return ord_verts

def get_friction_and_contact_areas(mat,slides,fixed_sides,n):
    friction = -1
    contact = -1
    ffaces = []
    cfaces = []
    if len(slides)>0:
        friction = 0
        contact = 0
        other_fixed_sides = []
        for n2 in range(len(fixed_sides)):
            if n==n2: continue
            other_fixed_sides.extend(fixed_sides[n2])
        # Define which axes are acting in friction
        friction_axes = [0,1,2]
        bad_axes = []
        for item in slides: bad_axes.append(item[0])
        friction_axes = [x for x in friction_axes if x not in bad_axes]
        # Check neighbors in relevant axes. If neighbor is other, friction is acting!
        indices = np.argwhere(mat==n)
        for ind in indices:
            for ax in range(3):
                for dir in range(2):
                    nind = ind.copy()
                    nind[ax]+=2*dir-1
                    cont = False
                    if nind[ax]<0:
                        if FixedSide(ax,0).unique(other_fixed_sides): cont=True
                    elif nind[ax]>=len(mat):
                        if FixedSide(ax,1).unique(other_fixed_sides): cont=True
                    elif mat[tuple(nind)]!=n: cont = True
                    if cont:
                        contact+=1
                        find = ind.copy()
                        find[ax]+=dir
                        cfaces.append([ax,list(find)])
                        if ax in friction_axes:
                            friction += 1
                            ffaces.append([ax,list(find)])
        # Check neighbors for each fixed side of the current material
        for side in fixed_sides[n]:
            for i in range(len(mat)):
                for j in range(len(mat)):
                    nind = [i,j]
                    axind = side.dir*(len(mat)-1)
                    nind.insert(side.ax,axind)
                    if mat[tuple(nind)]!=n: # neighboring another timber
                        contact+=1
                        find = nind.copy()
                        find[side.ax]+=side.dir
                        cfaces.append([side.ax,list(find)])
                        if side.ax in friction_axes:
                            friction+=1
                            ffaces.append([side.ax,list(find)])
    return friction, ffaces, contact, cfaces

def is_connected(mat,n):
    connected = False
    all_same = np.count_nonzero(mat==n) # Count number of ones in matrix
    if all_same>0:
        ind = tuple(np.argwhere(mat==n)[0]) # Pick a random one
        inds = get_all_same_connected(mat,[ind]) # Get all its neighbors (recursively)
        connected_same = len(inds)
        if connected_same==all_same: connected = True
    return connected

def is_bridged(mat,n):
    bridged = False
    all_same = np.count_nonzero(mat==n) # Count number of ones in matrix
    if all_same>0:
        ind = tuple(np.argwhere(mat==n)[0]) # Pick a random one
        inds = get_all_same_connected(mat,[ind]) # Get all its neighbors (recursively)
        connected_same = len(inds)
        if connected_same==all_same: bridged = True
    return bridged

def get_sliding_directions(mat,noc):
    sliding_directions = []
    number_of_sliding_directions = []
    for n in range(noc): # Browse the components
        mat_sliding = []
        for ax in range(3): # Browse the three possible sliding axes
            oax = [0,1,2]
            oax.remove(ax)
            for dir in range(2): # Browse the two possible directions of the axis
                slides_in_this_direction = True
                for i in range(mat.shape[oax[0]]):
                    for j in range(mat.shape[oax[1]]):
                        first_same = False
                        for k in range(mat.shape[ax]):
                            if dir==0: k = mat.shape[ax]-k-1
                            ind = [i,j]
                            ind.insert(ax,k)
                            val = mat[tuple(ind)]
                            if val==n:
                                first_same = True
                                continue
                            elif first_same and val!=-1:
                                slides_in_this_direction=False
                                break
                        if slides_in_this_direction==False: break
                    if slides_in_this_direction==False: break
                if slides_in_this_direction==True:
                    mat_sliding.append([ax,dir])
        sliding_directions.append(mat_sliding)
        number_of_sliding_directions.append(len(mat_sliding))
    return sliding_directions,number_of_sliding_directions

def get_sliding_directions_of_one_timber(mat,level):
    sliding_directions = []
    n = level
    for ax in range(3): # Browse the three possible sliding axes
        oax = [0,1,2]
        oax.remove(ax)
        for dir in range(2): # Browse the two possible directions of the axis
            slides_in_this_direction = True
            for i in range(mat.shape[oax[0]]):
                for j in range(mat.shape[oax[1]]):
                    first_same = False
                    for k in range(mat.shape[ax]):
                        if dir==0: k = mat.shape[ax]-k-1
                        ind = [i,j]
                        ind.insert(ax,k)
                        val = mat[tuple(ind)]
                        if val==n:
                            first_same = True
                            continue
                        elif first_same and val!=-1:
                            slides_in_this_direction=False
                            break
                    if slides_in_this_direction==False: break
                if slides_in_this_direction==False: break
            if slides_in_this_direction==True:
                sliding_directions.append([ax,dir])
    number_of_sliding_directions = len(sliding_directions)
    return sliding_directions,number_of_sliding_directions

def add_fixed_sides(mat,fixed_sides, add=0):
    dim = len(mat)
    pad_loc = [[0,0],[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1],[-1,-1]]
    for n in range(len(fixed_sides)):
        for side in fixed_sides[n]:
            pad_loc[side.ax][side.dir] = 1
            pad_val[side.ax][side.dir] = n+add
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)
    # Take care of corners ########################needs to be adjusted for 3 components....!!!!!!!!!!!!!!!!!!
    for fixed_sides_1 in fixed_sides:
        for fixed_sides_2 in fixed_sides:
            for side in fixed_sides_1:
                for side2 in fixed_sides_2:
                    if side.ax==side2.ax: continue
                    for i in range(dim+2):
                        ind = [i,i,i]
                        ind[side.ax] =  side.dir*(mat.shape[side.ax]-1)
                        ind[side2.ax] = side2.dir*(mat.shape[side2.ax]-1)
                        try:
                            mat[tuple(ind)] = -1
                        except:
                            a = 0
    return mat

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

def reverse_columns(cols):
    new_cols = []
    for i in range(len(cols)):
        temp = []
        for j in range(len(cols[i])):
            temp.append(cols[i][len(cols[i])-j-1].astype(int))
        new_cols.append(temp)
    return new_cols

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

def is_connected_to_fixed_side(indices,mat,fixed_sides):
    connected = False
    val = mat[tuple(indices[0])]
    d = len(mat)
    for ind in indices:
        for side in fixed_sides:
            if ind[side.ax]==0 and side.dir==0:
                connected=True
                break
            elif ind[side.ax]==d-1 and side.dir==1:
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

def get_same_neighbors_2d(mat2,inds,val):
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if ind2[ax]>=0 and ind2[ax]<mat2.shape[ax]:
                    val2 = mat2[tuple(ind2)]
                    if val2!=val: continue
                    unique = True
                    for ind3 in new_inds:
                        if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                            unique = False
                            break
                    if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_same_neighbors_2d(mat2,new_inds,val)
    return new_inds

def get_chessboard_vertics(mat,ax,noc,n):
    chess = False
    dim = len(mat)
    verts = []
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                ind3d = [i,j,k]
                ind2d = ind3d.copy()
                ind2d.pop(ax)
                if ind2d[0]<1 or ind2d[1]<1: continue
                neighbors = []
                flat_neigbors = []
                for x in range(-1,1,1):
                    temp = []
                    for y in range(-1,1,1):
                        nind = ind2d.copy()
                        nind[0]+=x
                        nind[1]+=y
                        nind.insert(ax,ind3d[ax])
                        val = mat[tuple(nind)]
                        temp.append(val)
                        flat_neigbors.append(val)
                    neighbors.append(temp)
                flat_neigbors = np.array(flat_neigbors)
                ## check THIS material
                cnt = np.sum(flat_neigbors==n)
                if cnt==2:
                    # cheack diagonal
                    if neighbors[0][1]==neighbors[1][0] and neighbors[0][0]==neighbors[1][1]:
                        chess = True
                        verts.append(ind3d)
    return chess,verts

def is_connected_to_fixed_side_2d(inds,fixed_sides,ax,dim):
    connected = False
    for side in fixed_sides:
        fax2d = [0,0,0]
        fax2d[side.ax] = 1
        fax2d.pop(ax)
        fax2d = fax2d.index(1)
        for ind in inds:
            if ind[fax2d]==side.dir*(dim-1):
                connected = True
                break
        if connected: break
    return connected

def get_neighbors_2d(ind,reg_inds,lay_mat,n):
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
                    type = 1 # free
                    val =-1
                elif lay_mat[tuple(nind)]<0:
                    type = 1 # free
                else: type = 1 # blocked

            if val==None:
                val=lay_mat[tuple(nind)]

            temp.append(type)
            temp2.append(val)
        in_out.append(temp)
        values.append(temp2)
    return in_out, values

def get_region_outline(reg_inds,lay_mat,fixed_neighbors,n):
    # also duplicate vertices on diagonal
    reg_verts = []
    for i in range(lay_mat.shape[0]+1):
        for j in range(lay_mat.shape[1]+1):
            ind = [i,j]
            neigbors,neighbor_values = get_neighbors_2d(ind,reg_inds,lay_mat,n)
            neigbors = np.array(neigbors)
            if np.any(neigbors.flatten()==0) and not np.all(neigbors.flatten()==0): # some but not all region neighbors
                dia1 = neigbors[0][1]==neigbors[1][0]
                dia2 = neigbors[0][0]==neigbors[1][1]
                if np.sum(neigbors.flatten()==0)==2 and  np.sum(neigbors.flatten()==1)==2 and dia1 and dia2: # diagonal detected
                    other_indices = np.argwhere(neigbors==0)
                    for oind in other_indices:
                        oneigbors = copy.deepcopy(neigbors)
                        oneigbors[tuple(oind)] = 1
                        oneigbors = np.array(oneigbors)
                        reg_verts.append(RegionVertex(ind,ind,oneigbors,neighbor_values,dia=True))
                else: # normal situation
                    reg_verts.append(RegionVertex(ind,ind,neigbors,neighbor_values))
    return reg_verts

def get_breakable_voxels(mat,fixed_sides,sax,n):
    breakable = False
    outline_indices = []
    voxel_indices = []
    dim = len(mat)
    gax = fixed_sides[0].ax # grain axis
    if gax!=sax: # if grain direction does not equal to the sliding direction

        paxes = [0,1,2]
        paxes.pop(gax) # perpendicular to grain axis

        for pax in paxes:

            potentially_fragile_reg_inds = []

            for lay_num in range(dim):
                temp = []

                lay_mat = layer_mat(mat,pax,dim,lay_num)

                for reg_num in range(dim*dim): # region number

                    # Get indices of a region
                    inds = np.argwhere((lay_mat!=-1) & (lay_mat==n))
                    if len(inds)==0: break

                    reg_inds = get_same_neighbors_2d(lay_mat,[inds[0]],n)

                    # Check if any item in this region is connected to a fixed side
                    fixed = is_connected_to_fixed_side_2d(reg_inds,fixed_sides,pax,dim)

                    if not fixed: temp.append(reg_inds)

                    # Overwrite detected regin in original matrix
                    for reg_ind in reg_inds: lay_mat[tuple(reg_ind)]=-1

                potentially_fragile_reg_inds.append(temp)


            for lay_num in range(dim):

                lay_mat = layer_mat(mat,pax,dim,lay_num)

                for reg_inds in potentially_fragile_reg_inds[lay_num]:

                    # Is any voxel of this region connected to fixed materials in any axial direction?
                    fixed_neighbors = [False,False]

                    for reg_ind in reg_inds:

                        # get 3d index
                        ind3d = reg_ind.copy()
                        ind3d = list(ind3d)
                        ind3d.insert(pax,lay_num)
                        for dir in range(-1,2,2): #-1/1

                            # check neigbor in direction
                            ind3d_dir = ind3d.copy()
                            ind3d_dir[pax] += dir
                            if ind3d_dir[pax]>=0 and ind3d_dir[pax]<dim:
                                # Is there any material at all?
                                val = mat[tuple(ind3d_dir)]
                                if val==n: # There is material
                                    # Is this material in the list of potentially fragile or not?
                                    attached_to_fragile=False
                                    ind2d_dir = ind3d_dir.copy()
                                    ind2d_dir.pop(pax)
                                    for dir_reg_inds in potentially_fragile_reg_inds[lay_num+dir]:
                                        for dir_ind in dir_reg_inds:
                                            if dir_ind[0]==ind2d_dir[0] and dir_ind[1]==ind2d_dir[1]:
                                                attached_to_fragile = True
                                                break
                                    # need to check more steps later...####################################!!!!!!!!!!!!!!!
                                    if not attached_to_fragile:
                                        fixed_neighbors[int((dir+1)/2)] = True

                    if fixed_neighbors[0]==False or fixed_neighbors[1]==False:
                        breakable = True

                        # Append to list of breakable voxel indices
                        for ind in reg_inds:
                            ind3d = list(ind.copy())
                            ind3d.insert(pax,lay_num)
                            voxel_indices.append(ind3d)

                        # Get region outline
                        outline = get_region_outline(reg_inds,lay_mat,fixed_neighbors,n)

                        # Order region outline
                        outline = get_ordered_outline(outline)
                        outline.append(outline[0])

                        for dir in range(0,2):
                            #if not fixed_neighbors[dir]: continue
                            for i in range(len(outline)-1):
                                for j in range(2):
                                    oind = outline[i+j].ind.copy()
                                    oind.insert(pax,lay_num)
                                    oind[pax]+=dir
                                    outline_indices.append(oind)

    return breakable,outline_indices,voxel_indices

def is_fab_direction_ok(mat,ax,n):
    fab_dir = 1
    dim = len(mat)
    for dir in range(2):
        is_ok = True
        for i in range(dim):
            for j in range(dim):
                found_first_same = False
                for k in range(dim):
                    if dir==0: k = dim-k-1
                    ind = [i,j]
                    ind.insert(ax,k)
                    val = mat[tuple(ind)]
                    if val==n: found_first_same=True
                    elif found_first_same: is_ok=False; break
                if not is_ok: break
            if not is_ok: break
        if is_ok:
            fab_dir=dir
            break
    return is_ok, fab_dir

def layer_mat(mat3d,ax,dim,lay_num):
    mat2d = np.ndarray(shape=(dim,dim), dtype=int)
    for i in range(dim):
        for j in range(dim):
            ind = [i,j]
            ind.insert(ax,lay_num)
            mat2d[i][j]=int(mat3d[tuple(ind)])
    return mat2d

def open_matrix(mat,sax,noc):
    # Pad matrix by correct number of rows top and bottom
    dim = len(mat)
    pad_loc = [[0,0],[0,0],[0,0]]
    pad_loc[sax] = [0,noc-1]
    pad_val = [[-1,-1],[-1,-1],[-1,-1]]
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)

    # Move integers one step at the time
    for i in range(noc-1,0,-1):
        inds = np.argwhere(mat==i)
        for ind in inds:
            mat[tuple(ind)]=-1
        for ind in inds:
            ind[sax]+=i
            mat[tuple(ind)]=i
    return mat

def flood_all_nonneg(mat,floodval):
    inds = np.argwhere(mat==floodval)
    start_len = len(inds)
    for ind in inds:
        for ax in range(3):
            for dir in range(-1,2,2):
                #define neighbor index
                ind2 = np.copy(ind)
                ind2[ax]+=dir
                #within bounds?
                if ind2[ax]<0: continue
                if ind2[ax]>=mat.shape[ax]: continue
                #relevant value?
                val = mat[tuple(ind2)]
                if val<0 or val==floodval: continue
                #overwrite
                mat[tuple(ind2)]=floodval
    end_len = len(np.argwhere(mat==floodval))
    if end_len>start_len:
        mat = flood_all_nonneg(mat,floodval)
    return mat

def is_potentially_connected(mat,dim,noc,level):
    potconn=True
    mat[mat==level] = -1
    mat[mat==level+10] = -1

    # 1. Check for connectivity
    floodval = 99
    mat_conn = np.copy(mat)
    flood_start_vals = []
    for n in range(noc):
        if n!=level: mat_conn[mat_conn==n+10] = floodval

    # Recursively add all positive neigbors
    mat_conn = flood_all_nonneg(mat_conn,floodval)

    # Get the count of all uncovered voxels
    uncovered_inds = np.argwhere((mat_conn!=floodval)&(mat_conn>=0))
    if len(uncovered_inds)>0: potconn=False


    if potconn:
        # 3. Check so that there are at least some (3) voxels that could connect to each fixed side
        for n in range(noc):
            if n==level: continue
            mat_conn = np.copy(mat)
            mat_conn[mat_conn==n+10] = floodval
            for n2 in range(noc):
                if n2==level or n2==n: continue
                mat_conn[mat_conn==n2+10] = -1
            start_len = len(np.argwhere(mat_conn==floodval))
            # Recursively add all positive neigbors
            mat_conn = flood_all_nonneg(mat_conn,floodval)
            end_len = len(np.argwhere(mat_conn==floodval))
            if end_len-start_len<3:
                potconn=False
                #print("too few potentially connected for",n,".difference:",end_len-start_len)
                #print(mat)
                break
        # 3. Check for potential bridging
        for n in range(noc):
            if n==level: continue
            inds = np.argwhere(mat==n+10)
            if len(inds)>dim*dim*dim: #i.e. if there are more than 1 fixed side
                mat_conn = np.copy(mat)
                mat_conn[tuple(inds[0])] = floodval #make 1 item 99
                for n2 in range(noc):
                    if n2==level or n2==n: continue
                    mat_conn[mat_conn==n2+10] = -1
                # Recursively add all positive neigbors
                mat_conn = flood_all_nonneg(mat_conn,floodval)
                for ind in inds:
                    if mat_conn[tuple(ind)]!=floodval:
                        potconn = False
                        #print("Not potentially bridgning")
                        break
    return potconn

class Evaluation:
    def __init__(self,voxel_matrix,type,mainmesh=True):
        self.mainmesh = mainmesh
        self.valid = True
        self.slides = []
        self.number_of_slides = []
        self.interlock = False
        self.interlocks = []
        self.connected = []
        self.bridged = []
        self.breakable = []
        self.checker = []
        self.checker_vertices = []
        self.fab_direction_ok = []
        self.voxel_matrix_connected = None
        self.voxel_matrix_unconnected = None
        self.voxel_matrices_unbridged = []
        self.breakable_outline_inds = []
        self.breakable_voxel_inds = []
        self.sliding_depths = []
        self.friction_nums = []
        self.friction_faces = []
        self.contact_nums = []
        self.contact_faces = []
        self.fab_directions = self.update(voxel_matrix,type)

    def update(self,voxel_matrix,type):
        self.voxel_matrix_with_sides = add_fixed_sides(voxel_matrix, type.fixed.sides)

        # Voxel connection and bridgeing
        self.connected = []
        self.bridged = []
        self.voxel_matrices_unbridged = []
        for n in range(type.noc):
            self.connected.append(is_connected(self.voxel_matrix_with_sides,n))
            self.bridged.append(True)
            self.voxel_matrices_unbridged.append(None)
        self.voxel_matrix_connected = voxel_matrix.copy()
        self.voxel_matrix_unconnected = None

        self.seperate_unconnected(voxel_matrix,type.fixed.sides,type.dim)

        # Bridging
        voxel_matrix_connected_with_sides = add_fixed_sides(self.voxel_matrix_connected, type.fixed.sides)
        for n in range(type.noc):
            self.bridged[n] = is_connected(voxel_matrix_connected_with_sides,n)
            if not self.bridged[n]:
                voxel_matrix_unbridged_1, voxel_matrix_unbridged_2 = self.seperate_unbridged(voxel_matrix,type.fixed.sides,type.dim,n)
                self.voxel_matrices_unbridged[n] = [voxel_matrix_unbridged_1, voxel_matrix_unbridged_2]

        # Fabricatability by direction constraint
        self.fab_direction_ok = []
        fab_directions = list(range(type.noc))
        for n in range(type.noc):
            if n==0 or n==type.noc-1:
                self.fab_direction_ok.append(True)
                if n==0: fab_directions[n]=0
                else: fab_directions[n]=1
            else:
                fab_ok,fab_dir = is_fab_direction_ok(voxel_matrix,type.sax,n)
                fab_directions[n] = fab_dir
                self.fab_direction_ok.append(fab_ok)

        # Chessboard
        self.checker = []
        self.checker_vertices = []
        for n in range(type.noc):
            check,verts = get_chessboard_vertics(voxel_matrix,type.sax,type.noc,n)
            self.checker.append(check)
            self.checker_vertices.append(verts)

        # Sliding directions
        self.slides,self.number_of_slides = get_sliding_directions(self.voxel_matrix_with_sides,type.noc)
        self.interlock = True
        for n in range(type.noc):
            if (n==0 or n==type.noc-1):
                if self.number_of_slides[n]<=1:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock=False
            else:
                if self.number_of_slides[n]==0:
                    self.interlocks.append(True)
                else:
                    self.interlocks.append(False)
                    self.interlock=False

        # Friction
        self.friction_nums = []
        self.friction_faces = []
        self.contact_nums = []
        self.contact_faces = []
        for n in range(type.noc):
            friction,ffaces,contact,cfaces, = get_friction_and_contact_areas(voxel_matrix,self.slides[n],type.fixed.sides,n)
            self.friction_nums.append(friction)
            self.friction_faces.append(ffaces)
            self.contact_nums.append(contact)
            self.contact_faces.append(cfaces)


        # Grain direction
        for n in range(type.noc):
            brk,brk_oinds,brk_vinds = get_breakable_voxels(voxel_matrix,type.fixed.sides[n],type.sax,n)
            self.breakable.append(brk)
            self.breakable_outline_inds.append(brk_oinds)
            self.breakable_voxel_inds.append(brk_vinds)
        self.non_breakable_voxmat, self.breakable_voxmat = self.seperate_voxel_matrix(voxel_matrix,self.breakable_voxel_inds)

        if not self.interlock or not all(self.connected) or not all(self.bridged):
            self.valid=False
        elif any(self.breakable) or any(self.checker) or not all(self.fab_direction_ok):
            self.valid=False

        """
        # Sliding depth
        sliding_depths = [3,3,3]
        open_mat = np.copy(self.voxel_matrix_with_sides)
        for depth in range(4):
            slds,nos = get_sliding_directions(open_mat,noc)
            for n in range(noc):
                if sliding_depths[n]!=3: continue
                if n==0 or n==noc-1:
                    if nos[n]>1: sliding_depths[n]=depth
                else:
                    if nos[n]>0: sliding_depths[n]=depth
            open_mat = open_matrix(open_mat,sax,noc)
        self.slide_depths = sliding_depths
        self.slide_depth_product = np.prod(np.array(sliding_depths))
        print(self.slide_depths,self.slide_depth_product)
        """
        return fab_directions

    def seperate_unconnected(self,voxel_matrix,fixed_sides,dim):
        connected_mat = np.zeros((dim,dim,dim))-1
        unconnected_mat = np.zeros((dim,dim,dim))-1
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    connected = False
                    ind = [i,j,k]
                    val = voxel_matrix[tuple(ind)]
                    connected = is_connected_to_fixed_side(np.array([ind]),voxel_matrix,fixed_sides[int(val)])
                    if connected: connected_mat[tuple(ind)] = val
                    else: unconnected_mat[tuple(ind)] = val
        self.voxel_matrix_connected = connected_mat
        self.voxel_matrix_unconnected = unconnected_mat

    def seperate_voxel_matrix(self,voxmat,inds):
        dim = len(voxmat)
        voxmat_a = copy.deepcopy(voxmat)
        voxmat_b = np.zeros((dim,dim,dim))-1
        for n in range(len(inds)):
            for ind in inds[n]:
                ind = tuple(ind)
                val = voxmat[ind]
                voxmat_a[ind] = -1
                voxmat_b[ind] = val
        return voxmat_a,voxmat_b

    def seperate_unbridged(self,voxel_matrix,fixed_sides,dim,n):
        unbridged_1 = np.zeros((dim,dim,dim))-1
        unbridged_2 = np.zeros((dim,dim,dim))-1
        for i in range(dim):
            for j in range(dim):
                for k in range(dim):
                    ind = [i,j,k]
                    val = voxel_matrix[tuple(ind)]
                    if val!=n: continue
                    conn_1 = is_connected_to_fixed_side(np.array([ind]),voxel_matrix,[fixed_sides[n][0]])
                    conn_2 = is_connected_to_fixed_side(np.array([ind]),voxel_matrix,[fixed_sides[n][1]])
                    if conn_1: unbridged_1[tuple(ind)] = val
                    if conn_2: unbridged_2[tuple(ind)] = val
        return unbridged_1, unbridged_2

class EvaluationOne:
    def __init__(self,voxel_matrix,fixed_sides,sax,noc,level,last):

        # Initiate metrics
        self.connected_and_bridged = True
        self.other_connected_and_bridged = True
        self.nocheck = True
        self.interlock = True
        self.nofragile = True
        self.valid = False

        # Add fixed sides to voxel matrix, get dimension
        self.voxel_matrix_with_sides = add_fixed_sides(voxel_matrix, fixed_sides)
        dim = len(voxel_matrix)

        #Connectivity and bridging
        self.connected_and_bridged = is_connected(self.voxel_matrix_with_sides,level)
        if not self.connected_and_bridged: return

        # Other connectivity and bridging
        if not last:
            other_level = 0
            if level==0: other_level = 1
            special_voxmat_with_sides = add_fixed_sides(voxel_matrix, fixed_sides, 10)
            self.other_connected_and_bridged = is_potentially_connected(special_voxmat_with_sides,dim,noc,level)
            if not self.other_connected_and_bridged: return

        # Checkerboard
        if last:
            check,verts = get_chessboard_vertics(voxel_matrix,sax,noc,level)
            if check: self.nocheck=False
            if not self.nocheck: return

        # Slidability
        self.slides,self.number_of_slides = get_sliding_directions_of_one_timber(self.voxel_matrix_with_sides,level)
        if level==0 or level==noc-1:
            if self.number_of_slides!=1: self.interlock=False
        else:
            if self.number_of_slides!=0: self.interlock=False
        if not self.interlock: return

        # Durability
        brk,brk_inds = get_breakable_voxels(voxel_matrix,fixed_sides[level],sax,level)
        if brk: self.nofragile = False
        if not self.nofragile: return

        self.valid=True

class EvaluationSlides:
    def __init__(self,voxel_matrix,fixed_sides,sax,noc):
        voxel_matrix_with_sides = add_fixed_sides(voxel_matrix, fixed_sides)
        # Sliding depth
        sliding_depths = [3,3,3]
        open_mat = np.copy(voxel_matrix_with_sides)
        for depth in range(4):
            slds,nos = get_sliding_directions(open_mat,noc)
            for n in range(noc):
                if sliding_depths[n]!=3: continue
                if n==0 or n==noc-1:
                    if nos[n]>1: sliding_depths[n]=depth
                else:
                    if nos[n]>0: sliding_depths[n]=depth
            open_mat = open_matrix(open_mat,sax,noc)
        self.slide_depths = sliding_depths
        #self.slide_depths_sorted = sliding_depths
        #self.slide_depth_product = np.prod(np.array(sliding_depths))
