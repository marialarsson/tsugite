import numpy as np

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

def get_sliding_directions(mat,noc):
    sliding_directions = []
    for n in range(noc): # Browse the components
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
                        elif first_same==True and (col[i]!=n and col[i]>0):
                            slides_in_this_direction=False; break
                    if slides_in_this_direction==False: break #stop checking further columns if one was blocking the slide
                if slides_in_this_direction==True:
                    mat_sliding.append([ax,dir])
        sliding_directions.append(mat_sliding)
    return sliding_directions

def add_fixed_sides(mat,fixed_sides):
    dim = len(mat)
    pad_loc = [[0,0],[0,0],[0,0]]
    pad_val = [[-1,-1],[-1,-1],[-1,-1]]
    for n in range(len(fixed_sides)):
        for ax,dir in fixed_sides[n]:
            pad_loc[ax][dir] = 1
            pad_val[ax][dir] = n
    pad_loc = tuple(map(tuple, pad_loc))
    pad_val = tuple(map(tuple, pad_val))
    mat = np.pad(mat, pad_loc, 'constant', constant_values=pad_val)
    # Take care of corners ########################needs to be adjusted for 3 components....!!!!!!!!!!!!!!!!!!
    for ax,dir in fixed_sides[0]:
        for ax2,dir2 in fixed_sides[1]:
            if ax==ax2: continue
            for i in range(dim):
                ind = [i,i,i]
                ind[ax] =  dir*(mat.shape[ax]-1)
                ind[ax2] = dir2*(mat.shape[ax2]-1)
                mat[tuple(ind)] = -1
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

class Evaluation:
    def __init__(self,parent):
        self.parent = parent
        self.slides = []
        self.connected = []
        self.bridged = []
        self.voxel_matrix_connected = None
        self.voxel_matrix_unconnected = None
        self.voxel_matrices_unbridged = []
        self.update(parent)

    def update(self,parent):
        self.voxel_matrix_with_sides = add_fixed_sides(parent.voxel_matrix, parent.fixed_sides)
        # Sliding directions
        self.slides = get_sliding_directions(self.voxel_matrix_with_sides,parent.noc)
        # Friction
        friciton = get_friction(self.voxel_matrix_with_sides,self.slides)
        # Voxel connection and bridgeing
        self.connected = []
        self.bridged = []
        self.voxel_matrices_unbridged = []
        for n in range(parent.noc):
            self.connected.append(is_connected(self.voxel_matrix_with_sides,n))
            self.bridged.append(True)
            self.voxel_matrices_unbridged.append(None)
        self.voxel_matrix_connected = parent.voxel_matrix.copy()
        self.voxel_matrix_unconnected = None
        if not all(self.connected):
            self.seperate_unconnected(parent)
            # Bridging
            if parent.joint_type=="T" or parent.joint_type=="X":
                voxel_matrix_connected_with_sides = add_fixed_sides(self.voxel_matrix_connected, parent.fixed_sides)
                for n in range(parent.noc):
                    self.bridged[n] = is_connected(voxel_matrix_connected_with_sides,n)
                    if not self.bridged[n]:
                        voxel_matrix_unbridged_1, voxel_matrix_unbridged_2 = self.seperate_unbridged(parent,n)
                        self.voxel_matrices_unbridged[n] = [voxel_matrix_unbridged_1, voxel_matrix_unbridged_2]

    def seperate_unconnected(self,parent):
        connected_mat = np.zeros((parent.dim,parent.dim,parent.dim))-1
        unconnected_mat = np.zeros((parent.dim,parent.dim,parent.dim))-1
        for i in range(parent.dim):
            for j in range(parent.dim):
                for k in range(parent.dim):
                    connected = False
                    ind = [i,j,k]
                    val = parent.voxel_matrix[tuple(ind)]
                    connected = is_connected_to_fixed_side(np.array([ind]),parent.voxel_matrix,parent.fixed_sides[int(val)])
                    if connected: connected_mat[tuple(ind)] = val
                    else: unconnected_mat[tuple(ind)] = val
        self.voxel_matrix_connected = connected_mat
        self.voxel_matrix_unconnected = unconnected_mat

    def seperate_unbridged(self,parent,n):
        unbridged_1 = np.zeros((parent.dim,parent.dim,parent.dim))-1
        unbridged_2 = np.zeros((parent.dim,parent.dim,parent.dim))-1
        for i in range(parent.dim):
            for j in range(parent.dim):
                for k in range(parent.dim):
                    ind = [i,j,k]
                    val = parent.voxel_matrix[tuple(ind)]
                    if val!=n: continue
                    conn_1 = is_connected_to_fixed_side(np.array([ind]),parent.voxel_matrix,[parent.fixed_sides[n][0]])
                    conn_2 = is_connected_to_fixed_side(np.array([ind]),parent.voxel_matrix,[parent.fixed_sides[n][1]])
                    if conn_1: unbridged_1[tuple(ind)] = val
                    if conn_2: unbridged_2[tuple(ind)] = val
        return unbridged_1, unbridged_2
