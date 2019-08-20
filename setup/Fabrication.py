import numpy as np

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

        self.rad = 0.015 #milling bit radius
        self.dep = 0.015 #milling depth
        self.milling_path_A = self.milling_path_B = None
        
class Fabrication:
    def __init__(self, path_vertices, virtual_component_size, sliding_direction, joint_type, n):
        self.path_vertices = path_vertices
        self.sliding_direction = sliding_direction
        self.n = n
        self.real_component_size = 36 #mm
        self.ratio = self.real_component_size/virtual_component_size
        if self.sliding_direction[0]==2: self.coords = [0,1,2]
        elif self.sliding_direction[0]==1: self.coords = [2,0,1]
        #if n==1: self.coords[2] = -self.coords[2]

    def export_gcode(self,file_name):
        d = 3 # =precision / no of decimals to write
        file = open("gcode/"+file_name+".gcode","w")
        ###initialization
        file.write("%\n")
        file.write("G90 (Absolute [G91 is incremental])\n")
        file.write("G21 (set unit[mm])\n")
        file.write("G54\n")
        file.write("F1000.0S6000 (Feeding 2000mm/min, Spindle 6000rpm)\n")
        file.write("G17 (set XY plane for circle path)\n")
        file.write("M03 (spindle start)\n")
        speed = 400
        ###content
        x_ = str(9999999999)
        y_ = str(9999999999)
        z_ = str(9999999999)
        for i in range(0,len(self.path_vertices),8):
            x = self.path_vertices[i]
            y = self.path_vertices[i+1]
            z = self.path_vertices[i+2]
            #convert from virtul dimensions to mm
            x = self.ratio*x
            y = self.ratio*y
            z = self.ratio*z
            #sawp
            xyz = [x,y,z]
            xyz = xyz[self.coords[0]],xyz[self.coords[1]],xyz[self.coords[2]]
            x,y,z = xyz[0],xyz[1],xyz[2]
            #move z down, flip if component b
            z = -(2*self.n-1)*z-0.5*self.real_component_size
            #string
            x = str(round(x,d))
            y = str(round(y,d))
            z = str(round(z,d))
            #write to file
            if x_!=x or y_!=y: file.write("G01 X "+x+" Y "+y+" F "+str(speed)+"\n")
            if z_!=z: file.write("G01 Z "+z+" F "+str(speed)+"\n")
            x_ = x
            y_ = y
            z_ = z
        ###end
        file.write("M05 (Spindle stop)\n")
        file.write("M02(end of program)\n")
        file.write("%\n")
        file.close()
