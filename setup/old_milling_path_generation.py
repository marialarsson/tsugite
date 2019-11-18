def milling_path_vertices(self,n):
    vertices = []
    # Parameters
    r = g = b = tx = ty = 0.0
    vrad = self.fab.rad/self.fab.ratio #virtual radius
    vext = self.fab.ext/self.fab.ratio
    # Define number of steps
    if (self.fab.real_voxel_size/self.fab.dia)<1.0: print("Could not generate milling path. The milling bit is too large")
    else:
        no_z = int(self.fab.real_voxel_size/self.fab.dep)
        dep = self.voxel_size/no_z
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
        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):
                    ind = [j,k]
                    ind.insert(ax,(self.dim-1)*(1-dir)+(2*n-1)*i) # 0 when n is 1, dim-1 when n is 0
                    val = self.voxel_matrix[tuple(ind)]
                    # Check weather the 4 sides are free or blocked
                    dir_sides,off_sides = check_free_sides(self,ind,dir_ax,off_ax,n)
                    if int(val)==n:
                        # check if it is on a fixed side of the other material
                        other_fixed_sides = self.fixed_sides.copy()
                        other_fixed_sides.pop(n)
                        on_side = False
                        for sides in other_fixed_sides:
                            for side in sides:
                                if ind[side[0]]==side[1]*(self.dim-1):
                                    on_side = True
                                    fside = side
                        if on_side==True:
                            if fside[0]==off_ax:
                                off_side_ax = dir_ax
                                dir_side_ax = off_ax
                                off_side_vec = dir_vec
                                dir_side_vec = off_vec
                            else:
                                off_side_ax = off_ax
                                dir_side_ax = dir_ax
                                off_side_vec = off_vec
                                dir_side_vec = dir_vec
                            line = []
                            #print(fside[1]*(self.dim-1))
                            add = [0,0,0]
                            add[ax] = 1-n
                            add[fside[0]] = fside[1]
                            i_pt = get_index(ind,add,self.dim)
                            pt1 = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
                            add[off_side_ax] = 1
                            i_pt = get_index(ind,add,self.dim)
                            pt2 = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
                            #
                            off_ratio = math.sqrt(2)-1
                            if ind[off_side_ax]>0 and dir_sides[0]==1:
                                outline = []
                                pt1a = pt1-off_ratio*vrad*off_side_vec
                                pt1b = pt1+off_ratio*vrad*dir_side_vec*(2*fside[1]-1)
                                outline.extend([pt1a,pt1b])
                                #vertices.extend(get_layered_vertices(self,outline,n,i,ax,no_z,dep,r,g,b,tx,ty))
                            #
                            if ind[off_side_ax]<self.dim-1 and dir_sides[1]==1:
                                outline = []
                                pt2a = pt2+off_ratio*vrad*(2*fside[1]-1)*dir_side_vec
                                pt2b = pt2+off_ratio*vrad*off_side_vec
                                outline.extend([pt2a,pt2b])
                                #vertices.extend(get_layered_vertices(self,outline,n,i,ax,no_z,dep,r,g,b,tx,ty))
                            #

                        continue # dont cut out if material is there, continue
                    # Calculate lane width and no of lanes based on if pre or next in off is free
                    #
                    # get 4 corners of the top of the voxel (a,b,c,d)
                    outline = []
                    for a in range(2):
                        temp = []
                        for b in range(2):
                            if a==1: b=1-b
                            add = [0,0,0]
                            add[dir_ax] = a
                            add[off_ax] = b
                            add[ax] = 1-n
                            i_pt = get_index(ind,add,self.dim)
                            pt = get_vertex(i_pt,self.jverts[n],self.vertex_no_info)
                            if dir_sides[a]==1 and off_sides[b]==1: #and blocked diagonal...
                                pt0 = pt+vrad*dir_vec*(1-2*a)+vext*off_vec*(2*b-1)
                                pt1 = pt+vrad*off_vec*(1-2*b)+vext*dir_vec*(2*a-1)
                                if a==b: outline.extend([pt0,pt1])
                                else: outline.extend([pt1,pt0])
                            else:
                                if dir_sides[a]==0: pt = pt+vrad*dir_vec*(1-2*a) #blocked
                                else: pt = pt+vext*dir_vec*(2*a-1) #unblocked
                                if off_sides[b]==0: pt = pt+vrad*off_vec*(1-2*b) #blocked
                                else: pt = pt+vext*off_vec*(2*b-1) #unblocked
                                outline.append(pt)
                    outline.append(outline[0])
                    # Offset outline
                    #
                    # Add zdept
                    vertices.extend(get_layered_vertices(self,outline,n,i,ax,no_z,dep,r,g,b,tx,ty))
        vertices = np.array(vertices, dtype = np.float32)
    return vertices
