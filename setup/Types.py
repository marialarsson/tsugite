from OpenGL.GL import *
import numpy as np
import math
import copy
from Buffer import Buffer
from Evaluation import Evaluation
from Geometries import Geometries
from Geometries import get_index
from Geometries import get_corner_indices

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
                tx = tex_coords[0]/self.dim
                ty = tex_coords[1]/self.dim
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

def produce_suggestions(type,hfs):
    valid_suggestions = []
    for i in range(len(hfs)):
        for j in range(type.dim):
            for k in range(type.dim):
                for add in range(-1,2,2):
                    sugg_hfs = copy.deepcopy(hfs)
                    sugg_hfs[i][j][k]+=add
                    val = sugg_hfs[i][j][k]
                    if val>=0 and val<type.dim:
                        sugg_voxmat = mat_from_fields(sugg_hfs,type.sax)
                        sugg_eval = Evaluation(sugg_voxmat,type)
                        if sugg_eval.valid:
                            valid_suggestions.append(sugg_hfs)
                            if len(valid_suggestions)==4: break
    return valid_suggestions

class Types:
    def __init__(self,fs=[[[2,0]],[[2,1]]],sax=2,dim=3):
        self.sax = sax
        self.fixed_sides = fs
        self.noc = len(fs) #number of components
        self.dim = dim
        self.component_size = 0.275
        self.voxel_size = self.component_size/self.dim
        self.component_length = 0.5*self.component_size
        self.vertex_no_info = 8
        self.buff = Buffer(self) #initiating the buffer
        self.vertices = self.create_and_buffer_vertices() # create and buffer vertices
        self.update_unblocked_fixed_sides()
        self.mesh = Geometries(self)
        self.sugs = []
        self.update_suggestions()
        self.combine_and_buffer_indices()

    def create_and_buffer_vertices(self, milling_path=False):
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

    def combine_and_buffer_indices(self):
        self.mesh.create_indices()
        for i in range(len(self.sugs)): self.sugs[i].create_indices(index=i)
        indices = []
        indices.extend(self.mesh.indices)
        for mesh in self.sugs: indices.extend(mesh.indices)
        self.indices = np.array(indices, dtype=np.uint32)
        Buffer.buffer_indices(self.buff)

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

    def update_sliding_direction(self,sax):
        blocked = False
        for i,sides in enumerate(self.fixed_sides):
            for ax,dir in sides:
                if ax==sax:
                    if dir==0 and i==0: continue
                    if dir==1 and i==self.noc-1: continue
                    blocked = True
        if blocked: print("Blocked sliding direction")
        else:
            self.sax = sax
            self.update_unblocked_fixed_sides()
            self.create_and_buffer_vertices()
            self.mesh.voxel_matrix_from_height_fields()
            for mesh in self.sugs: mesh.voxel_matrix_from_height_fields()
            self.combine_and_buffer_indices()

    def update_dimension(self,add):
        self.dim+=add
        self.voxel_size = self.component_size/self.dim
        #self.fab.real_voxel_size = self.fab.real_component_size/self.dim
        self.create_and_buffer_vertices()
        self.mesh.randomize_height_fields()

    def update_number_of_components(self,new_noc):
        if new_noc!=self.noc:
            # Increasing number of components
            if new_noc>self.noc:
                if len(self.unblocked_fixed_sides)>=(new_noc-self.noc):
                    for i in range(new_noc-self.noc):
                        nfs = next_fixed_sides(self.fixed_sides)
                        if self.fixed_sides[-1][0][0]==self.sax: # last component is aligned with the sliding axis
                            self.fixed_sides.insert(-1,nfs)
                        else:
                            self.fixed_sides.append(nfs)
                        #also consider if it is aligned and should be the first one in line... rare though...
                    self.noc = new_noc
            # Decreasing number of components
            elif new_noc<self.noc:
                for i in range(self.noc-new_noc):
                    self.fixed_sides.pop()
                self.noc = new_noc
            # Rebuffer
            self.update_unblocked_fixed_sides()
            self.create_and_buffer_vertices()
            mesh.randomize_height_fields()

    def update_component_position(self,new_fixed_sides,n):
        self.fixed_sides[n] = new_fixed_sides
        self.update_unblocked_fixed_sides()
        self.create_and_buffer_vertices()
        self.mesh.voxel_matrix_from_height_fields()
        self.combine_and_buffer_indices()

    def update_suggestions(self):
        self.sugs = [] # clear list of suggestions
        sugg_hfs = []
        if not self.mesh.eval.valid:
            print("Looking for valud suggestions...")
            sugg_hfs = produce_suggestions(self,self.mesh.height_fields)
            print(sugg_hfs)
            for i in range(len(sugg_hfs)): self.sugs.append(Geometries(self,mainmesh=False,hfs=sugg_hfs[i]))
