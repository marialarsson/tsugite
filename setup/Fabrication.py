import numpy as np
import math

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

class RegionVertex:
    def __init__(self,ind,abs_ind,neighbors,neighbor_values,dia=False):
        self.ind = ind
        self.i = ind[0]
        self.j = ind[1]
        self.neighbors = neighbors
        self.flat_neighbors = self.neighbors.flatten()
        self.region_count = np.sum(self.flat_neighbors==0)
        self.block_count = np.sum(self.flat_neighbors==1)
        self.free_count = np.sum(self.flat_neighbors==2)
        ##
        self.dia = dia
        ##
        self.neighbor_values = np.array(neighbor_values)
        self.flat_neighbor_values = self.neighbor_values.flatten()

    def set_pos(self,pos):
        self.pos = pos

class RoughPixel:
    def __init__(self,ind,mat,pad_loc,dim,n):
        self.ind = ind
        self.ind_abs = ind.copy()
        self.ind_abs[0] -= pad_loc[0][0]
        self.ind_abs[1] -= pad_loc[1][0]
        self.outside = False
        if self.ind_abs[0]<0 or self.ind_abs[0]>=dim:
            self.outside = True
        elif self.ind_abs[1]<0 or self.ind_abs[1]>=dim:
            self.outside = True
        self.neighbors = []
        # Region or free=0
        # Blocked=1
        for ax in range(2):
            temp = []
            for dir in range(-1,2,2):
                nind = self.ind.copy()
                nind[ax] += dir
                type = 0
                if nind[0]>=0 and nind[0]<mat.shape[0] and nind[1]>=0 and nind[1]<mat.shape[1]:
                    val = mat[tuple(nind)]
                    if val==n: type = 1
                temp.append(type)
            self.neighbors.append(temp)
        self.flat_neighbors = [x for sublist in self.neighbors for x in sublist]

class MillVertex:
    def __init__(self,pt,is_arc=False,arc_ctr=np.array([0,0,0])):
        self.pt = np.array(pt)
        self.x = pt[0]
        self.y = pt[1]
        self.z = pt[2]
        self.is_arc = is_arc
        self.arc_ctr = np.array(arc_ctr)

    def scale_and_swap(self,ax,dir,ratio,real_component_size,coords,d,n):
        #sawp
        xyz = [ratio*self.x,ratio*self.y,ratio*self.z]
        if ax==2: xyz[1] = -xyz[1]
        xyz = xyz[coords[0]],xyz[coords[1]],xyz[coords[2]]
        self.x,self.y,self.z = xyz[0],xyz[1],xyz[2]
        #move z down, flip if component b
        self.z = -(2*dir-1)*self.z-0.5*real_component_size
        self.y = -(2*dir-1)*self.y
        self.pt = np.array([self.x,self.y,self.z])
        self.pos = np.array([self.x,self.y,self.z],dtype=np.float64)
        self.xstr = str(round(self.x,d))
        self.ystr = str(round(self.y,d))
        self.zstr = str(round(self.z,d))
        ##
        if self.is_arc:
            self.arc_ctr = [ratio*self.arc_ctr[0],ratio*self.arc_ctr[1],ratio*self.arc_ctr[2]] #ratio*self.arc_ctr
            if ax==2: self.arc_ctr[1] = -self.arc_ctr[1]
            self.arc_ctr = [self.arc_ctr[coords[0]],self.arc_ctr[coords[1]],self.arc_ctr[coords[2]]]
            self.arc_ctr[2] = -(2*dir-1)*self.arc_ctr[2]-0.5*real_component_size
            self.arc_ctr[1] = -(2*dir-1)*self.arc_ctr[1]
            self.arc_ctr = np.array(self.arc_ctr)

    def rotate(self,ang,d):
        self.pt = np.array([self.x,self.y,self.z])
        self.pt = rotate_vector_around_axis(self.pt, [0,0,1], ang)
        self.x = self.pt[0]
        self.y = self.pt[1]
        self.z = self.pt[2]
        self.pos = np.array([self.x,self.y,self.z],dtype=np.float64)
        self.xstr = str(round(self.x,d))
        self.ystr = str(round(self.y,d))
        self.zstr = str(round(self.z,d))
        ##
        if self.is_arc:
            self.arc_ctr = rotate_vector_around_axis(self.arc_ctr, [0,0,1], ang)
            self.arc_ctr = np.array(self.arc_ctr)

class Fabrication:
    def __init__(self,parent,tol=0.15,rad=3.00):
        self.parent = parent
        self.real_component_size = self.parent.real_comp_width-0.5 #29.5 #44.45 #36.5 #mm
        self.real_voxel_size = self.real_component_size/self.parent.dim
        self.ratio = self.real_component_size/self.parent.component_size
        self.rad_real = rad #milling bit radius in mm
        self.rad = self.rad_real #milling bit radius in mm
        self.tol = tol #0.10 #tolerance in mm
        self.rad -= self.tol
        self.dia = 2*self.rad
        self.vdia = self.dia/self.ratio
        self.vrad = self.rad/self.ratio
        self.vtol = self.tol/self.ratio
        self.ext = 1.0
        self.vext = self.ext/self.ratio
        self.dep = 1.5 #milling depth in mm
        self.extra_rot = 0

    def export_gcode(self,filename_tsu="C:/Users/makal/Dropbox/gcode/joint.tsu"):
        # make sure that the z axis of the gcode is facing up
        fax = self.parent.sax
        coords = [0,1]
        coords.insert(fax,2)
        #
        d = 3 # =precision / no of decimals to write
        names = ["A","B","C","D","E","F"]
        for n in range(self.parent.noc):
            comp_ax = self.parent.fixed_sides[n][0][0]
            comp_dir = self.parent.fixed_sides[n][0][1] # component direction
            comp_vec = self.parent.pos_vecs[comp_ax]
            if comp_dir==0 and comp_ax!=self.parent.sax:
                comp_vec=-comp_vec
            comp_vec = np.array([comp_vec[coords[0]],comp_vec[coords[1]],comp_vec[coords[2]]])
            comp_vec = comp_vec/np.linalg.norm(comp_vec) #unitize
            xax = np.array([1,0,0])
            rot_ang = angle_between(xax,comp_vec)
            aaxis = np.cross(xax,comp_vec)
            if np.dot(comp_vec, np.cross(aaxis, xax))<0: #>
                rot_ang=-rot_ang
            fdir = self.parent.mesh.fab_directions[n]
            if fdir==0: rot_ang+=math.pi #1
            rot_ang+=self.extra_rot ###for roland machine etc###############################
            #
            file_name = filename_tsu[:-4] + "_"+names[n]+".gcode"
            file = open(file_name,"w")
            ###initialization
            file.write("%\n")
            file.write("G90 (Absolute [G91 is incremental])\n")
            file.write("G17 (set XY plane for circle path)\n")
            file.write("G94 (set unit/minute)\n")
            file.write("G21 (set unit[mm])\n")
            file.write("F400. (Feeding 400mm/min)\n")
            file.write("S6000 (Spindle 6000rpm)\n")
            file.write("M3 (spindle start)\n")
            file.write("G54\n")

            #speed = 400
            ###content
            currentg = ""
            for i,mv in enumerate(self.parent.gcodeverts[n]):
                mv.scale_and_swap(fax,fdir,self.ratio,self.real_component_size,coords,d,n)
                if comp_ax!=fax:
                    mv.rotate(rot_ang,d)
                if i>0: pmv = self.parent.gcodeverts[n][i-1]
                # check segment angle
                arc = False
                clockwise = False
                if i>0 and mv.is_arc and pmv.is_arc and np.array_equal(mv.arc_ctr,pmv.arc_ctr):
                    arc = True
                    vec1 = mv.pt-mv.arc_ctr
                    vec1 = vec1/np.linalg.norm(vec1)
                    zvec = np.array([0,0,1])
                    xvec = np.cross(vec1,zvec)
                    vec2 = pmv.pt-mv.arc_ctr
                    vec2 = vec2/np.linalg.norm(vec2)
                    diff_ang = angle_between(xvec,vec2)
                    if diff_ang>0.5*math.pi: clockwise = True

                #write to file
                if arc and clockwise:
                    if i==0: file.write("G2 X"+mv.xstr+" Y"+mv.ystr+" R"+str(self.dia)+" F400."+"\n")
                    else: file.write("G2 X"+mv.xstr+" Y"+mv.ystr+" R"+str(self.dia)+"\n")
                    currentg=="G2"
                elif arc and not clockwise:
                    if i==0: file.write("G3 X"+mv.xstr+" Y"+mv.ystr+" R"+str(self.dia)+" F400."+"\n")
                    else: file.write("G3 X"+mv.xstr+" Y"+mv.ystr+" R"+str(self.dia)+"\n")
                    currentg = "G3"
                elif i==0 or pmv.x!=mv.x or pmv.y!=mv.y:
                    if currentg=="G1":
                        if i==0: file.write("X"+mv.xstr+" Y"+mv.ystr+" F400."+"\n")
                        else: file.write("X"+mv.xstr+" Y"+mv.ystr+"\n")
                    else:
                        if i==0: file.write("G1 X"+mv.xstr+" Y"+mv.ystr+" F400."+"\n")
                        else: file.write("G1 X"+mv.xstr+" Y"+mv.ystr+"\n")
                        currentg=="G1"
                ##
                if i==0 or pmv.z!=mv.z:
                    if currentg=="G1":
                        file.write("Z"+mv.zstr+"\n")
                    else:
                        file.write("G1 Z"+mv.zstr+"\n")
                        currentg = "G1"
            #end
            file.write("M5 (Spindle stop)\n")
            file.write("M2 (end of program)\n")
            file.write("M30 (delete sd file)\n")
            file.write("%\n")
            print("Exported",file_name)
            file.close()
