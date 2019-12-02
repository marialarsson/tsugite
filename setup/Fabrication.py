import numpy as np
import math

class RegionVertex:
    def __init__(self,ind,neighbors,flat_neighbors):
        self.ind = ind
        self.i = ind[0]
        self.j = ind[1]
        self.neighbors = neighbors
        self.flat_neighbors = flat_neighbors
        self.region_count = np.sum(self.flat_neighbors==0)
        self.block_count = np.sum(self.flat_neighbors==1)
        self.free_count = np.sum(self.flat_neighbors==2)
        if self.block_count==2 and self.region_count==2 and self.neighbors[0][1]==self.neighbors[1][0]: self.dia = True
        else: self.dia = False

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

    def scale_and_swap(self,ratio,real_component_size,coords,d,n):
        #sawp
        xyz = [ratio*self.x,ratio*self.y,ratio*self.z]
        xyz = xyz[coords[0]],xyz[coords[1]],xyz[coords[2]]
        self.x,self.y,self.z = xyz[0],xyz[1],xyz[2]
        #move z down, flip if component b
        self.z = -(2*n-1)*self.z-0.5*real_component_size
        self.pt = np.array([self.x,self.y,self.z])
        self.pos = np.array([self.x,self.y,self.z],dtype=np.float64)
        self.xstr = str(round(self.x,d))
        self.ystr = str(round(self.y,d))
        self.zstr = str(round(self.z,d))
        ##
        if self.is_arc:
            self.arc_ctr = [ratio*self.arc_ctr[0],ratio*self.arc_ctr[1],ratio*self.arc_ctr[2]] #ratio*self.arc_ctr
            self.arc_ctr = [self.arc_ctr[coords[0]],self.arc_ctr[coords[1]],self.arc_ctr[coords[2]]]
            self.arc_ctr[2] = -(2*n-1)*self.arc_ctr[2]-0.5*real_component_size
            self.arc_ctr = np.array(self.arc_ctr)

class Fabrication:
    def __init__(self,parent):
        self.parent = parent
        self.real_component_size = 30 #44.45 #mm
        self.real_voxel_size = self.real_component_size/self.parent.dim
        self.ratio = self.real_component_size/self.parent.component_size
        self.rad = 2.0 #2.9 #milling bit radius in mm
        self.dia = 2*self.rad
        self.vdia = self.dia/self.ratio
        self.vrad = self.rad/self.ratio
        self.ext = 1.0
        self.vext = self.ext/self.ratio
        self.dep = 1.0 #milling depth in mm


    def export_gcode(self,file_name):
        if self.parent.sliding_directions[0][0][0]==2: coords = [0,1,2]
        elif self.parent.sliding_directions[0][0][0]==1: coords = [2,0,1]
        d = 3 # =precision / no of decimals to write
        names = ["A","B","C","D","E","F"]
        for n in range(self.parent.noc):
            file_name = "joint_"+names[n]
            file = open("C:/Users/makal/Dropbox/gcode/"+file_name+".gcode","w")
            ###initialization
            file.write("%\n")
            file.write("G90 (Absolute [G91 is incremental])\n")
            file.write("G21 (set unit[mm])\n")
            file.write("G54\n")
            file.write("F1000.0S6000 (Feeding 2000mm/min, Spindle 6000rpm)\n")
            file.write("G17 (set XY plane for circle path)\n")
            file.write("M03 (spindle start)\n")
            speed = 200
            ###content
            for i,mv in enumerate(self.parent.gcodeverts[n]):
                mv.scale_and_swap(self.ratio,self.real_component_size,coords,d,n)
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
                    dist = np.linalg.norm(xvec-vec2)
                    if dist>0: clockwise = True

                #write to file
                if arc and clockwise:
                    file.write("G02 X "+mv.xstr+" Y "+mv.ystr+" R "+str(self.dia)+" F "+str(speed)+"\n")
                elif arc and not clockwise:
                    file.write("G03 X "+mv.xstr+" Y "+mv.ystr+" R "+str(self.dia)+" F "+str(speed)+"\n")
                elif i==0 or pmv.x!=mv.x or pmv.y!=mv.y: file.write("G01 X "+mv.xstr+" Y "+mv.ystr+" F "+str(speed)+"\n")
                ##
                if i==0 or pmv.z!=mv.z: file.write("G01 Z "+mv.zstr+" F "+str(speed)+"\n")
            ###end
            file.write("M05 (Spindle stop)\n")
            file.write("M02(end of program)\n")
            file.write("%\n")
            print("Exported",file_name+".gcode.")
            file.close()
