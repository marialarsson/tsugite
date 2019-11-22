import numpy as np
import math

class RegionVertex:
    def __init__(self,ind,neighbors,flat_neighbors):
        self.ind = ind
        self.i = ind[0]
        self.j = ind[1]
        self.neighbors = neighbors
        self.flat_neighbors = flat_neighbors

    def set_pos(self,pos):
        self.pos = pos

class Fabrication:
    def __init__(self,parent):
        self.parent = parent
        self.real_component_size = 30 #44.45 #mm
        self.real_voxel_size = self.real_component_size/self.parent.dim
        self.ratio = self.real_component_size/self.parent.component_size
        self.rad = 2.9 #milling bit radius in mm
        self.dia = 2*self.rad
        self.vrad = self.rad/self.ratio
        self.ext = 1.0
        self.vext = self.ext/self.ratio
        self.dep = 1.0 #milling depth in mm


    def export_gcode(self,file_name):
        if self.parent.sliding_directions[0][0][0]==2: coords = [0,1,2]
        elif self.parent.sliding_directions[0][0][0]==1: coords = [2,0,1]
        d = 3 # =precision / no of decimals to write
        names = ["A","B","C","D"]
        for n in range(self.parent.noc):
            file_name = "joint_"+names[n]
            file = open("gcode/"+file_name+".gcode","w")
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
            x_ = str(9999999999)
            y_ = str(9999999999)
            z_ = str(9999999999)
            for i in range(0,len(self.parent.mverts[n]),8):
                x = self.parent.mverts[n][i]
                y = self.parent.mverts[n][i+1]
                z = self.parent.mverts[n][i+2]
                # check segment angle
                arc = False
                if i>0:
                    px = self.parent.mverts[n][i-8]
                    py = self.parent.mverts[n][i+1-8]
                    pz = self.parent.mverts[n][i+2-8]
                    pt = np.array([x,y,z],dtype=np.float64)
                    ppt = np.array([px,py,pz],dtype=np.float64)
                    if np.sum((ppt-pt)==0)==1:
                        arc = True
                        clockwise=False
                        ppx = self.parent.mverts[n][i-16]
                        ppy = self.parent.mverts[n][i+1-16]
                        ppz = self.parent.mverts[n][i+2-16]
                        pppt = np.array([ppx,ppy,ppz],dtype=np.float64)
                        vec1 = ppt-pppt
                        vec2 = pt-ppt
                        crossp = np.cross(vec1,vec2)
                        crossi = np.argwhere(crossp!=0)[0][0]
                        crossv = crossp[crossi]
                        if crossv<0 and n==1: clockwise=True
                        elif crossv>0 and n==0: clockwise=True
                #convert from virtul dimensions to mm
                x = self.ratio*x
                y = self.ratio*y
                z = self.ratio*z
                #sawp
                xyz = [x,y,z]
                xyz = xyz[coords[0]],xyz[coords[1]],xyz[coords[2]]
                x,y,z = xyz[0],xyz[1],xyz[2]
                #move z down, flip if component b
                z = -(2*n-1)*z-0.5*self.real_component_size
                #string
                x = str(round(x,d))
                y = str(round(y,d))
                z = str(round(z,d))
                #write to file
                if not arc:
                    if x_!=x or y_!=y: file.write("G01 X "+x+" Y "+y+" F "+str(speed)+"\n")
                    if z_!=z: file.write("G01 Z "+z+" F "+str(speed)+"\n")
                elif clockwise:
                    file.write("G02 X "+x+" Y "+y+" R "+str(self.dia)+" F "+str(speed)+"\n")
                else:
                    file.write("G03 X "+x+" Y "+y+" R "+str(self.dia)+" F "+str(speed)+"\n")
                x_ = x
                y_ = y
                z_ = z
            ###end
            file.write("M05 (Spindle stop)\n")
            file.write("M02(end of program)\n")
            file.write("%\n")
            print("Exported",file_name+".gcode.")
            file.close()
