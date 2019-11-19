import numpy as np
import math

class Region:
    def __init__(self,parent,layer):
        self.layer = layer()
        self.verts = []
        self.voxels = []

class VoxelVertex:
    def __init__(self):
        self.x = x
        self.y = y
        self.z = z
        self.i = i
        self.j = j
        self.lay_num = lay_num


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
                if x_!=x or y_!=y: file.write("G01 X "+x+" Y "+y+" F "+str(speed)+"\n")
                if z_!=z: file.write("G01 Z "+z+" F "+str(speed)+"\n")
                x_ = x
                y_ = y
                z_ = z
            ###end
            file.write("M05 (Spindle stop)\n")
            file.write("M02(end of program)\n")
            file.write("%\n")
            print("Exported",file_name+".gcode.")
            file.close()
