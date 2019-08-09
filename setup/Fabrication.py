import numpy as np

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
