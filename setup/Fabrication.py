import numpy as np

class Fabrication:
    def __init__(self, path_vertices):
        self.path_vertices = path_vertices

    def export_gcode(self):
        d = 3 # =precision / no of decimals to write
        file = open("gcode/test.gcode","w")
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
            #convert from cm to mm
            x = str(round(100*x,d))
            y = str(round(100*y,d))
            z = str(round(100*z,d))
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
