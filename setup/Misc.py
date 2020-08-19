import numpy as np
import random

def depth(l):
    if isinstance(l, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0

class FixedSides:
    def __init__(self,parent,side_str=None,fs=None):
        self.parent = parent
        if side_str!=None: self.sides_from_string(side_str)
        elif fs!=None: self.sides=fs
        else: self.sides=[[FixedSide(2,0)],[FixedSide(2,1)]]
        self.update_unblocked()

    def sides_from_string(self,side_str):
        self.sides = []
        for tim_fss in side_str.split(":"):
            temp = []
            for tim_fs in tim_fss.split("."):
                axdir = tim_fs.split(",")
                ax = int(float(axdir[0]))
                dir = int(float(axdir[1]))
                temp.append(FixedSide(ax,dir))
            self.sides.append(temp)

    def update_unblocked(self):
        # List unblocked POSITIONS
        self.unblocked = []
        for ax in range(3):
            for dir in range(2):
                blocked=False
                if self.sides!=None:
                    for sides in self.sides:
                        for side in sides:
                            if [side.ax,side.dir]==[ax,dir]:
                                blocked=True
                                break
                if not blocked: self.unblocked.append(FixedSide(ax,dir))

        # List unblocked ORIENTATIONS ??????????????
        self.parent.rot=True
        if self.sides!=None:
            for sides in self.sides:
                # if one or more component axes are aligned with the sliding axes (sax), rotation cannot be performed ?????????
                if sides[0].ax==self.parent.sax:
                    self.parent.rot=False
                    break

class FixedSide:
    def __init__(self, ax, dir):
        self.ax = ax
        self.dir = dir

    def unique(self, other_sides):
        unique=True
        if depth(other_sides)==1:
            for side in other_sides:
                if self.ax==side.ax and self.dir==side.dir:
                    unique=False
                    break
        elif depth(other_sides)==2:
            for sides in other_sides:
                for side in sides:
                    if self.ax==side.ax and self.dir==side.dir:
                        unique=False
                        break
        return unique
