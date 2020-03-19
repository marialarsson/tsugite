import numpy as np
import pyrr
from numpy import linalg
import copy
import math

def get_same_height_neighbors(hfield,inds):
    dim = len(hfield)
    val = hfield[tuple(inds[0])]
    new_inds = list(inds)
    for ind in inds:
        for ax in range(2):
            for dir in range(-1,2,2):
                ind2 = ind.copy()
                ind2[ax] += dir
                if np.all(ind2>=0) and np.all(ind2<dim):
                    val2 = hfield[tuple(ind2)]
                    if val2==val:
                        unique = True
                        for ind3 in new_inds:
                            if ind2[0]==ind3[0] and ind2[1]==ind3[1]:
                                unique = False
                                break
                        if unique: new_inds.append(ind2)
    if len(new_inds)>len(inds):
        new_inds = get_same_height_neighbors(hfield,new_inds)
    return new_inds

class Selection:
    def __init__(self,parent):
        self.state = -1 #-1: nothing, 0: hovered, 1: adding, 2: pulling, 10: timber hovered, 12: timber pulled
        self.suggstate = -1 #-1: nothing, 0: hovering first, 1: hovering secong, and so on.
        self.parent = parent
        self.n = self.x = self.y = None
        self.refresh = False
        self.shift = False
        self.faces = []
        self.new_fixed_sides_for_display = None
        self.val=0

    def update_pick(self,x,y,n,dir):
        self.n = n
        self.x = x
        self.y = y
        self.dir = dir
        if self.x!=None and self.y!=None:
            if self.shift:
                self.faces = get_same_height_neighbors(self.parent.height_fields[n-dir],[np.array([self.x,self.y])])
            else: self.faces = [np.array([self.x,self.y])]

    def start_pull(self,mouse_pos):
        self.state=2
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.start_height = self.parent.height_fields[self.n-self.dir][self.x][self.y]
        self.parent.parent.combine_and_buffer_indices() # for selection area

    def end_pull(self):
        if self.val!=0: self.parent.edit_height_fields(self.faces,self.current_height,self.n,self.dir)
        self.state=-1
        self.refresh = True

    def edit(self,mouse_pos,screen_xrot,screen_yrot):
        self.current_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.current_height = self.start_height
        ## Mouse vector
        mouse_vec = self.current_pos-self.start_pos
        mouse_vec[0] = mouse_vec[0]/800
        mouse_vec[1] = mouse_vec[1]/800
        ## Sliding direction vector
        sdir_vec = [0,0,0]
        sdir_vec[self.parent.parent.sax] = -self.parent.parent.voxel_size
        rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
        rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
        sdir_vec = np.dot(sdir_vec,rot_x*rot_y)
        sdir_vec = np.delete(sdir_vec,2) # delete Z-value
        ## Calculate angle between mouse vector and sliding direction vector
        cosang = np.dot(mouse_vec, sdir_vec) # Negative / positive depending on direction
        #sinang = linalg.norm(np.cross(mouse_vec, joint_vec))
        #ang = math.degrees(np.arctan2(sinang, cosang))
        val = int(linalg.norm(mouse_vec)/linalg.norm(sdir_vec)+0.5)
        if cosang!=None and cosang>0: val = -val
        if self.start_height + val>self.parent.parent.dim: val = self.parent.parent.dim-self.start_height
        elif self.start_height+val<0: val = -self.start_height
        self.current_height = self.start_height + val
        self.val = int(val)

    def start_move(self,mouse_pos):
        self.state=12
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.new_fixed_sides = self.parent.parent.fixed_sides[self.n]
        self.new_fixed_sides_for_display = self.parent.parent.fixed_sides[self.n]
        self.parent.parent.combine_and_buffer_indices # for move preview outline

    def end_move(self):
        self.parent.parent.update_component_position(self.new_fixed_sides,self.n)
        self.state=-1
        self.new_fixed_sides_for_display = None

    def move(self,mouse_pos,screen_xrot,screen_yrot): # actually move OR rotate
        sax = self.parent.parent.sax
        noc = self.parent.parent.noc
        self.new_fixed_sides = copy.deepcopy(self.parent.parent.fixed_sides[self.n])
        self.new_fixed_sides_for_display = copy.deepcopy(self.parent.parent.fixed_sides[self.n])
        self.current_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        ## Mouse vector
        mouse_vec = self.current_pos-self.start_pos
        mouse_vec[0] = mouse_vec[0]/800
        mouse_vec[1] = mouse_vec[1]/800
        ## Check that the move distance is above some treshhold
        move_dist = linalg.norm(mouse_vec)
        if move_dist>0.01:
            ## Get component direction vector
            comp_ax = self.parent.parent.fixed_sides[self.n][0][0] # component axis
            comp_vec = [0,0,0]
            comp_vec[comp_ax] = 3*self.parent.parent.component_size
            ## Flatten vector to screen
            rot_x = pyrr.Matrix33.from_x_rotation(screen_xrot)
            rot_y = pyrr.Matrix33.from_y_rotation(screen_yrot)
            comp_vec = np.dot(comp_vec,rot_x*rot_y)
            comp_vec = np.delete(comp_vec,2) # delete Z-value
            ## Calculate angle between mouse vector and sliding direction vector
            cosang = np.dot(mouse_vec, comp_vec) # Negative / positive depending on direction
            sinang = linalg.norm(np.cross(mouse_vec, comp_vec))
            ang = math.degrees(np.arctan2(sinang, cosang))
            oax = None
            if ang>45 and ang<135: #rotation mode
                # Check plane of rotating by checking which axis the vector is more alinged to
                other_axes = [0,1,2]
                other_axes.pop(comp_ax)
                # The axis that is flatter to the scren will be processed
                maxlen = 0
                for i in range(len(other_axes)):
                    other_vec = [0,0,0]
                    other_vec[other_axes[i]]=1
                    ## Flatten vector to screen
                    other_vec = np.dot(other_vec,rot_x*rot_y)
                    other_vec = np.delete(other_vec,2) # delete Z-value
                    ## Check length
                    other_length = linalg.norm(other_vec)
                    if other_length>maxlen:
                        maxlen = other_length
                        oax = other_axes[i]
                # check rotation direction
                clockwise = True
                if cosang<0: clockwise = False
                #
                self.new_fixed_sides_for_display = []
                blocked = False
                for i in range(len(self.parent.parent.fixed_sides[self.n])):
                    ndir = self.parent.parent.fixed_sides[self.n][i][1]
                    if not clockwise: ndir = 1-ndir
                    side = [oax,ndir]
                    self.new_fixed_sides_for_display.append(side)
                    if side==[sax,0] and self.n!=0: blocked=True; break
                    if side==[sax,1] and self.n!=noc-1: blocked=True; break
                    if side not in self.parent.parent.unblocked_fixed_sides:
                        blocked = True
                if not blocked: self.new_fixed_sides = self.new_fixed_sides_for_display
                else: print("Blocked rotation")
            else: # moveing mode
                length_ratio = linalg.norm(mouse_vec)/linalg.norm(comp_vec)
                if length_ratio>0.2: #treshhold
                    if len(self.parent.parent.fixed_sides[self.n])==1 and length_ratio<0.5: # moved just a bit, L to T
                        if [comp_ax,1] not in self.parent.parent.fixed_sides[self.n]:
                            self.new_fixed_sides_for_display.append([comp_ax,1])
                        elif [comp_ax,0] not in self.parent.parent.fixed_sides[self.n]:
                            self.new_fixed_sides_for_display.insert(0,[comp_ax,0])
                    # L long move or T short move
                    elif cosang!=None:
                        if cosang>0:
                            if [comp_ax,0] in self.parent.parent.fixed_sides[self.n]:
                                self.new_fixed_sides_for_display.pop(0)
                            if [comp_ax,1] not in self.parent.parent.fixed_sides[self.n]:
                                self.new_fixed_sides_for_display.append([comp_ax,1])
                        else:
                            if [comp_ax,1] in self.parent.parent.fixed_sides[self.n]:
                                self.new_fixed_sides_for_display.pop()
                            if [comp_ax,0] not in self.parent.parent.fixed_sides[self.n]:
                                self.new_fixed_sides_for_display.insert(0,[comp_ax,0])
                # check if the direction is blocked
                blocked = False
                for side in self.new_fixed_sides_for_display:
                    if side==[sax,0] and self.n!=0: blocked=True; break
                    if side==[sax,1] and self.n!=noc-1: blocked=True; break
                    if side not in self.parent.parent.fixed_sides[self.n]:
                        if side not in self.parent.parent.unblocked_fixed_sides:
                            blocked = True
                if not blocked: self.new_fixed_sides = self.new_fixed_sides_for_display
                else: print("Blocked move")
        if not np.equal(self.parent.parent.fixed_sides[self.n],np.array(self.new_fixed_sides_for_display)).all():
            self.parent.parent.combine_and_buffer_indices()# for move/rotate preview outline # cant you show this by tansformation instead?
