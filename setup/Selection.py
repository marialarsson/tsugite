import numpy as np
import pyrr
from numpy import linalg

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
        self.parent = parent
        self.n = self.x = self.y = None
        self.refresh = False
        self.shift = False

    def update_pick(self,x,y,n):
        self.n = n
        self.x = x
        self.y = y
        if self.shift:
            self.faces = get_same_height_neighbors(self.parent.height_fields[n],[np.array([x,y])])
        else: self.faces = [np.array([x,y])]

    def start_pull(self,mouse_pos):
        self.state=2
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.start_height = self.parent.height_fields[self.n][self.x][self.y]
        self.parent.create_indices() # for selection area

    def end_pull(self):
        if self.val!=0: self.parent.edit_height_fields(self.faces,self.current_height,self.n)
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
        ax = self.parent.sax
        dir = self.parent.fab_directions[self.n]
        sdir_vec[ax] = (2*dir-1)*self.parent.voxel_size
        if self.n==1: sdir_vec[ax] = -sdir_vec[ax]
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
        if self.start_height + val>self.parent.dim: val = self.parent.dim-self.start_height
        elif self.start_height+val<0: val = -self.start_height
        self.current_height = self.start_height + val
        self.val = int(val)

    def start_move(self,mouse_pos):
        self.state=12
        self.start_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        self.new_fixed_sides = self.parent.fixed_sides[self.n]
        #self.parent.create_indices() # for selection area

    def end_move(self):
        self.parent.update_component_position(self.new_fixed_sides,self.n)
        self.state=-1

    def move(self,mouse_pos,screen_xrot,screen_yrot):
        self.current_pos = np.array([mouse_pos[0],-mouse_pos[1]])
        ## Mouse vector
        mouse_vec = self.current_pos-self.start_pos
        mouse_vec[0] = mouse_vec[0]/800
        mouse_vec[1] = mouse_vec[1]/800
        ##
        self.new_fixed_sides = self.parent.fixed_sides[self.n]
        if len(self.new_fixed_sides)>1: self.new_fixed_sides.pop()
