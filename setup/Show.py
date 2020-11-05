from OpenGL.GL import *
from Buffer import ElementProperties
from ViewSettings import ViewSettings
import numpy as np
import pyrr

class Show:
    def __init__(self,parent,type):
        self.parent = parent
        self.type = type
        self.view = ViewSettings()
        self.create_color_shaders()
        self.create_texture_shaders()

    def update(self):
        self.init_shader(self.shader_col)
        if (self.view.open_joint and self.view.open_ratio<self.type.noc-1) or (not self.view.open_joint and self.view.open_ratio>0):
            self.view.set_joint_opening_distance(self.type.noc)

    def create_color_shaders(self):
        vertex_shader = """
        #version 330
        #extension GL_ARB_explicit_attrib_location : require
        #extension GL_ARB_explicit_uniform_location : require
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec2 inTexCoords;
        layout(location = 3) uniform mat4 transform;
        layout(location = 4) uniform mat4 translate;
        layout(location = 5) uniform vec3 myColor;
        out vec3 newColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = transform* translate* vec4(position, 1.0f);
            newColor = myColor;
            outTexCoords = inTexCoords;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 newColor;
        in vec2 outTexCoords;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            outColor = vec4(newColor, 1.0);
        }
        """
        # Compiling the shaders
        self.shader_col = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                  OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    def create_texture_shaders(self):
        vertex_shader = """
        #version 330
        #extension GL_ARB_explicit_attrib_location : require
        #extension GL_ARB_explicit_uniform_location : require
        layout(location = 0) in vec3 position;
        layout(location = 1) in vec3 color;
        layout(location = 2) in vec2 inTexCoords;
        layout(location = 3) uniform mat4 transform;
        layout(location = 4) uniform mat4 translate;
        out vec3 newColor;
        out vec2 outTexCoords;
        void main()
        {
            gl_Position = transform* translate* vec4(position, 1.0f);
            newColor = color;
            outTexCoords = inTexCoords;
        }
        """

        fragment_shader = """
        #version 330
        in vec3 newColor;
        in vec2 outTexCoords;
        out vec4 outColor;
        uniform sampler2D samplerTex;
        void main()
        {
            outColor = texture(samplerTex, outTexCoords);
        }
        """
        # Compiling the shaders
        self.shader_tex = OpenGL.GL.shaders.compileProgram(OpenGL.GL.shaders.compileShader(vertex_shader, GL_VERTEX_SHADER),
                                                  OpenGL.GL.shaders.compileShader(fragment_shader, GL_FRAGMENT_SHADER))

    def init_shader(self,shader):
        glUseProgram(shader)
        rot_x = pyrr.Matrix44.from_x_rotation(self.view.xrot)
        rot_y = pyrr.Matrix44.from_y_rotation(self.view.yrot)
        glUniformMatrix4fv(3, 1, GL_FALSE, np.array(rot_x * rot_y))

    def draw_geometries(self, geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.type.sax] = self.view.open_ratio*self.type.component_size
        move_vec = np.array(move_vec)
        moves = []
        for n in range(self.type.noc):
            tot_move_vec = (2*n+1-self.type.noc)/(self.type.noc-1)*move_vec
            move_mat = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
            moves.append(move_mat)
        if clear_depth_buffer: glClear(GL_DEPTH_BUFFER_BIT)
        for geo in geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

    def draw_geometries_with_excluded_area(self, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.type.sax] = self.view.open_ratio*self.type.component_size
        move_vec = np.array(move_vec)
        moves = []
        moves_show = []
        for n in range(self.type.noc):
            tot_move_vec = (2*n+1-self.type.noc)/(self.type.noc-1)*move_vec
            move_mat = pyrr.matrix44.create_from_translation(tot_move_vec)
            moves.append(move_mat)
            move_mat_show = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
            moves_show.append(move_mat_show)
        #
        glClear(GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glColorMask(GL_FALSE,GL_FALSE,GL_FALSE,GL_FALSE)
        glEnable(GL_STENCIL_TEST)
        glStencilFunc(GL_ALWAYS,1,1)
        glStencilOp(GL_REPLACE,GL_REPLACE,GL_REPLACE)
        glDepthRange (0.0, 0.9975)
        for geo in show_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
        glEnable(GL_DEPTH_TEST)
        glStencilFunc(GL_EQUAL,1,1)
        glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
        glDepthRange (0.0025, 1.0)
        for geo in screen_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
        glDisable(GL_STENCIL_TEST)
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
        glDepthRange (0.0, 0.9975)
        for geo in show_geos:
            if geo==None: continue
            if self.view.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

    def pick(self,xpos,ypos,height):

        if not self.view.gallery:
            ######################## COLOR SHADER ###########################
            glUseProgram(self.shader_col)
            glClearColor(1.0, 1.0, 1.0, 1.0) # white
            glEnable(GL_DEPTH_TEST)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            glMatrixMode(GL_MODELVIEW)
            rot_x = pyrr.Matrix44.from_x_rotation(self.view.xrot)
            rot_y = pyrr.Matrix44.from_y_rotation(self.view.yrot)
            glUniformMatrix4fv(3, 1, GL_FALSE, np.array(rot_x * rot_y))
            glPolygonOffset(1.0,1.0)

            ########################## Draw colorful top faces ##########################

            # Draw colorful geometries
            col_step = 1.0/(2+2*self.type.dim*self.type.dim)
            for n in range(self.type.noc):
                col = np.zeros(3, dtype=np.float64)
                col[n%3] = 1.0
                if n>2: col[(n+1)%self.type.dim] = 1.0
                glUniform3f(5, col[0], col[1], col[2])
                self.draw_geometries([self.type.mesh.indices_fpick_not_top[n]],clear_depth_buffer=False)
                if n==0 or n==self.type.noc-1: mos = 1
                else: mos = 2
                # mos is "number of sides"
                for m in range(mos):
                    # Draw top faces
                    for i in range(self.type.dim*self.type.dim):
                        col -= col_step
                        glUniform3f(5, col[0], col[1], col[2])
                        top = ElementProperties(GL_QUADS, 4, self.type.mesh.indices_fpick_top[n].start_index+mos*4*i+4*m, n)
                        self.draw_geometries([top],clear_depth_buffer=False)

        ############### Read pixel color at mouse position ###############
        mouse_pixel = glReadPixelsub(xpos, height-ypos, 1, 1, GL_RGB, outputType=None)[0][0]
        mouse_pixel = np.array(mouse_pixel)
        pick_n = pick_d = pick_x = pick_y = None
        self.type.mesh.select.suggstate = -1
        self.type.mesh.select.gallstate = -1
        if not self.view.gallery:
            if xpos>self.parent.width-self.parent.wstep: # suggestion side
                if ypos>0 and ypos<self.parent.height:
                    index = int(ypos/self.parent.hstep)
                    if self.type.mesh.select.suggstate!=index:
                        self.type.mesh.select.suggstate=index
            elif not np.all(mouse_pixel==255): # not white / background
                    non_zeros = np.where(mouse_pixel!=0)
                    if len(non_zeros)>0:
                        if len(non_zeros[0]>0):
                            pick_n = non_zeros[0][0]
                            if len(non_zeros[0])>1:
                                pick_n = pick_n+self.type.dim
                                if mouse_pixel[0]==mouse_pixel[2]: pick_n = 5
                            val = 255-mouse_pixel[non_zeros[0][0]]
                            i = int(0.5+val*(2+2*self.type.dim*self.type.dim)/255)-1
                            if i>=0:
                                pick_x = (int(i/self.type.dim))%self.type.dim
                                pick_y = i%self.type.dim
                            pick_d = 0
                            if pick_n==self.type.noc-1: pick_d = 1
                            elif int(i/self.type.dim)>=self.type.dim: pick_d = 1
                            #print(pick_n,pick_d,pick_x,pick_y)
        """
        else: #gallerymode
            if xpos>0 and xpos<2000 and ypos>0 and ypos<1600:
                i = int(xpos/400)
                j = int(ypos/400)
                index = i*4+j
                mesh.select.gallstate=index
                mesh.select.state = -1
                mesh.select.suggstate = -1
        """
        ### Update selection
        if pick_x !=None and pick_d!=None and pick_y!=None and pick_n!=None:
            ### Initialize selection
            new_pos = False
            if pick_x!=self.type.mesh.select.x or pick_y!=self.type.mesh.select.y or pick_n!=self.type.mesh.select.n or pick_d!=self.type.mesh.select.dir or self.type.mesh.select.refresh:
                self.type.mesh.select.update_pick(pick_x,pick_y,pick_n,pick_d)
                self.type.mesh.select.refresh = False
                self.type.mesh.select.state = 0 # hovering
        elif pick_n!=None:
            self.type.mesh.select.state = 10 # hovering component body
            self.type.mesh.select.update_pick(pick_x,pick_y,pick_n,pick_d)
        else: self.type.mesh.select.state = -1
        glClearColor(1.0, 1.0, 1.0, 1.0)

    def selected(self):
        ################### Draw top face that is currently being hovered ##########
        # Draw base face (hovered)
        if self.type.mesh.select.state==0:
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)
            glUniform3f(5, 0.2, 0.2, 0.2) #dark grey
            G1 = self.type.mesh.indices_fpick_not_top
            for face in self.type.mesh.select.faces:
                if self.type.mesh.select.n==0 or self.type.mesh.select.n==self.type.noc-1: mos = 1
                else: mos = 2
                index = int(self.type.dim*face[0]+face[1])
                top = ElementProperties(GL_QUADS, 4, self.type.mesh.indices_fpick_top[self.type.mesh.select.n].start_index+mos*4*index+(mos-1)*4*self.type.mesh.select.dir, self.type.mesh.select.n)
                #top = ElementProperties(GL_QUADS, 4, mesh.indices_fpick_top[mesh.select.n].start_index+4*index, mesh.select.n)
                self.draw_geometries_with_excluded_area([top],G1)
        # Draw pulled face
        if self.type.mesh.select.state==2:
            glPushAttrib(GL_ENABLE_BIT)
            glLineWidth(3)
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(2, 0xAAAA)
            for val in range(0,abs(self.type.mesh.select.val)+1):
                if self.type.mesh.select.val<0: val = -val
                pulled_vec = [0,0,0]
                pulled_vec[self.type.sax] = val*self.type.voxel_sizes[self.type.sax]
                self.draw_geometries([self.type.mesh.outline_selected_faces],translation_vec=np.array(pulled_vec))
            glPopAttrib()

    def difference_suggestion(self,index):
        glPushAttrib(GL_ENABLE_BIT)
        # draw faces of additional part
        #glUniform3f(5, 1.0, 1.0, 1.0) # white
        #for n in range(self.type.noc):
        #    G0 = [self.type.sugs[index].indices_fall[n]]
        #    G1 = self.type.mesh.indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw faces of subtracted part
        #glUniform3f(5, 1.0, 0.5, 0.5) # pink/red
        #for n in range(self.type.noc):
        #    G0 = [self.type.mesh.indices_fall[n]]
        #    G1 = self.type.sugs[index].indices_fall
        #    self.draw_geometries_with_excluded_area(G0,G1)

        # draw outlines
        glUniform3f(5, 0.0, 0.0, 0.0) # black
        glLineWidth(3)
        glEnable(GL_LINE_STIPPLE)
        glLineStipple(2, 0xAAAA)
        for n in range(self.type.noc):
            G0 = [self.type.sugs[index].indices_lns[n]]
            G1 = self.type.sugs[index].indices_fall
            self.draw_geometries_with_excluded_area(G0,G1)
        glPopAttrib()


    def moving_rotating(self):
        # Draw moved_rotated component before action is finalized
        if self.type.mesh.select.state==12 and self.type.mesh.outline_selected_component!=None:
            glPushAttrib(GL_ENABLE_BIT)
            glLineWidth(3)
            glEnable(GL_LINE_STIPPLE)
            glLineStipple(2, 0xAAAA)
            self.draw_geometries([self.type.mesh.outline_selected_component])
            glPopAttrib()

    def joint_geometry(self,mesh=None,lw=3,hidden=True,zoom=False):

        if mesh==None: mesh = self.type.mesh

        ############################# Draw hidden lines #############################
        glClear(GL_DEPTH_BUFFER_BIT)
        glUniform3f(5,0.0,0.0,0.0) # black
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(1)
        glLineStipple(3, 0xAAAA) #dashed line
        glEnable(GL_LINE_STIPPLE)
        if hidden and self.view.show_hidden_lines:
            for n in range(mesh.parent.noc):
                G0 = [mesh.indices_lns[n]]
                G1 = [mesh.indices_fall[n]]
                self.draw_geometries_with_excluded_area(G0,G1)
        glPopAttrib()

        ############################ Draw visible lines #############################
        for n in range(mesh.parent.noc):
            if not mesh.mainmesh or (mesh.eval.interlocks[n] and self.view.show_feedback) or not self.view.show_feedback:
                glUniform3f(5,0.0,0.0,0.0) # black
                glLineWidth(lw)
            else:
                glUniform3f(5,1.0,0.0,0.0) # red
                glLineWidth(lw+1)
            G0 = [mesh.indices_lns[n]]
            G1 = mesh.indices_fall
            self.draw_geometries_with_excluded_area(G0,G1)


        if mesh.mainmesh:
            ################ When joint is fully open, draw dahsed lines ################
            if hidden and not self.view.hidden[0] and not self.view.hidden[1] and self.view.open_ratio==1+0.5*(mesh.parent.noc-2):
                glUniform3f(5,0.0,0.0,0.0) # black
                glPushAttrib(GL_ENABLE_BIT)
                glLineWidth(2)
                glLineStipple(1, 0x00FF)
                glEnable(GL_LINE_STIPPLE)
                G0 = mesh.indices_open_lines
                G1 = mesh.indices_fall
                self.draw_geometries_with_excluded_area(G0,G1)
                glPopAttrib()

    def end_grains(self):
        self.init_shader(self.shader_tex)
        G0 = self.type.mesh.indices_fend
        G1 = self.type.mesh.indices_not_fend
        self.draw_geometries_with_excluded_area(G0,G1)
        self.init_shader(self.shader_col)

    def unfabricatable(self):
        col = [1.0, 0.8, 0.5] # orange
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.type.noc):
            if not self.type.mesh.eval.fab_direction_ok[n]:
                G0 = [self.type.mesh.indices_fall[n]]
                G1 = []
                for n2 in range(self.type.noc):
                    if n2!=n: G1.append(self.type.mesh.indices_fall[n2])
                self.draw_geometries_with_excluded_area(G0,G1)

    def unconnected(self):
        # 1. Draw hidden geometry
        col = [1.0, 0.8, 0.7]  # light red orange
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.type.mesh.parent.noc):
            if not self.type.mesh.eval.connected[n]:
                self.draw_geometries([self.type.mesh.indices_not_fcon[n]])

        # 1. Draw visible geometry
        col = [1.0, 0.2, 0.0] # red orange
        glUniform3f(5, col[0], col[1], col[2])
        G0 = self.type.mesh.indices_not_fcon
        G1 = self.type.mesh.indices_fcon
        self.draw_geometries_with_excluded_area(G0,G1)

    def unbridged(self):
        # Draw colored faces when unbridged
        for n in range(self.type.noc):
            if not self.type.mesh.eval.bridged[n]:
                for m in range(2): # browse the two parts
                    # a) Unbridge part 1
                    col = self.view.unbridge_colors[n][m]
                    glUniform3f(5, col[0], col[1], col[2])
                    G0 = [self.type.mesh.indices_not_fbridge[n][m]]
                    G1 = [self.type.mesh.indices_not_fbridge[n][1-m],
                          self.type.mesh.indices_fall[1-n],
                          self.type.mesh.indices_not_fcon[n]] # needs reformulation for 3 components
                    self.draw_geometries_with_excluded_area(G0,G1)

    def checker(self):
        # 1. Draw hidden geometry
        glUniform3f(5, 1.0, 0.2, 0.0) # red orange
        glLineWidth(8)
        for n in range(self.type.mesh.parent.noc):
            if self.type.mesh.eval.checker[n]:
                self.draw_geometries([self.type.mesh.indices_chess_lines[n]])
        glUniform3f(5, 0.0, 0.0, 0.0) # back to black

    def arrows(self):
        #glClear(GL_DEPTH_BUFFER_BIT)
        glUniform3f(5, 0.0, 0.0, 0.0)
        ############################## Direction arrows ################################
        for n in range(self.type.noc):
            if (self.type.mesh.eval.interlocks[n]): glUniform3f(5,0.0,0.0,0.0) # black
            else: glUniform3f(5,1.0,0.0,0.0) # red
            glLineWidth(3)
            G1 = self.type.mesh.indices_fall
            G0 = self.type.mesh.indices_arrows[n]
            d0 = 2.55*self.type.component_size
            d1 = 1.55*self.type.component_size
            if len(self.type.fixed.sides[n])==2: d0 = d1
            for side in self.type.fixed.sides[n]:
                vec = d0*(2*side.dir-1)*self.type.pos_vecs[side.ax]/np.linalg.norm(self.type.pos_vecs[side.ax])
                #draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)
                self.draw_geometries(G0,translation_vec=vec)

    def nondurable(self):
        # 1. Draw hidden geometry
        col = [1.0, 1.0, 0.8] # super light yellow
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(self.type.noc):
            self.draw_geometries_with_excluded_area([self.type.mesh.indices_fbrk[n]],[self.type.mesh.indices_not_fbrk[n]])

        # Draw visible geometry
        col = [1.0, 1.0, 0.4] # light yellow
        glUniform3f(5, col[0], col[1], col[2])
        self.draw_geometries_with_excluded_area(self.type.mesh.indices_fbrk,self.type.mesh.indices_not_fbrk)

    def milling_paths(self):
        if len(self.type.mesh.indices_milling_path)==0: self.view.show_milling_path = False
        if self.view.show_milling_path:
            cols = [[1.0,0,0],[0,1.0,0],[0,0,1.0],[1.0,1.0,0],[0.0,1.0,1.0],[1.0,0,1.0]]
            glLineWidth(3)
            for n in range(self.type.noc):
                if self.type.mesh.eval.fab_direction_ok[n]:
                    glUniform3f(5,cols[n][0],cols[n][1],cols[n][2])
                    self.draw_geometries([self.type.mesh.indices_milling_path[n]])
