from OpenGL.GL import *
import numpy as np
import pyrr

class Display:
    def __init__(self,type,view_opt):
        self.type = type
        self.view_opt = view_opt

    def draw_geometries(self, geos,clear_depth_buffer=True, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.type.sax] = self.view_opt.open_ratio*self.type.component_size
        move_vec = np.array(move_vec)
        moves = []
        for n in range(self.type.noc):
            tot_move_vec = (2*n+1-self.type.noc)/(self.type.noc-1)*move_vec
            move_mat = pyrr.matrix44.create_from_translation(tot_move_vec+translation_vec)
            moves.append(move_mat)
        if clear_depth_buffer: glClear(GL_DEPTH_BUFFER_BIT)
        for geo in geos:
            if geo==None: continue
            if self.view_opt.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

    def draw_geometries_with_excluded_area(self, show_geos, screen_geos, translation_vec=np.array([0,0,0])):
        # Define translation matrices for opening
        move_vec = [0,0,0]
        move_vec[self.type.sax] = self.view_opt.open_ratio*self.type.component_size
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
            if self.view_opt.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
        glEnable(GL_DEPTH_TEST)
        glStencilFunc(GL_EQUAL,1,1)
        glStencilOp(GL_KEEP,GL_KEEP,GL_KEEP)
        glDepthRange (0.0025, 1.0)
        for geo in screen_geos:
            if geo==None: continue
            if self.view_opt.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))
        glDisable(GL_STENCIL_TEST)
        glColorMask(GL_TRUE,GL_TRUE,GL_TRUE,GL_TRUE)
        glDepthRange (0.0, 0.9975)
        for geo in show_geos:
            if geo==None: continue
            if self.view_opt.hidden[geo.n]: continue
            glUniformMatrix4fv(4, 1, GL_FALSE, moves_show[geo.n])
            glDrawElements(geo.draw_type, geo.count, GL_UNSIGNED_INT,  ctypes.c_void_p(4*geo.start_index))

    def joint_geometry(self,mesh,lw=3,hidden=True,zoom=False):

        ############################# Draw hidden lines #############################
        glClear(GL_DEPTH_BUFFER_BIT)
        glUniform3f(5,0.0,0.0,0.0) # black
        glPushAttrib(GL_ENABLE_BIT)
        glLineWidth(1)
        glLineStipple(3, 0xAAAA) #dashed line
        glEnable(GL_LINE_STIPPLE)
        if hidden and self.view_opt.show_hidden_lines:
            for n in range(mesh.parent.noc):
                G0 = [mesh.indices_lns[n]]
                G1 = [mesh.indices_fall[n]]
                self.draw_geometries_with_excluded_area(G0,G1)
        glPopAttrib()

        ############################ Draw visible lines #############################
        for n in range(mesh.parent.noc):
            if not mesh.mainmesh or (mesh.eval.interlocks[n] and self.view_opt.show_feedback) or not self.view_opt.show_feedback:
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
            if hidden and not self.view_opt.hidden[0] and not self.view_opt.hidden[1] and self.view_opt.open_ratio==1+0.5*(mesh.parent.noc-2):
                glUniform3f(5,0.0,0.0,0.0) # black
                glPushAttrib(GL_ENABLE_BIT)
                glLineWidth(2)
                glLineStipple(1, 0x00FF)
                glEnable(GL_LINE_STIPPLE)
                G0 = mesh.indices_open_lines
                G1 = mesh.indices_fall
                self.draw_geometries_with_excluded_area(G0,G1)
                glPopAttrib()


"""
    def end_grains(window,mesh):
        G0 = mesh.indices_fend
        G1 = mesh.indices_not_fend
        draw_geometries_with_excluded_area(window,G0,G1)

    def unconnected(window,mesh):
        # 1. Draw hidden geometry
        col = [1.0, 0.8, 0.7]  # light red orange
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(mesh.parent.noc):
            if not mesh.eval.connected[n]: draw_geometries(window,[mesh.indices_not_fcon[n]])

        # 1. Draw visible geometry
        col = [1.0, 0.2, 0.0] # red orange
        glUniform3f(5, col[0], col[1], col[2])
        G0 = mesh.indices_not_fcon
        G1 = mesh.indices_fcon
        draw_geometries_with_excluded_area(window,G0,G1)

    def area(window,mesh,view_opt):
        # 1. Draw hidden geometry
        if view_opt.show_friction:
            #tin = mesh.eval.friction_nums[0]/50
            #col = [0.9-tin, 1.0, 0.9-tin]  # green
            G0 = mesh.indices_ffric
            G1 = mesh.indices_not_ffric
        else:
            #tin = mesh.eval.contact_nums[0]/50
            #col = [0.9-tin, 0.9-tin, 1.0]  # blue
            G0 = mesh.indices_fcont
            G1 = mesh.indices_not_fcont
        #glUniform3f(5, col[0], col[1], col[2])
        draw_geometries_with_excluded_area(window,G0,G1)

    def unbridged(window,mesh,view_opt):
        # Draw colored faces when unbridged
        for n in range(mesh.parent.noc):
            if not mesh.eval.bridged[n]:
                for m in range(2): # browse the two parts
                    # a) Unbridge part 1
                    col = view_opt.unbridge_colors[n][m]
                    glUniform3f(5, col[0], col[1], col[2])
                    G0 = [mesh.indices_not_fbridge[n][m]]
                    G1 = [mesh.indices_not_fbridge[n][1-m], mesh.indices_fall[1-n], mesh.indices_not_fcon[n]] # needs reformulation for 3 components
                    draw_geometries_with_excluded_area(window,G0,G1)

    def arrows(window,mesh,view_opt):
        #glClear(GL_DEPTH_BUFFER_BIT)
        glUniform3f(5, 0.0, 0.0, 0.0)
        ############################## Direction arrows ################################
        for n in range(mesh.parent.noc):
            if (mesh.eval.interlocks[n]): glUniform3f(5,0.0,0.0,0.0) # black
            else: glUniform3f(5,1.0,0.0,0.0) # red
            glLineWidth(3)
            G1 = mesh.indices_fall
            G0 = mesh.indices_arrows[n]
            d0 = 2.55*mesh.parent.component_size
            d1 = 1.55*mesh.parent.component_size
            if len(mesh.parent.fixed_sides[n])==2: d0 = d1
            for ax,dir in mesh.parent.fixed_sides[n]:
                vec = d0*(2*dir-1)*mesh.parent.pos_vecs[ax]/np.linalg.norm(mesh.parent.pos_vecs[ax])
                #draw_geometries_with_excluded_area(window,G0,G1,translation_vec=vec)
                draw_geometries(window,G0,translation_vec=vec)

    def checker(window,mesh,view_opt):
        # 1. Draw hidden geometry
        glUniform3f(5, 1.0, 0.2, 0.0) # red orange
        glLineWidth(8)
        for n in range(mesh.parent.noc):
            if mesh.eval.checker[n]:
                draw_geometries(window,[mesh.indices_chess_lines[n]])
        glUniform3f(5, 0.0, 0.0, 0.0) # back to black

    def nondurable(window,mesh,view_opt):
        # 1. Draw hidden geometry
        col = [1.0, 1.0, 0.8] # super light yellow
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(mesh.parent.noc):
            draw_geometries_with_excluded_area(window,[mesh.indices_fbrk[n]],[mesh.indices_not_fbrk[n]])

        # Draw visible geometry
        col = [1.0, 1.0, 0.4] # light yellow
        glUniform3f(5, col[0], col[1], col[2])
        draw_geometries_with_excluded_area(window,mesh.indices_fbrk,mesh.indices_not_fbrk)

    def unfabricatable(window,mesh,view_opt):
        col = [1.0, 0.8, 0.5] # orange
        glUniform3f(5, col[0], col[1], col[2])
        for n in range(mesh.parent.noc):
            if not mesh.eval.fab_direction_ok[n]:
                G0 = [mesh.indices_fall[n]]
                G1 = []
                for n2 in range(mesh.parent.noc):
                    if n2!=n: G1.append(mesh.indices_fall[n2])
                draw_geometries_with_excluded_area(window,G0,G1)
"""
