from OpenGL.GL import *
import numpy as np
from PIL import Image

class ElementProperties:
    def __init__(self, draw_type, count, start_index, n):
        self.draw_type = draw_type
        self.count = count
        self.start_index = start_index
        self.n = n

class Buffer:
    def __init__(self,parent):
        self.parent = parent
        self.VBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.VBO)
        self.EBO = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.EBO)
        self.vertex_no_info = 8
        image = Image.open("textures/end_grain.jpg")
        self.img_data = np.array(list(image.getdata()), np.uint8)
        image = Image.open("textures/friction_area.jpg")
        self.img_data_fric = np.array(list(image.getdata()), np.uint8)
        image = Image.open("textures/contact_area.jpg")
        self.img_data_cont = np.array(list(image.getdata()), np.uint8)

    def buffer_vertices(self):
        try:
            cnt = 6*len(self.parent.vertices)
            glBufferData(GL_ARRAY_BUFFER, cnt, self.parent.vertices, GL_DYNAMIC_DRAW)
        except:
            print("--------------------------ERROR IN ARRAY BUFFER WRAPPER -------------------------------------")

        # vertex attribute pointers
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0)) #position
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12)) #color
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24)) #texture
        glEnableVertexAttribArray(2)
        glGenTextures(3)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, 0)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 400, 400, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 400, 400, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data_fric)
        glActiveTexture(GL_TEXTURE2)
        glBindTexture(GL_TEXTURE_2D, 2)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 400, 400, 0, GL_RGB, GL_UNSIGNED_BYTE, self.img_data_cont)

    def buffer_indices(self):
        cnt = 4*len(self.parent.indices)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, cnt, self.parent.indices, GL_DYNAMIC_DRAW)
