class Model:
    def __init__(self):
        import ctypes
        from pyglet.gl import glGenVertexArrays, glBindVertexArray, glGenBuffers, glBindBuffer, GL_ARRAY_BUFFER, glBufferData, GL_DYNAMIC_DRAW, glVertexAttribPointer, glEnableVertexAttribArray, GL_FLOAT, GL_FALSE, GLuint
        self.batch_vao = GLuint(0)
        self.batch_vbo = GLuint(0)
        
        glGenVertexArrays(1, ctypes.byref(self.batch_vao))
        glBindVertexArray(self.batch_vao)
        
        glGenBuffers(1, ctypes.byref(self.batch_vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.batch_vbo)
        
        # Max expected vertices per entity model (e.g. 6 parts * 6 faces * 6 verts = 216 verts)
        # Allocate enough space for 1000 vertices * 15 floats * 4 bytes = 60KB
        glBufferData(GL_ARRAY_BUFFER, 1000 * 15 * 4, None, GL_DYNAMIC_DRAW)
        
        stride = 15 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(52))
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(56))
        glEnableVertexAttribArray(6)
        
        glBindVertexArray(0)

    def render(self, entity, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, parent_matrix, u_view_loc):
        self.setup_anim(walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale)
        
    def setup_anim(self, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale):
        pass

    def render_batched(self, parts, parent_matrix, u_view_loc, scale):
        import numpy as np
        import ctypes
        from pyglet.gl import glBindVertexArray, glDrawArrays, GL_TRIANGLES, glBindBuffer, GL_ARRAY_BUFFER, glBufferSubData, glUniformMatrix4fv, GL_FALSE, GLfloat
        
        all_verts = []
        for part in parts:
            verts = part.get_transformed_vertices(parent_matrix, scale)
            if verts is not None:
                all_verts.append(verts)
                
        if not all_verts:
            return
            
        combined = np.concatenate(all_verts)
        data_len = len(combined) * 4
        
        glBindVertexArray(self.batch_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.batch_vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, data_len, combined.ctypes.data)
        
        # We supply Identity matrix because vertices are already transformed to View Space
        identity = np.eye(4, dtype=np.float32).flatten()
        glUniformMatrix4fv(u_view_loc, 1, GL_FALSE, (GLfloat * 16)(*identity))
        
        glDrawArrays(GL_TRIANGLES, 0, len(combined) // 15)
        glBindVertexArray(0)
