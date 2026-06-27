import ctypes
import numpy as np
from pyglet.gl import *
from core.frustum import get_visible_chunk_indices
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT

class LegacyChunkRenderer:
    def __init__(self, capacity):
        self.capacity = capacity
        
        # Numba JIT Frustum Culling Arrayleri
        self.chunk_bounds = np.zeros((self.capacity, 6), dtype=np.float32)
        self.chunk_active = np.zeros(self.capacity, dtype=np.bool_)
        self.visible_indices = np.zeros(self.capacity, dtype=np.int32)
        
        self.chunk_indices = {} # (cx, cz) -> chunk_idx
        self.chunk_vaos_array = [None] * self.capacity # [ (o_vao, o_vbo, o_count, t_vao, t_vbo, t_count) ]
        self.free_chunk_indices = list(range(self.capacity))
        self.total_verts = 0
        self.visible_count = 0
        
    def add_chunk_bounds(self, cx, cz):
        if not self.free_chunk_indices:
            return -1
        chunk_idx = self.free_chunk_indices.pop()
        self.chunk_indices[(cx, cz)] = chunk_idx
        self.chunk_bounds[chunk_idx] = [
            cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE,
            (cx + 1) * CHUNK_SIZE, CHUNK_HEIGHT, (cz + 1) * CHUNK_SIZE
        ]
        return chunk_idx
        
    def add_chunk_mesh(self, cx, cz, meshes):
        if (cx, cz) not in self.chunk_indices:
            return
            
        opaque_mesh, trans_mesh = meshes
        o_count = len(opaque_mesh) // 15
        t_count = len(trans_mesh) // 15
        
        chunk_idx = self.chunk_indices[(cx, cz)]
        old_data = self.chunk_vaos_array[chunk_idx]
        
        o_vao, o_vbo, old_o_count, t_vao, t_vbo, old_t_count = 0, 0, 0, 0, 0, 0
        
        def create_vao(mesh, count):
            if count == 0: return GLuint(0), GLuint(0)
            vao = GLuint(0)
            glGenVertexArrays(1, ctypes.byref(vao))
            glBindVertexArray(vao)
            vbo = GLuint(0)
            glGenBuffers(1, ctypes.byref(vbo))
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh.ctypes.data, GL_STATIC_DRAW)
            
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
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            return vao, vbo

        if old_data is not None:
            o_vao, o_vbo, old_o_count, t_vao, t_vbo, old_t_count = old_data
            self.total_verts -= (old_o_count + old_t_count)
            
            if o_count > 0:
                if o_vbo.value != 0:
                    glBindBuffer(GL_ARRAY_BUFFER, o_vbo)
                    glBufferData(GL_ARRAY_BUFFER, opaque_mesh.nbytes, opaque_mesh.ctypes.data, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                else:
                    o_vao, o_vbo = create_vao(opaque_mesh, o_count)
            if t_count > 0:
                if t_vbo.value != 0:
                    glBindBuffer(GL_ARRAY_BUFFER, t_vbo)
                    glBufferData(GL_ARRAY_BUFFER, trans_mesh.nbytes, trans_mesh.ctypes.data, GL_STATIC_DRAW)
                    glBindBuffer(GL_ARRAY_BUFFER, 0)
                else:
                    t_vao, t_vbo = create_vao(trans_mesh, t_count)
                
            self.chunk_vaos_array[chunk_idx] = (o_vao, o_vbo, o_count, t_vao, t_vbo, t_count)
            self.total_verts += (o_count + t_count)
            self.chunk_active[chunk_idx] = (o_count > 0 or t_count > 0)
        else:
            o_vao, o_vbo = create_vao(opaque_mesh, o_count)
            t_vao, t_vbo = create_vao(trans_mesh, t_count)
            
            self.chunk_vaos_array[chunk_idx] = (o_vao, o_vbo, o_count, t_vao, t_vbo, t_count)
            self.total_verts += (o_count + t_count)
            self.chunk_active[chunk_idx] = (o_count > 0 or t_count > 0)

    def unload_chunk(self, cx, cz):
        if (cx, cz) not in self.chunk_indices: return
        
        chunk_idx = self.chunk_indices.pop((cx, cz))
        self.free_chunk_indices.append(chunk_idx)
        self.chunk_active[chunk_idx] = False
        
        # Free VRAM
        old_data = self.chunk_vaos_array[chunk_idx]
        if old_data is not None:
            o_vao, o_vbo, o_count, t_vao, t_vbo, t_count = old_data
            if o_vao.value != 0:
                glDeleteVertexArrays(1, ctypes.byref(o_vao))
            if o_vbo.value != 0:
                glDeleteBuffers(1, ctypes.byref(o_vbo))
            if t_vao.value != 0:
                glDeleteVertexArrays(1, ctypes.byref(t_vao))
            if t_vbo.value != 0:
                glDeleteBuffers(1, ctypes.byref(t_vbo))
            self.total_verts -= (o_count + t_count)
            self.chunk_vaos_array[chunk_idx] = None
            
    def update_frustum(self, proj_view):
        self.visible_count = get_visible_chunk_indices(proj_view, self.chunk_bounds, self.chunk_active, self.visible_indices)
        return self.visible_count
        
    def render_opaque(self):
        for i in range(self.visible_count):
            chunk_idx = self.visible_indices[i]
            o_vao, _, o_count, _, _, _ = self.chunk_vaos_array[chunk_idx]
            if o_count > 0:
                glBindVertexArray(o_vao)
                glDrawArrays(GL_TRIANGLES, 0, o_count)
                
    def render_transparent(self):
        for i in range(self.visible_count):
            chunk_idx = self.visible_indices[i]
            _, _, _, t_vao, _, t_count = self.chunk_vaos_array[chunk_idx]
            if t_count > 0:
                glBindVertexArray(t_vao)
                glDrawArrays(GL_TRIANGLES, 0, t_count)
