import math
import numpy as np
import ctypes
from world.terrain import BLOCK_TINT_ARRAY
from pyglet.gl import *

class ParticleRenderer:
    def __init__(self):
        self.vao = GLuint(0)
        self.vbo = GLuint(0)
        self.max_particles = 1000
        self.vertex_size = 15 # pos(3) + norm(3) + col(3) + tex(3) + ao(1) + light(1) + overlay(1)
        self.bytes_per_vertex = self.vertex_size * 4 # float32
        
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)
        
        glGenBuffers(1, ctypes.byref(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, self.max_particles * 6 * self.bytes_per_vertex, None, GL_DYNAMIC_DRAW)
        
        stride = self.bytes_per_vertex
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
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_particles(self, particles, engine, partial_tick):
        if not particles: return
        
        yRot = engine.camera.yaw
        xRot = engine.camera.pitch
        
        rad = math.pi / 180.0
        xa = math.cos(math.radians(yRot))
        za = math.sin(math.radians(yRot))
        
        xa2 = -za * math.sin(math.radians(xRot))
        za2 = xa * math.sin(math.radians(xRot))
        ya = math.cos(math.radians(xRot))
        
        buffer_data = []
        to_render = min(len(particles), self.max_particles)
        
        for i in range(to_render):
            p = particles[i]
            x = p.xo + (p.x - p.xo) * partial_tick
            y = p.yo + (p.y - p.yo) * partial_tick
            z = p.zo + (p.z - p.zo) * partial_tick
            
            r = 0.1 * p.size
            
            u0 = p.uo / 4.0
            u1 = u0 + 0.25
            v0 = p.vo / 4.0
            v1 = v0 + 0.25
            
            layer = 0.0
            if hasattr(engine, 'texture_manager'):
                uv_map = engine.texture_manager.get_uvs_for_blocks()
                if p.block_id < len(uv_map):
                    layer = float(uv_map[p.block_id][0]) # TOP Face
                    
            lighting = 15.0 # Max light
            ao = 1.0 # No ambient occlusion shadow
            overlay = 0.0 # No overlay
            
            cr, cg, cb = 1.0, 1.0, 1.0
            
            # Grass biome tinting (block_id == 3)
            if p.block_id == 3:
                cr, cg, cb = 0.5, 0.8, 0.4
            # Leaves (block_id == 12)
            elif BLOCK_TINT_ARRAY[p.block_id] == 2:
                cr, cg, cb = 0.3, 0.6, 0.2
            
            v1_pos = (x - xa * r - xa2 * r, y - ya * r, z - za * r - za2 * r)
            v2_pos = (x - xa * r + xa2 * r, y + ya * r, z - za * r + za2 * r)
            v3_pos = (x + xa * r + xa2 * r, y + ya * r, z + za * r + za2 * r)
            v4_pos = (x + xa * r - xa2 * r, y - ya * r, z + za * r - za2 * r)
            
            n = (0.0, 1.0, 0.0)
            col = (cr, cg, cb)
            
            def vert(pos, u, v):
                return [*pos, *n, *col, u, v, layer, ao, lighting, overlay]
            
            buffer_data.extend(vert(v1_pos, u1, v1))
            buffer_data.extend(vert(v2_pos, u1, v0))
            buffer_data.extend(vert(v3_pos, u0, v0))
            
            buffer_data.extend(vert(v1_pos, u1, v1))
            buffer_data.extend(vert(v3_pos, u0, v0))
            buffer_data.extend(vert(v4_pos, u0, v1))
            
        if buffer_data:
            data_arr = (ctypes.c_float * len(buffer_data))(*buffer_data)
            
            glBindVertexArray(self.vao)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferSubData(GL_ARRAY_BUFFER, 0, len(buffer_data) * 4, data_arr)
            
            glActiveTexture(GL_TEXTURE0)
            if hasattr(engine, 'texture_id'):
                glBindTexture(GL_TEXTURE_2D_ARRAY, engine.texture_id)
                
            glDrawArrays(GL_TRIANGLES, 0, to_render * 6)
            
            glBindVertexArray(0)
