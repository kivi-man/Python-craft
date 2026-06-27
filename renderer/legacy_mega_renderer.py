import ctypes
import numpy as np
from pyglet.gl import *
from core.frustum import get_multidraw_indices
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT

STRIDE = 15 * 4

def get_gpu_vram_mb():
    try:
        total_kb = GLint(0)
        glGetIntegerv(0x9048, ctypes.byref(total_kb))
        if total_kb.value > 0: return total_kb.value / 1024
    except: pass
    
    try:
        vbo_free = (GLint * 4)()
        glGetIntegerv(0x87FB, vbo_free)
        if vbo_free[0] > 0: return vbo_free[0] / 1024
    except: pass
    
    return 2048 # Fallback

class VRAMAllocator:
    def __init__(self, capacity_vertices):
        self.capacity = capacity_vertices
        self.free_blocks = [(0, capacity_vertices)] 

    def allocate(self, size):
        for i, (offset, bsize) in enumerate(self.free_blocks):
            if bsize >= size:
                self.free_blocks.pop(i)
                if bsize > size:
                    self.free_blocks.insert(i, (offset + size, bsize - size))
                return offset
        return -1 

    def free(self, offset, size):
        self.free_blocks.append((offset, size))
        self.free_blocks.sort(key=lambda x: x[0])
        merged = []
        for block in self.free_blocks:
            if not merged:
                merged.append(block)
            else:
                last_offset, last_size = merged[-1]
                if last_offset + last_size == block[0]:
                    merged[-1] = (last_offset, last_size + block[1])
                else:
                    merged.append(block)
        self.free_blocks = merged

class LegacyMegaRenderer:
    @property
    def visible_count(self):
        return self.o_visible_count

    def _init_arrays_and_vbos(self):
        pass

    def __init__(self, capacity):
        self.capacity = capacity
        
        # --- DYNAMIC VRAM ALLOCATION ---
        total_vram_mb = get_gpu_vram_mb()
        usable_vram_mb = max(256, total_vram_mb - 512)
        
        opaque_vram_mb = usable_vram_mb * 0.70
        trans_vram_mb = usable_vram_mb * 0.30
        
        self.vram_opaque_capacity = int((opaque_vram_mb * 1024 * 1024) / STRIDE)
        self.vram_trans_capacity = int((trans_vram_mb * 1024 * 1024) / STRIDE)
        
        print(f"[VRAM-LegacyMega] Detected GPU VRAM: {total_vram_mb:.0f} MB")
        print(f"[VRAM-LegacyMega] Dedicated to Engine: {usable_vram_mb:.0f} MB")
        
        self.o_allocator = VRAMAllocator(self.vram_opaque_capacity)
        self.t_allocator = VRAMAllocator(self.vram_trans_capacity)
        
        # CPU Culling Arrays
        self.chunk_bounds = np.zeros((self.capacity, 6), dtype=np.float32)
        self.chunk_active = np.zeros(self.capacity, dtype=np.bool_)
        
        # New Array for MultiDraw offsets
        self.chunk_vram_data = np.zeros((self.capacity, 4), dtype=np.int32)
        self.chunk_vram_data.fill(-1)
        
        self.chunk_indices = {}
        self.free_chunk_indices = list(range(self.capacity))
        
        self.total_verts = 0
        self.o_visible_count = 0
        self.t_visible_count = 0
        
        # Pre-allocated MultiDraw Arrays
        self.o_firsts = np.zeros(self.capacity, dtype=np.int32)
        self.o_counts = np.zeros(self.capacity, dtype=np.int32)
        self.t_firsts = np.zeros(self.capacity, dtype=np.int32)
        self.t_counts = np.zeros(self.capacity, dtype=np.int32)
        
        # 1. Mega VBOs (Opaque and Transparent)
        self.mega_vao_o = GLuint(0)
        self.mega_vbo_o = GLuint(0)
        self.mega_vao_t = GLuint(0)
        self.mega_vbo_t = GLuint(0)
        
        print(f"[VRAM-LegacyMega] Allocating {self.vram_opaque_capacity * STRIDE / (1024*1024):.1f} MB for Opaque Mega-VBO...")
        self._init_mega_vbo(self.vram_opaque_capacity * STRIDE, self.mega_vao_o, self.mega_vbo_o)
        
        print(f"[VRAM-LegacyMega] Allocating {self.vram_trans_capacity * STRIDE / (1024*1024):.1f} MB for Transparent Mega-VBO...")
        self._init_mega_vbo(self.vram_trans_capacity * STRIDE, self.mega_vao_t, self.mega_vbo_t)

    def _init_mega_vbo(self, size_bytes, vao_ref, vbo_ref):
        glGenVertexArrays(1, ctypes.byref(vao_ref))
        glBindVertexArray(vao_ref)
        
        glGenBuffers(1, ctypes.byref(vbo_ref))
        glBindBuffer(GL_ARRAY_BUFFER, vbo_ref)
        glBufferData(GL_ARRAY_BUFFER, size_bytes, None, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(36))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(48))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(52))
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, STRIDE, ctypes.c_void_p(56))
        glEnableVertexAttribArray(6)
        
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def add_chunk_bounds(self, cx, cz):
        if not self.free_chunk_indices: return -1
        chunk_idx = self.free_chunk_indices.pop()
        self.chunk_indices[(cx, cz)] = chunk_idx
        self.chunk_bounds[chunk_idx] = [
            cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE,
            (cx + 1) * CHUNK_SIZE, CHUNK_HEIGHT, (cz + 1) * CHUNK_SIZE
        ]
        return chunk_idx

    def add_chunk_mesh(self, cx, cz, meshes):
        if (cx, cz) not in self.chunk_indices:
            return False
            
        opaque_mesh, trans_mesh = meshes
        o_count = len(opaque_mesh) // 15
        t_count = len(trans_mesh) // 15
        
        chunk_idx = self.chunk_indices[(cx, cz)]
        
        old_o_offset = self.chunk_vram_data[chunk_idx, 0]
        old_o_count  = self.chunk_vram_data[chunk_idx, 1]
        old_t_offset = self.chunk_vram_data[chunk_idx, 2]
        old_t_count  = self.chunk_vram_data[chunk_idx, 3]
        
        # Free old allocations
        if old_o_offset != -1 and old_o_count > 0:
            self.o_allocator.free(old_o_offset, old_o_count)
        if old_t_offset != -1 and old_t_count > 0:
            self.t_allocator.free(old_t_offset, old_t_count)
            
        self.total_verts -= (old_o_count + old_t_count)
        
        success = True
        
        # Allocate new blocks
        o_offset = -1
        if o_count > 0:
            o_offset = self.o_allocator.allocate(o_count)
            if o_offset == -1:
                print(f"[VRAM-LegacyMega] OUT OF VRAM (Opaque) for chunk {cx},{cz}")
                success = False
                o_count = 0
        
        t_offset = -1
        if t_count > 0:
            t_offset = self.t_allocator.allocate(t_count)
            if t_offset == -1:
                print(f"[VRAM-LegacyMega] OUT OF VRAM (Transparent) for chunk {cx},{cz}")
                success = False
                t_count = 0
                
        self.chunk_vram_data[chunk_idx, 0] = o_offset
        self.chunk_vram_data[chunk_idx, 1] = o_count
        self.chunk_vram_data[chunk_idx, 2] = t_offset
        self.chunk_vram_data[chunk_idx, 3] = t_count
        
        self.chunk_active[chunk_idx] = (o_count > 0 or t_count > 0)
        
        # Upload Mega-VBO data
        if o_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.mega_vbo_o)
            glBufferSubData(GL_ARRAY_BUFFER, o_offset * STRIDE, opaque_mesh.nbytes, opaque_mesh.ctypes.data)
        if t_count > 0:
            glBindBuffer(GL_ARRAY_BUFFER, self.mega_vbo_t)
            glBufferSubData(GL_ARRAY_BUFFER, t_offset * STRIDE, trans_mesh.nbytes, trans_mesh.ctypes.data)
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        
        self.total_verts += (o_count + t_count)
        return success

    def unload_chunk(self, cx, cz):
        if (cx, cz) not in self.chunk_indices: return
        
        chunk_idx = self.chunk_indices.pop((cx, cz))
        self.free_chunk_indices.append(chunk_idx)
        self.chunk_active[chunk_idx] = False
        
        old_o_offset = self.chunk_vram_data[chunk_idx, 0]
        old_o_count  = self.chunk_vram_data[chunk_idx, 1]
        old_t_offset = self.chunk_vram_data[chunk_idx, 2]
        old_t_count  = self.chunk_vram_data[chunk_idx, 3]
        
        if old_o_offset != -1 and old_o_count > 0:
            self.o_allocator.free(old_o_offset, old_o_count)
        if old_t_offset != -1 and old_t_count > 0:
            self.t_allocator.free(old_t_offset, old_t_count)
            
        self.total_verts -= (old_o_count + old_t_count)
        
        self.chunk_vram_data[chunk_idx, 0] = -1
        self.chunk_vram_data[chunk_idx, 1] = -1
        self.chunk_vram_data[chunk_idx, 2] = -1
        self.chunk_vram_data[chunk_idx, 3] = -1

    def update_frustum(self, proj_view):
        self.o_visible_count, self.t_visible_count = get_multidraw_indices(
            proj_view, self.chunk_bounds, self.chunk_active, self.chunk_vram_data,
            self.o_firsts, self.o_counts, self.t_firsts, self.t_counts
        )
        return self.o_visible_count

    def render_opaque(self):
        if self.o_visible_count > 0:
            glBindVertexArray(self.mega_vao_o)
            glMultiDrawArrays(GL_TRIANGLES, 
                              self.o_firsts.ctypes.data_as(ctypes.POINTER(GLint)), 
                              self.o_counts.ctypes.data_as(ctypes.POINTER(GLsizei)), 
                              self.o_visible_count)
            glBindVertexArray(0)
            
    def render_transparent(self):
        if self.t_visible_count > 0:
            glBindVertexArray(self.mega_vao_t)
            glMultiDrawArrays(GL_TRIANGLES, 
                              self.t_firsts.ctypes.data_as(ctypes.POINTER(GLint)), 
                              self.t_counts.ctypes.data_as(ctypes.POINTER(GLsizei)), 
                              self.t_visible_count)
            glBindVertexArray(0)
