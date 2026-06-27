import ctypes
import numpy as np
from pyglet.gl import *
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT

MAX_O_VERTS = 15000
MAX_T_VERTS = 5000
STRIDE = 15 * 4

# Fallback for OpenGL 4.3 constants if missing in Pyglet older versions
GL_DRAW_INDIRECT_BUFFER = 0x8F3F
GL_SHADER_STORAGE_BUFFER = 0x90D2
GL_COMMAND_BARRIER_BIT = 0x00000040
GL_SHADER_STORAGE_BARRIER_BIT = 0x00002000

def get_gpu_vram_mb():
    try:
        # NVIDIA (GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX)
        total_kb = GLint(0)
        glGetIntegerv(0x9048, ctypes.byref(total_kb))
        if total_kb.value > 0: return total_kb.value / 1024
    except: pass
    
    try:
        # AMD (GL_VBO_FREE_MEMORY_ATI)
        vbo_free = (GLint * 4)()
        glGetIntegerv(0x87FB, vbo_free)
        if vbo_free[0] > 0: return vbo_free[0] / 1024
    except: pass
    
    return 2048 # Fallback (2 GB) if query fails


class VRAMAllocator:
    def __init__(self, capacity_vertices):
        self.capacity = capacity_vertices
        self.free_blocks = [(0, capacity_vertices)] # list of (offset, size)

    def allocate(self, size):
        for i, (offset, bsize) in enumerate(self.free_blocks):
            if bsize >= size:
                self.free_blocks.pop(i)
                if bsize > size:
                    self.free_blocks.insert(i, (offset + size, bsize - size))
                return offset
        return -1 # Out of VRAM

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


class ModernChunkRenderer:
    def __init__(self, capacity):
        self.capacity = capacity
        
        # --- DYNAMIC VRAM ALLOCATION ---
        total_vram_mb = get_gpu_vram_mb()
        # Leave 1.5 GB for OS, background apps, and Game Textures
        usable_vram_mb = max(512, total_vram_mb - 1500)
        
        # 70% for Opaque Blocks, 30% for Transparent Blocks
        opaque_vram_mb = usable_vram_mb * 0.70
        trans_vram_mb = usable_vram_mb * 0.30
        
        # Convert MB to Vertices (each vertex is STRIDE=60 bytes)
        self.vram_opaque_capacity = int((opaque_vram_mb * 1024 * 1024) / STRIDE)
        self.vram_trans_capacity = int((trans_vram_mb * 1024 * 1024) / STRIDE)
        
        print(f"[VRAM] Detected GPU VRAM: {total_vram_mb:.0f} MB")
        print(f"[VRAM] Dedicated to Engine: {usable_vram_mb:.0f} MB")
        # -------------------------------
        
        self.chunk_indices = {} # (cx, cz) -> chunk_idx
        self.free_chunk_indices = list(range(self.capacity))
        
        # Track memory offsets for each chunk so we can free them on unload
        self.chunk_vram_offsets = {} # chunk_idx -> (o_offset, o_count, t_offset, t_count)
        
        self.total_verts = 0
        self.visible_count = 0
        
        self.o_allocator = VRAMAllocator(self.vram_opaque_capacity)
        self.t_allocator = VRAMAllocator(self.vram_trans_capacity)
        
        # 1. Mega VBOs (Opaque and Transparent)
        self.mega_vao_o = GLuint(0)
        self.mega_vbo_o = GLuint(0)
        self.mega_vao_t = GLuint(0)
        self.mega_vbo_t = GLuint(0)
        
        print(f"[VRAM] Allocating {self.vram_opaque_capacity * STRIDE / (1024*1024):.1f} MB for Opaque Mega-VBO ({self.vram_opaque_capacity:,} Verts)...")
        self._init_mega_vbo(self.vram_opaque_capacity * STRIDE, self.mega_vao_o, self.mega_vbo_o)
        
        print(f"[VRAM] Allocating {self.vram_trans_capacity * STRIDE / (1024*1024):.1f} MB for Transparent Mega-VBO ({self.vram_trans_capacity:,} Verts)...")
        self._init_mega_vbo(self.vram_trans_capacity * STRIDE, self.mega_vao_t, self.mega_vbo_t)
        
        # 2. SSBOs
        # Chunk Data: min_x, min_y, min_z, active, max_x, max_y, max_z, pad, o_count, o_first, t_count, t_first
        # Total 12 floats/uints per chunk = 48 bytes
        self.chunk_data = np.zeros(self.capacity * 12, dtype=np.float32)
        self.chunk_data_uint = self.chunk_data.view(np.uint32)
        
        self.chunk_ssbo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self.chunk_ssbo))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.chunk_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.capacity * 48, self.chunk_data.ctypes.data, GL_DYNAMIC_DRAW)
        
        # Opaque and Transparent Indirect Command Buffers
        self.o_cmd_ssbo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self.o_cmd_ssbo))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.o_cmd_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.capacity * 16, None, GL_DYNAMIC_DRAW)
        
        self.t_cmd_ssbo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self.t_cmd_ssbo))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.t_cmd_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.capacity * 16, None, GL_DYNAMIC_DRAW)
        
        # Counter Buffer (opaqueCount, transCount)
        self.counter_ssbo = GLuint(0)
        glGenBuffers(1, ctypes.byref(self.counter_ssbo))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.counter_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, 8, None, GL_DYNAMIC_DRAW)
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        # Load Compute Shader
        self._load_compute_shader()
        
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

    def _load_compute_shader(self):
        with open("shaders/frustum.comp", "r") as f:
            comp_src = f.read()
        
        shader = glCreateShader(GL_COMPUTE_SHADER)
        c_str = ctypes.c_char_p(comp_src.encode('utf-8'))
        c_str_ptr = ctypes.cast(ctypes.pointer(c_str), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
        glShaderSource(shader, 1, c_str_ptr, None)
        glCompileShader(shader)
        
        status = GLint(0)
        glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(status))
        if not status:
            log_length = GLint(0)
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
            log = ctypes.create_string_buffer(log_length.value)
            glGetShaderInfoLog(shader, log_length, None, log)
            raise RuntimeError(f"Compute shader compilation failed: {log.value.decode('utf-8')}")
            
        self.compute_program = glCreateProgram()
        glAttachShader(self.compute_program, shader)
        glLinkProgram(self.compute_program)
        
        glGetProgramiv(self.compute_program, GL_LINK_STATUS, ctypes.byref(status))
        if not status:
            log_length = GLint(0)
            glGetProgramiv(self.compute_program, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
            log = ctypes.create_string_buffer(log_length.value)
            glGetProgramInfoLog(self.compute_program, log_length, None, log)
            raise RuntimeError(f"Compute program linking failed: {log.value.decode('utf-8')}")
            
        glDeleteShader(shader)
        self.u_proj_view = glGetUniformLocation(self.compute_program, b"projView")
        self.u_total_chunks = glGetUniformLocation(self.compute_program, b"totalChunks")

    def add_chunk_bounds(self, cx, cz):
        if not self.free_chunk_indices:
            print(f"[WARNING] No free chunk indices left in capacity {self.capacity}!")
            return -1
        chunk_idx = self.free_chunk_indices.pop()
        self.chunk_indices[(cx, cz)] = chunk_idx
        
        base = chunk_idx * 12
        self.chunk_data[base + 0] = cx * CHUNK_SIZE
        self.chunk_data[base + 1] = 0.0
        self.chunk_data[base + 2] = cz * CHUNK_SIZE
        self.chunk_data[base + 3] = 1.0 # active
        self.chunk_data[base + 4] = (cx + 1) * CHUNK_SIZE
        self.chunk_data[base + 5] = CHUNK_HEIGHT
        self.chunk_data[base + 6] = (cz + 1) * CHUNK_SIZE
        
        self.chunk_vram_offsets[chunk_idx] = (-1, 0, -1, 0) # init
        
        # Upload just this chunk's bounds to SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.chunk_ssbo)
        offset = base * 4
        size = 12 * 4
        sub_data = self.chunk_data[base:base+12].ctypes.data_as(ctypes.c_void_p)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset, size, sub_data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        return chunk_idx

    def add_chunk_mesh(self, cx, cz, meshes):
        if (cx, cz) not in self.chunk_indices:
            return False
            
        opaque_mesh, trans_mesh = meshes
        o_count = len(opaque_mesh) // 15
        t_count = len(trans_mesh) // 15
        
        chunk_idx = self.chunk_indices[(cx, cz)]
        old_o_offset, old_o_count, old_t_offset, old_t_count = self.chunk_vram_offsets[chunk_idx]
        
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
                print(f"[VRAM] OUT OF VRAM (Opaque) for chunk {cx},{cz} (Needs {o_count} verts)")
                success = False
                o_count = 0 # Drop mesh
        
        t_offset = -1
        if t_count > 0:
            t_offset = self.t_allocator.allocate(t_count)
            if t_offset == -1:
                print(f"[VRAM] OUT OF VRAM (Transparent) for chunk {cx},{cz} (Needs {t_count} verts)")
                success = False
                t_count = 0 # Drop mesh
                
        self.chunk_vram_offsets[chunk_idx] = (o_offset, o_count, t_offset, t_count)
        
        # Update SSBO metadata
        base = chunk_idx * 12
        self.chunk_data_uint[base + 8] = o_count
        self.chunk_data_uint[base + 9] = max(0, o_offset)
        self.chunk_data_uint[base + 10] = t_count
        self.chunk_data_uint[base + 11] = max(0, t_offset)
        
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.chunk_ssbo)
        offset_ssbo = (base + 8) * 4
        size_ssbo = 4 * 4
        sub_data = self.chunk_data_uint[base+8:base+12].ctypes.data_as(ctypes.c_void_p)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, offset_ssbo, size_ssbo, sub_data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
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
        
        old_o_offset, old_o_count, old_t_offset, old_t_count = self.chunk_vram_offsets.pop(chunk_idx)
        
        # Free allocations
        if old_o_offset != -1 and old_o_count > 0:
            self.o_allocator.free(old_o_offset, old_o_count)
        if old_t_offset != -1 and old_t_count > 0:
            self.t_allocator.free(old_t_offset, old_t_count)
            
        base = chunk_idx * 12
        self.chunk_data[base + 3] = 0.0 # set active to false
        self.chunk_data_uint[base + 8] = 0
        self.chunk_data_uint[base + 10] = 0
        
        # Update SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.chunk_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, base * 4, 12 * 4, self.chunk_data[base:base+12].ctypes.data)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        
        self.total_verts -= (old_o_count + old_t_count)

    def update_frustum(self, proj_view):
        # 1. Clear Counter Buffer
        zero_counters = np.zeros(2, dtype=np.uint32)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.counter_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8, zero_counters.ctypes.data)
        
        try:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.o_cmd_ssbo)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, ctypes.c_void_p(0))
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.t_cmd_ssbo)
            glClearBufferData(GL_SHADER_STORAGE_BUFFER, GL_R32UI, GL_RED_INTEGER, GL_UNSIGNED_INT, ctypes.c_void_p(0))
        except:
            zero_cmd = np.zeros(self.capacity * 4, dtype=np.uint32)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.o_cmd_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.capacity * 16, zero_cmd.ctypes.data)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.t_cmd_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.capacity * 16, zero_cmd.ctypes.data)
        
        # 2. Bind SSBOs
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.chunk_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.o_cmd_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.t_cmd_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.counter_ssbo)
        
        # 3. Execute Compute Shader
        glUseProgram(self.compute_program)
        glUniformMatrix4fv(self.u_proj_view, 1, GL_TRUE, (GLfloat * 16)(*proj_view))
        glUniform1i(self.u_total_chunks, self.capacity)
        
        num_groups = (self.capacity + 63) // 64
        glDispatchCompute(num_groups, 1, 1)
        
        # 4. Memory Barrier
        glMemoryBarrier(GL_COMMAND_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT)
        glUseProgram(0)
        
        # Read back counter for debug info (optional)
        counters = (GLuint * 2)()
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.counter_ssbo)
        glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, 8, ctypes.byref(counters))
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.visible_count = counters[0] + counters[1]

    def render_opaque(self):
        glBindVertexArray(self.mega_vao_o)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self.o_cmd_ssbo)
        glMultiDrawArraysIndirect(GL_TRIANGLES, 0, self.capacity, 0)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)
        glBindVertexArray(0)

    def render_transparent(self):
        glBindVertexArray(self.mega_vao_t)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, self.t_cmd_ssbo)
        glMultiDrawArraysIndirect(GL_TRIANGLES, 0, self.capacity, 0)
        glBindBuffer(GL_DRAW_INDIRECT_BUFFER, 0)
        glBindVertexArray(0)
