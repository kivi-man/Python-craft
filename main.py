"""
PythonCraft God-Tier Engine
Ana dosya — Pencere, kamera, dünya üretimi ve render döngüsü.

WASD: Hareket
Mouse: Bakış yönü
Space: Yukarı
Shift: Aşağı
ESC: Çıkış
"""
import math
import os
import sys
import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse
import concurrent.futures
from renderer.mesh_builder import build_chunk_mesh_bg
import ctypes
import numpy as np
import math
import time
import os

from world.mc_terrain import generate_chunk, recalculate_chunk_light, CHUNK_SIZE, CHUNK_HEIGHT, AIR, WATER
from world.terrain import CACTUS, SAND
from renderer.mesh_builder import build_chunk_mesh
from core.player import Player
from core.raycast import raycast

# ─────────────────────────── SHADER DERLEME ───────────────────────────

def compile_shader(source, shader_type):
    shader = glCreateShader(shader_type)
    source_bytes = source.encode('utf-8')
    src_buffer = ctypes.create_string_buffer(source_bytes)
    buf_pointer = ctypes.cast(ctypes.pointer(ctypes.pointer(src_buffer)),
                              ctypes.POINTER(ctypes.POINTER(ctypes.c_char)))
    length = ctypes.c_int(len(source_bytes))
    glShaderSource(shader, 1, buf_pointer, ctypes.byref(length))
    glCompileShader(shader)
    
    # Hata kontrolü
    status = ctypes.c_int(0)
    glGetShaderiv(shader, GL_COMPILE_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int(0)
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        glGetShaderInfoLog(shader, log_length, None, log)
        raise RuntimeError(f"Shader compile error: {log.value.decode()}")
    return shader

def create_shader_program(vert_path, frag_path):
    with open(vert_path, 'r') as f:
        vert_src = f.read()
    # İlk satırı atla (Python comment)
    if vert_src.startswith('#'):
        pass  # GLSL # ile başlar, sorun yok
    # Python yorumunu temizle
    lines = vert_src.split('\n')
    clean_lines = [l for l in lines if not l.strip().startswith('# ')]
    vert_src = '\n'.join(clean_lines)
    
    with open(frag_path, 'r') as f:
        frag_src = f.read()
    lines = frag_src.split('\n')
    clean_lines = [l for l in lines if not l.strip().startswith('# ')]
    frag_src = '\n'.join(clean_lines)
    
    vs = compile_shader(vert_src, GL_VERTEX_SHADER)
    fs = compile_shader(frag_src, GL_FRAGMENT_SHADER)
    
    program = glCreateProgram()
    glAttachShader(program, vs)
    glAttachShader(program, fs)
    glLinkProgram(program)
    
    status = ctypes.c_int(0)
    glGetProgramiv(program, GL_LINK_STATUS, ctypes.byref(status))
    if not status.value:
        log_length = ctypes.c_int(0)
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, ctypes.byref(log_length))
        log = ctypes.create_string_buffer(log_length.value)
        glGetProgramInfoLog(program, log_length, None, log)
        raise RuntimeError(f"Shader link error: {log.value.decode()}")
    
    glDeleteShader(vs)
    glDeleteShader(fs)
    return program

# ─────────────────────── MATRİS HESAPLAMA ─────────────────────────────

def perspective_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    return (GLfloat * 16)(
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), -1,
        0, 0, (2 * far * near) / (near - far), 0
    )

def look_at_matrix(eye, center, up):
    f = normalize_vec(sub_vec(center, eye))
    s = normalize_vec(cross_vec(f, up))
    u = cross_vec(s, f)
    
    return (GLfloat * 16)(
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -dot_vec(s, eye), -dot_vec(u, eye), dot_vec(f, eye), 1
    )

def normalize_vec(v):
    l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if l < 1e-8: return [0, 0, 0]
    return [v[0]/l, v[1]/l, v[2]/l]

def sub_vec(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def cross_vec(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def dot_vec(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

# ─────────────────────── KAMERA SİSTEMİ ──────────────────────────────

class Camera:
    def __init__(self):
        self.x, self.y, self.z = 32.0, 45.0, 32.0
        self.yaw = -90.0   # Yatay bakış açısı
        self.pitch = -25.0  # Dikey bakış açısı
        self.speed = 30.0
        self.sensitivity = 0.15
        
    def get_front(self):
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        fx = math.cos(rp) * math.cos(ry)
        fy = math.sin(rp)
        fz = math.cos(rp) * math.sin(ry)
        return normalize_vec([fx, fy, fz])
    
    def get_view_matrix(self):
        f = self.get_front()
        center = [self.x + f[0], self.y + f[1], self.z + f[2]]
        return look_at_matrix([self.x, self.y, self.z], center, [0, 1, 0])

# ────────────────────────── ANA MOTOR ─────────────────────────────────

def async_log(message):
    try:
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass

from core.frustum import get_visible_chunk_indices
from world.mc_terrain import load_or_generate_chunk
from core.world_db import save_chunk

class PythonCraftEngine(pyglet.window.Window):
    def __init__(self, render_distance=4, fast_leaves=False, debug_mode=False):
        super().__init__(width=1280, height=720, caption="PythonCraft God-Tier Engine",
                         resizable=True, vsync=False)
        self.debug_mode = debug_mode
        
        self.set_exclusive_mouse(True)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        glClearColor(0.47, 0.65, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')
        self.program = create_shader_program(
            os.path.join(shader_dir, 'vertex.glsl'),
            os.path.join(shader_dir, 'fragment.glsl')
        )
        print("[GPU] Shader program compiled & linked.")
        
        self.u_projection = glGetUniformLocation(self.program, b"projection")
        self.u_view = glGetUniformLocation(self.program, b"view")
        self.u_texture = glGetUniformLocation(self.program, b"u_texture")
        
        glUseProgram(self.program)
        glUniform1i(self.u_texture, 0)
        glUseProgram(0)
        
        from core.texture_manager import TextureManager
        texture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'textures')
        self.texture_manager = TextureManager(texture_dir, fast_leaves=fast_leaves)
        self.texture_manager.load_textures()
        self.block_layers = self.texture_manager.get_uvs_for_blocks()
        self.block_overlays = self.texture_manager.get_overlays_for_blocks()
        
        tex_data, num_layers = self.texture_manager.get_texture_array_data()
        
        self.texture_id = GLuint(0)
        glGenTextures(1, ctypes.byref(self.texture_id))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        from pyglet.gl import glTexImage3D
        tex_buffer = (ctypes.c_ubyte * len(tex_data)).from_buffer_copy(tex_data)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 16, 16, num_layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_buffer)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        
        self.camera = Camera()
        self.player = Player(32.0, 100.0, 32.0)
        self.selected_block_id = 3 # GRASS
        
        # Mouse basılı tutma kontrolü ve cooldown takibi
        self.mouse_held = {mouse.LEFT: False, mouse.RIGHT: False}
        self.mouse_action_cooldown = 0.0
        
        self.world_chunks = {}
        self.world_light_maps = {}
        self.world_biomes = {}
        self.pending_decorations = {}
        self.total_verts = 0
        self.rendered_chunks = 0
        
        self._init_world_system(render_distance)
        
        self._frame_count = 0
        pyglet.clock.schedule_interval(self._update_title, 0.5)
        pyglet.clock.schedule_interval(self.update, 1.0 / 120.0)
        
        print("=============================================")
        print("   ENGINE READY. ENTERING MAIN GAME LOOP.    ")
        print("   WASD: Hareket | Mouse: Bakış | ESC: Çıkış ")
        print("=============================================")
    def _init_world_system(self, render_distance):
        self.RENDER_DISTANCE = render_distance
        # Add a +2 buffer so we have enough capacity for loading chunks before old ones unload
        # Also double it for full diameter
        diameter = (self.RENDER_DISTANCE + 2) * 2
        self.TOTAL_CHUNKS = diameter * diameter
        
        # Numba JIT Frustum Culling Arrayleri
        self.chunk_bounds = np.zeros((self.TOTAL_CHUNKS, 6), dtype=np.float32)
        self.chunk_active = np.zeros(self.TOTAL_CHUNKS, dtype=np.bool_)
        self.visible_indices = np.zeros(self.TOTAL_CHUNKS, dtype=np.int32)
        
        self.chunk_indices = {}
        self.chunk_vaos_array = [None] * self.TOTAL_CHUNKS # [ (vao, vbo, vertex_count) ]
        self.free_chunk_indices = list(range(self.TOTAL_CHUNKS))
        
        # Multithreading İşçi Havuzu
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.future_to_chunk = {}      # future -> (cx, cz)
        self.mesh_future_to_chunk = {} # future -> (cx, cz)
        
        self.chunk_load_queue = []
        self.chunk_load_queue_set = set()
        self.chunk_unload_queue = []
        self.chunk_unload_queue_set = set()
        self.chunk_mesh_queue = [] # cx, cz tuples
        self.chunk_mesh_queue_set = set()
        self.modified_chunks = set()
        
        self.empty_chunk = np.zeros((0,0,0), dtype=np.uint8)
        
        self.last_player_cx = None
        self.last_player_cz = None
        
        import datetime
        self.log(f"\n=== NEW SESSION STARTED AT {datetime.datetime.now()} ===")
        self.log(f"[WORLD] Dynamic Chunk System Initialized. Pool size: {self.TOTAL_CHUNKS}")
        
    def log(self, message):
        if hasattr(self, 'executor'):
            self.executor.submit(async_log, message)
        else:
            async_log(message)
        if self.debug_mode:
            print(message)
            
    def _unload_chunk(self, cx, cz):
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
            
        if (cx, cz) in self.world_chunks:
            if (cx, cz) in self.modified_chunks:
                # Arka planda veritabanına kaydet
                b_copy = self.world_chunks[(cx, cz)].copy()
                l_copy = self.world_light_maps[(cx, cz)].copy()
                self.executor.submit(save_chunk, cx, cz, b_copy, l_copy)
                self.modified_chunks.remove((cx, cz))
            del self.world_chunks[(cx, cz)]
        if (cx, cz) in self.world_light_maps:
            del self.world_light_maps[(cx, cz)]
        if (cx, cz) in self.world_biomes:
            del self.world_biomes[(cx, cz)]
            
    def _update_chunk_loading(self):
        import time
        t_start = time.perf_counter()
        px = int(self.player.x / CHUNK_SIZE)
        pz = int(self.player.z / CHUNK_SIZE)
        
        if px == self.last_player_cx and pz == self.last_player_cz:
            return
            
        self.last_player_cx = px
        self.last_player_cz = pz
        
        # 1. Queue old chunks for unloading
        unload_dist = self.RENDER_DISTANCE + 2
        for (cx, cz) in list(self.chunk_indices.keys()):
            if abs(cx - px) > unload_dist or abs(cz - pz) > unload_dist:
                if (cx, cz) not in self.chunk_unload_queue_set:
                    self.chunk_unload_queue.append((cx, cz))
                    self.chunk_unload_queue_set.add((cx, cz))
                # Diğer kuyruklardan hemen temizle ki boşuna işlem yapılmasın
                if (cx, cz) in self.chunk_load_queue_set:
                    self.chunk_load_queue_set.remove((cx, cz))
                    if (cx, cz) in self.chunk_load_queue:
                        self.chunk_load_queue.remove((cx, cz))
                if (cx, cz) in self.chunk_mesh_queue_set:
                    self.chunk_mesh_queue_set.remove((cx, cz))
                    if (cx, cz) in self.chunk_mesh_queue:
                        self.chunk_mesh_queue.remove((cx, cz))
                
        # 2. Queue new chunks
        new_queue = []
        for cx in range(px - self.RENDER_DISTANCE, px + self.RENDER_DISTANCE + 1):
            for cz in range(pz - self.RENDER_DISTANCE, pz + self.RENDER_DISTANCE + 1):
                if (cx, cz) not in self.world_chunks and (cx, cz) not in self.chunk_indices:
                    # check if it's already queued or generating
                    if (cx, cz) not in self.chunk_load_queue_set and (cx, cz) not in self.future_to_chunk.values():
                        dist_sq = (cx - px)**2 + (cz - pz)**2
                        new_queue.append((dist_sq, cx, cz))
                        
        if new_queue:
            new_queue.sort(key=lambda item: item[0])
            for item in new_queue:
                self.chunk_load_queue.append((item[1], item[2]))
                self.chunk_load_queue_set.add((item[1], item[2]))
                
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 1.0:
            self.log(f"  [_update_chunk_loading] took {dur:.2f}ms")
        
    def _process_chunk_queues(self):
        import time
        t_start = time.perf_counter()
        # 0. Eski chunk'ları zamana yayarak sil (Kare başına maks 2 adet)
        px = int(self.player.x / CHUNK_SIZE)
        pz = int(self.player.z / CHUNK_SIZE)
        unload_dist = self.RENDER_DISTANCE + 2
        
        unloaded = 0
        t_unload_start = time.perf_counter()
        while self.chunk_unload_queue and unloaded < 2:
            cx, cz = self.chunk_unload_queue.pop(0)
            self.chunk_unload_queue_set.discard((cx, cz)) # Set'ten de çıkar
            if abs(cx - px) > unload_dist or abs(cz - pz) > unload_dist:
                self._unload_chunk(cx, cz)
                unloaded += 1
        t_unload_end = time.perf_counter()

        # 1. Bekleyen generation'ları kontrol et
        t_gen_start = time.perf_counter()
        done_gen = [f for f in self.future_to_chunk if f.done()]
        for f in done_gen[:1]: # Her frame'de en fazla 1 chunk generation işle
            cx, cz = self.future_to_chunk.pop(f)
            blocks, light_map, out_of_bounds, biomes = f.result()
            
            # Eğer yüklenirken oyuncu uzaklaştıysa, hemen çöpe at
            if abs(cx - self.last_player_cx) > self.RENDER_DISTANCE + 2 or abs(cz - self.last_player_cz) > self.RENDER_DISTANCE + 2:
                continue
                
            from world.terrain import LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, AIR, SNOW
            
            # Handle out_of_bounds (pending_decorations)
            for i in range(len(out_of_bounds)):
                wx, wy, wz, block_id = out_of_bounds[i]
                tcx, tcz = wx // 16, wz // 16
                if (tcx, tcz) not in self.pending_decorations:
                    self.pending_decorations[(tcx, tcz)] = []
                self.pending_decorations[(tcx, tcz)].append((wx, wy, wz, block_id))
                
                # If target chunk is already generated, apply it directly and schedule remesh
                if (tcx, tcz) in self.world_chunks:
                    lx, lz = wx - tcx * 16, wz - tcz * 16
                    current = self.world_chunks[(tcx, tcz)][lx, wy, lz]
                    if block_id in (LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES):
                        if current in (AIR, LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, SNOW):
                            self.world_chunks[(tcx, tcz)][lx, wy, lz] = block_id
                            if (tcx, tcz) not in self.chunk_mesh_queue_set:
                                self.chunk_mesh_queue.append((tcx, tcz))
                                self.chunk_mesh_queue_set.add((tcx, tcz))
                    else:
                        self.world_chunks[(tcx, tcz)][lx, wy, lz] = block_id
                        if (tcx, tcz) not in self.chunk_mesh_queue_set:
                            self.chunk_mesh_queue.append((tcx, tcz))
                            self.chunk_mesh_queue_set.add((tcx, tcz))

            # Apply any pending decorations destined for THIS chunk
            if (cx, cz) in self.pending_decorations:
                for wx, wy, wz, block_id in self.pending_decorations[(cx, cz)]:
                    lx, lz = wx - cx * 16, wz - cz * 16
                    current = blocks[lx, wy, lz]
                    if block_id in (LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES):
                        if current in (AIR, LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, SNOW):
                            blocks[lx, wy, lz] = block_id
                    else:
                        blocks[lx, wy, lz] = block_id
                del self.pending_decorations[(cx, cz)]
                
            self.world_chunks[(cx, cz)] = blocks
            self.world_light_maps[(cx, cz)] = light_map
            self.world_biomes[(cx, cz)] = biomes
            
            if not self.free_chunk_indices:
                self.log(f"[WARNING] No free chunk indices for ({cx}, {cz})!")
                continue
                
            chunk_idx = self.free_chunk_indices.pop()
            self.chunk_indices[(cx, cz)] = chunk_idx
            self.chunk_bounds[chunk_idx] = [
                cx * CHUNK_SIZE, 0, cz * CHUNK_SIZE,
                (cx + 1) * CHUNK_SIZE, CHUNK_HEIGHT, (cz + 1) * CHUNK_SIZE
            ]
            
            if (cx, cz) not in self.chunk_mesh_queue_set:
                self.chunk_mesh_queue.append((cx, cz))
                self.chunk_mesh_queue_set.add((cx, cz))
            
            # Komşuları da mesh sırasına ekle
            for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                ncx, ncz = cx+dx, cz+dz
                if (ncx, ncz) in self.world_chunks and (ncx, ncz) not in self.chunk_mesh_queue_set:
                    self.chunk_mesh_queue.append((ncx, ncz))
                    self.chunk_mesh_queue_set.add((ncx, ncz))
        t_gen_end = time.perf_counter()
                    
        # 2. Yeni generation'lar yolla
        t_submit_gen_start = time.perf_counter()
        submitted_gen = 0
        while self.chunk_load_queue and len(self.future_to_chunk) < 2 and submitted_gen < 1:
            cx, cz = self.chunk_load_queue.pop(0)
            self.chunk_load_queue_set.discard((cx, cz)) # Set'ten de çıkar
            future = self.executor.submit(load_or_generate_chunk, cx, cz)
            self.future_to_chunk[future] = (cx, cz)
            submitted_gen += 1
        t_submit_gen_end = time.perf_counter()
            
        # 3. Bekleyen mesh'leri kontrol et (GPU'ya yükle)
        t_mesh_start = time.perf_counter()
        done_mesh = [f for f in self.mesh_future_to_chunk if f.done()]
        for f in done_mesh[:1]: # Her frame'de en fazla 1 mesh yükle
            cx, cz = self.mesh_future_to_chunk.pop(f)
            mesh = f.result()
            self._apply_chunk_mesh(cx, cz, mesh)
        t_mesh_end = time.perf_counter()
            
        # 4. Yeni mesh'ler yolla
        t_submit_mesh_start = time.perf_counter()
        submitted_mesh = 0
        while self.chunk_mesh_queue and len(self.mesh_future_to_chunk) < 4 and submitted_mesh < 1:
            cx, cz = self.chunk_mesh_queue.pop(0)
            self.chunk_mesh_queue_set.remove((cx, cz))
            blocks = self.world_chunks.get((cx, cz))
            if blocks is None: continue
            
            # Background mesh builder
            future = self.executor.submit(
                build_chunk_mesh, 
                blocks, 
                self.world_light_maps[(cx, cz)], 
                cx, cz, 
                self.world_chunks, 
                self.world_light_maps,
                self.world_biomes,
                self.block_layers,
                self.block_overlays
            )
            self.mesh_future_to_chunk[future] = (cx, cz)
            submitted_mesh += 1
        t_submit_mesh_end = time.perf_counter()
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 2.0:
            self.log(f"  [_process_chunk_queues] took {dur:.2f}ms | Unload: {(t_unload_end-t_unload_start)*1000.0:.2f}ms | DoneGen: {(t_gen_end-t_gen_start)*1000.0:.2f}ms | SubmitGen: {(t_submit_gen_end-t_submit_gen_start)*1000.0:.2f}ms | ApplyMesh: {(t_mesh_end-t_mesh_start)*1000.0:.2f}ms | SubmitMesh: {(t_submit_mesh_end-t_submit_mesh_start)*1000.0:.2f}ms")

    def _apply_chunk_mesh(self, cx, cz, meshes):
        import time
        t_start = time.perf_counter()
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
            
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 1.0:
            self.log(f"    [_apply_chunk_mesh] ({cx}, {cz}) took {dur:.2f}ms")
    
    def get_block(self, x, y, z):
        if y < 0 or y >= CHUNK_HEIGHT: return 0
        cx = int(math.floor(x / CHUNK_SIZE))
        cz = int(math.floor(z / CHUNK_SIZE))
        chunk = self.world_chunks.get((cx, cz))
        if chunk is None: return 0
        return chunk[int(math.floor(x)) % CHUNK_SIZE, int(math.floor(y)), int(math.floor(z)) % CHUNK_SIZE]
    
    def update(self, dt):
        import time
        t_start = time.perf_counter()
        dt = min(dt, 0.05)
        
        # Mouse basılı tutma aksiyonları (Sürekli kırma/koyma)
        if self.mouse_action_cooldown > 0.0:
            self.mouse_action_cooldown -= dt
            
        if self.mouse_action_cooldown <= 0.005:
            if self.mouse_held[mouse.LEFT]:
                self._handle_mouse_action(mouse.LEFT)
                self.mouse_action_cooldown = 0.20
            elif self.mouse_held[mouse.RIGHT]:
                self._handle_mouse_action(mouse.RIGHT)
                self.mouse_action_cooldown = 0.20
        
        t_load_start = time.perf_counter()
        self._update_chunk_loading()
        t_load_end = time.perf_counter()
        
        t_queue_start = time.perf_counter()
        self._process_chunk_queues()
        t_queue_end = time.perf_counter()
        
        front = self.camera.get_front()
        flat_front = normalize_vec([front[0], 0, front[2]])
        right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
        
        dx, dz = 0.0, 0.0
        if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
        if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
        if self.keys[key.A]: dx -= right[0]; dz -= right[2]
        if self.keys[key.D]: dx += right[0]; dz += right[2]
            
        length = math.sqrt(dx*dx + dz*dz)
        if length > 0.001:
            dx /= length
            dz /= length
            
        jump = self.keys[key.SPACE]
        crouch = self.keys[key.LSHIFT]
        sprint = self.keys[key.LCTRL]
        
        # Oyuncu yüklenmemiş bir chunk'taysa düşmesini engelle
        pcx = int(math.floor(self.player.x / CHUNK_SIZE))
        pcz = int(math.floor(self.player.z / CHUNK_SIZE))
        
        t_player_start = time.perf_counter()
        if (pcx, pcz) in self.world_chunks:
            self.player.update(dt, dx, dz, jump, crouch, sprint, self.get_block)
        else:
            self.player.vy = 0.0 # Yerçekimi birikmesini sıfırla
        t_player_end = time.perf_counter()
            
        eye_pos = self.player.get_eye_position()
        self.camera.x, self.camera.y, self.camera.z = eye_pos[0], eye_pos[1], eye_pos[2]
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 2.0:
            self.log(f"[SLOW UPDATE] Total: {dur:.2f}ms | Load: {(t_load_end-t_load_start)*1000.0:.2f}ms | Queues: {(t_queue_end-t_queue_start)*1000.0:.2f}ms | Player: {(t_player_end-t_player_start)*1000.0:.2f}ms")
    
    def _handle_mouse_action(self, button):
        eye_pos = self.player.get_eye_position()
        direction = self.camera.get_front()
        hx, hy, hz, px, py, pz = raycast(eye_pos, direction, self.get_block)
        
        if hx is None: return
        target_x, target_y, target_z = None, None, None
        new_block_id = 0
        
        if button == mouse.LEFT:
            target_x, target_y, target_z = hx, hy, hz
            new_block_id = 0
        elif button == mouse.RIGHT and px is not None:
            # GÜVENLİK KATMANI: Oyuncunun kendi içine katı blok koymasını engelle (0: Hava, 4: Su hariç)
            if self.selected_block_id > 0 and self.selected_block_id != 4:
                player_aabb = self.player._get_player_aabb(self.player.x, self.player.y, self.player.z)
                block_aabb = (px, py, pz, px + 1.0, py + 1.0, pz + 1.0)
                
                # AABB kesişim kontrolü
                intersects = (
                    player_aabb[0] < block_aabb[3] and player_aabb[3] > block_aabb[0] and
                    player_aabb[1] < block_aabb[4] and player_aabb[4] > block_aabb[1] and
                    player_aabb[2] < block_aabb[5] and player_aabb[5] > block_aabb[2]
                )
                if intersects:
                    return
            
            target_x, target_y, target_z = px, py, pz
            new_block_id = self.selected_block_id
                 
        if target_x is not None:
            self.set_block(target_x, target_y, target_z, new_block_id)
            
    def on_mouse_press(self, x, y, button, modifiers):
        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = True
            self._handle_mouse_action(button)
            self.mouse_action_cooldown = 0.20 # 4 tick (0.2s) cooldown
            
    def on_mouse_release(self, x, y, button, modifiers):
        if button in (mouse.LEFT, mouse.RIGHT):
            self.mouse_held[button] = False
            
    def set_block(self, wx, wy, wz, block_id):
        wx = int(math.floor(wx))
        wy = int(math.floor(wy))
        wz = int(math.floor(wz))
        
        if not (0 <= wy < CHUNK_HEIGHT): return
        
        cx, cz = wx // CHUNK_SIZE, wz // CHUNK_SIZE
        chunk = self.world_chunks.get((cx, cz))
        if chunk is None: return
        
        lx, lz = wx % CHUNK_SIZE, wz % CHUNK_SIZE
        
        if chunk[lx, wy, lz] == block_id: return
        
        chunk[lx, wy, lz] = block_id
        self.modified_chunks.add((cx, cz))
        
        light_map = self.world_light_maps.get((cx, cz))
        if light_map is not None:
            recalculate_chunk_light(chunk, light_map)
            
        def add_urgent_mesh(ucx, ucz):
            if (ucx, ucz) in self.world_chunks:
                if (ucx, ucz) in self.chunk_mesh_queue_set:
                    self.chunk_mesh_queue.remove((ucx, ucz))
                else:
                    self.chunk_mesh_queue_set.add((ucx, ucz))
                self.chunk_mesh_queue.insert(0, (ucx, ucz))
        
        add_urgent_mesh(cx, cz)
        if lx == 0: add_urgent_mesh(cx - 1, cz)
        elif lx == CHUNK_SIZE - 1: add_urgent_mesh(cx + 1, cz)
        if lz == 0: add_urgent_mesh(cx, cz - 1)
        elif lz == CHUNK_SIZE - 1: add_urgent_mesh(cx, cz + 1)
        
        # Block update logic (Cactus survival)
        neighbors = [
            (wx+1, wy, wz), (wx-1, wy, wz),
            (wx, wy, wz+1), (wx, wy, wz-1),
            (wx, wy+1, wz) # Above
        ]
        
        for nx, ny, nz in neighbors:
            if self.get_block(nx, ny, nz) == CACTUS: # CACTUS
                below = self.get_block(nx, ny - 1, nz)
                survives = (below == SAND or below == CACTUS) # SAND or CACTUS
                if survives:
                    for dx, dz in [(1,0), (-1,0), (0,1), (0,-1)]:
                        adj = self.get_block(nx + dx, ny, nz + dz)
                        if adj != 0 and adj != 4: # Not AIR and not WATER
                            survives = False
                            break
                if not survives:
                    self.set_block(nx, ny, nz, 0) # Break cactus
    
    def on_mouse_motion(self, x, y, dx, dy):
        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
        
    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        self.camera.yaw += dx * self.camera.sensitivity
        self.camera.pitch += dy * self.camera.sensitivity
        self.camera.pitch = max(-89.0, min(89.0, self.camera.pitch))
    
    def on_key_press(self, symbol, modifiers):
        if symbol == key.ESCAPE:
            self.set_exclusive_mouse(False)
            pyglet.app.exit()
        elif symbol == key.TAB:
            self.player.is_flying = not self.player.is_flying
            mode = "AÇIK" if self.player.is_flying else "KAPALI"
            print(f"[PLAYER] Fly Modu: {mode}")
        elif symbol == key._1:
            self.selected_block_id = 1 # STONE
            print("[PLAYER] Seçilen Blok: TAŞ")
        elif symbol == key._2:
            self.selected_block_id = 3 # GRASS
            print("[PLAYER] Seçilen Blok: ÇİMEN")
        elif symbol == key._3:
            self.selected_block_id = 20 # GLASS
            print("[PLAYER] Seçilen Blok: CAM")
        elif symbol == key._4:
            self.selected_block_id = 12 # LEAVES
            print("[PLAYER] Seçilen Blok: YAPRAK")
        elif symbol == key._5:
            self.selected_block_id = 4 # WATER
            print("[PLAYER] Seçilen Blok: SU")
        elif symbol == key._6:
            self.selected_block_id = CACTUS # CACTUS
            print("[PLAYER] Seçilen Blok: KAKTÜS")
    
    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        return pyglet.event.EVENT_HANDLED
    
    def on_draw(self):
        import time
        t_start = time.perf_counter()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        
        aspect = self.width / max(self.height, 1)
        proj = perspective_matrix(70.0, aspect, 0.1, 1000.0)
        view = self.camera.get_view_matrix()
        
        glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, view)
        
        t_cull_start = time.perf_counter()
        # Proj * View matris çarpımı (Manuel)
        clip = [0.0] * 16
        for i in range(4):
            for j in range(4):
                clip[i*4+j] = proj[0*4+j]*view[i*4+0] + proj[1*4+j]*view[i*4+1] + proj[2*4+j]*view[i*4+2] + proj[3*4+j]*view[i*4+3]
        proj_view = np.array(clip, dtype=np.float32)
        
        # Numba ile ışık hızında Culling
        visible_count = get_visible_chunk_indices(proj_view, self.chunk_bounds, self.chunk_active, self.visible_indices)
        self.rendered_chunks = visible_count
        t_cull_end = time.perf_counter()
        
        # Yalnızca görünür chunk'ların draw call'larını tetikle
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        t_opaque_start = time.perf_counter()
        # OPAQUE PASS
        for i in range(visible_count):
            chunk_idx = self.visible_indices[i]
            o_vao, _, o_count, _, _, _ = self.chunk_vaos_array[chunk_idx]
            if o_count > 0:
                glBindVertexArray(o_vao)
                glDrawArrays(GL_TRIANGLES, 0, o_count)
        t_opaque_end = time.perf_counter()
                
        t_trans_start = time.perf_counter()
        # TRANSPARENT PASS
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for i in range(visible_count):
            chunk_idx = self.visible_indices[i]
            _, _, _, t_vao, _, t_count = self.chunk_vaos_array[chunk_idx]
            if t_count > 0:
                glBindVertexArray(t_vao)
                glDrawArrays(GL_TRIANGLES, 0, t_count)
                
        glDisable(GL_BLEND)
        glBindVertexArray(0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        glUseProgram(0)
        t_trans_end = time.perf_counter()
        
        self._frame_count += 1
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 4.0:
            self.log(f"[SLOW DRAW] Total: {dur:.2f}ms | Cull: {(t_cull_end-t_cull_start)*1000.0:.2f}ms | Opaque: {(t_opaque_end-t_opaque_start)*1000.0:.2f}ms | Trans: {(t_trans_end-t_trans_start)*1000.0:.2f}ms")
    
    def _update_title(self, dt):
        fps = self._frame_count / max(dt, 0.001)
        self._frame_count = 0
        cam = self.camera
        queued = len(self.chunk_load_queue) + len(self.chunk_mesh_queue)
        
        self.set_caption(
            f"Pythoncraft | FPS: {fps:.0f} | "
            f"Pos: ({cam.x:.0f}, {cam.y:.0f}, {cam.z:.0f}) | "
            f"Chunks: {self.rendered_chunks}/{len(self.world_chunks)} (Q:{queued}) | "
            f"Verts: {self.total_verts:,}"
        )

def main():
    import sys
    
    print("=============================================")
    print("      PYTHONCRAFT ENGINE INITIATING...       ")
    print("=============================================")
    
    distance = 4
    fast_leaves = False
    debug_mode = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "-fast":
                fast_leaves = True
                print("Fast mode enabled: Leaves are now opaque.")
            elif arg == "-debug":
                debug_mode = True
                print("Debug mode enabled: Performance metrics will be printed to console.")
            else:
                try:
                    distance = int(arg)
                    print(f"User requested Render Distance: {distance} ({distance*2}x{distance*2} = {distance*distance*4} chunks)")
                except ValueError:
                    pass
                    
    if fast_leaves:
        from world.terrain import BLOCK_OPAQUE_ARRAY
        BLOCK_OPAQUE_ARRAY[12] = True
        BLOCK_OPAQUE_ARRAY[16] = True
        BLOCK_OPAQUE_ARRAY[17] = True
    
    engine = PythonCraftEngine(render_distance=distance, fast_leaves=fast_leaves, debug_mode=debug_mode)
    pyglet.app.run()

if __name__ == '__main__':
    main()
