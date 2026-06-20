import concurrent.futures
import numpy as np
import math
import ctypes
from pyglet.gl import *
from world.mc_terrain import load_or_generate_chunk, CHUNK_SIZE, CHUNK_HEIGHT, recalculate_chunk_light
from core.world_db import save_chunk
from renderer.mesh_builder import build_chunk_mesh
from world.terrain import CACTUS, SAND

def async_log(message):
    try:
        with open("log.txt", "a", encoding="utf-8") as f:
            f.write(message + "\n")
    except Exception:
        pass


class ChunkMixin:

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
        
        # Multithreading Worker Pool
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
                # Save to database in the background
                b_copy = self.world_chunks[(cx, cz)].copy()
                l_copy = self.world_light_maps[(cx, cz)].copy()
                self.executor.submit(save_chunk, cx, cz, b_copy, l_copy)
                self.modified_chunks.remove((cx, cz))
            del self.world_chunks[(cx, cz)]
        if (cx, cz) in self.world_light_maps:
            del self.world_light_maps[(cx, cz)]
        if (cx, cz) in self.world_biomes:
            del self.world_biomes[(cx, cz)]
            
        # Save and remove entities
        if hasattr(self, 'entities'):
            import json
            from core.world_db import save_chunk_entities
            entities_to_save = []
            keep_entities = []
            for e in self.entities:
                ecx = int(math.floor(e.x / CHUNK_SIZE))
                ecz = int(math.floor(e.z / CHUNK_SIZE))
                if ecx == cx and ecz == cz:
                    if hasattr(e, 'to_dict') and not getattr(e, 'dead', False):
                        entities_to_save.append(e.to_dict())
                else:
                    keep_entities.append(e)
            self.entities = keep_entities
            if entities_to_save:
                self.executor.submit(save_chunk_entities, cx, cz, json.dumps(entities_to_save))
            else:
                self.executor.submit(save_chunk_entities, cx, cz, "[]")

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
                # Clean up from other queues to avoid redundant processing
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
        # 0. Unload old chunks gradually (max 2 per frame)
        px = int(self.player.x / CHUNK_SIZE)
        pz = int(self.player.z / CHUNK_SIZE)
        unload_dist = self.RENDER_DISTANCE + 2
        
        unloaded = 0
        t_unload_start = time.perf_counter()
        while self.chunk_unload_queue and unloaded < 2:
            cx, cz = self.chunk_unload_queue.pop(0)
            self.chunk_unload_queue_set.discard((cx, cz)) # Remove from set as well
            if abs(cx - px) > unload_dist or abs(cz - pz) > unload_dist:
                self._unload_chunk(cx, cz)
                unloaded += 1
        t_unload_end = time.perf_counter()

        # 1. Check pending generation tasks
        t_gen_start = time.perf_counter()
        done_gen = [f for f in self.future_to_chunk if f.done()]
        for f in done_gen[:1]: # Process at most 1 chunk generation task per frame
            cx, cz = self.future_to_chunk.pop(f)
            blocks, light_map, out_of_bounds, biomes = f.result()
            
            # If player moved too far during load, discard the chunk immediately
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
            
            # Load entities
            if hasattr(self, 'entities'):
                import json
                from core.world_db import load_chunk_entities
                from core.entities.pig import Pig
                
                entities_json = load_chunk_entities(cx, cz)
                if entities_json:
                    try:
                        entities_data = json.loads(entities_json)
                        for e_data in entities_data:
                            if e_data.get('type') == 'Pig':
                                p = Pig()
                                p.from_dict(e_data)
                                self.entities.append(p)
                    except Exception as e:
                        self.log(f"[ERROR] Failed to load entities for {cx}, {cz}: {e}")
            
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
            
            # Add neighbors to the mesh queue
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
            self.chunk_load_queue_set.discard((cx, cz)) # Remove from set as well
            if hasattr(self, 'flat_mode') and self.flat_mode:
                from world.mc_flat_terrain import load_or_generate_flat_chunk
                func = load_or_generate_flat_chunk
            else:
                func = load_or_generate_chunk
            future = self.executor.submit(func, cx, cz)
            self.future_to_chunk[future] = (cx, cz)
            submitted_gen += 1
        t_submit_gen_end = time.perf_counter()
            
        # 3. Check pending meshes (upload to GPU)
        t_mesh_start = time.perf_counter()
        done_mesh = [f for f in self.mesh_future_to_chunk if f.done()]
        for f in done_mesh[:1]: # Upload at most 1 mesh to GPU per frame
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
        if chunk is None: return -1
        return chunk[int(math.floor(x)) % CHUNK_SIZE, int(math.floor(y)), int(math.floor(z)) % CHUNK_SIZE]
    

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
    
