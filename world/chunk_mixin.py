import concurrent.futures
import numpy as np
import math
import ctypes
import collections
import datetime
import time
import json
from pyglet.gl import *
from world.mc_terrain import load_or_generate_chunk, CHUNK_SIZE, CHUNK_HEIGHT, recalculate_chunk_light
from world.mc_flat_terrain import load_or_generate_chunk as load_flat_chunk
from core.world_db import save_chunk, save_chunk_entities, load_chunk_entities
from renderer.mesh_builder import build_chunk_mesh
from world.terrain import CACTUS, SAND, LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, AIR, SNOW
from core.entities.pig import Pig

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
        
        # Initialize Renderer based on GPU mode
        if getattr(self, 'gpu_mode', False):
            from renderer.modern_chunk_renderer import ModernChunkRenderer
            self.chunk_renderer = ModernChunkRenderer(self.TOTAL_CHUNKS)
        else:
            from renderer.legacy_chunk_renderer import LegacyChunkRenderer
            self.chunk_renderer = LegacyChunkRenderer(self.TOTAL_CHUNKS)

        
        # Multithreading Worker Pool
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.future_to_chunk = {}      # future -> (cx, cz)
        self.mesh_future_to_chunk = {} # future -> (cx, cz)
        
        self.chunk_load_queue = collections.deque()
        self.chunk_load_queue_set = set()
        self.chunk_unload_queue = collections.deque()
        self.chunk_unload_queue_set = set()
        self.chunk_mesh_queue = collections.deque()
        self.chunk_mesh_queue_set = set()
        self.pending_vram_chunks = set() # Chunks that failed VRAM allocation
        self.modified_chunks = set()
        
        self.empty_chunk = np.zeros((0,0,0), dtype=np.uint8)
        
        self.last_player_cx = None
        self.last_player_cz = None
        
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
        self.chunk_renderer.unload_chunk(cx, cz)
            
        if (cx, cz) in self.world_chunks:
            if (cx, cz) in self.modified_chunks:
                # Save to database in the background
                b_copy = self.world_chunks[(cx, cz)].copy()
                d_copy = self.world_data_maps[(cx, cz)].copy()
                l_copy = self.world_light_maps[(cx, cz)].copy()
                self.executor.submit(save_chunk, cx, cz, b_copy, d_copy, l_copy)
                self.modified_chunks.remove((cx, cz))
            del self.world_chunks[(cx, cz)]
        if (cx, cz) in self.world_data_maps:
            del self.world_data_maps[(cx, cz)]
        if (cx, cz) in self.world_light_maps:
            del self.world_light_maps[(cx, cz)]
        if (cx, cz) in self.world_biomes:
            del self.world_biomes[(cx, cz)]
            
        # Save and remove entities
        if hasattr(self, 'entities'):
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
        t_start = time.perf_counter()
        px = int(self.player.x / CHUNK_SIZE)
        pz = int(self.player.z / CHUNK_SIZE)
        
        if px == self.last_player_cx and pz == self.last_player_cz:
            return
            
        self.last_player_cx = px
        self.last_player_cz = pz
        
        # 1. Queue old chunks for unloading
        unload_dist = self.RENDER_DISTANCE + 2
        for (cx, cz) in list(self.world_chunks.keys()):
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
                if (cx, cz) not in self.world_chunks:
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
        t_start = time.perf_counter()
        # 0. Unload old chunks gradually (max 2 per frame)
        px = int(self.player.x / CHUNK_SIZE)
        pz = int(self.player.z / CHUNK_SIZE)
        unload_dist = self.RENDER_DISTANCE + 2
        
        unloaded = 0
        t_unload_start = time.perf_counter()
        while self.chunk_unload_queue and unloaded < 2:
            cx, cz = self.chunk_unload_queue.popleft()
            self.chunk_unload_queue_set.discard((cx, cz)) # Remove from set as well
            if abs(cx - px) > unload_dist or abs(cz - pz) > unload_dist:
                self._unload_chunk(cx, cz)
                unloaded += 1
        t_unload_end = time.perf_counter()
        
        # If we unloaded chunks, VRAM is freed. Let's retry 1 pending chunk if any!
        if unloaded > 0 and self.pending_vram_chunks:
            retry_cx, retry_cz = self.pending_vram_chunks.pop()
            if (retry_cx, retry_cz) in self.world_chunks and (retry_cx, retry_cz) not in self.chunk_mesh_queue_set:
                self.chunk_mesh_queue.append((retry_cx, retry_cz))
                self.chunk_mesh_queue_set.add((retry_cx, retry_cz))

        # 1. Check pending generation tasks
        t_gen_start = time.perf_counter()
        done_gen = [f for f in self.future_to_chunk if f.done()]
        for f in done_gen[:1]: # Process at most 1 chunk generation task per frame
            cx, cz = self.future_to_chunk.pop(f)
            res = f.result()
            if len(res) == 6:
                blocks, data, light_map, out_of_bounds, biomes, generated = res
            else:
                blocks, data, light_map, out_of_bounds, biomes = res
                generated = False
                
            if generated:
                self.modified_chunks.add((cx, cz))
            
            # If player moved too far during load, discard the chunk immediately
            if abs(cx - self.last_player_cx) > self.RENDER_DISTANCE + 2 or abs(cz - self.last_player_cz) > self.RENDER_DISTANCE + 2:
                continue
                

            
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
            self.world_data_maps[(cx, cz)] = data
            self.world_light_maps[(cx, cz)] = light_map
            self.world_biomes[(cx, cz)] = biomes
            
            # Load entities
            if hasattr(self, 'entities'):
                entities_json = load_chunk_entities(cx, cz)
                if entities_json:
                    try:
                        entities_data = json.loads(entities_json)
                        for e_data in entities_data:
                            if e_data.get('type') == 'Pig':
                                p = Pig()
                                p.from_dict(e_data)
                                if hasattr(p, 'set_level'):
                                    p.set_level(self)
                                self.entities.append(p)
                    except Exception as e:
                        self.log(f"[ERROR] Failed to load entities for {cx}, {cz}: {e}")
            
            self.chunk_renderer.add_chunk_bounds(cx, cz)

            
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
            cx, cz = self.chunk_load_queue.popleft()
            self.chunk_load_queue_set.discard((cx, cz)) # Remove from set as well
            if hasattr(self, 'flat_mode') and self.flat_mode:
                future = self.executor.submit(load_flat_chunk, cx, cz)
            else:
                future = self.executor.submit(load_or_generate_chunk, cx, cz)
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
            cx, cz = self.chunk_mesh_queue.popleft()
            self.chunk_mesh_queue_set.remove((cx, cz))
            blocks = self.world_chunks.get((cx, cz))
            if blocks is None: continue
            
            # Background mesh builder
            future = self.executor.submit(
                build_chunk_mesh, 
                blocks, 
                self.world_data_maps[(cx, cz)],
                self.world_light_maps[(cx, cz)], 
                cx, cz, 
                self.world_chunks, 
                self.world_data_maps,
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
        t_start = time.perf_counter()
        if getattr(self, 'gpu_mode', False):
            success = self.chunk_renderer.add_chunk_mesh(cx, cz, meshes)
            if not success:
                self.pending_vram_chunks.add((cx, cz))
            else:
                self.pending_vram_chunks.discard((cx, cz))
        else:
            self.chunk_renderer.add_chunk_mesh(cx, cz, meshes)
            
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 1.0:
            self.log(f"    [_apply_chunk_mesh] ({cx}, {cz}) took {dur:.2f}ms")
    

    def get_block_info(self, x, y, z):
        if y < 0 or y >= CHUNK_HEIGHT: return 0, 0
        cx = int(math.floor(x / CHUNK_SIZE))
        cz = int(math.floor(z / CHUNK_SIZE))
        chunk = self.world_chunks.get((cx, cz))
        if chunk is None: return -1, 0
        data_chunk = self.world_data_maps.get((cx, cz))
        lx, ly, lz = int(math.floor(x)) % CHUNK_SIZE, int(math.floor(y)), int(math.floor(z)) % CHUNK_SIZE
        d = data_chunk[lx, ly, lz] if data_chunk is not None else 0
        return chunk[lx, ly, lz], d
        
    def get_block(self, x, y, z):
        if y < 0 or y >= CHUNK_HEIGHT: return 0
        cx = int(math.floor(x / CHUNK_SIZE))
        cz = int(math.floor(z / CHUNK_SIZE))
        chunk = self.world_chunks.get((cx, cz))
        if chunk is None: return -1
        return chunk[int(math.floor(x)) % CHUNK_SIZE, int(math.floor(y)), int(math.floor(z)) % CHUNK_SIZE]
    

    def set_block(self, wx, wy, wz, block_id, data=0):
        wx = int(math.floor(wx))
        wy = int(math.floor(wy))
        wz = int(math.floor(wz))
        
        if not (0 <= wy < CHUNK_HEIGHT): return
        
        cx, cz = wx // CHUNK_SIZE, wz // CHUNK_SIZE
        chunk = self.world_chunks.get((cx, cz))
        if chunk is None: return
        
        lx, lz = wx % CHUNK_SIZE, wz % CHUNK_SIZE
        
        if chunk[lx, wy, lz] == block_id: 
            # Check data
            d_chunk = self.world_data_maps.get((cx, cz))
            if d_chunk is not None and d_chunk[lx, wy, lz] == data:
                return
        
        chunk[lx, wy, lz] = block_id
        d_chunk = self.world_data_maps.get((cx, cz))
        if d_chunk is not None:
            d_chunk[lx, wy, lz] = data
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
                self.chunk_mesh_queue.appendleft((ucx, ucz))
        
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
    
