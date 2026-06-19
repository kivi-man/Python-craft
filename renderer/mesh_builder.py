"""
PythonCraft - Greedy Mesh Builder (Numba JIT Optimized)
Includes Voxel Ambient Occlusion and SkyLight
"""
import numpy as np
from numba import njit
import math
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT
from world.terrain import BLOCK_COLORS_ARRAY, BLOCK_OPAQUE_ARRAY, WATER, GLASS, AIR
BIOME_COLORS = np.zeros((30, 3), dtype=np.float32)
BIOME_COLORS[1] = [0.55, 0.78, 0.35] # PLAINS
BIOME_COLORS[2] = [0.80, 0.80, 0.30] # DESERT
BIOME_COLORS[3] = [0.45, 0.60, 0.40] # EXTREME_HILLS
BIOME_COLORS[4] = [0.35, 0.65, 0.25] # FOREST
BIOME_COLORS[5] = [0.30, 0.50, 0.40] # TAIGA
BIOME_COLORS[6] = [0.40, 0.45, 0.20] # SWAMP
BIOME_COLORS[12] = [0.60, 0.80, 0.60] # ICE_PLAINS
BIOME_COLORS[21] = [0.30, 0.70, 0.15] # JUNGLE

@njit(nogil=True)
def _get_block_jit(blocks, n_left, n_right, n_front, n_back, x, y, z):
    if y < 0 or y >= CHUNK_HEIGHT:
        return 0
    if x < 0:
        if n_left.shape[0] == 0: return 0
        return n_left[CHUNK_SIZE - 1, y, z]
    if x >= CHUNK_SIZE:
        if n_right.shape[0] == 0: return 0
        return n_right[0, y, z]
    if z < 0:
        if n_back.shape[0] == 0: return 0
        return n_back[x, y, CHUNK_SIZE - 1]
    if z >= CHUNK_SIZE:
        if n_front.shape[0] == 0: return 0
        return n_front[x, y, 0]
    return blocks[x, y, z]

@njit(nogil=True)
def _get_light_jit(lights, l_left, l_right, l_front, l_back, x, y, z):
    if y < 0 or y >= CHUNK_HEIGHT:
        return 15
    if x < 0:
        if l_left.shape[0] == 0: return 15
        return l_left[CHUNK_SIZE - 1, y, z]
    if x >= CHUNK_SIZE:
        if l_right.shape[0] == 0: return 15
        return l_right[0, y, z]
    if z < 0:
        if l_back.shape[0] == 0: return 15
        return l_back[x, y, CHUNK_SIZE - 1]
    if z >= CHUNK_SIZE:
        if l_front.shape[0] == 0: return 15
        return l_front[x, y, 0]
    return lights[x, y, z]

@njit(nogil=True)
def _is_solid(blocks, n_left, n_right, n_front, n_back, x, y, z):
    b = _get_block_jit(blocks, n_left, n_right, n_front, n_back, x, y, z)
    if not _is_opaque(b):
        return 0
    return 1

@njit(nogil=True)
def _calc_ao(s1, s2, c):
    if s1 == 1 and s2 == 1:
        return 0
    return 3 - (s1 + s2 + c)

@njit(nogil=True)
def _get_ao_values(blocks, n_left, n_right, n_front, n_back, px, py, pz, u_dir, v_dir):
    def solid(du, dv):
        return _is_solid(blocks, n_left, n_right, n_front, n_back, 
                         px + u_dir[0]*du + v_dir[0]*dv, 
                         py + u_dir[1]*du + v_dir[1]*dv, 
                         pz + u_dir[2]*du + v_dir[2]*dv)
    
    s00 = solid(-1, -1)
    s10 = solid(0, -1)
    s20 = solid(1, -1)
    s01 = solid(-1, 0)
    s21 = solid(1, 0)
    s02 = solid(-1, 1)
    s12 = solid(0, 1)
    s22 = solid(1, 1)
    
    ao00 = _calc_ao(s01, s10, s00)
    ao10 = _calc_ao(s21, s10, s20)
    ao11 = _calc_ao(s21, s12, s22)
    ao01 = _calc_ao(s01, s12, s02)
    return ao00, ao10, ao11, ao01

@njit(nogil=True)
def _fix_chunk_light_boundaries(lights, blocks, l_left, l_right, l_front, l_back):
    queue_x = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * 4, dtype=np.int32)
    queue_y = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * 4, dtype=np.int32)
    queue_z = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * 4, dtype=np.int32)
    head = 0
    tail = 0
    
    for y in range(CHUNK_HEIGHT):
        for z in range(CHUNK_SIZE):
            if l_left.shape[0] > 0:
                L = l_left[CHUNK_SIZE - 1, y, z]
                if L > 1 and lights[0, y, z] < L - 1:
                    b = blocks[0, y, z]
                    if not _is_opaque(b):
                        lights[0, y, z] = L - 1
                        queue_x[tail] = 0; queue_y[tail] = y; queue_z[tail] = z; tail += 1
            if l_right.shape[0] > 0:
                L = l_right[0, y, z]
                if L > 1 and lights[CHUNK_SIZE - 1, y, z] < L - 1:
                    b = blocks[CHUNK_SIZE - 1, y, z]
                    if not _is_opaque(b):
                        lights[CHUNK_SIZE - 1, y, z] = L - 1
                        queue_x[tail] = CHUNK_SIZE - 1; queue_y[tail] = y; queue_z[tail] = z; tail += 1
                    
        for x in range(CHUNK_SIZE):
            if l_back.shape[0] > 0:
                L = l_back[x, y, CHUNK_SIZE - 1]
                if L > 1 and lights[x, y, 0] < L - 1:
                    b = blocks[x, y, 0]
                    if not _is_opaque(b):
                        lights[x, y, 0] = L - 1
                        queue_x[tail] = x; queue_y[tail] = y; queue_z[tail] = 0; tail += 1
            if l_front.shape[0] > 0:
                L = l_front[x, y, 0]
                if L > 1 and lights[x, y, CHUNK_SIZE - 1] < L - 1:
                    b = blocks[x, y, CHUNK_SIZE - 1]
                    if not _is_opaque(b):
                        lights[x, y, CHUNK_SIZE - 1] = L - 1
                        queue_x[tail] = x; queue_y[tail] = y; queue_z[tail] = CHUNK_SIZE - 1; tail += 1
                    
    dxs = np.array([1, -1, 0, 0, 0, 0], dtype=np.int32)
    dys = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
    dzs = np.array([0, 0, 0, 0, 1, -1], dtype=np.int32)
    
    while head < tail:
        x = queue_x[head]; y = queue_y[head]; z = queue_z[head]
        head += 1
        
        L = lights[x, y, z]
        if L <= 1: continue
        new_light = L - 1
        
        for i in range(6):
            nx = x + dxs[i]; ny = y + dys[i]; nz = z + dzs[i]
            if 0 <= nx < CHUNK_SIZE and 0 <= ny < CHUNK_HEIGHT and 0 <= nz < CHUNK_SIZE:
                b = blocks[nx, ny, nz]
                if not _is_opaque(b):
                    if lights[nx, ny, nz] < new_light:
                        lights[nx, ny, nz] = new_light
                        if tail < queue_x.shape[0]:
                            queue_x[tail] = nx; queue_y[tail] = ny; queue_z[tail] = nz; tail += 1

@njit(nogil=True)
def _get_biome_jit(biomes, b_left, b_right, b_front, b_back, x, z):
    if x < 0:
        if z < 0: return biomes[0, 0]
        if z >= 16: return biomes[0, 15]
        if b_left.shape[0] == 0: return biomes[0, z]
        return b_left[15, z]
    if x >= 16:
        if z < 0: return biomes[15, 0]
        if z >= 16: return biomes[15, 15]
        if b_right.shape[0] == 0: return biomes[15, z]
        return b_right[0, z]
    if z < 0:
        if b_back.shape[0] == 0: return biomes[x, 0]
        return b_back[x, 15]
    if z >= 16:
        if b_front.shape[0] == 0: return biomes[x, 15]
        return b_front[x, 0]
    return biomes[x, z]

@njit(nogil=True)
def _get_smooth_biome_color(biomes, b_left, b_right, b_front, b_back, x, z):
    r_sum, g_sum, b_sum = 0.0, 0.0, 0.0
    count = 0
    # Average 5x5 area centered on (x, z)
    for dx in range(-2, 3):
        for dz in range(-2, 3):
            nx = x + dx
            nz = z + dz
            biome_id = _get_biome_jit(biomes, b_left, b_right, b_front, b_back, nx, nz)
            if 0 <= biome_id < 30:
                c = BIOME_COLORS[biome_id]
                if c[0] == 0.0 and c[1] == 0.0 and c[2] == 0.0:
                    c = BIOME_COLORS[1]
            else:
                c = BIOME_COLORS[1]
            r_sum += c[0]
            g_sum += c[1]
            b_sum += c[2]
            count += 1
            
    return r_sum / count, g_sum / count, b_sum / count

@njit(nogil=True)
def _is_opaque(b):
    if b < 1024:
        return BLOCK_OPAQUE_ARRAY[b]
    return True

@njit(nogil=True)
def _build_chunk_mesh_jit(blocks, lights_in, n_left, n_right, n_front, n_back, l_left, l_right, l_front, l_back, b_left, b_right, b_front, b_back, wx, wz, biomes, block_layers, block_overlays):
    lights = lights_in.copy()
    _fix_chunk_light_boundaries(lights, blocks, l_left, l_right, l_front, l_back)
    
    opaque_verts = np.empty(2000000, dtype=np.float32)
    trans_verts = np.empty(2000000, dtype=np.float32)
    o_idx = 0
    t_idx = 0
    
    faces = [
        (1,  1, 0, 2,  0.0,  1.0,  0.0), # top
        (1, -1, 0, 2,  0.0, -1.0,  0.0), # bottom
        (0,  1, 2, 1,  1.0,  0.0,  0.0), # right
        (0, -1, 2, 1, -1.0,  0.0,  0.0), # left
        (2,  1, 0, 1,  0.0,  0.0,  1.0), # front
        (2, -1, 0, 1,  0.0,  0.0, -1.0)  # back
    ]
    
    for f in range(6):
        axis, direction, u_axis, v_axis, nx, ny, nz = faces[f]
        
        axis_size = CHUNK_HEIGHT if axis == 1 else CHUNK_SIZE
        u_size = CHUNK_HEIGHT if u_axis == 1 else CHUNK_SIZE
        v_size = CHUNK_HEIGHT if v_axis == 1 else CHUNK_SIZE
        
        u_dir = [0,0,0]
        v_dir = [0,0,0]
        u_dir[u_axis] = 1
        v_dir[v_axis] = 1
        
        for d in range(axis_size):
            mask = np.zeros((u_size, v_size), dtype=np.uint32)
            
            for u in range(u_size):
                for v in range(v_size):
                    pos = [0, 0, 0]
                    pos[axis] = d
                    pos[u_axis] = u
                    pos[v_axis] = v
                    x, y, z = pos[0], pos[1], pos[2]
                    
                    block_id = blocks[x, y, z]
                    if block_id == AIR or block_id in (31, 37, 38, 175, 176, 177, 178): continue
                        
                    npos = [x, y, z]
                    npos[axis] += direction
                    neighbor_id = _get_block_jit(blocks, n_left, n_right, n_front, n_back, npos[0], npos[1], npos[2])
                    
                    # Culling logic:
                    render_face = False
                    if not _is_opaque(neighbor_id):
                        if block_id == neighbor_id:
                            # Cull faces between same transparent blocks (Water-Water, Glass-Glass)
                            # BUT do not cull Leaves (12, 16, 17) so canopy looks dense inside
                            if block_id == 12 or block_id == 16 or block_id == 17:
                                render_face = True
                            else:
                                render_face = False
                        else:
                            render_face = True
                        
                    if render_face:
                        face_light = _get_light_jit(lights, l_left, l_right, l_front, l_back, npos[0], npos[1], npos[2])
                        ao00, ao10, ao11, ao01 = _get_ao_values(blocks, n_left, n_right, n_front, n_back, npos[0], npos[1], npos[2], u_dir, v_dir)
                        
                        biome_bits = 0
                        if block_id == 3 or block_id == 12 or block_id == 16 or block_id == 17:
                            biome_bits = biomes[x, z] & 0xF
                            
                        mask_val = block_id | (ao00 << 8) | (ao10 << 12) | (ao11 << 16) | (ao01 << 20) | (face_light << 24) | (biome_bits << 28)
                        mask[u, v] = mask_val
            
            visited = np.zeros((u_size, v_size), dtype=np.bool_)
            
            for u in range(u_size):
                for v in range(v_size):
                    if visited[u, v] or mask[u, v] == 0: continue
                    mask_val = mask[u, v]
                    block_id = mask_val & 0xFF
                    ao00 = (mask_val >> 8) & 0xF
                    ao10 = (mask_val >> 12) & 0xF
                    ao11 = (mask_val >> 16) & 0xF
                    ao01 = (mask_val >> 20) & 0xF
                    face_light = (mask_val >> 24) & 0xF
                    biome_bits = (mask_val >> 28) & 0xF
                    
                    width = 1
                    while u + width < u_size and not visited[u + width, v] and mask[u + width, v] == mask_val:
                        width += 1
                    height = 1
                    while v + height < v_size:
                        done = False
                        for w in range(width):
                            if visited[u + w, v + height] or mask[u + w, v + height] != mask_val:
                                done = True
                                break
                        if done: break
                        height += 1
                        
                    for du in range(width):
                        for dv in range(height):
                            visited[u + du, v + dv] = True
                            
                    is_trans = not _is_opaque(block_id)
                    verts = trans_verts if is_trans else opaque_verts
                    if is_trans: v_idx = t_idx
                    else: v_idx = o_idx
                    
                    if v_idx + 90 > verts.shape[0]: continue
                    
                    d_val = float(d + (1 if direction > 0 else 0))
                    
                    c0 = [0.0, 0.0, 0.0]; c0[axis] = d_val; c0[u_axis] = float(u); c0[v_axis] = float(v)
                    c1 = [0.0, 0.0, 0.0]; c1[axis] = d_val; c1[u_axis] = float(u + width); c1[v_axis] = float(v)
                    c2 = [0.0, 0.0, 0.0]; c2[axis] = d_val; c2[u_axis] = float(u + width); c2[v_axis] = float(v + height)
                    c3 = [0.0, 0.0, 0.0]; c3[axis] = d_val; c3[u_axis] = float(u); c3[v_axis] = float(v + height)
                    
                    e1x = c1[0] - c0[0]; e1y = c1[1] - c0[1]; e1z = c1[2] - c0[2]
                    e2x = c3[0] - c0[0]; e2y = c3[1] - c0[1]; e2z = c3[2] - c0[2]
                    cross_x = e1y*e2z - e1z*e2y
                    cross_y = e1z*e2x - e1x*e2z
                    cross_z = e1x*e2y - e1y*e2x
                    dot_prod = cross_x*nx + cross_y*ny + cross_z*nz
                    
                    if dot_prod > 0: tri_order = [0, 1, 2, 0, 2, 3]
                    else: tri_order = [0, 2, 1, 0, 3, 2]
                        
                    corners = [c0, c1, c2, c3]
                    aos = [ao00, ao10, ao11, ao01]
                    
                    uvs = [
                        [0.0, 0.0],
                        [float(width), 0.0],
                        [float(width), float(height)],
                        [0.0, float(height)]
                    ]
                    
                    layer_idx = float(block_layers[block_id, f])
                    overlay_idx = float(block_overlays[block_id, f])
                    
                    for idx in tri_order:
                        c = corners[idx]
                        ao_val = float(aos[idx])
                        u_val, v_val = uvs[idx]
                        
                        if block_id == 4:
                            vx, vy, vz = c[0] + wx, c[1], c[2] + wz
                            if abs(c[1] - round(c[1])) < 0.01:
                                bx = int(math.floor(c[0] + 0.01)) if nx <= 0 else int(math.floor(c[0] - 0.01))
                                bz = int(math.floor(c[2] + 0.01)) if nz <= 0 else int(math.floor(c[2] - 0.01))
                                if f == 0:
                                    bx = int(math.floor(c[0] + 0.01))
                                    bz = int(math.floor(c[2] + 0.01))
                                by = int(round(c[1]))
                                above = _get_block_jit(blocks, n_left, n_right, n_front, n_back, bx, by, bz)
                                if above == AIR:
                                    vy -= 0.125
                            verts[v_idx] = vx
                            verts[v_idx+1] = vy
                            verts[v_idx+2] = vz
                        elif block_id == 13:
                            vx, vy, vz = c[0] + wx, c[1], c[2] + wz
                            if f == 2: vx -= 0.0625
                            elif f == 3: vx += 0.0625
                            elif f == 4: vz -= 0.0625
                            elif f == 5: vz += 0.0625
                            verts[v_idx] = vx
                            verts[v_idx+1] = vy
                            verts[v_idx+2] = vz
                        else:
                            verts[v_idx] = c[0] + wx
                            verts[v_idx+1] = c[1]
                            verts[v_idx+2] = c[2] + wz
                            
                        verts[v_idx+3] = float(nx)
                        verts[v_idx+4] = float(ny)
                        verts[v_idx+5] = float(nz)
                        
                        if block_id == 3 and (f == 0 or (f >= 2 and overlay_idx > 0.0)):
                            vr, vg, vb = _get_smooth_biome_color(biomes, b_left, b_right, b_front, b_back, int(c[0]), int(c[2]))
                            verts[v_idx+6] = vr; verts[v_idx+7] = vg; verts[v_idx+8] = vb
                        elif block_id == 12 or block_id == 16 or block_id == 17:
                            vr, vg, vb = _get_smooth_biome_color(biomes, b_left, b_right, b_front, b_back, int(c[0]), int(c[2]))
                            verts[v_idx+6] = vr; verts[v_idx+7] = vg; verts[v_idx+8] = vb
                        else:
                            if layer_idx > 0.0:
                                r, g, b = 1.0, 1.0, 1.0
                            else:
                                r, g, b = BLOCK_COLORS_ARRAY[block_id]
                            verts[v_idx+6] = r; verts[v_idx+7] = g; verts[v_idx+8] = b
                            
                        verts[v_idx+9] = u_val
                        verts[v_idx+10] = v_val
                        verts[v_idx+11] = layer_idx
                        verts[v_idx+12] = ao_val
                        verts[v_idx+13] = float(face_light)
                        verts[v_idx+14] = overlay_idx
                        v_idx += 15
                        
                    if is_trans: t_idx = v_idx
                    else: o_idx = v_idx

    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_HEIGHT):
            for z in range(CHUNK_SIZE):
                b = blocks[x, y, z]
                if b == 31 or b == 37 or b == 38 or b == 175 or b == 176:
                    if t_idx + 15 * 24 > trans_verts.shape[0]: continue
                    
                    vx = float(x) + wx
                    vy = float(y)
                    vz = float(z) + wz
                    
                    face_light = _get_light_jit(lights, l_left, l_right, l_front, l_back, x, y, z)
                    layer_idx = float(block_layers[b, 0])
                    
                    if b == 31 or b == 175 or b == 176:
                        vr, vg, vb = _get_smooth_biome_color(biomes, b_left, b_right, b_front, b_back, x, z)
                    else:
                        vr, vg, vb = 1.0, 1.0, 1.0
                        
                    quads = [
                        [ [0.0,0.0,0.0], [1.0,0.0,1.0], [1.0,1.0,1.0], [0.0,1.0,0.0] ],
                        [ [1.0,0.0,1.0], [0.0,0.0,0.0], [0.0,1.0,0.0], [1.0,1.0,1.0] ],
                        [ [1.0,0.0,0.0], [0.0,0.0,1.0], [0.0,1.0,1.0], [1.0,1.0,0.0] ],
                        [ [0.0,0.0,1.0], [1.0,0.0,0.0], [1.0,1.0,0.0], [0.0,1.0,1.0] ]
                    ]
                    
                    for q in quads:
                        c0, c1, c2, c3 = q
                        tri_order = [0, 1, 2, 0, 2, 3]
                        corners = [c0, c1, c2, c3]
                        uvs = [ [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0] ]
                        for idx in tri_order:
                            c = corners[idx]
                            u_val, v_val = uvs[idx]
                            trans_verts[t_idx]   = vx + c[0]
                            trans_verts[t_idx+1] = vy + c[1]
                            trans_verts[t_idx+2] = vz + c[2]
                            trans_verts[t_idx+3] = 0.0
                            trans_verts[t_idx+4] = 1.0
                            trans_verts[t_idx+5] = 0.0
                            trans_verts[t_idx+6] = vr
                            trans_verts[t_idx+7] = vg
                            trans_verts[t_idx+8] = vb
                            trans_verts[t_idx+9] = u_val
                            trans_verts[t_idx+10]= v_val
                            trans_verts[t_idx+11]= layer_idx
                            trans_verts[t_idx+12]= 3.0
                            trans_verts[t_idx+13]= float(face_light)
                            trans_verts[t_idx+14]= 0.0
                            t_idx += 15

    return opaque_verts[:o_idx], trans_verts[:t_idx]

def build_chunk_mesh_bg(blocks, lights, n_left, n_right, n_front, n_back, l_left, l_right, l_front, l_back, b_left, b_right, b_front, b_back, cx, cz, biomes, block_layers, block_overlays):
    wx = float(cx * CHUNK_SIZE)
    wz = float(cz * CHUNK_SIZE)
    return _build_chunk_mesh_jit(blocks, lights, n_left, n_right, n_front, n_back, l_left, l_right, l_front, l_back, b_left, b_right, b_front, b_back, wx, wz, biomes, block_layers, block_overlays)

def build_chunk_mesh(blocks, lights, cx, cz, world_chunks, world_light_maps, world_biomes, block_layers, block_overlays):
    empty = np.zeros((0,0,0), dtype=np.uint8)
    n_left = world_chunks.get((cx - 1, cz), empty)
    n_right = world_chunks.get((cx + 1, cz), empty)
    n_front = world_chunks.get((cx, cz + 1), empty)
    n_back = world_chunks.get((cx, cz - 1), empty)
    
    l_left = world_light_maps.get((cx - 1, cz), empty)
    l_right = world_light_maps.get((cx + 1, cz), empty)
    l_front = world_light_maps.get((cx, cz + 1), empty)
    l_back = world_light_maps.get((cx, cz - 1), empty)
    
    empty_biomes = np.zeros((0,0), dtype=np.int32)
    b_left = world_biomes.get((cx - 1, cz), empty_biomes)
    b_right = world_biomes.get((cx + 1, cz), empty_biomes)
    b_front = world_biomes.get((cx, cz + 1), empty_biomes)
    b_back = world_biomes.get((cx, cz - 1), empty_biomes)
    
    wx = float(cx * CHUNK_SIZE)
    wz = float(cz * CHUNK_SIZE)
    
    empty_b = np.zeros((16, 16), dtype=np.int32)
    biomes = world_biomes.get((cx, cz), empty_b)
    
    return _build_chunk_mesh_jit(blocks, lights, n_left, n_right, n_front, n_back, l_left, l_right, l_front, l_back, b_left, b_right, b_front, b_back, wx, wz, biomes, block_layers, block_overlays)
