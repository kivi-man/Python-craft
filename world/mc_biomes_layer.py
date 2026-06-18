import numpy as np
from numba import njit
import math

# Biome IDs
OCEAN = 0
PLAINS = 1
DESERT = 2
EXTREME_HILLS = 3
FOREST = 4
TAIGA = 5
SWAMPLAND = 6
RIVER = 7
ICE_PLAINS = 12
JUNGLE = 21
MUSHROOM_ISLAND = 14
BEACH = 16

@njit(nogil=True, cache=True)
def get_chunk_seed(x, z, base_seed, layer_seed):
    chunkSeed = np.int64(x) * np.int64(341873128712) + np.int64(z) * np.int64(132897987541) + np.int64(base_seed) + np.int64(layer_seed)
    chunkSeed = chunkSeed * chunkSeed * np.int64(68922841597) + chunkSeed * np.int64(11)
    return chunkSeed

@njit(nogil=True, cache=True)
def next_int(chunkSeed, max_val):
    r = int((chunkSeed >> 24) % max_val)
    if r < 0: r += max_val
    chunkSeed = chunkSeed * chunkSeed * np.int64(68922841597) + chunkSeed * np.int64(11)
    return r, chunkSeed

@njit(nogil=True, cache=True)
def layer_island(x, z, w, h, base_seed):
    layer_seed = 1
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
            r, cs = next_int(cs, 10)
            if rx == 0 and rz == 0:
                res[i, j] = 1 # Land
            else:
                res[i, j] = 1 if r == 0 else OCEAN
    return res

@njit(nogil=True, cache=True)
def layer_fuzzy_zoom(x, z, w, h, parent_data, base_seed):
    layer_seed = 2000
    res = np.zeros((w, h), dtype=np.int32)
    xo = x >> 1
    zo = z >> 1
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            px = (rx >> 1) - xo
            pz = (rz >> 1) - zo
            
            v1 = parent_data[px, pz]
            v2 = parent_data[px + 1, pz]
            v3 = parent_data[px, pz + 1]
            v4 = parent_data[px + 1, pz + 1]
            
            cs = get_chunk_seed(rx >> 1 << 1, rz >> 1 << 1, base_seed, layer_seed)
            
            bx = rx & 1
            bz = rz & 1
            if bx == 0 and bz == 0:
                res[i, j] = v1
            elif bx == 1 and bz == 0:
                idx, cs = next_int(cs, 2)
                res[i, j] = v1 if idx == 0 else v2
            elif bx == 0 and bz == 1:
                idx, cs = next_int(cs, 2)
                res[i, j] = v1 if idx == 0 else v3
            else:
                idx, cs = next_int(cs, 4)
                if idx == 0: res[i, j] = v1
                elif idx == 1: res[i, j] = v2
                elif idx == 2: res[i, j] = v3
                else: res[i, j] = v4
    return res

@njit(nogil=True, cache=True)
def layer_zoom(x, z, w, h, parent_data, base_seed, layer_seed):
    res = np.zeros((w, h), dtype=np.int32)
    xo = x >> 1
    zo = z >> 1
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            px = (rx >> 1) - xo
            pz = (rz >> 1) - zo
            
            v1 = parent_data[px, pz]
            v2 = parent_data[px + 1, pz]
            v3 = parent_data[px, pz + 1]
            v4 = parent_data[px + 1, pz + 1]
            
            cs = get_chunk_seed(rx >> 1 << 1, rz >> 1 << 1, base_seed, layer_seed)
            
            bx = rx & 1
            bz = rz & 1
            if bx == 0 and bz == 0:
                res[i, j] = v1
            elif bx == 1 and bz == 0:
                idx, cs = next_int(cs, 2)
                res[i, j] = v1 if idx == 0 else v2
            elif bx == 0 and bz == 1:
                idx, cs = next_int(cs, 2)
                res[i, j] = v1 if idx == 0 else v3
            else:
                # Actual zoom logic for diagonal selects randomly, but handles identical neighbors smartly
                if v1 == v2 and v3 == v4:
                    idx, cs = next_int(cs, 2)
                    res[i, j] = v1 if idx == 0 else v3
                elif v1 == v3 and v2 == v4:
                    idx, cs = next_int(cs, 2)
                    res[i, j] = v1 if idx == 0 else v2
                elif v1 == v4 and v2 == v3:
                    idx, cs = next_int(cs, 2)
                    res[i, j] = v1 if idx == 0 else v2
                elif v1 == v2 or v1 == v3 or v1 == v4:
                    res[i, j] = v1
                elif v2 == v3 or v2 == v4:
                    res[i, j] = v2
                elif v3 == v4:
                    res[i, j] = v3
                else:
                    idx, cs = next_int(cs, 4)
                    if idx == 0: res[i, j] = v1
                    elif idx == 1: res[i, j] = v2
                    elif idx == 2: res[i, j] = v3
                    else: res[i, j] = v4
    return res

@njit(nogil=True, cache=True)
def layer_add_island(x, z, w, h, parent_data, base_seed, layer_seed):
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            px = i + 1
            pz = j + 1
            
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]
            n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]
            n4 = parent_data[px, pz + 1]
            
            if c == OCEAN and (n1 != OCEAN or n2 != OCEAN or n3 != OCEAN or n4 != OCEAN):
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 5)
                if r == 0:
                    c = 1 # Land
            elif c != OCEAN and (n1 == OCEAN or n2 == OCEAN or n3 == OCEAN or n4 == OCEAN):
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 5)
                if r == 0:
                    c = OCEAN # Eroded by ocean
            res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_biome_init(x, z, w, h, parent_data, base_seed):
    layer_seed = 200
    res = np.zeros((w, h), dtype=np.int32)
    # Jungle, Desert, Forest, ExtremeHills, Swampland, Plains, Taiga
    biomes = np.array([JUNGLE, DESERT, FOREST, EXTREME_HILLS, SWAMPLAND, PLAINS, TAIGA], dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            c = parent_data[i, j]
            if c == 0:
                res[i, j] = OCEAN
            else:
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                idx, cs = next_int(cs, len(biomes))
                res[i, j] = biomes[idx]
    return res

@njit(nogil=True, cache=True)
def layer_river_init(x, z, w, h, parent_data, base_seed):
    layer_seed = 100
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            c = parent_data[i, j]
            if c == 0:
                res[i, j] = 0
            else:
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 299999)
                res[i, j] = r + 2 # Arbitrary large number to trigger rivers later
    return res

@njit(nogil=True, cache=True)
def layer_river(x, z, w, h, parent_data, base_seed):
    layer_seed = 1
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i
            rz = z + j
            px = i + 1
            pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]
            n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]
            n4 = parent_data[px, pz + 1]
            
            # Helper logic inline
            def river_filter(v):
                return v >= 2
                
            if river_filter(c) and (not river_filter(n1) or not river_filter(n2) or not river_filter(n3) or not river_filter(n4)):
                res[i, j] = RIVER
            else:
                res[i, j] = -1
    return res

@njit(nogil=True, cache=True)
def layer_smooth(x, z, w, h, parent_data, base_seed, layer_seed):
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1
            pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]
            n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]
            n4 = parent_data[px, pz + 1]
            
            if n1 == n2 and n3 == n4:
                cs = get_chunk_seed(x + i, z + j, base_seed, layer_seed)
                r, cs = next_int(cs, 2)
                res[i, j] = n1 if r == 0 else n3
            elif n1 == n2: res[i, j] = n1
            elif n3 == n4: res[i, j] = n3
            else: res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_river_mixer(x, z, w, h, biome_data, river_data):
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            b = biome_data[i, j]
            r = river_data[i, j]
            if b == OCEAN:
                res[i, j] = OCEAN
            elif r == RIVER:
                if b == ICE_PLAINS:
                    res[i, j] = RIVER # frozen river eventually
                else:
                    res[i, j] = RIVER
            else:
                res[i, j] = b
    return res

# --- Pipeline Evaluator ---
@njit(nogil=True, cache=True)
def get_biome_layer_data(bx, bz, bw, bh, base_seed):
    # BX, BZ are the requested coordinates.
    # To correctly calculate the required parent sizes backward:
    
    # Smooth (1000)
    req_s_w, req_s_h = bw + 2, bh + 2
    
    # Zoom 6 (1005)
    req_z6_w, req_z6_h = (req_s_w >> 1) + 2, (req_s_h >> 1) + 2
    # Zoom 5 (1004)
    req_z5_w, req_z5_h = (req_z6_w >> 1) + 2, (req_z6_h >> 1) + 2
    # Zoom 4 (1003)
    req_z4_w, req_z4_h = (req_z5_w >> 1) + 2, (req_z5_h >> 1) + 2
    # Zoom 3 (1002)
    req_z3_w, req_z3_h = (req_z4_w >> 1) + 2, (req_z4_h >> 1) + 2
    # Zoom 2 (1001)
    req_z2_w, req_z2_h = (req_z3_w >> 1) + 2, (req_z3_h >> 1) + 2
    # Zoom 1 (1000)
    req_z1_w, req_z1_h = (req_z2_w >> 1) + 2, (req_z2_h >> 1) + 2
    
    # BiomeInit (200) -> requires same size
    req_bi_w, req_bi_h = req_z1_w, req_z1_h
    
    # Zoom pre-biome (2001)
    req_zp_w, req_zp_h = (req_bi_w >> 1) + 2, (req_bi_h >> 1) + 2
    # AddIsland 2 (2)
    req_ai2_w, req_ai2_h = req_zp_w + 2, req_zp_h + 2
    # FuzzyZoom (2000)
    req_fz_w, req_fz_h = (req_ai2_w >> 1) + 2, (req_ai2_h >> 1) + 2
    # Island (1)
    req_is_w, req_is_h = req_fz_w, req_fz_h

    # To calculate exact offsets, we need to trace X, Z backward
    x_s = bx - 1; z_s = bz - 1
    x_z6 = x_s >> 1; z_z6 = z_s >> 1
    x_z5 = x_z6 >> 1; z_z5 = z_z6 >> 1
    x_z4 = x_z5 >> 1; z_z4 = z_z5 >> 1
    x_z3 = x_z4 >> 1; z_z3 = z_z4 >> 1
    x_z2 = x_z3 >> 1; z_z2 = z_z3 >> 1
    x_z1 = x_z2 >> 1; z_z1 = z_z2 >> 1
    x_bi = x_z1; z_bi = z_z1
    x_zp = x_bi >> 1; z_zp = z_bi >> 1
    x_ai2 = x_zp - 1; z_ai2 = z_zp - 1
    x_fz = x_ai2 >> 1; z_fz = z_ai2 >> 1
    x_is = x_fz; z_is = z_fz
    
    # 1. Island
    d_is = layer_island(x_is, z_is, req_is_w, req_is_h, base_seed)
    
    # 2. Fuzzy Zoom
    d_fz = layer_fuzzy_zoom(x_ai2, z_ai2, req_ai2_w, req_ai2_h, d_is, base_seed)
    
    # 3. AddIsland 2
    d_ai2 = layer_add_island(x_zp, z_zp, req_zp_w, req_zp_h, d_fz, base_seed, 2)
    
    # 4. Zoom Pre-Biome
    d_zp = layer_zoom(x_bi, z_bi, req_bi_w, req_bi_h, d_ai2, base_seed, 2001)
    
    # 5. Biome Init
    d_bi = layer_biome_init(x_bi, z_bi, req_bi_w, req_bi_h, d_zp, base_seed)
    
    # 6. Zooms
    d_z1 = layer_zoom(x_z2, z_z2, req_z2_w, req_z2_h, d_bi, base_seed, 1000)
    d_z2 = layer_zoom(x_z3, z_z3, req_z3_w, req_z3_h, d_z1, base_seed, 1001)
    d_z3 = layer_zoom(x_z4, z_z4, req_z4_w, req_z4_h, d_z2, base_seed, 1002)
    d_z4 = layer_zoom(x_z5, z_z5, req_z5_w, req_z5_h, d_z3, base_seed, 1003)
    d_z5 = layer_zoom(x_z6, z_z6, req_z6_w, req_z6_h, d_z4, base_seed, 1004)
    d_z6 = layer_zoom(x_s, z_s, req_s_w, req_s_h, d_z5, base_seed, 1005)
    
    # 7. Smooth
    d_s = layer_smooth(bx, bz, bw, bh, d_z6, base_seed, 1000)
    
    return d_s
