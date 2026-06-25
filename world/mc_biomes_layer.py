import numpy as np
from numba import njit
import math
from world.mc_biomes import *

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
            rx = x + i; rz = z + j
            cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
            r, cs = next_int(cs, 10)
            if rx == 0 and rz == 0: res[i, j] = 1
            else: res[i, j] = 1 if r == 0 else OCEAN
    return res

@njit(nogil=True, cache=True)
def layer_fuzzy_zoom(x, z, w, h, parent_data, base_seed):
    layer_seed = 2000
    res = np.zeros((w, h), dtype=np.int32)
    xo = x >> 1; zo = z >> 1
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            px = (rx >> 1) - xo; pz = (rz >> 1) - zo
            v1 = parent_data[px, pz]; v2 = parent_data[px + 1, pz]
            v3 = parent_data[px, pz + 1]; v4 = parent_data[px + 1, pz + 1]
            cs = get_chunk_seed(rx >> 1 << 1, rz >> 1 << 1, base_seed, layer_seed)
            bx = rx & 1; bz = rz & 1
            if bx == 0 and bz == 0: res[i, j] = v1
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
    xo = x >> 1; zo = z >> 1
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            px = (rx >> 1) - xo; pz = (rz >> 1) - zo
            v1 = parent_data[px, pz]; v2 = parent_data[px + 1, pz]
            v3 = parent_data[px, pz + 1]; v4 = parent_data[px + 1, pz + 1]
            cs = get_chunk_seed(rx >> 1 << 1, rz >> 1 << 1, base_seed, layer_seed)
            bx = rx & 1; bz = rz & 1
            if bx == 0 and bz == 0: res[i, j] = v1
            elif bx == 1 and bz == 0:
                idx, cs = next_int(cs, 2); res[i, j] = v1 if idx == 0 else v2
            elif bx == 0 and bz == 1:
                idx, cs = next_int(cs, 2); res[i, j] = v1 if idx == 0 else v3
            else:
                if v1 == v2 and v3 == v4:
                    idx, cs = next_int(cs, 2); res[i, j] = v1 if idx == 0 else v3
                elif v1 == v3 and v2 == v4:
                    idx, cs = next_int(cs, 2); res[i, j] = v1 if idx == 0 else v2
                elif v1 == v4 and v2 == v3:
                    idx, cs = next_int(cs, 2); res[i, j] = v1 if idx == 0 else v2
                elif v1 == v2 or v1 == v3 or v1 == v4: res[i, j] = v1
                elif v2 == v3 or v2 == v4: res[i, j] = v2
                elif v3 == v4: res[i, j] = v3
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
            rx = x + i; rz = z + j
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]; n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]; n4 = parent_data[px, pz + 1]
            if c == OCEAN and (n1 != OCEAN or n2 != OCEAN or n3 != OCEAN or n4 != OCEAN):
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 5)
                if r == 0: c = 1
            elif c != OCEAN and (n1 == OCEAN or n2 == OCEAN or n3 == OCEAN or n4 == OCEAN):
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 5)
                if r == 0: c = OCEAN
            res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_add_snow(x, z, w, h, parent_data, base_seed):
    layer_seed = 2
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            c = parent_data[i, j]
            if c == 0: res[i, j] = 0
            else:
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 6)
                if r == 0: res[i, j] = 4 # Icy
                else:
                    r, cs = next_int(cs, 3)
                    res[i, j] = r + 1 # 1 (Hot), 2 (Warm), 3 (Cool)
    return res

@njit(nogil=True, cache=True)
def layer_deep_ocean(x, z, w, h, parent_data, base_seed):
    layer_seed = 4
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px-1, pz]; n2 = parent_data[px+1, pz]
            n3 = parent_data[px, pz-1]; n4 = parent_data[px, pz+1]
            if c == 0:
                if n1 == 0 and n2 == 0 and n3 == 0 and n4 == 0:
                    res[i, j] = DEEP_OCEAN
                else: res[i, j] = OCEAN
            else: res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_biome_init(x, z, w, h, parent_data, base_seed):
    layer_seed = 200
    res = np.zeros((w, h), dtype=np.int32)
    desert_biomes = np.array([DESERT, DESERT, DESERT, SAVANNA, SAVANNA, PLAINS], dtype=np.int32)
    warm_biomes = np.array([FOREST, ROOFED_FOREST, EXTREME_HILLS, PLAINS, BIRCH_FOREST, SWAMPLAND], dtype=np.int32)
    cool_biomes = np.array([FOREST, EXTREME_HILLS, TAIGA, PLAINS], dtype=np.int32)
    icy_biomes = np.array([ICE_FLATS, ICE_FLATS, ICE_FLATS, COLD_TAIGA], dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            c = parent_data[i, j]
            if c == 0: res[i, j] = OCEAN
            elif c == DEEP_OCEAN or c == MUSHROOM_ISLAND: res[i, j] = c
            else:
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                if c == 1:
                    r, cs = next_int(cs, len(desert_biomes)); res[i, j] = desert_biomes[r]
                elif c == 3:
                    r, cs = next_int(cs, len(cool_biomes)); res[i, j] = cool_biomes[r]
                elif c == 4:
                    r, cs = next_int(cs, len(icy_biomes)); res[i, j] = icy_biomes[r]
                else:
                    r, cs = next_int(cs, len(warm_biomes)); res[i, j] = warm_biomes[r]
    return res

@njit(nogil=True, cache=True)
def layer_biome_edge(x, z, w, h, parent_data, base_seed):
    layer_seed = 1000
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px-1, pz]; n2 = parent_data[px+1, pz]
            n3 = parent_data[px, pz-1]; n4 = parent_data[px, pz+1]
            if c == EXTREME_HILLS:
                if n1 != EXTREME_HILLS or n2 != EXTREME_HILLS or n3 != EXTREME_HILLS or n4 != EXTREME_HILLS:
                    res[i, j] = SMALLER_EXTREME_HILLS
                else: res[i, j] = c
            elif c == DESERT:
                if n1 == ICE_FLATS or n2 == ICE_FLATS or n3 == ICE_FLATS or n4 == ICE_FLATS:
                    res[i, j] = EXTREME_HILLS
                else: res[i, j] = c
            else: res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_rare_biome(x, z, w, h, parent_data, base_seed):
    layer_seed = 1001
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            c = parent_data[i, j]
            cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
            r, cs = next_int(cs, 57)
            if r == 0:
                if c == PLAINS: res[i, j] = SUNFLOWERS_PLAINS
                elif c == DESERT: res[i, j] = DESERT_M
                elif c == FOREST: res[i, j] = FLOWER_FOREST
                elif c == TAIGA: res[i, j] = TAIGA_M
                elif c == SWAMPLAND: res[i, j] = SWAMPLAND_M
                elif c == EXTREME_HILLS: res[i, j] = EXTREME_HILLS_M
                else: res[i, j] = c + 128 if c + 128 < 256 else c
            else: res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_shore(x, z, w, h, parent_data, base_seed):
    layer_seed = 1000
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px-1, pz]; n2 = parent_data[px+1, pz]
            n3 = parent_data[px, pz-1]; n4 = parent_data[px, pz+1]
            if c != OCEAN and c != DEEP_OCEAN and c != RIVER and c != SWAMPLAND and c != EXTREME_HILLS:
                if n1 == OCEAN or n2 == OCEAN or n3 == OCEAN or n4 == OCEAN:
                    if c == MUSHROOM_ISLAND: res[i, j] = MUSHROOM_ISLAND_SHORE
                    elif c == ICE_FLATS or c == COLD_TAIGA: res[i, j] = COLD_BEACH
                    else: res[i, j] = BEACHES
                else: res[i, j] = c
            else: res[i, j] = c
    return res

@njit(nogil=True, cache=True)
def layer_river_init(x, z, w, h, parent_data, base_seed):
    layer_seed = 100
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            rx = x + i; rz = z + j
            c = parent_data[i, j]
            if c == 0: res[i, j] = 0
            else:
                cs = get_chunk_seed(rx, rz, base_seed, layer_seed)
                r, cs = next_int(cs, 299999)
                res[i, j] = r + 2
    return res

@njit(nogil=True, cache=True)
def layer_river(x, z, w, h, parent_data, base_seed):
    layer_seed = 1
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]; n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]; n4 = parent_data[px, pz + 1]
            if c != 0 and n1 != 0 and n2 != 0 and n3 != 0 and n4 != 0 and c == n1 and c == n2 and c == n3 and c == n4:
                res[i, j] = -1
            else:
                res[i, j] = RIVER
    return res

@njit(nogil=True, cache=True)
def layer_smooth(x, z, w, h, parent_data, base_seed, layer_seed):
    res = np.zeros((w, h), dtype=np.int32)
    for i in range(w):
        for j in range(h):
            px = i + 1; pz = j + 1
            c = parent_data[px, pz]
            n1 = parent_data[px - 1, pz]; n2 = parent_data[px + 1, pz]
            n3 = parent_data[px, pz - 1]; n4 = parent_data[px, pz + 1]
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
            if b == OCEAN or b == DEEP_OCEAN: res[i, j] = b
            elif r == RIVER:
                if b == ICE_FLATS or b == COLD_TAIGA: res[i, j] = FROZEN_RIVER
                else: res[i, j] = RIVER
            else: res[i, j] = b
    return res

@njit(nogil=True, cache=True)
def get_biome_layer_data(bx, bz, bw, bh, base_seed):
    req_s_w, req_s_h = bw + 2, bh + 2
    req_z8_w, req_z8_h = (req_s_w >> 1) + 2, (req_s_h >> 1) + 2
    req_z7_w, req_z7_h = (req_z8_w >> 1) + 2, (req_z8_h >> 1) + 2
    req_z6_w, req_z6_h = (req_z7_w >> 1) + 2, (req_z7_h >> 1) + 2
    req_z5_w, req_z5_h = (req_z6_w >> 1) + 2, (req_z6_h >> 1) + 2
    req_z4_w, req_z4_h = (req_z5_w >> 1) + 2, (req_z5_h >> 1) + 2
    req_z3_w, req_z3_h = (req_z4_w >> 1) + 2, (req_z4_h >> 1) + 2
    req_z2_w, req_z2_h = (req_z3_w >> 1) + 2, (req_z3_h >> 1) + 2
    req_z1_w, req_z1_h = (req_z2_w >> 1) + 2, (req_z2_h >> 1) + 2
    
    req_shore_w, req_shore_h = req_z1_w + 2, req_z1_h + 2
    req_rare_w, req_rare_h = req_shore_w, req_shore_h
    req_edge_w, req_edge_h = req_rare_w + 2, req_rare_h + 2
    req_bi_w, req_bi_h = req_edge_w, req_edge_h
    
    req_do_w, req_do_h = req_bi_w + 2, req_bi_h + 2
    req_ais_w, req_ais_h = req_do_w + 2, req_do_h + 2
    req_zp_w, req_zp_h = (req_ais_w >> 1) + 2, (req_ais_h >> 1) + 2
    req_as_w, req_as_h = req_zp_w, req_zp_h
    req_ai2_w, req_ai2_h = req_as_w + 2, req_as_h + 2
    req_fz_w, req_fz_h = (req_ai2_w >> 1) + 2, (req_ai2_h >> 1) + 2
    req_is_w, req_is_h = req_fz_w, req_fz_h

    x_s = bx - 1; z_s = bz - 1
    x_z8 = x_s >> 1; z_z8 = z_s >> 1
    x_z7 = x_z8 >> 1; z_z7 = z_z8 >> 1
    x_z6 = x_z7 >> 1; z_z6 = z_z7 >> 1
    x_z5 = x_z6 >> 1; z_z5 = z_z6 >> 1
    x_z4 = x_z5 >> 1; z_z4 = z_z5 >> 1
    x_z3 = x_z4 >> 1; z_z3 = z_z4 >> 1
    x_z2 = x_z3 >> 1; z_z2 = z_z3 >> 1
    x_z1 = x_z2 >> 1; z_z1 = z_z2 >> 1
    
    x_shore = x_z1 - 1; z_shore = z_z1 - 1
    x_rare = x_shore; z_rare = z_shore
    x_edge = x_rare - 1; z_edge = z_rare - 1
    x_bi = x_edge; z_bi = z_edge
    
    x_do = x_bi - 1; z_do = z_bi - 1
    x_ais = x_do - 1; z_ais = z_do - 1
    x_zp = x_ais >> 1; z_zp = z_ais >> 1
    x_as = x_zp; z_as = z_zp
    x_ai2 = x_as - 1; z_ai2 = z_as - 1
    x_fz = x_ai2 >> 1; z_fz = z_ai2 >> 1
    x_is = x_fz; z_is = z_fz
    
    d_is = layer_island(x_is, z_is, req_is_w, req_is_h, base_seed)
    d_fz = layer_fuzzy_zoom(x_ai2, z_ai2, req_ai2_w, req_ai2_h, d_is, base_seed)
    d_ai2 = layer_add_island(x_as, z_as, req_as_w, req_as_h, d_fz, base_seed, 2)
    d_as = layer_add_snow(x_as, z_as, req_as_w, req_as_h, d_ai2, base_seed)
    d_zp = layer_zoom(x_ais, z_ais, req_ais_w, req_ais_h, d_as, base_seed, 2001)
    d_ais = layer_add_island(x_do, z_do, req_do_w, req_do_h, d_zp, base_seed, 3)
    d_do = layer_deep_ocean(x_bi, z_bi, req_bi_w, req_bi_h, d_ais, base_seed)
    
    d_bi = layer_biome_init(x_bi, z_bi, req_bi_w, req_bi_h, d_do, base_seed)
    d_edge = layer_biome_edge(x_rare, z_rare, req_rare_w, req_rare_h, d_bi, base_seed)
    d_rare = layer_rare_biome(x_z1, z_z1, req_z1_w, req_z1_h, d_edge, base_seed)
    
    d_z1 = layer_zoom(x_z2, z_z2, req_z2_w, req_z2_h, d_rare, base_seed, 1000)
    d_z2 = layer_zoom(x_z3, z_z3, req_z3_w, req_z3_h, d_z1, base_seed, 1001)
    d_z3 = layer_zoom(x_z4, z_z4, req_z4_w, req_z4_h, d_z2, base_seed, 1002)
    d_z4 = layer_zoom(x_z5, z_z5, req_z5_w, req_z5_h, d_z3, base_seed, 1003)
    d_z5 = layer_zoom(x_z6, z_z6, req_z6_w, req_z6_h, d_z4, base_seed, 1004)
    d_z6 = layer_zoom(x_z7, z_z7, req_z7_w, req_z7_h, d_z5, base_seed, 1005)
    d_z7 = layer_zoom(x_z8, z_z8, req_z8_w, req_z8_h, d_z6, base_seed, 1006)
    d_z8 = layer_zoom(x_s, z_s, req_s_w, req_s_h, d_z7, base_seed, 1007)
    
    d_shore = layer_shore(x_s, z_s, req_s_w, req_s_h, d_z8, base_seed)
    d_s = layer_smooth(bx, bz, bw, bh, d_shore, base_seed, 1000)
    
    r_ri_input = np.zeros((req_z1_w, req_z1_h), dtype=np.int32)
    for i in range(req_z1_w):
        for j in range(req_z1_h):
            r_ri_input[i, j] = d_do[i + (x_z1 - x_bi), j + (z_z1 - z_bi)]
            
    r_ri = layer_river_init(x_z1, z_z1, req_z1_w, req_z1_h, r_ri_input, base_seed)
    r_z1 = layer_zoom(x_z2, z_z2, req_z2_w, req_z2_h, r_ri, base_seed, 1000)
    r_z2 = layer_zoom(x_z3, z_z3, req_z3_w, req_z3_h, r_z1, base_seed, 1001)
    r_z3 = layer_zoom(x_z4, z_z4, req_z4_w, req_z4_h, r_z2, base_seed, 1002)
    r_z4 = layer_zoom(x_z5, z_z5, req_z5_w, req_z5_h, r_z3, base_seed, 1003)
    r_z5 = layer_zoom(x_z6, z_z6, req_z6_w, req_z6_h, r_z4, base_seed, 1004)
    r_z6 = layer_zoom(x_z7, z_z7, req_z7_w, req_z7_h, r_z5, base_seed, 1005)
    r_z7 = layer_zoom(x_z8, z_z8, req_z8_w, req_z8_h, r_z6, base_seed, 1006)
    r_z8 = layer_zoom(x_s, z_s, req_s_w, req_s_h, r_z7, base_seed, 1007)
    r_r = layer_river(x_s, z_s, req_s_w, req_s_h, r_z8, base_seed)
    r_s = layer_smooth(bx, bz, bw, bh, r_r, base_seed, 1000)
    
    return layer_river_mixer(bx, bz, bw, bh, d_s, r_s)
