"""
PythonCraft - World Terrain Generator (Numba JIT Optimized)
Tüm noise hesaplamaları ve döngüler Numba ile makine koduna derlenir.
İlk çalıştırmada ~2sn derleme süresi olur, sonraki çağrılar C hızında çalışır.
"""
import numpy as np
from numba import njit, prange
from core.world_db import load_chunk

# Blok Türleri
AIR   = 0
STONE = 1
DIRT  = 2
GRASS = 3
WATER = 4
SAND  = 5
SNOW  = 6
ICE   = 7
GRAVEL = 8
SANDSTONE = 9
MYCELIUM = 10
WOOD  = 11
LEAVES = 12
CACTUS = 13
BIRCH_WOOD = 14
SPRUCE_WOOD = 15
BIRCH_LEAVES = 16
SPRUCE_LEAVES = 17
GLASS = 20

# Blok renkleri (R, G, B)
BLOCK_COLORS = {
    STONE: (0.45, 0.45, 0.45),
    DIRT:  (0.40, 0.26, 0.13),
    GRASS: (0.30, 0.65, 0.20),
    WATER: (0.20, 0.40, 0.75),
    SAND:  (0.85, 0.80, 0.55),
    SNOW:  (0.95, 0.95, 0.95),
    ICE:   (0.60, 0.80, 1.00),
    GRAVEL:(0.50, 0.50, 0.50),
    SANDSTONE:(0.80, 0.75, 0.50),
    MYCELIUM: (0.45, 0.35, 0.40),
    WOOD:     (0.40, 0.30, 0.15),
    LEAVES:   (0.15, 0.50, 0.15),
    CACTUS:   (0.10, 0.60, 0.20),
    BIRCH_WOOD:(0.90, 0.90, 0.85),
    SPRUCE_WOOD:(0.30, 0.20, 0.10),
    BIRCH_LEAVES:(0.25, 0.55, 0.25),
    SPRUCE_LEAVES:(0.10, 0.35, 0.15),
}

CHUNK_SIZE = 16
CHUNK_HEIGHT = 64
WATER_LEVEL = 20

# ─────────────── Numba JIT Perlin Noise ───────────────
# Saf Python ile yazılmış, Numba ile makine koduna derlenen noise.
# C kütüphanesine bağımlılık yok. Tüm döngüler CPU SIMD ile hızlanır.

# Permütasyon tablosu (sabit, Perlin'in orijinal tablosu)
_PERM = np.array([
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,
    69,142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,
    252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,
    171,168,68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,
    122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,65,25,
    63,161,1,216,80,73,209,76,132,187,208,89,18,169,200,196,135,130,116,
    188,159,86,164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,
    38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,
    42,223,183,170,213,119,248,152,2,44,154,163,70,221,153,101,155,167,
    43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,178,185,112,104,
    218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241,81,51,
    145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,176,
    115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,
    141,128,195,78,66,215,61,156,180
], dtype=np.int32)

@njit(cache=True, nogil=True)
def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

@njit(cache=True, nogil=True)
def _grad2d(h, x, y):
    h = h & 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return -x - y

@njit(cache=True, nogil=True)
def _perlin2d(x, y, perm):
    """Tek nokta için 2D Perlin Noise (Numba JIT ile makine kodu)"""
    xi = int(np.floor(x)) & 255
    yi = int(np.floor(y)) & 255
    xf = x - np.floor(x)
    yf = y - np.floor(y)
    
    u = _fade(xf)
    v = _fade(yf)
    
    aa = perm[(perm[xi] + yi) & 255]
    ab = perm[(perm[xi] + yi + 1) & 255]
    ba = perm[(perm[(xi + 1) & 255] + yi) & 255]
    bb = perm[(perm[(xi + 1) & 255] + yi + 1) & 255]
    
    x1 = _grad2d(aa, xf, yf) * (1 - u) + _grad2d(ba, xf - 1, yf) * u
    x2 = _grad2d(ab, xf, yf - 1) * (1 - u) + _grad2d(bb, xf - 1, yf - 1) * u
    
    return x1 * (1 - v) + x2 * v

@njit(cache=True, nogil=True)
def _fbm2d(x, y, perm, octaves, persistence):
    """Fractal Brownian Motion — gerçekçi dağlar için katmanlı noise"""
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    max_val = 0.0
    
    for _ in range(octaves):
        total += _perlin2d(x * frequency, y * frequency, perm) * amplitude
        max_val += amplitude
        amplitude *= persistence
        frequency *= 2.0
    
    return total / max_val

# ─────────────── Numba JIT Chunk Üretici ───────────────

@njit(cache=True, nogil=True)
def _generate_chunk_data(cx, cz, perm):
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    # 1. Arazi Üretimi
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = cx * CHUNK_SIZE + lx
            wz = cz * CHUNK_SIZE + lz
            
            h1 = _fbm2d(wx * 0.005, wz * 0.005, perm, 6, 0.5) * 32.0
            h2 = _fbm2d(wx * 0.02,  wz * 0.02,  perm, 3, 0.4) * 8.0
            height = int(28 + h1 + h2)
            if height < 1: height = 1
            if height >= CHUNK_HEIGHT: height = CHUNK_HEIGHT - 1
            
            for y in range(height):
                if y < height - 4:
                    blocks[lx, y, lz] = STONE
                elif y < height - 1:
                    blocks[lx, y, lz] = DIRT
                else:
                    if y <= WATER_LEVEL + 1:
                        blocks[lx, y, lz] = SAND
                    else:
                        blocks[lx, y, lz] = GRASS
            
            for y in range(height, WATER_LEVEL):
                if blocks[lx, y, lz] == AIR:
                    blocks[lx, y, lz] = WATER

@njit(cache=True, nogil=True)
def _calc_light_jit(blocks, light_map):
    # 1. Dikey Güneş Işığı (SkyLight)
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            light = 15
            for y in range(CHUNK_HEIGHT - 1, -1, -1):
                b = blocks[x, y, z]
                if b != AIR and b != WATER:
                    light = 0
                light_map[x, y, z] = light

    # 2. Yatay Işık Yayılımı (BFS)
    queue_x = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    queue_y = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    queue_z = np.empty(CHUNK_SIZE * CHUNK_HEIGHT * CHUNK_SIZE, dtype=np.int32)
    head = 0
    tail = 0
    
    for x in range(CHUNK_SIZE):
        for y in range(CHUNK_HEIGHT):
            for z in range(CHUNK_SIZE):
                if light_map[x, y, z] == 15:
                    queue_x[tail] = x
                    queue_y[tail] = y
                    queue_z[tail] = z
                    tail += 1
                    
    dxs = np.array([1, -1, 0, 0, 0, 0], dtype=np.int32)
    dys = np.array([0, 0, 1, -1, 0, 0], dtype=np.int32)
    dzs = np.array([0, 0, 0, 0, 1, -1], dtype=np.int32)
    
    while head < tail:
        x = queue_x[head]
        y = queue_y[head]
        z = queue_z[head]
        head += 1
        
        L = light_map[x, y, z]
        if L <= 1: continue
        
        new_light = L - 1
        for i in range(6):
            nx = x + dxs[i]
            ny = y + dys[i]
            nz = z + dzs[i]
            
            if 0 <= nx < CHUNK_SIZE and 0 <= ny < CHUNK_HEIGHT and 0 <= nz < CHUNK_SIZE:
                b = blocks[nx, ny, nz]
                if b == AIR or b == WATER:
                    if light_map[nx, ny, nz] < new_light:
                        light_map[nx, ny, nz] = new_light
                        queue_x[tail] = nx
                        queue_y[tail] = ny
                        queue_z[tail] = nz
                        tail += 1
                        
@njit(cache=True, nogil=True)
def _generate_chunk_data(cx, cz, perm):
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    # 1. Arazi Üretimi
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            wx = cx * CHUNK_SIZE + lx
            wz = cz * CHUNK_SIZE + lz
            
            h1 = _fbm2d(wx * 0.005, wz * 0.005, perm, 6, 0.5) * 32.0
            h2 = _fbm2d(wx * 0.02,  wz * 0.02,  perm, 3, 0.4) * 8.0
            height = int(28 + h1 + h2)
            if height < 1: height = 1
            if height >= CHUNK_HEIGHT: height = CHUNK_HEIGHT - 1
            
            for y in range(height):
                if y < height - 4:
                    blocks[lx, y, lz] = STONE
                elif y < height - 1:
                    blocks[lx, y, lz] = DIRT
                else:
                    if y <= WATER_LEVEL + 1:
                        blocks[lx, y, lz] = SAND
                    else:
                        blocks[lx, y, lz] = GRASS
            
            for y in range(height, WATER_LEVEL):
                if blocks[lx, y, lz] == AIR:
                    blocks[lx, y, lz] = WATER

    _calc_light_jit(blocks, light_map)
    return blocks, light_map

_PERM_DOUBLE = np.concatenate([_PERM, _PERM])

def generate_chunk(cx, cz):
    return _generate_chunk_data(cx, cz, _PERM_DOUBLE)

def load_or_generate_chunk(cx, cz):
    data = load_chunk(cx, cz)
    if data is not None:
        return data
    return generate_chunk(cx, cz)

def recalculate_chunk_light(blocks, light_map):
    _calc_light_jit(blocks, light_map)
