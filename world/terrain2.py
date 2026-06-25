"""
PythonCraft - World Terrain Generator (Numba JIT Optimized)
Tüm noise hesaplamaları ve döngüler Numba ile makine koduna derlenir.
İlk çalıştırmada ~2sn derleme süresi olur, sonraki çağrılar C hızında çalışır.
"""
import numpy as np
from numba import njit, prange
from core.world_db import load_chunk

# Blok Kayıt Sistemi (Block Registry)
BLOCK_REGISTRY = {
}

# Dinamik yerine statik tanımlama (Flake8 ve Numba için daha güvenli)
AIR = 0
STONE = 1
DIRT = 2
GRASS = 3
WATER = 4
SAND = 5
SNOW = 6
COBBLESTONE = 43
PLANKS_OAK = 44
PLANKS_SPRUCE = 45
PLANKS_BIRCH = 46
PLANKS_JUNGLE = 47
PLANKS_ACACIA = 48
PLANKS_DARK_OAK = 49
SPONGE = 50
LAPIS_BLOCK = 51
GOLD_BLOCK = 52
IRON_BLOCK = 53
BRICKS = 54
TNT = 55
OAK_STAIRS = 200
COBBLESTONE_STAIRS = 201
WOODEN_SLAB = 202
STONE_SLAB = 203
SPRUCE_STAIRS = 204
BIRCH_STAIRS = 205
JUNGLE_STAIRS = 206
ACACIA_STAIRS = 207
DARK_OAK_STAIRS = 208
BRICK_STAIRS = 209
STONE_BRICK_STAIRS = 210
NETHER_BRICK_STAIRS = 211
SANDSTONE_STAIRS = 212
QUARTZ_STAIRS = 213
SPRUCE_SLAB = 214
BIRCH_SLAB = 215
JUNGLE_SLAB = 216
ACACIA_SLAB = 217
DARK_OAK_SLAB = 218
BRICK_SLAB = 219
STONE_BRICK_SLAB = 220
NETHER_BRICK_SLAB = 221
SANDSTONE_SLAB = 222
QUARTZ_SLAB = 223
COBBLESTONE_SLAB = 224
WOOD_DOOR = 162
IRON_DOOR = 163
BOOKSHELF = 57
MOSSY_COBBLESTONE = 58
OBSIDIAN = 59
DIAMOND_BLOCK = 60
CLAY_BLOCK = 61
JUKEBOX = 62
NETHERRACK = 63
SOUL_SAND = 64
GLOWSTONE = 65
STONEBRICK = 66
STONEBRICK_MOSSY = 67
STONEBRICK_CRACKED = 68
STONEBRICK_CARVED = 69
MELON_BLOCK = 70
NETHER_BRICK = 71
END_STONE = 72
EMERALD_BLOCK_SOLID = 74
REDSTONE_BLOCK = 75
QUARTZ_BLOCK = 76
QUARTZ_CHISELED = 77
QUARTZ_PILLAR = 78
COAL_BLOCK_SOLID = 80
PACKED_ICE = 81
RED_SAND = 82
PODZOL = 83
COARSE_DIRT = 84
ANDESITE = 85
ANDESITE_POLISHED = 86
DIORITE = 87
DIORITE_POLISHED = 88
GRANITE = 89
GRANITE_POLISHED = 90
PRISMARINE = 91
PRISMARINE_BRICKS = 92
PRISMARINE_DARK = 93
SEA_LANTERN = 94
SLIME_BLOCK = 95
PORKCHOP_RAW = 1000

# Items
STICK = 1001
DIAMOND = 1002
IRON_INGOT = 1003
GOLD_INGOT = 1004
COAL = 1005
CHARCOAL = 1006
BOWL = 1007
MUSHROOM_STEW = 1008
WOODEN_SWORD = 1009
WOODEN_SHOVEL = 1010
WOODEN_PICKAXE = 1011
WOODEN_AXE = 1012
WOODEN_HOE = 1013
STONE_SWORD = 1014
STONE_SHOVEL = 1015
STONE_PICKAXE = 1016
STONE_AXE = 1017
STONE_HOE = 1018
IRON_SWORD = 1019
IRON_SHOVEL = 1020
IRON_PICKAXE = 1021
IRON_AXE = 1022
IRON_HOE = 1023
DIAMOND_SWORD = 1024
DIAMOND_SHOVEL = 1025
DIAMOND_PICKAXE = 1026
DIAMOND_AXE = 1027
DIAMOND_HOE = 1028
GOLD_SWORD = 1029
GOLD_SHOVEL = 1030
GOLD_PICKAXE = 1031
GOLD_AXE = 1032
GOLD_HOE = 1033
HAY_BLOCK = 96
WOOL_WHITE = 100
GLASS_WHITE = 120
STAINED_CLAY_WHITE = 140
WOOL_ORANGE = 101
GLASS_ORANGE = 121
STAINED_CLAY_ORANGE = 141
WOOL_MAGENTA = 102
GLASS_MAGENTA = 122
STAINED_CLAY_MAGENTA = 142
WOOL_LIGHT_BLUE = 103
GLASS_LIGHT_BLUE = 123
STAINED_CLAY_LIGHT_BLUE = 143
WOOL_YELLOW = 104
GLASS_YELLOW = 124
STAINED_CLAY_YELLOW = 144
WOOL_LIME = 105
GLASS_LIME = 125
STAINED_CLAY_LIME = 145
WOOL_PINK = 106
GLASS_PINK = 126
STAINED_CLAY_PINK = 146
WOOL_GRAY = 107
GLASS_GRAY = 127
STAINED_CLAY_GRAY = 147
WOOL_SILVER = 108
GLASS_SILVER = 128
STAINED_CLAY_SILVER = 148
WOOL_CYAN = 109
GLASS_CYAN = 225
STAINED_CLAY_CYAN = 149
WOOL_PURPLE = 110
GLASS_PURPLE = 130
STAINED_CLAY_PURPLE = 150
WOOL_BLUE = 111
GLASS_BLUE = 131
STAINED_CLAY_BLUE = 151
WOOL_BROWN = 112
GLASS_BROWN = 132
STAINED_CLAY_BROWN = 152
WOOL_GREEN = 113
GLASS_GREEN = 133
STAINED_CLAY_GREEN = 153
WOOL_RED = 114
GLASS_RED = 134
STAINED_CLAY_RED = 154
WOOL_BLACK = 115
GLASS_BLACK = 135
STAINED_CLAY_BLACK = 155
CRAFTING_TABLE = 116
HARDENED_CLAY = 160
BEDROCK = 7
ICE = 79
GRAVEL = 8
SANDSTONE = 9
MYCELIUM = 10
WOOD = 11
LEAVES = 12
CACTUS = 13
BIRCH_WOOD = 14
SPRUCE_WOOD = 15
BIRCH_LEAVES = 16
SPRUCE_LEAVES = 17
GLASS = 20
LAVA = 22

DEADBUSH = 32
MUSHROOM_BROWN = 39
MUSHROOM_RED = 230
VINE = 231
REEDS = 232
WATERLILY = 233
PUMPKIN = 234
SNOW_LAYER = 235
JUNGLE_WOOD = 236
ACACIA_WOOD = 237
DARK_OAK_WOOD = 238
JUNGLE_LEAVES = 239
ACACIA_LEAVES = 240
DARK_OAK_LEAVES = 241

GOLD_ORE = 40
IRON_ORE = 41
COAL_ORE = 42
LAPIS_ORE = 21
DIAMOND_ORE = 56
REDSTONE_ORE = 73
EMERALD_ORE = 129

TALLGRASS = 31
DANDELION = 37
ROSE = 38
DOUBLE_GRASS_BTM = 175
DOUBLE_GRASS_TOP = 176
DOUBLE_ROSE_BTM = 177
DOUBLE_ROSE_TOP = 178
PORKCHOP_RAW = 1000

# Numba için hızlı erişim dizileri (0-2048 ID'ler için)
BLOCK_OPAQUE_ARRAY = np.ones(2048, dtype=np.bool_)
BLOCK_LIGHT_EMISSION_ARRAY = np.zeros(2048, dtype=np.uint8)
BLOCK_COLORS_ARRAY = np.ones((2048, 3), dtype=np.float32)
BLOCK_HARDNESS_ARRAY = np.zeros(2048, dtype=np.float32)

# Tool Types mapping to integer for Numba:
# 0: NONE, 1: PICKAXE, 2: AXE, 3: SHOVEL
TOOL_TYPE_MAP = {"NONE": 0, "PICKAXE": 1, "AXE": 2, "SHOVEL": 3}
BLOCK_TOOL_ARRAY = np.zeros(2048, dtype=np.uint8)
BLOCK_MAX_STACK_ARRAY = np.full(2048, 64, dtype=np.uint8)

for name, data in BLOCK_REGISTRY.items():
    bid = data["id"]
    if bid < 2048:
        BLOCK_OPAQUE_ARRAY[bid] = not data["transparent"]
        BLOCK_LIGHT_EMISSION_ARRAY[bid] = data["light"]
        BLOCK_COLORS_ARRAY[bid] = data["color"]
        BLOCK_HARDNESS_ARRAY[bid] = data["hardness"]
        BLOCK_TOOL_ARRAY[bid] = TOOL_TYPE_MAP.get(data.get("tool_type", "NONE"), 0)
        
        if bid >= 1009 and bid <= 1033: # Tools
            BLOCK_MAX_STACK_ARRAY[bid] = 1
        else:
            BLOCK_MAX_STACK_ARRAY[bid] = data.get("max_stack", 64)

CHUNK_SIZE = 16
CHUNK_HEIGHT = 256
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
                if BLOCK_OPAQUE_ARRAY[b]:
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
                if not BLOCK_OPAQUE_ARRAY[blocks[nx, ny, nz]]:
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
        # data is (blocks, lights)
        return data[0], data[1], np.zeros((0, 4), dtype=np.int32), np.full((16, 16), 1, dtype=np.int32), False
    blocks, lights, oob, chunk_biomes = generate_chunk(cx, cz)
    return blocks, lights, oob, chunk_biomes, True

def recalculate_chunk_light(blocks, light_map):
    _calc_light_jit(blocks, light_map)

def get_break_time(block_id, held_id):
    hardness = BLOCK_HARDNESS_ARRAY[block_id]
    if hardness < 0:
        return 999999.0
        
    base_time = hardness * 5.0
    
    if held_id == 0 or held_id < 1000:
        return base_time
        
    block_tool_type = BLOCK_TOOL_ARRAY[block_id]
    if block_tool_type == 0:
        return base_time
        
    # Determine held tool type and multiplier
    held_tool_type = 0
    multiplier = 1.0
    
    # 1: PICKAXE, 2: AXE, 3: SHOVEL
    if held_id in (1011, 1016, 1021, 1026, 1031):
        held_tool_type = 1
    elif held_id in (1012, 1017, 1022, 1027, 1032):
        held_tool_type = 2
    elif held_id in (1010, 1015, 1020, 1025, 1030):
        held_tool_type = 3
        
    if held_tool_type > 0 and held_tool_type == block_tool_type:
        if held_id in (1009, 1010, 1011, 1012, 1013): # Wood
            multiplier = 2.0
        elif held_id in (1014, 1015, 1016, 1017, 1018): # Stone
            multiplier = 4.0
        elif held_id in (1019, 1020, 1021, 1022, 1023): # Iron
            multiplier = 6.0
        elif held_id in (1024, 1025, 1026, 1027, 1028): # Diamond
            multiplier = 8.0
        elif held_id in (1029, 1030, 1031, 1032, 1033): # Gold
            multiplier = 12.0
            
        return base_time / multiplier
        
    return base_time
