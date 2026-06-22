import numpy as np
from numba import njit, prange
from world.mc_noise import JavaRandom, PerlinNoise
from world.mc_biomes import get_biome, get_biome_properties, BIOME_DATA
from world.mc_biomes import PLAINS, DESERT, FOREST, TAIGA, EXTREME_HILLS, JUNGLE, SWAMPLAND, FOREST_HILLS, TAIGA_HILLS, DESERT_HILLS, JUNGLE_HILLS, SMALLER_EXTREME_HILLS, OCEAN, RIVER, ICE_FLATS, BEACHES
from world.mc_biomes_layer import get_biome_layer_data
from world.mc_trees import generate_oak_tree, generate_birch_tree, generate_spruce_tree, generate_cactus, generate_grass_cluster, generate_flower_cluster, generate_double_plant_cluster
from core.world_db import load_chunk
from world.mc_caves import carve_caves, carve_canyons
from world.mc_ores import carve_ores

from world.terrain import *

CHUNK_SIZE = 16
CHUNK_HEIGHT = 256  # Minecraft 1.7 standard height
WATER_LEVEL = 63

@njit(cache=True, nogil=True)
def get_biome_grid(cx, cz, tempNoise, downNoise):
    # base_seed = 12345 (could be passed from world_seed later)
    return get_biome_layer_data(cx * 4 - 2, cz * 4 - 2, 10, 10, 12345)

@njit(cache=True, nogil=True)
def get_noise_buffer(cx, cz, lperlin1, lperlin2, perlin1, depthNoise, biome_grid):
    xSize = 5
    ySize = (CHUNK_HEIGHT // 8) + 1 # 17
    zSize = 5

    buffer = np.zeros(xSize * ySize * zSize, dtype=np.float64)
    
    s = 684.412
    hs = 684.412
    
    p = 0
    for xx in range(xSize):
        for zz in range(zSize):
            wx = cx * 4 + xx
            wz = cz * 4 + zz
            
            sss = 0.0
            ddd = 0.0
            weight_sum = 0.0
            
            center_biome = biome_grid[xx + 2, zz + 2]
            center_depth, center_scale, _, _ = get_biome_properties(center_biome)
            
            for xb in range(-2, 3):
                for zb in range(-2, 3):
                    b_id = biome_grid[xx + xb + 2, zz + zb + 2]
                    b_depth, b_scale, _, _ = get_biome_properties(b_id)
                    
                    dist = xb*xb + zb*zb
                    weight = 10.0 / np.sqrt(dist + 0.2) / (b_depth + 2.0)
                    
                    if b_depth > center_depth:
                        weight /= 2.0
                        
                    sss += b_scale * weight
                    ddd += b_depth * weight
                    weight_sum += weight
                    
            sss /= weight_sum
            ddd /= weight_sum
            
            sss = sss * 0.9 + 0.1
            ddd = (ddd * 4.0 - 1.0) / 8.0
            
            rdepth = (depthNoise.getValue2D(wx, wz) / 8000.0) # Dummy
            rdepth = rdepth * 3.0 - 2.0
            if rdepth < 0:
                rdepth /= 2.0
                if rdepth < -1.0: rdepth = -1.0
                rdepth /= 1.4
                rdepth /= 2.0
            else:
                if rdepth > 1.0: rdepth = 1.0
                rdepth /= 8.0
                
            for yy in range(ySize):
                wy = yy
                
                d = ddd
                s_factor = sss
                
                d += rdepth * 0.2
                base_ySize = 17.0
                d = d * base_ySize / 16.0
                yCenter = (base_ySize / 2.0) + (d * 4.0)
                
                yOffs = (yy - yCenter) * 12.0 / s_factor
                if yOffs < 0: yOffs *= 4.0
                
                bb = lperlin1.getValue(wx * s, wy * hs, wz * s) / 512.0
                cc = lperlin2.getValue(wx * s, wy * hs, wz * s) / 512.0
                v = (perlin1.getValue(wx * s / 80.0, wy * hs / 160.0, wz * s / 80.0) / 10.0 + 1.0) / 2.0
                
                if v < 0.0: val = bb
                elif v > 1.0: val = cc
                else: val = bb + (cc - bb) * v
                
                val -= yOffs
                
                if yy > ySize - 4:
                    slide = (yy - (ySize - 4)) / 3.0
                    val = val * (1.0 - slide) + (-10.0 * slide)
                    
                buffer[p] = val
                p += 1
                
    return buffer

@njit(cache=True, nogil=True)
def prepare_heights(cx, cz, buffer):
    blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    xChunks = 4
    yChunks = CHUNK_HEIGHT // 8 # 16
    zChunks = 4
    
    xSize = 5
    ySize = yChunks + 1
    zSize = 5
    
    for xc in range(xChunks):
        for zc in range(zChunks):
            for yc in range(yChunks):
                yStep = 1.0 / 8.0
                
                idx0 = ((xc + 0) * zSize + (zc + 0)) * ySize + (yc + 0)
                idx1 = ((xc + 0) * zSize + (zc + 1)) * ySize + (yc + 0)
                idx2 = ((xc + 1) * zSize + (zc + 0)) * ySize + (yc + 0)
                idx3 = ((xc + 1) * zSize + (zc + 1)) * ySize + (yc + 0)
                
                s0 = buffer[idx0]; s1 = buffer[idx1]; s2 = buffer[idx2]; s3 = buffer[idx3]
                
                s0a = (buffer[idx0 + 1] - s0) * yStep
                s1a = (buffer[idx1 + 1] - s1) * yStep
                s2a = (buffer[idx2 + 1] - s2) * yStep
                s3a = (buffer[idx3 + 1] - s3) * yStep
                
                for y in range(8):
                    xStep = 1.0 / 4.0
                    _s0 = s0
                    _s1 = s1
                    _s0a = (s2 - s0) * xStep
                    _s1a = (s3 - s1) * xStep
                    
                    for x in range(4):
                        zStep = 1.0 / 4.0
                        val = _s0
                        vala = (_s1 - _s0) * zStep
                        
                        for z in range(4):
                            bx = xc * 4 + x
                            by = yc * 8 + y
                            bz = zc * 4 + z
                            
                            if val > 0.0:
                                blocks[bx, by, bz] = STONE
                            elif by < WATER_LEVEL:
                                blocks[bx, by, bz] = WATER
                            else:
                                blocks[bx, by, bz] = AIR
                                
                            val += vala
                        _s0 += _s0a
                        _s1 += _s1a
                    s0 += s0a
                    s1 += s1a
                    s2 += s2a
                    s3 += s3a
                    
    return blocks

@njit(cache=True, nogil=True)
def build_surfaces(cx, cz, blocks, random, biome_grid):
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            wx = cx * CHUNK_SIZE + x
            wz = cz * CHUNK_SIZE + z
            
            # Map chunk local coords to 4-block grid (0-15 -> 0-3)
            # biome_grid has offset of 2, so index is x//4 + 2
            gx = (x // 4) + 2
            gz = (z // 4) + 2
            biome_id = biome_grid[gx, gz]
            _, _, top_block, filler_block = get_biome_properties(biome_id)
            
            # Use random to vary grass/dirt depth
            run_depth = random.nextInt(3) + 2 # 2 to 4 blocks of dirt/sand
            run = -1
                
            for y in range(CHUNK_HEIGHT - 1, -1, -1):
                b = blocks[x, y, z]
                
                if b == AIR:
                    run = -1
                elif b == STONE:
                    if run == -1:
                        if y >= WATER_LEVEL - 1:
                            blocks[x, y, z] = top_block
                        else:
                            blocks[x, y, z] = filler_block
                        run = run_depth
                    elif run > 0:
                        run -= 1
                        blocks[x, y, z] = filler_block
            
            # Bedrock layer (Y=0 to 4)
            for y in range(5):
                if y == 0 or random.nextInt(5) > y:
                    blocks[x, y, z] = BEDROCK


@njit(cache=True, nogil=True)
def generate_trees_for_chunk(cx, cz, blocks, random, biome_grid):
    out_of_bounds = np.zeros((2000, 4), dtype=np.int32)
    out_count = 0
    
    # Chunk center biome
    center_biome = biome_grid[4, 4]
    
    treeCount = 0
    cactusCount = 0
    grassCount = 0
    flowerCount = 0
    doublePlantCount = 0
    
    if center_biome == PLAINS:
        treeCount = 0
        flowerCount = 4
        grassCount = 10
        doublePlantCount = 1
    elif center_biome == FOREST or center_biome == FOREST_HILLS:
        treeCount = 5
        flowerCount = 2
        grassCount = 2
        doublePlantCount = 1
    elif center_biome == TAIGA or center_biome == TAIGA_HILLS:
        treeCount = 6
        grassCount = 1
    elif center_biome == EXTREME_HILLS or center_biome == SMALLER_EXTREME_HILLS:
        treeCount = 1
        grassCount = 1
    elif center_biome == SWAMPLAND:
        treeCount = 2
        flowerCount = 1
        grassCount = 5
    elif center_biome == JUNGLE or center_biome == JUNGLE_HILLS:
        treeCount = 15
        grassCount = 25
        flowerCount = 4
    elif center_biome == DESERT or center_biome == DESERT_HILLS:
        cactusCount = 10
        
    forests = treeCount
    if random.nextInt(10) == 0:
        forests += 1
        
    for i in range(forests):
        x = random.nextInt(16)
        z = random.nextInt(16)
        
        # Determine the EXACT biome at this specific block column
        b_id = biome_grid[(x // 4) + 2, (z // 4) + 2]
        
        # Filter out invalid biomes for trees
        if b_id == DESERT or b_id == OCEAN or b_id == RIVER or b_id == ICE_FLATS or b_id == BEACHES:
            continue
        
        wx = cx * CHUNK_SIZE + x
        wz = cz * CHUNK_SIZE + z
        
        wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            b = blocks[x, y, z]
            if b == GRASS or b == SAND or b == DIRT or b == SNOW:
                wy = y + 1
                break
                
        if wy != -1 and wy < CHUNK_HEIGHT - 15:
            # Generate tree based on the local block's biome (b_id), NOT the center_biome
            if b_id == FOREST or b_id == FOREST_HILLS:
                if random.nextInt(5) == 0:
                    out_count = generate_birch_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                else:
                    out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == TAIGA or b_id == TAIGA_HILLS:
                out_count = generate_spruce_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            else:
                out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                
    for i in range(cactusCount):
        x = random.nextInt(16)
        z = random.nextInt(16)
        wx = cx * CHUNK_SIZE + x
        wz = cz * CHUNK_SIZE + z
        
        wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            if blocks[x, y, z] == SAND:
                wy = y + 1
                break
                
        if wy != -1 and wy < CHUNK_HEIGHT - 5:
            out_count = generate_cactus(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)

    for i in range(grassCount):
        x = random.nextInt(16)
        y = random.nextInt(128)
        z = random.nextInt(16)
        out_count = generate_grass_cluster(cx, cz, cx * CHUNK_SIZE + x, y, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count)

    for i in range(flowerCount):
        x = random.nextInt(16)
        y = random.nextInt(128)
        z = random.nextInt(16)
        f_type = 37 if random.nextInt(3) == 0 else 38 # 1/3 Dandelion, 2/3 Rose
        out_count = generate_flower_cluster(cx, cz, cx * CHUNK_SIZE + x, y, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count, f_type)
        
    for i in range(doublePlantCount):
        x = random.nextInt(16)
        y = random.nextInt(128)
        z = random.nextInt(16)
        is_rose = (random.nextInt(3) == 0)
        btm = 177 if is_rose else 175
        top = 178 if is_rose else 176
        out_count = generate_double_plant_cluster(cx, cz, cx * CHUNK_SIZE + x, y, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count, btm, top)

    return out_of_bounds[:out_count]

@njit(cache=True, nogil=True)
def _calc_light_jit(blocks, light_map):
    for x in range(CHUNK_SIZE):
        for z in range(CHUNK_SIZE):
            light = 15
            for y in range(CHUNK_HEIGHT - 1, -1, -1):
                b = blocks[x, y, z]
                if b == 22: # LAVA emits light
                    light = 15
                elif b < 2048 and BLOCK_OPAQUE_ARRAY[b]:
                    light = 0
                light_map[x, y, z] = light

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
                if b < 2048 and not BLOCK_OPAQUE_ARRAY[b]:
                    if light_map[nx, ny, nz] < new_light:
                        light_map[nx, ny, nz] = new_light
                        queue_x[tail] = nx
                        queue_y[tail] = ny
                        queue_z[tail] = nz
                        tail += 1

# Instantiate singletons for world generation
_WORLD_SEED = 12345
_GLOBAL_RANDOM = JavaRandom(_WORLD_SEED)
_LPERLIN1 = PerlinNoise(JavaRandom(_WORLD_SEED), 16)
_LPERLIN2 = PerlinNoise(JavaRandom(_WORLD_SEED), 16)
_PERLIN1 = PerlinNoise(JavaRandom(_WORLD_SEED), 8)
_DEPTH = PerlinNoise(JavaRandom(_WORLD_SEED), 16)
_TEMP = PerlinNoise(JavaRandom(_WORLD_SEED * 2), 4)
_DOWN = PerlinNoise(JavaRandom(_WORLD_SEED * 3), 4)

def generate_chunk(cx, cz):
    # Use chunk specific random
    random = JavaRandom(cx * 341873128712 + cz * 132897987541)
    
    # Generate biome grid
    biome_grid = get_biome_grid(cx, cz, _TEMP, _DOWN)
    
    # Generate height buffer
    buffer = get_noise_buffer(cx, cz, _LPERLIN1, _LPERLIN2, _PERLIN1, _DEPTH, biome_grid)
    
    # Prepare heights (3D interpolation)
    blocks = prepare_heights(cx, cz, buffer)
    
    # Build surfaces (top grass, filler dirt, bedrock)
    build_surfaces(cx, cz, blocks, random, biome_grid)
    
    # Generate ores
    carve_ores(cx, cz, blocks, _WORLD_SEED)
    
    # Carve caves and canyons
    carve_caves(cx, cz, blocks, _WORLD_SEED, biome_grid)
    carve_canyons(cx, cz, blocks, _WORLD_SEED, biome_grid)
    
    # Generate trees and decorations
    out_of_bounds = generate_trees_for_chunk(cx, cz, blocks, random, biome_grid)
    
    # Lighting
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    _calc_light_jit(blocks, light_map)
    chunk_biomes = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            chunk_biomes[lx, lz] = biome_grid[(lx//4)+2, (lz//4)+2]
            
    # Data metadata array
    data = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
            
    return blocks, data, light_map, out_of_bounds, chunk_biomes

def load_or_generate_chunk(cx, cz):
    chunk_data = load_chunk(cx, cz)
    if chunk_data is not None:
        biome_grid = get_biome_grid(cx, cz, _TEMP, _DOWN)
        chunk_biomes = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
        for lx in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):
                chunk_biomes[lx, lz] = biome_grid[(lx//4)+2, (lz//4)+2]
        # chunk_data is (blocks, data, lights)
        return chunk_data[0], chunk_data[1], chunk_data[2], np.zeros((0, 4), dtype=np.int32), chunk_biomes, False
    blocks, data, lights, oob, chunk_biomes = generate_chunk(cx, cz)
    return blocks, data, lights, oob, chunk_biomes, True

def recalculate_chunk_light(blocks, light_map):
    _calc_light_jit(blocks, light_map)
