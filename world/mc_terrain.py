import numpy as np
from numba import njit, prange
from world.mc_noise import JavaRandom, PerlinNoise
from world.mc_biomes import get_biome, get_biome_properties, BIOME_DATA
from world.mc_biomes import *
from world.mc_biomes_layer import get_biome_layer_data
from world.mc_trees import *
from world.mc_decorations import *
from core.world_db import load_chunk
from world.mc_caves import carve_caves, carve_canyons
from world.mc_ores import carve_ores

from world.terrain import *

CHUNK_SIZE = 16
CHUNK_HEIGHT = 256  # Minecraft 1.7 standard height
WATER_LEVEL = 63

@njit(cache=True, nogil=True)
def get_biome_grid(cx, cz, tempNoise, downNoise, seed=12345):
    return get_biome_layer_data(cx * 4 - 2, cz * 4 - 2, 10, 10, seed)

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
                
                yOffs = (yy - yCenter) * 6.0 / s_factor
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
            
            gx = (x // 4) + 2
            gz = (z // 4) + 2
            biome_id = biome_grid[gx, gz]
            _, _, top_block, filler_block = get_biome_properties(biome_id)
            
            run_depth = random.nextInt(3) + 2
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
            
            for y in range(5):
                if y == 0 or random.nextInt(5) > y:
                    blocks[x, y, z] = BEDROCK


@njit(cache=True, nogil=True)
def decorate_chunk(cx, cz, blocks, data, random, biome_grid):
    out_of_bounds = np.zeros((2000, 4), dtype=np.int32)
    out_count = 0
    center_biome = biome_grid[4, 4]
    
    waterlilyPerChunk = 0
    treesPerChunk = 0
    flowersPerChunk = 2
    grassPerChunk = 1
    deadBushPerChunk = 0
    mushroomsPerChunk = 0
    reedsPerChunk = 0
    cactiPerChunk = 0
    sandPerChunk = 1
    sandPerChunk2 = 3
    clayPerChunk = 1
    bigMushroomsPerChunk = 0
    
    b = center_biome
    if b == PLAINS or b == 129:
        treesPerChunk = 0
        flowersPerChunk = 4
        grassPerChunk = 10
    elif b == DESERT or b == DESERT_HILLS or b == 130:
        treesPerChunk = 0
        deadBushPerChunk = 2
        reedsPerChunk = 50
        cactiPerChunk = 10
    elif b == EXTREME_HILLS or b == SMALLER_EXTREME_HILLS or b == 131:
        treesPerChunk = 1
        flowersPerChunk = 2
        grassPerChunk = 1
    elif b == FOREST or b == FOREST_HILLS or b == 132:
        treesPerChunk = 10
        grassPerChunk = 2
        if b == 132: flowersPerChunk = 100
    elif b == TAIGA or b == TAIGA_HILLS or b == 133:
        treesPerChunk = 10
        grassPerChunk = 7
        deadBushPerChunk = 1
    elif b == SWAMPLAND or b == 134:
        treesPerChunk = 2
        flowersPerChunk = 1
        deadBushPerChunk = 1
        mushroomsPerChunk = 8
        reedsPerChunk = 10
        clayPerChunk = 1
        waterlilyPerChunk = 4
        sandPerChunk = 0
        sandPerChunk2 = 0
    elif b == RIVER or b == FROZEN_RIVER:
        treesPerChunk = 0
        flowersPerChunk = 0
        grassPerChunk = 1
    elif b == ICE_FLATS or b == COLD_TAIGA:
        treesPerChunk = 0 if b == ICE_FLATS else 10
        flowersPerChunk = 0
        grassPerChunk = 1
    elif b == MUSHROOM_ISLAND:
        treesPerChunk = 0
        flowersPerChunk = 0
        grassPerChunk = 0
        mushroomsPerChunk = 1
        bigMushroomsPerChunk = 1
    elif b == BEACHES:
        treesPerChunk = 0
        flowersPerChunk = 0
        grassPerChunk = 0
    elif b == JUNGLE or b == JUNGLE_HILLS or b == JUNGLE_EDGE:
        treesPerChunk = 50
        grassPerChunk = 25
        flowersPerChunk = 4
        reedsPerChunk = 50
    elif b == SAVANNA or b == 163:
        treesPerChunk = 1
        flowersPerChunk = 4
        grassPerChunk = 20
    elif b == ROOFED_FOREST or b == 157:
        treesPerChunk = 50
        grassPerChunk = 2
    elif b == MEGA_TAIGA or b == MEGA_TAIGA_HILLS:
        treesPerChunk = 10
        grassPerChunk = 16
        deadBushPerChunk = 0
        mushroomsPerChunk = 1
        
    for i in range(sandPerChunk2):
        x = random.nextInt(16); z = random.nextInt(16); wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            if blocks[x, y, z] == WATER: wy = y; break
        if wy != -1: generate_sand_disk(blocks, data, x, wy, z, random, SAND, 7)
            
    for i in range(clayPerChunk):
        x = random.nextInt(16); z = random.nextInt(16); wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            if blocks[x, y, z] == WATER: wy = y; break
        if wy != -1: generate_sand_disk(blocks, data, x, wy, z, random, CLAY_BLOCK, 4)
            
    for i in range(sandPerChunk):
        x = random.nextInt(16); z = random.nextInt(16); wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            if blocks[x, y, z] == WATER: wy = y; break
        if wy != -1: generate_sand_disk(blocks, data, x, wy, z, random, GRAVEL, 6)

    forests = treesPerChunk
    if random.nextInt(10) == 0: forests += 1
        
    for i in range(forests):
        x = random.nextInt(16); z = random.nextInt(16)
        b_id = biome_grid[(x // 4) + 2, (z // 4) + 2]
        if b_id == DESERT or b_id == DESERT_HILLS or b_id == OCEAN or b_id == RIVER or b_id == ICE_FLATS or b_id == BEACHES: continue
        wx = cx * CHUNK_SIZE + x; wz = cz * CHUNK_SIZE + z
        wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            bl = blocks[x, y, z]
            if bl == GRASS or bl == SAND or bl == DIRT or bl == SNOW_LAYER or bl == PODZOL or bl == MYCELIUM:
                wy = y + 1; break
                
        if wy != -1 and wy < CHUNK_HEIGHT - 15:
            if b_id == FOREST or b_id == FOREST_HILLS:
                if random.nextInt(5) == 0: out_count = generate_birch_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                else: out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == BIRCH_FOREST or b_id == BIRCH_FOREST_HILLS:
                out_count = generate_birch_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == TAIGA or b_id == TAIGA_HILLS:
                out_count = generate_spruce_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == MEGA_TAIGA or b_id == MEGA_TAIGA_HILLS:
                if random.nextInt(10) == 0: out_count = generate_mega_pine_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                else: out_count = generate_spruce_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, pine=True)
            elif b_id == SWAMPLAND or b_id == 134:
                out_count = generate_swamp_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == JUNGLE or b_id == JUNGLE_HILLS or b_id == JUNGLE_EDGE:
                if random.nextInt(10) == 0: out_count = generate_mega_jungle_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                elif random.nextInt(2) == 0: out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, add_vines=True)
                else: out_count = generate_jungle_bush(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == SAVANNA or b_id == 163:
                if random.nextInt(5) == 0: out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                else: out_count = generate_savanna_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            elif b_id == ROOFED_FOREST or b_id == 157:
                if random.nextInt(5) == 0: out_count = generate_roof_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                else: out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
            else:
                out_count = generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count)
                
    for i in range(bigMushroomsPerChunk):
        x = random.nextInt(16); z = random.nextInt(16); wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            bl = blocks[x, y, z]
            if bl == GRASS or bl == MYCELIUM or bl == DIRT: wy = y + 1; break
        if wy != -1 and wy < CHUNK_HEIGHT - 10:
            out_count = generate_huge_mushroom(cx, cz, cx * CHUNK_SIZE + x, wy, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count, random.nextInt(2) == 0)

    for i in range(grassPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        out_count = generate_grass_cluster(cx, cz, cx * CHUNK_SIZE + x, y, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count)

    for i in range(flowersPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        f_type = 37 if random.nextInt(3) == 0 else 38
        out_count = generate_flower_cluster(cx, cz, cx * CHUNK_SIZE + x, y, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count, f_type)
        
    for i in range(deadBushPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        generate_dead_bush(blocks, data, x, y, z, random)
        
    for i in range(waterlilyPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        generate_waterlily(blocks, data, x, y, z, random)
        
    for i in range(mushroomsPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        generate_mushroom(blocks, data, x, y, z, random, MUSHROOM_BROWN if random.nextInt(2)==0 else MUSHROOM_RED)

    for i in range(reedsPerChunk):
        x = random.nextInt(16); y = random.nextInt(128); z = random.nextInt(16)
        generate_reeds(blocks, data, x, y, z, random)
        
    for i in range(cactiPerChunk):
        x = random.nextInt(16); z = random.nextInt(16); wy = -1
        for y in range(CHUNK_HEIGHT - 1, WATER_LEVEL - 2, -1):
            if blocks[x, y, z] == SAND: wy = y + 1; break
        if wy != -1: out_count = generate_cactus(cx, cz, cx * CHUNK_SIZE + x, wy, cz * CHUNK_SIZE + z, blocks, random, out_of_bounds, out_count)
        
    if random.nextInt(32) == 0: generate_pumpkin(blocks, data, random.nextInt(16), 64, random.nextInt(16), random)
    if center_biome == JUNGLE and random.nextInt(32) == 0: generate_melon(blocks, data, random.nextInt(16), 64, random.nextInt(16), random)

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
_WORLD_SEED = 1357
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
    biome_grid = get_biome_grid(cx, cz, _TEMP, _DOWN, _WORLD_SEED)
    
    # Generate height buffer
    buffer = get_noise_buffer(cx, cz, _LPERLIN1, _LPERLIN2, _PERLIN1, _DEPTH, biome_grid)
    
    # Prepare heights (3D interpolation)
    blocks = prepare_heights(cx, cz, buffer)
    
    # Build surfaces (top grass, filler dirt, bedrock)
    build_surfaces(cx, cz, blocks, random, biome_grid)
    
    # Generate caves and ores in proper order (caves -> canyons -> ores)
    carve_caves(cx, cz, blocks, _WORLD_SEED, biome_grid)
    carve_canyons(cx, cz, blocks, _WORLD_SEED, biome_grid)
    carve_ores(cx, cz, blocks, _WORLD_SEED)
    
    data = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    
    # Generate trees and decorations based on biome
    out_of_bounds = decorate_chunk(cx, cz, blocks, data, random, biome_grid)
    
    # Lighting
    light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
    _calc_light_jit(blocks, light_map)
    chunk_biomes = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
    for lx in range(CHUNK_SIZE):
        for lz in range(CHUNK_SIZE):
            chunk_biomes[lx, lz] = biome_grid[(lx//4)+2, (lz//4)+2]
            
    return blocks, data, light_map, out_of_bounds, chunk_biomes

def load_or_generate_chunk(cx, cz):
    chunk_data = load_chunk(cx, cz)
    if chunk_data is not None:
        biome_grid = get_biome_grid(cx, cz, _TEMP, _DOWN, _WORLD_SEED)
        chunk_biomes = np.zeros((CHUNK_SIZE, CHUNK_SIZE), dtype=np.int32)
        for lx in range(CHUNK_SIZE):
            for lz in range(CHUNK_SIZE):
                chunk_biomes[lx, lz] = biome_grid[(lx//4)+2, (lz//4)+2]
        # chunk_data is (blocks, data, lights)
        return chunk_data[0], chunk_data[1], chunk_data[2], np.zeros((0, 4), dtype=np.int32), chunk_biomes, False
    blocks, data, lights, oob, chunk_biomes = generate_chunk(cx, cz)
    return blocks, data, lights, oob, chunk_biomes, True

def recalculate_chunk_light(blocks, light_map):
    # Call JIT to recalculate light
    _calc_light_jit(blocks, light_map)
