import numpy as np
from numba import njit
import math
from world.mc_noise import JavaRandom
from world.terrain import CHUNK_SIZE, CHUNK_HEIGHT, STONE, DIRT, GRAVEL, COAL_ORE, IRON_ORE, GOLD_ORE, REDSTONE_ORE, DIAMOND_ORE, LAPIS_ORE

@njit(cache=True, nogil=True)
def _ore_feature(blocks, random, tile, count, x, y, z):
    dir_ = random.nextFloat() * math.pi
    
    x0 = x + 8.0 + math.sin(dir_) * count / 8.0
    x1 = x + 8.0 - math.sin(dir_) * count / 8.0
    z0 = z + 8.0 + math.cos(dir_) * count / 8.0
    z1 = z + 8.0 - math.cos(dir_) * count / 8.0
    
    y0 = y + random.nextInt(3) - 2.0
    y1 = y + random.nextInt(3) - 2.0
    
    for d in range(count + 1):
        xx = x0 + (x1 - x0) * d / count
        yy = y0 + (y1 - y0) * d / count
        zz = z0 + (z1 - z0) * d / count
        
        ss = random.nextDouble() * count / 16.0
        r = (math.sin(d * math.pi / count) + 1.0) * ss + 1.0
        hr = (math.sin(d * math.pi / count) + 1.0) * ss + 1.0
        
        xt0 = int(math.floor(xx - r / 2.0))
        yt0 = int(math.floor(yy - hr / 2.0))
        zt0 = int(math.floor(zz - r / 2.0))
        
        xt1 = int(math.floor(xx + r / 2.0))
        yt1 = int(math.floor(yy + hr / 2.0))
        zt1 = int(math.floor(zz + r / 2.0))
        
        # Clamp to chunk bounds
        xt0_c = max(0, min(15, xt0))
        xt1_c = max(0, min(15, xt1))
        yt0_c = max(0, min(CHUNK_HEIGHT - 1, yt0))
        yt1_c = max(0, min(CHUNK_HEIGHT - 1, yt1))
        zt0_c = max(0, min(15, zt0))
        zt1_c = max(0, min(15, zt1))
        
        for x2 in range(xt0_c, xt1_c + 1):
            xd = ((x2 + 0.5) - xx) / (r / 2.0)
            if xd * xd < 1.0:
                for y2 in range(yt0_c, yt1_c + 1):
                    yd = ((y2 + 0.5) - yy) / (hr / 2.0)
                    if xd * xd + yd * yd < 1.0:
                        for z2 in range(zt0_c, zt1_c + 1):
                            zd = ((z2 + 0.5) - zz) / (r / 2.0)
                            if xd * xd + yd * yd + zd * zd < 1.0:
                                if blocks[x2, y2, z2] == STONE:
                                    blocks[x2, y2, z2] = tile

@njit(cache=True, nogil=True)
def _generate_ore_type(blocks, random, tile, count, iterations, min_y, max_y):
    for i in range(iterations):
        x = random.nextInt(16)
        y = random.nextInt(max_y - min_y) + min_y
        z = random.nextInt(16)
        _ore_feature(blocks, random, tile, count, x, y, z)

@njit(cache=True, nogil=True)
def carve_ores(cx, cz, blocks, world_seed):
    random = JavaRandom(world_seed)
    xScale = random.nextLong()
    zScale = random.nextLong()
    xx = cx * xScale
    zz = cz * zScale
    chunk_seed = (xx ^ zz ^ world_seed) & 0xFFFFFFFFFFFFFFFF
    random.setSeed(chunk_seed)
    
    # Dirt pockets (20 per chunk, size 32)
    _generate_ore_type(blocks, random, DIRT, 32, 20, 0, CHUNK_HEIGHT)
    
    # Gravel pockets (10 per chunk, size 32)
    _generate_ore_type(blocks, random, GRAVEL, 32, 10, 0, CHUNK_HEIGHT)
    
    # Coal (20 per chunk, size 16, any height)
    _generate_ore_type(blocks, random, COAL_ORE, 16, 20, 0, CHUNK_HEIGHT)
    
    # Iron (20 per chunk, size 8, height < 64)
    _generate_ore_type(blocks, random, IRON_ORE, 8, 20, 0, min(64, CHUNK_HEIGHT))
    
    # Gold (2 per chunk, size 8, height < 32)
    _generate_ore_type(blocks, random, GOLD_ORE, 8, 2, 0, min(32, CHUNK_HEIGHT))
    
    # Redstone (8 per chunk, size 7, height < 16)
    _generate_ore_type(blocks, random, REDSTONE_ORE, 7, 8, 0, min(16, CHUNK_HEIGHT))
    
    # Diamond (1 per chunk, size 7, height < 16)
    _generate_ore_type(blocks, random, DIAMOND_ORE, 7, 1, 0, min(16, CHUNK_HEIGHT))
    
    # Lapis (1 per chunk, size 6, height < 32)
    _generate_ore_type(blocks, random, LAPIS_ORE, 6, 1, 0, min(32, CHUNK_HEIGHT))
