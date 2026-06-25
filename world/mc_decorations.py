import numpy as np
from numba import njit
from world.terrain import *

@njit(cache=True, nogil=True)
def is_empty(block_id):
    return block_id == AIR or block_id == SNOW_LAYER or block_id == TALLGRASS or block_id == DEADBUSH or block_id == VINE

@njit(cache=True, nogil=True)
def generate_reeds(chunk_blocks, chunk_data, start_x, start_y, start_z, random):
    for _ in range(20):
        x = start_x + random.nextInt(4) - random.nextInt(4)
        z = start_z + random.nextInt(4) - random.nextInt(4)
        y = start_y
        
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT - 4:
            continue
            
        if not is_empty(chunk_blocks[x, y, z]):
            continue
            
        has_water = False
        if x > 0 and chunk_blocks[x-1, y-1, z] == WATER: has_water = True
        if x < CHUNK_SIZE-1 and chunk_blocks[x+1, y-1, z] == WATER: has_water = True
        if z > 0 and chunk_blocks[x, y-1, z-1] == WATER: has_water = True
        if z < CHUNK_SIZE-1 and chunk_blocks[x, y-1, z+1] == WATER: has_water = True
        
        if has_water:
            below = chunk_blocks[x, y-1, z]
            if below == GRASS or below == DIRT or below == SAND or below == PODZOL or below == COARSE_DIRT:
                height = 2 + random.nextInt(random.nextInt(3) + 1)
                for i in range(height):
                    if is_empty(chunk_blocks[x, y+i, z]):
                        chunk_blocks[x, y+i, z] = REEDS

@njit(cache=True, nogil=True)
def generate_dead_bush(chunk_blocks, chunk_data, start_x, start_y, start_z, random):
    for _ in range(4):
        x = start_x + random.nextInt(8) - random.nextInt(8)
        z = start_z + random.nextInt(8) - random.nextInt(8)
        y = start_y + random.nextInt(4) - random.nextInt(4)
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT:
            continue
        
        if is_empty(chunk_blocks[x, y, z]):
            below = chunk_blocks[x, y-1, z]
            if below == SAND or below == RED_SAND or below == HARDENED_CLAY or below == STAINED_CLAY_ORANGE or below == DIRT or below == PODZOL or below == COARSE_DIRT:
                chunk_blocks[x, y, z] = DEADBUSH

@njit(cache=True, nogil=True)
def generate_waterlily(chunk_blocks, chunk_data, start_x, start_y, start_z, random):
    for _ in range(10):
        x = start_x + random.nextInt(8) - random.nextInt(8)
        z = start_z + random.nextInt(8) - random.nextInt(8)
        y = start_y + random.nextInt(4) - random.nextInt(4)
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT:
            continue
            
        if is_empty(chunk_blocks[x, y, z]) and chunk_blocks[x, y-1, z] == WATER:
            chunk_blocks[x, y, z] = WATERLILY

@njit(cache=True, nogil=True)
def generate_pumpkin(chunk_blocks, chunk_data, start_x, start_y, start_z, random):
    for _ in range(64):
        x = start_x + random.nextInt(8) - random.nextInt(8)
        z = start_z + random.nextInt(8) - random.nextInt(8)
        y = start_y + random.nextInt(4) - random.nextInt(4)
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT:
            continue
        if is_empty(chunk_blocks[x, y, z]) and chunk_blocks[x, y-1, z] == GRASS:
            chunk_blocks[x, y, z] = PUMPKIN
            chunk_data[x, y, z] = random.nextInt(4)

@njit(cache=True, nogil=True)
def generate_melon(chunk_blocks, chunk_data, start_x, start_y, start_z, random):
    for _ in range(64):
        x = start_x + random.nextInt(8) - random.nextInt(8)
        z = start_z + random.nextInt(8) - random.nextInt(8)
        y = start_y + random.nextInt(4) - random.nextInt(4)
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT:
            continue
        if is_empty(chunk_blocks[x, y, z]) and chunk_blocks[x, y-1, z] == GRASS:
            chunk_blocks[x, y, z] = MELON_BLOCK

@njit(cache=True, nogil=True)
def generate_sand_disk(chunk_blocks, chunk_data, start_x, start_y, start_z, random, block_id, radius):
    if chunk_blocks[start_x, start_y, start_z] != WATER:
        return
        
    r = random.nextInt(radius - 2) + 2
    for x in range(start_x - r, start_x + r + 1):
        for z in range(start_z - r, start_z + r + 1):
            xd = x - start_x
            zd = z - start_z
            if xd*xd + zd*zd > r*r:
                continue
            
            for y in range(start_y - 2, start_y + 3):
                if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 0 or y >= CHUNK_HEIGHT:
                    continue
                block = chunk_blocks[x, y, z]
                if block == DIRT or block == GRASS:
                    chunk_blocks[x, y, z] = block_id

@njit(cache=True, nogil=True)
def generate_mushroom(chunk_blocks, chunk_data, start_x, start_y, start_z, random, block_id):
    for _ in range(64):
        x = start_x + random.nextInt(8) - random.nextInt(8)
        z = start_z + random.nextInt(8) - random.nextInt(8)
        y = start_y + random.nextInt(4) - random.nextInt(4)
        if x < 0 or x >= CHUNK_SIZE or z < 0 or z >= CHUNK_SIZE or y < 1 or y >= CHUNK_HEIGHT:
            continue
        if is_empty(chunk_blocks[x, y, z]):
            below = chunk_blocks[x, y-1, z]
            if below == GRASS or below == DIRT or below == MYCELIUM or below == PODZOL:
                chunk_blocks[x, y, z] = block_id

@njit(cache=True, nogil=True)
def generate_vines_on_trunk(chunk_blocks, chunk_data, x, start_y, z, trunk_height, random):
    for y in range(start_y, start_y + trunk_height):
        if x > 0 and is_empty(chunk_blocks[x-1, y, z]) and random.nextInt(3) > 0:
            chunk_blocks[x-1, y, z] = VINE
            chunk_data[x-1, y, z] = 8
        if x < CHUNK_SIZE-1 and is_empty(chunk_blocks[x+1, y, z]) and random.nextInt(3) > 0:
            chunk_blocks[x+1, y, z] = VINE
            chunk_data[x+1, y, z] = 4
        if z > 0 and is_empty(chunk_blocks[x, y, z-1]) and random.nextInt(3) > 0:
            chunk_blocks[x, y, z-1] = VINE
            chunk_data[x, y, z-1] = 1
        if z < CHUNK_SIZE-1 and is_empty(chunk_blocks[x, y, z+1]) and random.nextInt(3) > 0:
            chunk_blocks[x, y, z+1] = VINE
            chunk_data[x, y, z+1] = 2

@njit(cache=True, nogil=True)
def add_hanging_vine(chunk_blocks, chunk_data, x, start_y, z, meta, random):
    if not is_empty(chunk_blocks[x, start_y, z]):
        return
    chunk_blocks[x, start_y, z] = VINE
    chunk_data[x, start_y, z] = meta
    
    length = 4
    y = start_y - 1
    while y > 0 and length > 0 and is_empty(chunk_blocks[x, y, z]):
        chunk_blocks[x, y, z] = VINE
        chunk_data[x, y, z] = meta
        y -= 1
        length -= 1
