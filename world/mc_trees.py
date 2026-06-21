import numpy as np
from numba import njit
from world.terrain import *
CHUNK_HEIGHT = 128

@njit(cache=True)
def place_block(blocks, wx, wy, wz, cx, cz, block_id, out_of_bounds, out_count):
    # Calculate local chunk coordinates
    lx = wx - (cx * CHUNK_SIZE)
    lz = wz - (cz * CHUNK_SIZE)
    ly = wy
    
    if ly < 0 or ly >= CHUNK_HEIGHT:
        return out_count
        
    if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE:
        # Don't replace solid blocks with leaves
        current = blocks[lx, ly, lz]
        if block_id == LEAVES or block_id == BIRCH_LEAVES or block_id == SPRUCE_LEAVES:
            if current != AIR and current != LEAVES and current != BIRCH_LEAVES and current != SPRUCE_LEAVES and current != SNOW:
                return out_count
        blocks[lx, ly, lz] = block_id
    else:
        # Save to out_of_bounds
        if out_count < len(out_of_bounds):
            out_of_bounds[out_count, 0] = wx
            out_of_bounds[out_count, 1] = wy
            out_of_bounds[out_count, 2] = wz
            out_of_bounds[out_count, 3] = block_id
            out_count += 1
            
    return out_count

@njit(cache=True)
def generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(3) + 4
    
    # Leaves
    grassHeight = 3
    for yy in range(wy + treeHeight - grassHeight, wy + treeHeight + 1):
        yo = yy - (wy + treeHeight)
        offs = 1 - int(yo / 2)
        for xx in range(wx - offs, wx + offs + 1):
            xo = xx - wx
            for zz in range(wz - offs, wz + offs + 1):
                zo = zz - wz
                if abs(xo) == offs and abs(zo) == offs and (random.nextInt(2) == 0 or yo == 0):
                    continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, LEAVES, out_of_bounds, out_count)
                
    # Trunk
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True)
def generate_birch_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(3) + 5
    
    # Leaves
    grassHeight = 3
    for yy in range(wy + treeHeight - grassHeight, wy + treeHeight + 1):
        yo = yy - (wy + treeHeight)
        offs = 1 - int(yo / 2)
        for xx in range(wx - offs, wx + offs + 1):
            xo = xx - wx
            for zz in range(wz - offs, wz + offs + 1):
                zo = zz - wz
                if abs(xo) == offs and abs(zo) == offs and (random.nextInt(2) == 0 or yo == 0):
                    continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, BIRCH_LEAVES, out_of_bounds, out_count)
                
    # Trunk
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, BIRCH_WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True)
def generate_spruce_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(4) + 6
    trunkHeight = 1 + random.nextInt(2)
    topHeight = treeHeight - trunkHeight
    leafRadius = 2 + random.nextInt(2)
    
    currentRadius = random.nextInt(2)
    maxRadius = 1
    minRadius = 0
    
    for heightPos in range(topHeight + 1):
        yy = wy + treeHeight - heightPos
        
        for xx in range(wx - currentRadius, wx + currentRadius + 1):
            xo = xx - wx
            for zz in range(wz - currentRadius, wz + currentRadius + 1):
                zo = zz - wz
                if abs(xo) == currentRadius and abs(zo) == currentRadius and currentRadius > 0:
                    continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, SPRUCE_LEAVES, out_of_bounds, out_count)
                
        if currentRadius >= maxRadius:
            currentRadius = minRadius
            minRadius = 1
            maxRadius += 1
            if maxRadius > leafRadius:
                maxRadius = leafRadius
        else:
            currentRadius += 1
            
    # Trunk
    topOffset = random.nextInt(3)
    for hh in range(treeHeight - topOffset):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True)
def can_cactus_survive(blocks, lx, wy, lz):
    if wy < 0 or wy >= CHUNK_HEIGHT: return False
    # Check neighbors (must be AIR)
    if lx > 0 and blocks[lx - 1, wy, lz] != AIR: return False
    if lx < 15 and blocks[lx + 1, wy, lz] != AIR: return False
    if lz > 0 and blocks[lx, wy, lz - 1] != AIR: return False
    if lz < 15 and blocks[lx, wy, lz + 1] != AIR: return False
    # Check below (must be SAND or CACTUS)
    if wy > 0:
        below = blocks[lx, wy - 1, lz]
        if below != SAND and below != CACTUS: # 5=SAND, 13=CACTUS
            return False
    return True

@njit(cache=True)
def generate_cactus(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    lx = wx - (cx * 16)
    lz = wz - (cz * 16)
    if lx < 0 or lx >= 16 or lz < 0 or lz >= 16:
        return out_count # Skip edge crossing for simplicity
        
    height = random.nextInt(3) + 1
    for hh in range(height):
        # Check survival BEFORE placing
        if can_cactus_survive(blocks, lx, wy + hh, lz):
            out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, CACTUS, out_of_bounds, out_count) # 13=CACTUS
        else:
            break # Stop growing if it can't survive
            
    return out_count

@njit(cache=True)
def generate_grass_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    # wx, wy, wz is the center
    for i in range(128):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 1:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny - 1, lz] == 3: # 0=AIR, 3=GRASS
                blocks[lx, ny, lz] = 31 # TALL_GRASS
    return out_count

@njit(cache=True)
def generate_flower_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, flower_type):
    for i in range(64):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 1:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny - 1, lz] == 3:
                blocks[lx, ny, lz] = flower_type
    return out_count
    
@njit(cache=True)
def generate_double_plant_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, btm_type, top_type):
    for i in range(64):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 2:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny + 1, lz] == 0 and blocks[lx, ny - 1, lz] == 3:
                blocks[lx, ny, lz] = btm_type
                blocks[lx, ny + 1, lz] = top_type
    return out_count
