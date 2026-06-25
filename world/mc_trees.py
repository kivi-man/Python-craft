import numpy as np
from numba import njit
from world.terrain import *
import math

@njit(cache=True, nogil=True)
def place_block(blocks, wx, wy, wz, cx, cz, block_id, out_of_bounds, out_count):
    # Calculate local chunk coordinates
    lx = wx - (cx * CHUNK_SIZE)
    lz = wz - (cz * CHUNK_SIZE)
    ly = wy
    
    if ly < 0 or ly >= CHUNK_HEIGHT:
        return out_count
        
    if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE:
        current = blocks[lx, ly, lz]
        # Allow overwriting leaves and air
        if block_id in (LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, JUNGLE_LEAVES, ACACIA_LEAVES, DARK_OAK_LEAVES):
            if current != AIR and current not in (LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, JUNGLE_LEAVES, ACACIA_LEAVES, DARK_OAK_LEAVES, SNOW_LAYER, VINE):
                return out_count
        # Trunk can overwrite leaves
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

@njit(cache=True, nogil=True)
def generate_oak_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, add_vines=False):
    treeHeight = random.nextInt(3) + 4
    
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
                out_count = place_block(blocks, xx, yy, zz, cx, cz, JUNGLE_LEAVES if add_vines else LEAVES, out_of_bounds, out_count)
                
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, JUNGLE_WOOD if add_vines else WOOD, out_of_bounds, out_count)
        
    if add_vines:
        for hh in range(treeHeight):
            if random.nextInt(3) > 0: out_count = place_block(blocks, wx-1, wy+hh, wz, cx, cz, VINE, out_of_bounds, out_count)
            if random.nextInt(3) > 0: out_count = place_block(blocks, wx+1, wy+hh, wz, cx, cz, VINE, out_of_bounds, out_count)
            if random.nextInt(3) > 0: out_count = place_block(blocks, wx, wy+hh, wz-1, cx, cz, VINE, out_of_bounds, out_count)
            if random.nextInt(3) > 0: out_count = place_block(blocks, wx, wy+hh, wz+1, cx, cz, VINE, out_of_bounds, out_count)

    return out_count

@njit(cache=True, nogil=True)
def generate_birch_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, tall=False):
    treeHeight = random.nextInt(3) + 5
    if tall:
        treeHeight += random.nextInt(7)
        
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
                
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, BIRCH_WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True, nogil=True)
def generate_spruce_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, pine=False):
    treeHeight = random.nextInt(4) + 6
    if pine:
        treeHeight = random.nextInt(5) + 7
        
    trunkHeight = 1 + random.nextInt(2)
    if pine:
        trunkHeight = treeHeight - random.nextInt(2) - 3
        
    topHeight = treeHeight - trunkHeight
    leafRadius = 2 + random.nextInt(2)
    
    currentRadius = random.nextInt(2) if not pine else 0
    maxRadius = 1
    minRadius = 0
    
    for heightPos in range(topHeight + 1):
        yy = wy + treeHeight - heightPos
        r = currentRadius if not pine else (1 + int(heightPos / topHeight * 2))
        
        for xx in range(wx - r, wx + r + 1):
            xo = xx - wx
            for zz in range(wz - r, wz + r + 1):
                zo = zz - wz
                if abs(xo) == r and abs(zo) == r and r > 0:
                    continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, SPRUCE_LEAVES, out_of_bounds, out_count)
                
        if not pine:
            if currentRadius >= maxRadius:
                currentRadius = minRadius
                minRadius = 1
                maxRadius += 1
                if maxRadius > leafRadius:
                    maxRadius = leafRadius
            else:
                currentRadius += 1
            
    topOffset = random.nextInt(3) if not pine else 1
    for hh in range(treeHeight - topOffset):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True, nogil=True)
def generate_swamp_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(4) + 5
    
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
                
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, WOOD, out_of_bounds, out_count)
        
    for yy in range(wy + treeHeight - grassHeight, wy + treeHeight + 1):
        yo = yy - (wy + treeHeight)
        offs = 1 - int(yo / 2)
        for xx in range(wx - offs, wx + offs + 1):
            for zz in range(wz - offs, wz + offs + 1):
                if random.nextInt(4) == 0:
                    for v in range(1, 5):
                        out_count = place_block(blocks, xx, yy - v, zz, cx, cz, VINE, out_of_bounds, out_count)
                        
    return out_count


@njit(cache=True, nogil=True)
def place_leaf_at(blocks, xx, yy, zz, cx, cz, leaf_id, out_of_bounds, out_count):
    # Only place if air or existing leaf
    lx = xx - (cx * CHUNK_SIZE)
    lz = zz - (cz * CHUNK_SIZE)
    if 0 <= lx < CHUNK_SIZE and 0 <= lz < CHUNK_SIZE:
        if 0 <= yy < CHUNK_HEIGHT:
            current = blocks[lx, yy, lz]
            if current == AIR or current in (LEAVES, BIRCH_LEAVES, SPRUCE_LEAVES, JUNGLE_LEAVES, ACACIA_LEAVES, DARK_OAK_LEAVES):
                blocks[lx, yy, lz] = leaf_id
                return out_count
    return place_block(blocks, xx, yy, zz, cx, cz, leaf_id, out_of_bounds, out_count)

@njit(cache=True, nogil=True)
def generate_savanna_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    height = random.nextInt(3) + random.nextInt(3) + 5
    branchStart = height - random.nextInt(4) - 1
    branchLen = 3 - random.nextInt(3)
    
    # 0=S, 1=W, 2=N, 3=E
    facing1 = random.nextInt(4)
    dx1, dz1 = 0, 0
    if facing1 == 0: dz1 = 1
    elif facing1 == 1: dx1 = -1
    elif facing1 == 2: dz1 = -1
    elif facing1 == GRASS: dx1 = 1
    
    curX, curZ = wx, wz
    topY = wy
    
    for l1 in range(height):
        curY = wy + l1
        if l1 >= branchStart and branchLen > 0:
            curX += dx1
            curZ += dz1
            branchLen -= 1
        out_count = place_block(blocks, curX, curY, curZ, cx, cz, ACACIA_WOOD, out_of_bounds, out_count)
        topY = curY
        
    # Layer 3
    for dx in range(-3, 4):
        for dz in range(-3, 4):
            if abs(dx) != 3 or abs(dz) != 3:
                out_count = place_leaf_at(blocks, curX + dx, topY, curZ + dz, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
                
    # Layer 1
    for dx in range(-1, 2):
        for dz in range(-1, 2):
            out_count = place_leaf_at(blocks, curX + dx, topY + 1, curZ + dz, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
    out_count = place_leaf_at(blocks, curX + 2, topY + 1, curZ, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
    out_count = place_leaf_at(blocks, curX - 2, topY + 1, curZ, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
    out_count = place_leaf_at(blocks, curX, topY + 1, curZ + 2, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
    out_count = place_leaf_at(blocks, curX, topY + 1, curZ - 2, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
    
    curX2, curZ2 = wx, wz
    facing2 = random.nextInt(4)
    
    if facing2 != facing1:
        start2 = branchStart - random.nextInt(2) - 1
        steps2 = 1 + random.nextInt(3)
        topY2 = 0
        
        dx2, dz2 = 0, 0
        if facing2 == 0: dz2 = 1
        elif facing2 == 1: dx2 = -1
        elif facing2 == 2: dz2 = -1
        elif facing2 == GRASS: dx2 = 1
        
        l4 = start2
        while l4 < height and steps2 > 0:
            if l4 >= 1:
                curY2 = wy + l4
                curX2 += dx2
                curZ2 += dz2
                out_count = place_block(blocks, curX2, curY2, curZ2, cx, cz, ACACIA_WOOD, out_of_bounds, out_count)
                topY2 = curY2
            l4 += 1
            steps2 -= 1
            
        if topY2 > 0:
            for dx in range(-2, 3):
                for dz in range(-2, 3):
                    if abs(dx) != 2 or abs(dz) != 2:
                        out_count = place_leaf_at(blocks, curX2 + dx, topY2, curZ2 + dz, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)
            for dx in range(-1, 2):
                for dz in range(-1, 2):
                    out_count = place_leaf_at(blocks, curX2 + dx, topY2 + 1, curZ2 + dz, cx, cz, ACACIA_LEAVES, out_of_bounds, out_count)

    return out_count
@njit(cache=True, nogil=True)
def generate_mega_jungle_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(15) + 15
    
    for yy in range(wy + treeHeight - 2, wy + treeHeight + 2):
        r = 3 if yy < wy + treeHeight + 1 else 2
        for xx in range(wx - r, wx + r + 2):
            for zz in range(wz - r, wz + r + 2):
                if abs(xx - wx) == r and abs(zz - wz) == r: continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, JUNGLE_LEAVES, out_of_bounds, out_count)
                
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, JUNGLE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz, cx, cz, JUNGLE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx, wy + hh, wz+1, cx, cz, JUNGLE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz+1, cx, cz, JUNGLE_WOOD, out_of_bounds, out_count)
        
    return out_count

@njit(cache=True, nogil=True)
def generate_mega_pine_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    treeHeight = random.nextInt(15) + 15
    
    crownHeight = random.nextInt(5) + 10
    for yy in range(wy + treeHeight - crownHeight, wy + treeHeight + 1):
        distFromTop = (wy + treeHeight) - yy
        r = int(distFromTop / crownHeight * 3.5)
        for xx in range(wx - r, wx + r + 2):
            for zz in range(wz - r, wz + r + 2):
                if (xx - wx)**2 + (zz - wz)**2 <= (r + 0.5)**2:
                    out_count = place_block(blocks, xx, yy, zz, cx, cz, SPRUCE_LEAVES, out_of_bounds, out_count)
                    
    for hh in range(treeHeight):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx, wy + hh, wz+1, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz+1, cx, cz, SPRUCE_WOOD, out_of_bounds, out_count)
        
    for xx in range(wx - 2, wx + 4):
        for zz in range(wz - 2, wz + 4):
            out_count = place_block(blocks, xx, wy - 1, zz, cx, cz, PODZOL, out_of_bounds, out_count)
            
    return out_count

@njit(cache=True, nogil=True)
def generate_roof_tree(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    height = random.nextInt(3) + 6
    
    for hh in range(height):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, DARK_OAK_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz, cx, cz, DARK_OAK_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx, wy + hh, wz+1, cx, cz, DARK_OAK_WOOD, out_of_bounds, out_count)
        out_count = place_block(blocks, wx+1, wy + hh, wz+1, cx, cz, DARK_OAK_WOOD, out_of_bounds, out_count)
        
    for yy in range(wy + height - 2, wy + height + 2):
        r = 3 if yy < wy + height + 1 else 2
        for xx in range(wx - r, wx + r + 2):
            for zz in range(wz - r, wz + r + 2):
                if abs(xx - wx) == r and abs(zz - wz) == r: continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, DARK_OAK_LEAVES, out_of_bounds, out_count)
                
    return out_count

@njit(cache=True, nogil=True)
def generate_jungle_bush(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    out_count = place_block(blocks, wx, wy, wz, cx, cz, JUNGLE_WOOD, out_of_bounds, out_count)
    for yy in range(wy, wy + 3):
        r = 2 - (yy - wy)
        for xx in range(wx - r, wx + r + 1):
            for zz in range(wz - r, wz + r + 1):
                if abs(xx - wx) == r and abs(zz - wz) == r and random.nextInt(2) == 0: continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, JUNGLE_LEAVES, out_of_bounds, out_count)
    return out_count

@njit(cache=True, nogil=True)
def generate_huge_mushroom(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, is_red):
    height = random.nextInt(3) + 4
    
    for hh in range(height):
        out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, MUSHROOM_BROWN if not is_red else MUSHROOM_RED, out_of_bounds, out_count)
        
    if not is_red:
        yy = wy + height
        for xx in range(wx - 3, wx + 4):
            for zz in range(wz - 3, wz + 4):
                if abs(xx - wx) == 3 and abs(zz - wz) == GRASS: continue
                out_count = place_block(blocks, xx, yy, zz, cx, cz, MUSHROOM_BROWN, out_of_bounds, out_count)
    else:
        for yy in range(wy + height - 3, wy + height + 1):
            r = 2 if yy < wy + height else 1
            for xx in range(wx - r, wx + r + 1):
                for zz in range(wz - r, wz + r + 1):
                    if abs(xx - wx) == r and abs(zz - wz) == r: continue
                    out_count = place_block(blocks, xx, yy, zz, cx, cz, MUSHROOM_RED, out_of_bounds, out_count)
                    
    return out_count

@njit(cache=True, nogil=True)
def can_cactus_survive(blocks, lx, wy, lz):
    if wy < 0 or wy >= CHUNK_HEIGHT: return False
    if lx > 0 and blocks[lx - 1, wy, lz] != AIR: return False
    if lx < 15 and blocks[lx + 1, wy, lz] != AIR: return False
    if lz > 0 and blocks[lx, wy, lz - 1] != AIR: return False
    if lz < 15 and blocks[lx, wy, lz + 1] != AIR: return False
    if wy > 0:
        below = blocks[lx, wy - 1, lz]
        if below != SAND and below != RED_SAND and below != CACTUS:
            return False
    return True

@njit(cache=True, nogil=True)
def generate_cactus(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    lx = wx - (cx * 16)
    lz = wz - (cz * 16)
    if lx < 0 or lx >= 16 or lz < 0 or lz >= 16:
        return out_count
        
    height = random.nextInt(3) + 1
    for hh in range(height):
        if can_cactus_survive(blocks, lx, wy + hh, lz):
            out_count = place_block(blocks, wx, wy + hh, wz, cx, cz, CACTUS, out_of_bounds, out_count)
        else:
            break
            
    return out_count

@njit(cache=True, nogil=True)
def generate_grass_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count):
    for i in range(128):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 1:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny - 1, lz] == GRASS:
                blocks[lx, ny, lz] = TALLGRASS
    return out_count

@njit(cache=True, nogil=True)
def generate_flower_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, flower_type):
    for i in range(64):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 1:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny - 1, lz] == GRASS:
                blocks[lx, ny, lz] = flower_type
    return out_count
    
@njit(cache=True, nogil=True)
def generate_double_plant_cluster(cx, cz, wx, wy, wz, blocks, random, out_of_bounds, out_count, btm_type, top_type):
    for i in range(64):
        nx = wx + random.nextInt(8) - random.nextInt(8)
        ny = wy + random.nextInt(4) - random.nextInt(4)
        nz = wz + random.nextInt(8) - random.nextInt(8)
        lx = nx - cx * 16
        lz = nz - cz * 16
        if 0 <= lx < 16 and 0 <= lz < 16 and 0 < ny < CHUNK_HEIGHT - 2:
            if blocks[lx, ny, lz] == 0 and blocks[lx, ny + 1, lz] == 0 and blocks[lx, ny - 1, lz] == GRASS:
                blocks[lx, ny, lz] = btm_type
                blocks[lx, ny + 1, lz] = top_type
    return out_count
