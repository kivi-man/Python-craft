"""
PythonCraft - Special Blocks System
Translates C++ StairTile and Slab AABB/Shape logic to Python.
"""
import numpy as np
from numba import njit
from world.terrain import (
    OAK_STAIRS, COBBLESTONE_STAIRS, SPRUCE_STAIRS, BIRCH_STAIRS, JUNGLE_STAIRS, 
    ACACIA_STAIRS, DARK_OAK_STAIRS, BRICK_STAIRS, STONE_BRICK_STAIRS, NETHER_BRICK_STAIRS,
    SANDSTONE_STAIRS, QUARTZ_STAIRS,
    WOODEN_SLAB, STONE_SLAB, SPRUCE_SLAB, BIRCH_SLAB, JUNGLE_SLAB,
    ACACIA_SLAB, DARK_OAK_SLAB, BRICK_SLAB, STONE_BRICK_SLAB, NETHER_BRICK_SLAB,
    SANDSTONE_SLAB, QUARTZ_SLAB, COBBLESTONE_SLAB
)

UPSIDEDOWN_BIT = 4
DIR_EAST = 0
DIR_WEST = 1
DIR_SOUTH = 2
DIR_NORTH = 3

STAIRS_IDS = (
    OAK_STAIRS, COBBLESTONE_STAIRS, SPRUCE_STAIRS, BIRCH_STAIRS, JUNGLE_STAIRS, 
    ACACIA_STAIRS, DARK_OAK_STAIRS, BRICK_STAIRS, STONE_BRICK_STAIRS, NETHER_BRICK_STAIRS,
    SANDSTONE_STAIRS, QUARTZ_STAIRS
)
SLAB_IDS = (
    WOODEN_SLAB, STONE_SLAB, SPRUCE_SLAB, BIRCH_SLAB, JUNGLE_SLAB,
    ACACIA_SLAB, DARK_OAK_SLAB, BRICK_SLAB, STONE_BRICK_SLAB, NETHER_BRICK_SLAB,
    SANDSTONE_SLAB, QUARTZ_SLAB, COBBLESTONE_SLAB
)

SLAB_TO_FULL = {
    WOODEN_SLAB: 44,       # PLANKS_OAK
    STONE_SLAB: 1,         # STONE
    SPRUCE_SLAB: 45,       # PLANKS_SPRUCE
    BIRCH_SLAB: 46,        # PLANKS_BIRCH
    JUNGLE_SLAB: 47,       # PLANKS_JUNGLE
    ACACIA_SLAB: 48,       # PLANKS_ACACIA
    DARK_OAK_SLAB: 49,     # PLANKS_DARK_OAK
    BRICK_SLAB: 54,        # BRICKS
    STONE_BRICK_SLAB: 66,  # STONEBRICK
    NETHER_BRICK_SLAB: 71, # NETHER_BRICK
    SANDSTONE_SLAB: 9,     # SANDSTONE
    QUARTZ_SLAB: 76,       # QUARTZ_BLOCK
    COBBLESTONE_SLAB: 43   # COBBLESTONE
}

BLOCK_IS_STAIR = np.zeros(256, dtype=np.bool_)
for s_id in STAIRS_IDS:
    BLOCK_IS_STAIR[s_id] = True

BLOCK_IS_SLAB = np.zeros(256, dtype=np.bool_)
for s_id in SLAB_IDS:
    BLOCK_IS_SLAB[s_id] = True

@njit(nogil=True)
def is_stairs(block_id):
    if block_id < 0 or block_id > 255: return False
    return BLOCK_IS_STAIR[block_id]

@njit(nogil=True)
def is_slab(block_id):
    if block_id < 0 or block_id > 255: return False
    return BLOCK_IS_SLAB[block_id]

@njit(nogil=True)
def is_lock_attached(neighbor_id, neighbor_data, target_data):
    if is_stairs(neighbor_id) and neighbor_data == target_data:
        return True
    return False

@njit(nogil=True)
def get_slab_aabbs(x, y, z, data):
    # Returns (minX, minY, minZ, maxX, maxY, maxZ)
    if (data & UPSIDEDOWN_BIT) != 0:
        return np.array([[x, y + 0.5, z, x + 1.0, y + 1.0, z + 1.0]], dtype=np.float32)
    else:
        return np.array([[x, y, z, x + 1.0, y + 0.5, z + 1.0]], dtype=np.float32)

@njit(nogil=True)
def get_stair_aabbs(x, y, z, data, 
                    w_id, w_data, e_id, e_data, 
                    n_id, n_data, s_id, s_data):
    """
    Returns list of AABBs for the stair.
    Uses exactly the same logic as StairTile.cpp's addAABBs, setBaseShape, setStepShape, setInnerPieceShape.
    Since Numba needs a fixed return type, we return a (3, 6) float32 array, with empty AABBs as zeros.
    """
    aabbs = np.zeros((3, 6), dtype=np.float32)
    dir_val = data & 0x3
    
    # 1. Base Shape
    bottom = 0.5
    top = 1.0
    if (data & UPSIDEDOWN_BIT) != 0:
        bottom = 0.0
        top = 0.5
        aabbs[0] = [x, y + 0.5, z, x + 1.0, y + 1.0, z + 1.0]
    else:
        aabbs[0] = [x, y, z, x + 1.0, y + 0.5, z + 1.0]

    # 2. Step Shape
    west = 0.0
    east = 1.0
    north = 0.0
    south = 0.5
    check_inner_piece = True

    if dir_val == DIR_EAST:
        west = 0.5
        south = 1.0
        if is_stairs(e_id) and ((data & UPSIDEDOWN_BIT) == (e_data & UPSIDEDOWN_BIT)):
            back_dir = e_data & 0x3
            if back_dir == DIR_NORTH and not is_lock_attached(s_id, s_data, data):
                south = 0.5
                check_inner_piece = False
            elif back_dir == DIR_SOUTH and not is_lock_attached(n_id, n_data, data):
                north = 0.5
                check_inner_piece = False
    elif dir_val == DIR_WEST:
        east = 0.5
        south = 1.0
        if is_stairs(w_id) and ((data & UPSIDEDOWN_BIT) == (w_data & UPSIDEDOWN_BIT)):
            back_dir = w_data & 0x3
            if back_dir == DIR_NORTH and not is_lock_attached(s_id, s_data, data):
                south = 0.5
                check_inner_piece = False
            elif back_dir == DIR_SOUTH and not is_lock_attached(n_id, n_data, data):
                north = 0.5
                check_inner_piece = False
    elif dir_val == DIR_SOUTH:
        north = 0.5
        south = 1.0
        if is_stairs(s_id) and ((data & UPSIDEDOWN_BIT) == (s_data & UPSIDEDOWN_BIT)):
            back_dir = s_data & 0x3
            if back_dir == DIR_WEST and not is_lock_attached(e_id, e_data, data):
                east = 0.5
                check_inner_piece = False
            elif back_dir == DIR_EAST and not is_lock_attached(w_id, w_data, data):
                west = 0.5
                check_inner_piece = False
    elif dir_val == DIR_NORTH:
        if is_stairs(n_id) and ((data & UPSIDEDOWN_BIT) == (n_data & UPSIDEDOWN_BIT)):
            back_dir = n_data & 0x3
            if back_dir == DIR_WEST and not is_lock_attached(e_id, e_data, data):
                east = 0.5
                check_inner_piece = False
            elif back_dir == DIR_EAST and not is_lock_attached(w_id, w_data, data):
                west = 0.5
                check_inner_piece = False

    aabbs[1] = [x + west, y + bottom, z + north, x + east, y + top, z + south]

    # 3. Inner Piece Shape
    if check_inner_piece:
        has_inner_piece = False
        west = 0.0
        east = 0.5
        north = 0.5
        south = 1.0
        
        if dir_val == DIR_EAST:
            if is_stairs(w_id) and ((data & UPSIDEDOWN_BIT) == (w_data & UPSIDEDOWN_BIT)):
                front_dir = w_data & 0x3
                if front_dir == DIR_NORTH and not is_lock_attached(n_id, n_data, data):
                    north = 0.0
                    south = 0.5
                    has_inner_piece = True
                elif front_dir == DIR_SOUTH and not is_lock_attached(s_id, s_data, data):
                    north = 0.5
                    south = 1.0
                    has_inner_piece = True
        elif dir_val == DIR_WEST:
            if is_stairs(e_id) and ((data & UPSIDEDOWN_BIT) == (e_data & UPSIDEDOWN_BIT)):
                west = 0.5
                east = 1.0
                front_dir = e_data & 0x3
                if front_dir == DIR_NORTH and not is_lock_attached(n_id, n_data, data):
                    north = 0.0
                    south = 0.5
                    has_inner_piece = True
                elif front_dir == DIR_SOUTH and not is_lock_attached(s_id, s_data, data):
                    north = 0.5
                    south = 1.0
                    has_inner_piece = True
        elif dir_val == DIR_SOUTH:
            if is_stairs(n_id) and ((data & UPSIDEDOWN_BIT) == (n_data & UPSIDEDOWN_BIT)):
                north = 0.0
                south = 0.5
                front_dir = n_data & 0x3
                if front_dir == DIR_WEST and not is_lock_attached(w_id, w_data, data):
                    west = 0.0
                    east = 0.5
                    has_inner_piece = True
                elif front_dir == DIR_EAST and not is_lock_attached(e_id, e_data, data):
                    west = 0.5
                    east = 1.0
                    has_inner_piece = True
        elif dir_val == DIR_NORTH:
            if is_stairs(s_id) and ((data & UPSIDEDOWN_BIT) == (s_data & UPSIDEDOWN_BIT)):
                front_dir = s_data & 0x3
                if front_dir == DIR_WEST and not is_lock_attached(w_id, w_data, data):
                    west = 0.0
                    east = 0.5
                    has_inner_piece = True
                elif front_dir == DIR_EAST and not is_lock_attached(e_id, e_data, data):
                    west = 0.5
                    east = 1.0
                    has_inner_piece = True
        
        if has_inner_piece:
            aabbs[2] = [x + west, y + bottom, z + north, x + east, y + top, z + south]
            
    return aabbs
