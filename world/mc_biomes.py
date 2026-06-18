import numpy as np
from numba import njit
from world.terrain import *

# Biome IDs
OCEAN = 0
PLAINS = 1
DESERT = 2
EXTREME_HILLS = 3
FOREST = 4
TAIGA = 5
SWAMPLAND = 6
RIVER = 7
HELL = 8
SKY = 9
FROZEN_OCEAN = 10
FROZEN_RIVER = 11
ICE_FLATS = 12
ICE_MOUNTAINS = 13
MUSHROOM_ISLAND = 14
MUSHROOM_ISLAND_SHORE = 15
BEACHES = 16
DESERT_HILLS = 17
FOREST_HILLS = 18
TAIGA_HILLS = 19
SMALLER_EXTREME_HILLS = 20
JUNGLE = 21
JUNGLE_HILLS = 22

BIOME_COUNT = 23

# Array to store biome properties:
# [depth, scale, temperature, downfall, topMaterial, material]
biome_data = np.zeros((BIOME_COUNT, 6), dtype=np.float64)

def _init_biome(b_id, depth, scale, temp, down, top, filler):
    biome_data[b_id, 0] = depth
    biome_data[b_id, 1] = scale
    biome_data[b_id, 2] = temp
    biome_data[b_id, 3] = down
    biome_data[b_id, 4] = top
    biome_data[b_id, 5] = filler

# Define all biomes based on Bedrock original values
_init_biome(OCEAN,                 -1.0,  0.4,  0.5,  0.5, DIRT, DIRT)
_init_biome(PLAINS,                 0.125,0.05, 0.8,  0.4, GRASS, DIRT)
_init_biome(DESERT,                 0.1,  0.2,  2.0,  0.0, SAND, SAND)
_init_biome(EXTREME_HILLS,          0.3,  1.5,  0.2,  0.3, GRASS, DIRT)
_init_biome(FOREST,                 0.1,  0.3,  0.7,  0.8, GRASS, DIRT)
_init_biome(TAIGA,                  0.1,  0.4,  0.05, 0.8, GRASS, DIRT)
_init_biome(SWAMPLAND,             -0.2,  0.1,  0.8,  0.9, GRASS, DIRT)
_init_biome(RIVER,                 -0.5,  0.0,  0.5,  0.5, DIRT, DIRT)
_init_biome(HELL,                   0.1,  0.2,  2.0,  0.0, STONE, STONE)
_init_biome(SKY,                    0.1,  0.2,  0.5,  0.5, STONE, STONE)
_init_biome(FROZEN_OCEAN,          -1.0,  0.5,  0.0,  0.5, DIRT, DIRT)
_init_biome(FROZEN_RIVER,          -0.5,  0.0,  0.0,  0.5, DIRT, DIRT)
_init_biome(ICE_FLATS,              0.125,0.05, 0.0,  0.5, SNOW, DIRT)
_init_biome(ICE_MOUNTAINS,          0.3,  1.3,  0.0,  0.5, SNOW, DIRT)
_init_biome(MUSHROOM_ISLAND,        0.2,  1.0,  0.9,  1.0, MYCELIUM, DIRT)
_init_biome(MUSHROOM_ISLAND_SHORE, -1.0,  0.1,  0.9,  1.0, MYCELIUM, DIRT)
_init_biome(BEACHES,                0.0,  0.1,  0.8,  0.4, SAND, SAND)
_init_biome(DESERT_HILLS,           0.3,  0.8,  2.0,  0.0, SAND, SAND)
_init_biome(FOREST_HILLS,           0.3,  0.7,  0.7,  0.8, GRASS, DIRT)
_init_biome(TAIGA_HILLS,            0.3,  0.8,  0.05, 0.8, GRASS, DIRT)
_init_biome(SMALLER_EXTREME_HILLS,  0.2,  0.8,  0.2,  0.3, GRASS, DIRT)
_init_biome(JUNGLE,                 0.2,  0.4,  1.2,  0.9, GRASS, DIRT)
_init_biome(JUNGLE_HILLS,           1.8,  0.5,  1.2,  0.9, GRASS, DIRT)

# Compile the static array into numba
BIOME_DATA = biome_data.copy()

@njit(cache=True)
def get_biome(temp, downfall):
    """
    Finds the biome that best matches the given temperature and downfall.
    Distance is calculated using simple Euclidean distance in temp/downfall space.
    """
    best_biome = PLAINS
    best_dist = 999999.0
    
    for i in range(BIOME_COUNT):
        # Skip special biomes that shouldn't generate from normal temp/downfall
        if i == HELL or i == SKY or i == MUSHROOM_ISLAND or i == MUSHROOM_ISLAND_SHORE or i == RIVER or i == FROZEN_RIVER or i == BEACHES or i == OCEAN or i == FROZEN_OCEAN:
            continue
            
        b_temp = BIOME_DATA[i, 2]
        b_down = BIOME_DATA[i, 3]
        
        # Calculate distance squared
        dt = temp - b_temp
        dd = downfall - b_down
        dist = dt*dt + dd*dd
        
        if dist < best_dist:
            best_dist = dist
            best_biome = i
            
    return best_biome

@njit(cache=True)
def get_biome_properties(biome_id):
    """Returns depth, scale, topMaterial, material for the given biome ID"""
    return BIOME_DATA[biome_id, 0], BIOME_DATA[biome_id, 1], int(BIOME_DATA[biome_id, 4]), int(BIOME_DATA[biome_id, 5])
