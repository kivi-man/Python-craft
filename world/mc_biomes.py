import numpy as np
from numba import njit
from world.terrain import *
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class MobSpawnerData:
    mob_type: str
    weight: int
    min_count: int
    max_count: int

# Default lists from Biome.cpp
DEFAULT_FRIENDLIES = [
    MobSpawnerData("Sheep", 12, 4, 4),
    MobSpawnerData("Pig", 10, 4, 4),
    MobSpawnerData("Chicken", 10, 4, 4),
    MobSpawnerData("Cow", 8, 4, 4),
]

DEFAULT_ENEMIES = [
    MobSpawnerData("Spider", 10, 4, 4),
    MobSpawnerData("Zombie", 10, 4, 4),
    MobSpawnerData("Skeleton", 10, 4, 4),
    MobSpawnerData("Creeper", 10, 4, 4),
    MobSpawnerData("Slime", 10, 4, 4),
    MobSpawnerData("Enderman", 1, 1, 4),
]

DEFAULT_WATER_FRIENDLIES = [
    MobSpawnerData("Squid", 10, 4, 4),
]

BIOME_SPAWN_LISTS: Dict[int, Dict[str, List[MobSpawnerData]]] = {}

def _init_biome_spawns(b_id, friendlies=None, enemies=None, water_friendlies=None):
    if friendlies is None: friendlies = list(DEFAULT_FRIENDLIES)
    if enemies is None: enemies = list(DEFAULT_ENEMIES)
    if water_friendlies is None: water_friendlies = list(DEFAULT_WATER_FRIENDLIES)
    
    BIOME_SPAWN_LISTS[b_id] = {
        "friendlies": friendlies,
        "enemies": enemies,
        "water_friendlies": water_friendlies
    }

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
JUNGLE_EDGE = 23
DEEP_OCEAN = 24
COLD_BEACH = 26
BIRCH_FOREST = 27
BIRCH_FOREST_HILLS = 28
ROOFED_FOREST = 29
COLD_TAIGA = 30
COLD_TAIGA_HILLS = 31
MEGA_TAIGA = 32
MEGA_TAIGA_HILLS = 33
EXTREME_HILLS_PLUS = 34
SAVANNA = 35
SAVANNA_PLATEAU = 36
MESA = 37
MESA_PLATEAU_F = 38
MESA_PLATEAU = 39

SUNFLOWERS_PLAINS = 129
DESERT_M = 130
EXTREME_HILLS_M = 131
FLOWER_FOREST = 132
TAIGA_M = 133
SWAMPLAND_M = 134
ICE_SPIKES = 140
JUNGLE_M = 149
JUNGLE_EDGE_M = 151
BIRCH_FOREST_M = 155
BIRCH_FOREST_HILLS_M = 156
ROOFED_FOREST_M = 157
COLD_TAIGA_M = 158
MEGA_SPRUCE_TAIGA = 160
MEGA_SPRUCE_TAIGA_HILLS = 161
EXTREME_HILLS_PLUS_M = 162
SAVANNA_M = 163
SAVANNA_PLATEAU_M = 164
MESA_BRYCE = 165
MESA_PLATEAU_F_M = 166
MESA_PLATEAU_M = 167

BIOME_COUNT = 256

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
_init_biome(OCEAN,                 -1.0,  0.4,  0.5,  0.5, GRAVEL, GRAVEL)
_init_biome(PLAINS,                 0.1,  0.3,  0.8,  0.4, GRASS, DIRT)
_init_biome(DESERT,                 0.1,  0.2,  2.0,  0.0, SAND, SAND)
_init_biome(EXTREME_HILLS,          0.3,  1.5,  0.2,  0.3, GRASS, DIRT)
_init_biome(FOREST,                 0.1,  0.3,  0.7,  0.8, GRASS, DIRT)
_init_biome(TAIGA,                  0.1,  0.4,  0.25, 0.8, GRASS, DIRT)
_init_biome(SWAMPLAND,             -0.2,  0.1,  0.8,  0.9, GRASS, DIRT)
_init_biome(RIVER,                 -0.5,  0.0,  0.5,  0.5, GRASS, DIRT)
_init_biome(HELL,                   0.1,  0.3,  2.0,  0.0, GRASS, DIRT)
_init_biome(SKY,                    0.1,  0.3,  0.5,  0.5, DIRT, DIRT)
_init_biome(FROZEN_OCEAN,          -1.0,  0.5,  0.0,  0.5, GRAVEL, GRAVEL)
_init_biome(FROZEN_RIVER,          -0.5,  0.0,  0.0,  0.5, GRASS, DIRT)
_init_biome(ICE_FLATS,              0.1,  0.3,  0.0,  0.5, GRASS, DIRT)
_init_biome(ICE_MOUNTAINS,          0.3,  1.3,  0.0,  0.5, GRASS, DIRT)
_init_biome(MUSHROOM_ISLAND,        0.2,  1.0,  0.9,  1.0, MYCELIUM, DIRT)
_init_biome(MUSHROOM_ISLAND_SHORE, -1.0,  0.1,  0.9,  1.0, MYCELIUM, DIRT)
_init_biome(BEACHES,                0.0,  0.1,  0.8,  0.4, SAND, SAND)
_init_biome(DESERT_HILLS,           0.3,  0.8,  2.0,  0.0, SAND, SAND)
_init_biome(FOREST_HILLS,           0.3,  0.7,  0.7,  0.8, GRASS, DIRT)
_init_biome(TAIGA_HILLS,            0.3,  0.8,  0.25, 0.8, GRASS, DIRT)
_init_biome(SMALLER_EXTREME_HILLS,  0.2,  0.8,  0.2,  0.3, GRASS, DIRT)
_init_biome(JUNGLE,                 0.2,  0.4,  1.2,  0.9, GRASS, DIRT)
_init_biome(JUNGLE_HILLS,           1.8,  0.5,  1.2,  0.9, GRASS, DIRT)
_init_biome(JUNGLE_EDGE,            0.1,  0.3,  0.95, 0.8, GRASS, DIRT)
_init_biome(DEEP_OCEAN,            -1.8,  0.1,  0.5,  0.5, GRAVEL, GRAVEL)
_init_biome(COLD_BEACH,             0.0,  0.025, 0.05, 0.3, SAND, SAND)
_init_biome(BIRCH_FOREST,           0.1,  0.3,  0.6,  0.6, GRASS, DIRT)
_init_biome(BIRCH_FOREST_HILLS,     0.45, 0.3,  0.6,  0.6, GRASS, DIRT)
_init_biome(ROOFED_FOREST,          0.1,  0.3,  0.7,  0.8, GRASS, DIRT)
_init_biome(COLD_TAIGA,             0.1,  0.4, -0.5,  0.4, GRASS, DIRT)
_init_biome(COLD_TAIGA_HILLS,       0.3,  0.8, -0.5,  0.4, GRASS, DIRT)
_init_biome(MEGA_TAIGA,             0.1,  0.4,  0.3,  0.8, GRASS, DIRT)
_init_biome(MEGA_TAIGA_HILLS,       0.3,  0.8,  0.3,  0.8, GRASS, DIRT)
_init_biome(EXTREME_HILLS_PLUS,     0.3,  1.5,  0.2,  0.3, GRASS, DIRT)
_init_biome(SAVANNA,                0.1,  0.3,  1.2,  0.0, GRASS, DIRT)
_init_biome(SAVANNA_PLATEAU,        1.5,  0.025,1.0,  0.0, GRASS, DIRT)
_init_biome(MESA,                   0.1,  0.3,  2.0,  0.0, RED_SAND, STAINED_CLAY_ORANGE)
_init_biome(MESA_PLATEAU_F,         1.5,  0.025,2.0,  0.0, RED_SAND, STAINED_CLAY_ORANGE)
_init_biome(MESA_PLATEAU,           1.5,  0.025,2.0,  0.0, RED_SAND, STAINED_CLAY_ORANGE)

def _init_mutated(b_id, base_id, depth_add=0.1, scale_add=0.2):
    biome_data[b_id, 0] = biome_data[base_id, 0] + depth_add
    biome_data[b_id, 1] = biome_data[base_id, 1] + scale_add
    biome_data[b_id, 2] = biome_data[base_id, 2]
    biome_data[b_id, 3] = biome_data[base_id, 3]
    biome_data[b_id, 4] = biome_data[base_id, 4]
    biome_data[b_id, 5] = biome_data[base_id, 5]

_init_mutated(SUNFLOWERS_PLAINS, PLAINS, 0, 0)
_init_mutated(DESERT_M, DESERT, 0.125, 0.05)
_init_mutated(EXTREME_HILLS_M, EXTREME_HILLS)
_init_mutated(FLOWER_FOREST, FOREST, 0, 0)
_init_mutated(TAIGA_M, TAIGA, 0.2, 0)
_init_mutated(SWAMPLAND_M, SWAMPLAND, 0.1, 0.2)
_init_mutated(ICE_SPIKES, ICE_FLATS, 0, 0)
biome_data[ICE_SPIKES, 4] = SNOW_LAYER
_init_mutated(JUNGLE_M, JUNGLE)
_init_mutated(JUNGLE_EDGE_M, JUNGLE_EDGE)
_init_mutated(BIRCH_FOREST_M, BIRCH_FOREST)
_init_mutated(BIRCH_FOREST_HILLS_M, BIRCH_FOREST_HILLS)
_init_mutated(ROOFED_FOREST_M, ROOFED_FOREST)
_init_mutated(COLD_TAIGA_M, COLD_TAIGA)
_init_mutated(MEGA_SPRUCE_TAIGA, MEGA_TAIGA, 0.1, -0.2)
_init_mutated(MEGA_SPRUCE_TAIGA_HILLS, MEGA_TAIGA_HILLS, -0.1, -0.6)
_init_mutated(EXTREME_HILLS_PLUS_M, EXTREME_HILLS_PLUS)
_init_mutated(SAVANNA_M, SAVANNA, 0.25, 1.0)
_init_mutated(SAVANNA_PLATEAU_M, SAVANNA_PLATEAU, -0.45, 1.1875)
_init_mutated(MESA_BRYCE, MESA_PLATEAU)
_init_mutated(MESA_PLATEAU_F_M, MESA_PLATEAU_F)
_init_mutated(MESA_PLATEAU_M, MESA_PLATEAU)

# Initialize spawn lists for all biomes
for b_id in range(BIOME_COUNT):
    _init_biome_spawns(b_id)

# Special overrides
# Wolves in forests and taigas
forest_friendlies = list(DEFAULT_FRIENDLIES) + [MobSpawnerData("Wolf", 5, 4, 4)]
BIOME_SPAWN_LISTS[FOREST]["friendlies"] = forest_friendlies
BIOME_SPAWN_LISTS[FOREST_HILLS]["friendlies"] = forest_friendlies
BIOME_SPAWN_LISTS[TAIGA]["friendlies"] = forest_friendlies
BIOME_SPAWN_LISTS[TAIGA_HILLS]["friendlies"] = forest_friendlies

# Mushroom islands have no regular enemies or friendlies by default, just mooshrooms (added if needed later)
BIOME_SPAWN_LISTS[MUSHROOM_ISLAND]["friendlies"] = [MobSpawnerData("MushroomCow", 8, 4, 8)]
BIOME_SPAWN_LISTS[MUSHROOM_ISLAND]["enemies"] = []
BIOME_SPAWN_LISTS[MUSHROOM_ISLAND_SHORE]["friendlies"] = [MobSpawnerData("MushroomCow", 8, 4, 8)]
BIOME_SPAWN_LISTS[MUSHROOM_ISLAND_SHORE]["enemies"] = []

# Hell has ghasts, pigzombies, lavaslimes
BIOME_SPAWN_LISTS[HELL]["friendlies"] = []
BIOME_SPAWN_LISTS[HELL]["water_friendlies"] = []
BIOME_SPAWN_LISTS[HELL]["enemies"] = [
    MobSpawnerData("Ghast", 50, 4, 4),
    MobSpawnerData("PigZombie", 100, 4, 4),
    MobSpawnerData("LavaSlime", 1, 4, 4)
]

# Sky (The End) has Endermen
BIOME_SPAWN_LISTS[SKY]["friendlies"] = []
BIOME_SPAWN_LISTS[SKY]["water_friendlies"] = []
BIOME_SPAWN_LISTS[SKY]["enemies"] = [MobSpawnerData("Enderman", 10, 4, 4)]

# Compile the static array into numba
BIOME_DATA = biome_data.copy()

@njit(cache=True, nogil=True)
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

@njit(cache=True, nogil=True)
def get_biome_properties(biome_id):
    """Returns depth, scale, topMaterial, material for the given biome ID"""
    return BIOME_DATA[biome_id, 0], BIOME_DATA[biome_id, 1], int(BIOME_DATA[biome_id, 4]), int(BIOME_DATA[biome_id, 5])
