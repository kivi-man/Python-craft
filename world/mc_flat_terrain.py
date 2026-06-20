import numpy as np
from world.terrain import AIR, BEDROCK, DIRT, GRASS
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT, _calc_light_jit
from core.world_db import load_chunk
from world.mc_biomes import PLAINS

# Cache for the flat chunk data
_FLAT_BLOCKS_CACHE = None
_FLAT_LIGHT_CACHE = None
_FLAT_BIOMES_CACHE = None

def _get_or_create_flat_cache():
    global _FLAT_BLOCKS_CACHE, _FLAT_LIGHT_CACHE, _FLAT_BIOMES_CACHE
    if _FLAT_BLOCKS_CACHE is None:
        # Generate base blocks instantly using numpy broadcasting
        blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        blocks[:, 0, :] = BEDROCK
        blocks[:, 1, :] = DIRT
        blocks[:, 2, :] = GRASS
        
        # Calculate lighting once
        light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        _calc_light_jit(blocks, light_map)
        
        # Create biome map once
        biomes = np.full((CHUNK_SIZE, CHUNK_SIZE), PLAINS, dtype=np.int32)
        
        _FLAT_BLOCKS_CACHE = blocks
        _FLAT_LIGHT_CACHE = light_map
        _FLAT_BIOMES_CACHE = biomes
        
    return _FLAT_BLOCKS_CACHE, _FLAT_LIGHT_CACHE, _FLAT_BIOMES_CACHE

def generate_flat_chunk(cx, cz):
    blocks, light_map, chunk_biomes = _get_or_create_flat_cache()
    
    # Return copies so modifications to one chunk don't affect others
    out_of_bounds = np.zeros((0, 4), dtype=np.int32)
    return blocks.copy(), light_map.copy(), out_of_bounds, chunk_biomes.copy()

def load_or_generate_flat_chunk(cx, cz):
    data = load_chunk(cx, cz)
    if data is not None:
        chunk_biomes = np.full((CHUNK_SIZE, CHUNK_SIZE), PLAINS, dtype=np.int32)
        return data[0], data[1], np.zeros((0, 4), dtype=np.int32), chunk_biomes
    return generate_flat_chunk(cx, cz)
