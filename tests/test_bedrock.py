import numpy as np
from world.mc_terrain import generate_chunk

def test_chunk_generation():
    blocks, lights, decorations, biomes = generate_chunk(0, 0)
    
    total_stone = np.sum(blocks == 1)
    total_dirt = np.sum(blocks == 2)
    total_grass = np.sum(blocks == 3)
    total_air = np.sum(blocks == 0)

    print(f"Total Stone: {total_stone}")
    print(f"Total Dirt: {total_dirt}")
    print(f"Total Grass: {total_grass}")
    print(f"Total Air: {total_air}")

    # Basic assertions to ensure chunk generation works
    assert blocks.shape == (16, 64, 16), "Chunk shape is incorrect"
    assert total_air > 0, "Chunk should have some air"
    assert total_stone > 0, "Chunk should have some stone"
    
    # Check y slices (up to 64, which is CHUNK_HEIGHT)
    for y in range(0, 64, 16):
        slice_blocks = blocks[:, y, :]
        print(f"Y={y}: Air={np.sum(slice_blocks==0)}, Solid={np.sum(slice_blocks>0)}")
