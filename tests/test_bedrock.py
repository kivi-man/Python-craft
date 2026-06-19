import numpy as np
from world.mc_terrain import generate_chunk

blocks, lights, decorations, biomes = generate_chunk(0, 0)
total_stone = np.sum(blocks == 1)
total_dirt = np.sum(blocks == 2)
total_grass = np.sum(blocks == 3)
total_air = np.sum(blocks == 0)

print(f"Total Stone: {total_stone}")
print(f"Total Dirt: {total_dirt}")
print(f"Total Grass: {total_grass}")
print(f"Total Air: {total_air}")

# Check y slices
for y in range(0, 128, 16):
    slice_blocks = blocks[:, y, :]
    print(f"Y={y}: Air={np.sum(slice_blocks==0)}, Solid={np.sum(slice_blocks>0)}")
