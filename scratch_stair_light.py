import numpy as np
from world.mc_terrain import CHUNK_SIZE, CHUNK_HEIGHT
from renderer.mesh_builder import build_chunk_mesh_bg
from world.mc_biomes import BIOME_DATA

# Create a dummy chunk
cx, cz = 0, 0
blocks = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint16)
data = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
light_map = np.zeros((CHUNK_SIZE, CHUNK_HEIGHT, CHUNK_SIZE), dtype=np.uint8)

# Place a grass block at y=63
blocks[:, 0:64, :] = 3 # Dirt
blocks[:, 64, :] = 2 # Grass

# Place a stair at x=5, y=65, z=5
stair_x, stair_y, stair_z = 5, 65, 5
blocks[stair_x, stair_y, stair_z] = 200 # Oak stairs
data[stair_x, stair_y, stair_z] = 1 # DIR_EAST

# Update light map (simple version for test)
for x in range(CHUNK_SIZE):
    for z in range(CHUNK_SIZE):
        light_map[x, 65:, z] = 15

from world.terrain import BLOCK_LAYERS_ARRAY, BLOCK_OVERLAYS_ARRAY

# Call mesh builder
verts, t_verts = build_chunk_mesh_bg(
    cx, cz, blocks, data, light_map,
    blocks, blocks, blocks, blocks,
    data, data, data, data,
    light_map, light_map, light_map, light_map,
    blocks, blocks, blocks, blocks,
    BIOME_DATA, BLOCK_LAYERS_ARRAY, BLOCK_OVERLAYS_ARRAY
)

# Find the vertices for the stair
print(f"Total vertices: {len(verts)//15}")
for i in range(0, len(verts), 15):
    vx, vy, vz = verts[i:i+3]
    if vx >= stair_x and vx <= stair_x+1 and vy >= stair_y and vy <= stair_y+1 and vz >= stair_z and vz <= stair_z+1:
        nx, ny, nz = verts[i+3:i+6]
        vLight = verts[i+13]
        print(f"Stair Vertex: pos=({vx},{vy},{vz}) normal=({nx},{ny},{nz}) vLight={vLight}")
