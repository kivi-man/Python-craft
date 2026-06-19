import numpy as np
from world.mc_terrain import get_noise_buffer, _LPERLIN1, _LPERLIN2, _PERLIN1, _DEPTH, _SCALE

buffer = get_noise_buffer(0, 0, _LPERLIN1, _LPERLIN2, _PERLIN1, _DEPTH, _SCALE)

# Buffer is size 5 x 17 x 5
# Print a few values for yy=0, yy=8, yy=16
buffer_3d = buffer.reshape((5, 5, 17)) # Wait, layout is xx, zz, yy
# Flat index: (xx * zSize + zz) * ySize + yy

print("YY = 0:")
print(buffer_3d[0, 0, 0])
print("YY = 8:")
print(buffer_3d[0, 0, 8])
print("YY = 16:")
print(buffer_3d[0, 0, 16])

for yy in range(17):
    print(f"YY={yy}, Val={buffer_3d[0, 0, yy]}")
