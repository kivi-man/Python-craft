import os
import sys
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from core.world_db import save_chunk, _get_region_file

blocks = np.zeros((16, 256, 16), dtype=np.uint8)
lights = np.zeros((16, 256, 16), dtype=np.uint8)
blocks[0, 64, 0] = 1

save_chunk(0, 0, blocks, lights)
print("Save complete.")

rx = 0
rz = 0
filename = os.path.join("world_data", "region", f"r.{rx}.{rz}.mca")
if os.path.exists(filename):
    print("File size:", os.path.getsize(filename))
else:
    print("File not found")
