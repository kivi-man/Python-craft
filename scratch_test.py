import numpy as np
from renderer.mesh_builder import _emit_aabb_faces

verts = np.zeros(15 * 6 * 6, dtype=np.float32)
aabb = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 1.0], dtype=np.float32)
lights = np.ones((16, 256, 16), dtype=np.uint8) * 15
blocks = np.zeros((16, 256, 16), dtype=np.int32)
empty_b = np.zeros((0,0,0), dtype=np.int32)
empty_l = np.zeros((0,0,0), dtype=np.uint8)

o_idx = _emit_aabb_faces.py_func(
    verts, 0, aabb, lights, blocks, empty_b, empty_b, empty_b, empty_b,
    empty_l, empty_l, empty_l, empty_l,
    8, 64, 8,
    8.0, 64.0, 8.0,
    200, 0.6, 0.5, 0.3,
    1.0, 0.0
)

print("LEFT FACE (nx=-1):")
for i in range(0, o_idx, 15):
    if verts[i+3] == -1.0:
        print(f"  Pos: {verts[i:i+3]}, UV: {verts[i+9:i+11]}")

print("FRONT FACE (nz=1):")
for i in range(0, o_idx, 15):
    if verts[i+5] == 1.0:
        print(f"  Pos: {verts[i:i+3]}, UV: {verts[i+9:i+11]}")
