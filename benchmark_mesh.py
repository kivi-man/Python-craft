"""Greedy Meshing Benchmark — Numba JIT Warmup Included"""
import time
import numpy as np
from world.terrain import generate_chunk
from renderer.mesh_builder import build_chunk_mesh

# 1. Chunk Üret (Numba Warmup Terrain)
print("Warming up Terrain JIT...")
c = generate_chunk(0, 0)
world_chunks = {(0, 0): c}

# 2. Mesh Üret (Numba Warmup Mesher)
print("Warming up Mesher JIT...")
build_chunk_mesh(c, 0, 0, world_chunks)

print("=" * 55)
print("  GREEDY MESHING BENCHMARK (JIT WARMED UP)")
print("=" * 55)

# 64 chunk üret (8x8)
world_chunks = {}
for cx in range(-4, 4):
    for cz in range(-4, 4):
        world_chunks[(cx, cz)] = generate_chunk(cx, cz)

total_greedy = 0

t = time.perf_counter()
for (cx, cz), blocks in world_chunks.items():
    mesh = build_chunk_mesh(blocks, cx, cz, world_chunks)
    greedy = len(mesh) // 9
    total_greedy += greedy
elapsed = time.perf_counter() - t

print(f"  Greedy Meshing:         {total_greedy:>10,} vertex")
print(f"  64 chunk mesh süresi:   {elapsed:.5f}s ({(elapsed/64)*1000:.2f}ms per chunk)")
print("\n" + "=" * 55)
