"""Numba JIT Terrain Benchmark"""
import time
from world.terrain import generate_chunk

print("=" * 50)
print("  NUMBA JIT TERRAIN BENCHMARK")
print("=" * 50)

# İlk çağrı — Numba derleme süresi dahil
print("\n[1] İlk çağrı (Numba derleme dahil)...")
t = time.perf_counter()
generate_chunk(0, 0)
print(f"    Süre: {time.perf_counter() - t:.3f}s (derleme dahil, tek seferlik)")

# Gerçek hız testi — 64 chunk üret
print("\n[2] 64 chunk üretimi (8x8 alan)...")
t = time.perf_counter()
for cx in range(-4, 4):
    for cz in range(-4, 4):
        generate_chunk(cx, cz)
elapsed = time.perf_counter() - t
per_chunk = elapsed / 64 * 1000
print(f"    Toplam: {elapsed:.3f}s")
print(f"    Chunk başına: {per_chunk:.2f}ms")
print(f"    Saniyede: {64/elapsed:.0f} chunk/s")

print("\n" + "=" * 50)
print("  BENCHMARK TAMAMLANDI")
print("=" * 50)
