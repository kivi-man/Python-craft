"""
PythonCraft - Kuş Bakışı Dünya Haritası Oluşturucu
===================================================
Tüm chunk'ları sıfırdan generate edip yüksek çözünürlüklü PNG olarak kaydeder.

Kullanım: python generate_map.py
Çıktı:    world_map.png  (proje kök dizininde)
"""

import os
import sys
import time
import math
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────── Proje kökünü path'e ekle ───────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from numba import njit
from world.mc_terrain import load_or_generate_chunk, generate_chunk, CHUNK_SIZE, CHUNK_HEIGHT
from world.terrain import BLOCK_COLORS_ARRAY, BLOCK_OPAQUE_ARRAY
from world.mc_id_converter import BLOCK_TINT_ARRAY

# ─────────────────── AYARLAR ───────────────────────
HALF_SIZE = 512              # Merkezden her yöne chunk sayısı (toplam 1024x1024)
NUM_WORKERS = 32             # Maksimum CPU/Disk kullanımı için yüksek iş parçacığı sayısı

# Sabit blok ID'leri (internal ID'ler, mc_id_converter tarafından atanmış)
_AIR   = 0
_WATER = 9
_ICE   = 79
_LAVA  = 11

# ─────────── RENK PALETİ ───────────────────────────
# BLOCK_COLORS_ARRAY'den kopyala (float32, 0-1 aralığında RGB)
COLORS = BLOCK_COLORS_ARRAY.astype(np.float64).copy()

# Daha güzel bir renk paleti için bazı renkleri override et
COLORS[_AIR]   = [0.47, 0.65, 1.00]   # Gökyüzü mavisi (boş sütunlar için)
COLORS[_WATER] = [0.18, 0.38, 0.72]   # Okyanus mavisi
COLORS[_ICE]   = [0.60, 0.80, 1.00]   # Buz açık mavisi
COLORS[_LAVA]  = [0.95, 0.45, 0.10]   # Lav turuncusu

# Su renk gradyanı
WATER_SHALLOW = np.array([0.25, 0.50, 0.85])
WATER_DEEP    = np.array([0.08, 0.18, 0.42])


# ─────────── HEIGHTMAP ÇIKARICI (Numba JIT) ───────────
@njit(cache=True)
def _extract_heightmap(blocks):
    """
    Chunk bloklarından kuş bakışı heightmap verisini çıkarır.
    Her (x,z) sütunu için en üstteki katı bloğu bulur.
    
    Returns:
        heights    [16,16] int32  - Yüzey yüksekliği
        top_blocks [16,16] int32  - Yüzey blok ID'si 
        water_depths [16,16] int32 - Su derinliği (0 = su yok)
    """
    heights      = np.zeros((16, 16), dtype=np.int32)
    top_blocks   = np.zeros((16, 16), dtype=np.int32)
    water_depths = np.zeros((16, 16), dtype=np.int32)

    for x in range(16):
        for z in range(16):
            w_depth = 0
            found   = False
            for y in range(255, -1, -1):
                b = int(blocks[x, y, z])
                if b == 0:          # AIR → atla
                    continue
                if b == 9:          # WATER → derinlik say
                    w_depth += 1
                    continue
                # Katı blok bulundu
                heights[x, z]      = y
                top_blocks[x, z]   = b
                water_depths[x, z] = w_depth
                found = True
                break

            if not found and w_depth > 0:
                # Tamamen su olan sütun (okyanus tabanı yok veya çok derin)
                heights[x, z]      = 63
                top_blocks[x, z]   = 9   # WATER
                water_depths[x, z] = w_depth

    return heights, top_blocks, water_depths


# ----------- CHUNK ISLEME FONKSIYONU -----------
def _process_chunk(cx, cz):
    """
    Tek bir chunk'i diskten yukle veya sifirdan olustur.
    Heightmap verisini dondur.
    """
    try:
        blocks, _data, _lights, _oob, chunk_biomes, is_new = load_or_generate_chunk(cx, cz)
        source = "gen" if is_new else "disk"

        heights, top_blocks, water_depths = _extract_heightmap(blocks)
        return cx, cz, heights, top_blocks, water_depths, chunk_biomes, source
    except Exception as e:
        return cx, cz, None, None, None, None, f"error: {e}"


# ─────────── RENDER FONKSİYONU ───────────
def _render_image(all_heights, all_blocks, all_water, all_biomes, map_size, colors):
    """
    Numpy vektörel işlemlerle haritayı renklendir.
    Hillshading + su derinliği + dağ vurgusu efektleri uygular.
    """
    print("       Hillshading hesaplaniyor...")

    # -- 1) Hillshading (güneş ışığı gölgelendirmesi) --
    h_float = all_heights.astype(np.float64)

    # Sobel-benzeri gradyan
    dy = np.zeros_like(h_float)
    dx = np.zeros_like(h_float)
    if map_size > 2:
        dy[1:-1, :] = h_float[2:, :] - h_float[:-2, :]
        dx[:, 1:-1] = h_float[:, 2:] - h_float[:, :-2]

    # Güneş yönü: Kuzeybatıdan 45° yükseklik açısı
    azimuth  = np.radians(315)
    altitude = np.radians(45)

    slope  = np.sqrt(dx**2 + dy**2)
    aspect = np.arctan2(-dy, dx)

    zen = np.radians(90) - altitude
    shade = np.cos(zen) + np.sin(zen) * np.cos(azimuth - aspect) * np.minimum(slope / 6.0, 1.0)
    shade = shade * 0.4 + 0.6                      # Kontrast ayari
    shade = np.clip(shade, 0.35, 1.20)

    print("       Blok renkleri ataniyor...")

    # -- 2) Vektorel renk atamasi --
    ids_flat = np.clip(all_blocks.flatten(), 0, 2047).astype(np.intp)
    img = colors[ids_flat].reshape(map_size, map_size, 3).copy()

    print("       Biyom renkleri uygulaniyor...")
    # -- Biyom Renklendirme --
    biome_grass = np.zeros((256, 3), dtype=np.float64)
    biome_grass[:] = [0.57, 0.74, 0.35]
    biome_grass[[2, 17, 35, 36, 130, 37, 38, 39, 165, 166, 167]] = [0.75, 0.72, 0.33] # Desert
    biome_grass[[4, 18, 27, 28, 132, 155, 156]] = [0.47, 0.75, 0.35] # Forest
    biome_grass[[6, 134]] = [0.42, 0.44, 0.22] # Swamp
    biome_grass[[3, 20, 34, 131, 162, 5, 19, 32, 33, 133, 160, 161, 10, 11, 12, 13, 26, 30, 31, 140, 158]] = [0.54, 0.72, 0.53] # Taiga / Snow
    biome_grass[[21, 22, 23, 149, 151]] = [0.35, 0.79, 0.24] # Jungle
    biome_foliage = biome_grass * 0.75

    tints_flat = BLOCK_TINT_ARRAY[ids_flat]
    tints = tints_flat.reshape(map_size, map_size)

    is_grass = tints == 1
    if np.any(is_grass):
        img[is_grass] = biome_grass[np.clip(all_biomes[is_grass], 0, 255)]

    is_foliage = tints == 2
    if np.any(is_foliage):
        img[is_foliage] = biome_foliage[np.clip(all_biomes[is_foliage], 0, 255)]

    print("       Su efektleri uygulaniyor...")

    # -- 3) Su altindaki kati bloklar --
    underwater = (all_water > 0) & (all_blocks != _WATER)
    if np.any(underwater):
        wd = all_water[underwater].astype(np.float64)
        depth_f = np.maximum(0.25, 1.0 - wd * 0.025)
        # Su rengiyle harmanlama
        t = np.minimum(wd / 30.0, 1.0)             # Derinlik interpolasyonu
        for ch in range(3):
            water_c = WATER_SHALLOW[ch] * (1.0 - t) + WATER_DEEP[ch] * t
            c = img[:, :, ch]
            c[underwater] = c[underwater] * 0.20 + water_c * 0.80
            c[underwater] *= depth_f

    # -- 4) Saf su sutunlari --
    pure_water = (all_blocks == _WATER)
    if np.any(pure_water):
        wd = all_water[pure_water].astype(np.float64)
        depth_f = np.maximum(0.18, 1.0 - wd * 0.012)
        t = np.minimum(wd / 40.0, 1.0)
        for ch in range(3):
            c_val = WATER_SHALLOW[ch] * (1.0 - t) + WATER_DEEP[ch] * t
            img[:, :, ch][pure_water] = c_val * depth_f

    print("       Golgelendirme uygulaniyor...")

    # -- 5) Hillshading uygula --
    for ch in range(3):
        img[:, :, ch] *= shade

    # -- 6) Yuksek daglarda hafif parlaklik artisi --
    mountain = (all_heights > 90) & (all_blocks != _WATER) & (all_water == 0)
    if np.any(mountain):
        bonus = 1.0 + (all_heights[mountain].astype(np.float64) - 90.0) * 0.0025
        for ch in range(3):
            img[:, :, ch][mountain] *= bonus

    # -- 7) uint8'e donustur --
    image = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return image


# ----------- ANA FONKSIYON -----------
def main():
    map_size     = HALF_SIZE * 2 * CHUNK_SIZE       # piksel
    total_chunks = (HALF_SIZE * 2) ** 2

    print()
    print("=" * 64)
    print("  PYTHONCRAFT DUNYA HARITASI OLUSTURUCU")
    print("  Kus Bakisi Top-Down Render")
    print(f"  {HALF_SIZE*2}x{HALF_SIZE*2} chunk = {map_size}x{map_size} piksel")
    print(f"  Toplam: {total_chunks:,} chunk")
    print(f"  Is parcacigi: {NUM_WORKERS}")
    print("=" * 64)

    t_global = time.time()

    # ------ 1) Numba JIT Isinma ------
    print("\n[1/4] Numba JIT derleniyor (ilk calistirmada ~30-60sn)...")
    t0 = time.time()
    _wb, _, _, _, _ = generate_chunk(9999, 9999)
    _extract_heightmap(_wb)
    del _wb
    jit_time = time.time() - t0
    print(f"       [OK] JIT derleme tamamlandi ({jit_time:.1f}s)")

    # ------ 2) Bellek Ayir ------
    print("\n[2/4] Bellek ayriliyor...")
    all_heights = np.zeros((map_size, map_size), dtype=np.int32)
    all_blocks  = np.zeros((map_size, map_size), dtype=np.int32)
    all_water   = np.zeros((map_size, map_size), dtype=np.int32)
    all_biomes  = np.zeros((map_size, map_size), dtype=np.int32)
    mem_mb = (all_heights.nbytes + all_blocks.nbytes + all_water.nbytes + all_biomes.nbytes) / (1024 * 1024)
    print(f"       [OK] {mem_mb:.0f} MB ayrildi")

    # ------ 3) Chunk Olusturma ------
    print(f"\n[3/4] Chunk'lar olusturuluyor ({total_chunks:,} adet)...")
    print(f"       Tahmini sure: ~{total_chunks * 0.15 / NUM_WORKERS / 60:.0f}-{total_chunks * 0.35 / NUM_WORKERS / 60:.0f} dakika")
    print()

    gen_start   = time.time()
    processed   = 0
    from_disk   = 0
    from_gen    = 0
    errors      = 0

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        # Chunk'lari merkezden disariya dogru sirala
        chunk_list = []
        for cx in range(-HALF_SIZE, HALF_SIZE):
            for cz in range(-HALF_SIZE, HALF_SIZE):
                chunk_list.append((cx, cz))

        # Tum chunk'lari gonder
        futures = {}
        for cx, cz in chunk_list:
            f = pool.submit(_process_chunk, cx, cz)
            futures[f] = (cx, cz)

        for future in as_completed(futures):
            cx, cz, heights, top_blocks, water_depths, chunk_biomes, source = future.result()
            processed += 1

            if heights is not None:
                col_start = (cx + HALF_SIZE) * 16
                row_start = (cz + HALF_SIZE) * 16
                # heightmap [local_x, local_z] -> image [row=z, col=x]
                all_heights[row_start:row_start+16, col_start:col_start+16] = heights.T
                all_blocks [row_start:row_start+16, col_start:col_start+16] = top_blocks.T
                all_water  [row_start:row_start+16, col_start:col_start+16] = water_depths.T
                if chunk_biomes is not None:
                    all_biomes [row_start:row_start+16, col_start:col_start+16] = chunk_biomes.T

                if source == "disk":
                    from_disk += 1
                else:
                    from_gen += 1
            else:
                errors += 1

            # Ilerleme raporu (her 500 chunk'ta)
            if processed % 500 == 0 or processed == total_chunks:
                elapsed = time.time() - gen_start
                rate = processed / elapsed if elapsed > 0 else 1
                eta = (total_chunks - processed) / rate if rate > 0 else 0
                pct = processed * 100 // total_chunks

                bar_len = 35
                filled = bar_len * processed // total_chunks
                bar = "#" * filled + "-" * (bar_len - filled)

                eta_str = f"{eta/60:.1f}dk" if eta > 60 else f"{eta:.0f}s"
                print(f"\r       [{bar}] {pct:3d}% | {processed:,}/{total_chunks:,} | "
                      f"{rate:.0f} c/s | ETA: {eta_str}  ", end="", flush=True)

    gen_elapsed = time.time() - gen_start
    print()
    print(f"\n       [OK] Tamamlandi: {gen_elapsed/60:.1f} dakika")
    print(f"         Diskten: {from_disk:,} | Olusturulan: {from_gen:,} | Hata: {errors}")

    # ------ 4) Render ------
    print(f"\n[4/4] Harita render ediliyor ({map_size}x{map_size} piksel)...")
    t0 = time.time()
    image_data = _render_image(all_heights, all_blocks, all_water, all_biomes, map_size, COLORS)
    render_time = time.time() - t0
    print(f"       [OK] Render tamamlandi ({render_time:.1f}s)")

    # PNG olarak kaydet
    print("       PNG dosyasi kaydediliyor...")
    img = Image.fromarray(image_data, "RGB")
    output_path = os.path.join(ROOT_DIR, "world_map.png")
    img.save(output_path, "PNG", optimize=False)
    file_size = os.path.getsize(output_path) / (1024 * 1024)

    total_elapsed = time.time() - t_global

    print()
    print("=" * 64)
    print(f"  [OK] Harita basariyla olusturuldu!")
    print(f"  Dosya:      world_map.png")
    print(f"  Cozunurluk: {map_size}x{map_size} piksel")
    print(f"  Boyut:      {file_size:.1f} MB")
    print(f"  Sure:       {total_elapsed/60:.1f} dakika")
    print(f"  Kapsam:     {map_size} x {map_size} blok ({HALF_SIZE*2}x{HALF_SIZE*2} chunk)")
    print("=" * 64)


if __name__ == "__main__":
    main()
