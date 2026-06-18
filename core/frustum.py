"""
PythonCraft - Frustum Culling (Numba JIT Optimized)
Kameranın görüş açısını hesaplar ve görünmeyen nesneleri ışık hızında eler.
"""
import numpy as np
from numba import njit

@njit(cache=True)
def extract_planes(clip):
    planes = np.zeros((6, 4), dtype=np.float32)
    # Sağ
    planes[0] = [clip[3] - clip[0], clip[7] - clip[4], clip[11] - clip[8], clip[15] - clip[12]]
    # Sol
    planes[1] = [clip[3] + clip[0], clip[7] + clip[4], clip[11] + clip[8], clip[15] + clip[12]]
    # Alt
    planes[2] = [clip[3] + clip[1], clip[7] + clip[5], clip[11] + clip[9], clip[15] + clip[13]]
    # Üst
    planes[3] = [clip[3] - clip[1], clip[7] - clip[5], clip[11] - clip[9], clip[15] - clip[13]]
    # Arka
    planes[4] = [clip[3] - clip[2], clip[7] - clip[6], clip[11] - clip[10], clip[15] - clip[14]]
    # Ön
    planes[5] = [clip[3] + clip[2], clip[7] + clip[6], clip[11] + clip[10], clip[15] + clip[14]]
    
    # Normalize
    for i in range(6):
        mag = np.sqrt(planes[i][0]**2 + planes[i][1]**2 + planes[i][2]**2)
        if mag > 0:
            planes[i] /= mag
            
    return planes

@njit(cache=True)
def get_visible_chunk_indices(proj_view, chunk_bounds, active_flags, out_indices):
    """
    Tüm chunk'ları Frustum testinden geçirir. Görünür ve aktif olanların indekslerini out_indices içine yazar.
    Döndürür: Görünür chunk sayısı (int)
    """
    planes = extract_planes(proj_view)
    visible_count = 0
    
    n_chunks = chunk_bounds.shape[0]
    for i in range(n_chunks):
        if not active_flags[i]:
            continue
            
        # AABB: min_x, min_y, min_z, max_x, max_y, max_z
        min_x, min_y, min_z, max_x, max_y, max_z = chunk_bounds[i]
        
        is_visible = True
        for p in range(6):
            px = max_x if planes[p][0] > 0 else min_x
            py = max_y if planes[p][1] > 0 else min_y
            pz = max_z if planes[p][2] > 0 else min_z
            
            if (planes[p][0]*px + planes[p][1]*py + planes[p][2]*pz + planes[p][3]) < -0.1:
                is_visible = False
                break
                
        if is_visible:
            out_indices[visible_count] = i
            visible_count += 1
            
    return visible_count
