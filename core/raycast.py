"""
PythonCraft - Raycasting
Kameranın baktığı yöne doğru ışın gönderip ilk çarpılan katı bloğu bulur.
Basit bir step algoritması kullanır (Voxel Traversal için yeterince hızlıdır).
"""
import math

def raycast(start_pos, direction, get_block_func, max_distance=5.0, step_size=0.05):
    """
    start_pos: (x, y, z) Işının çıkış noktası (Kamera pozisyonu)
    direction: (dx, dy, dz) Normalize edilmiş bakış yönü vektörü
    get_block_func: Verilen (x, y, z) koordinatındaki block ID'sini döndüren fonksiyon
    max_distance: Maksimum kırma/erişim mesafesi (blok cinsinden)
    step_size: Işının her adımda ne kadar ilerleyeceği
    
    Döndürür:
        (hit_x, hit_y, hit_z, prev_x, prev_y, prev_z) eğer bir bloğa çarparsa.
        (None, None, None, None, None, None) çarpmazsa.
        prev_* değerleri bloğun hemen önündeki hava bloğudur (blok koymak için kullanılır).
    """
    x, y, z = start_pos
    dx, dy, dz = direction
    
    distance_traveled = 0.0
    
    # Işının bir önceki adımında bulunduğu tamsayı koordinatları
    prev_bx, prev_by, prev_bz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
    
    while distance_traveled < max_distance:
        x += dx * step_size
        y += dy * step_size
        z += dz * step_size
        distance_traveled += step_size
        
        bx = int(math.floor(x))
        by = int(math.floor(y))
        bz = int(math.floor(z))
        
        # Bir blok sınırı geçildiyse kontrol et
        if bx != prev_bx or by != prev_by or bz != prev_bz:
            block_id = get_block_func(bx, by, bz)
            
            # 0: AIR, 4: WATER (Suyu veya havayı kıramazsınız)
            if block_id > 0 and block_id != 4:
                return (bx, by, bz, prev_bx, prev_by, prev_bz)
                
            prev_bx, prev_by, prev_bz = bx, by, bz
            
    return (None, None, None, None, None, None)
