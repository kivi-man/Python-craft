"""
PythonCraft - Player Physics and Collision Controller
Oyuncu fiziğini (yerçekimi, zıplama, eğilme) ve blok çarpışmalarını (AABB) yönetir.
"""
import math

class Player:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        
        # Fizik Sabitleri (Bedrock C++ Birebir Uyarlama)
        self.gravity = -32.0
        self.jump_force = 8.95
        self.walk_speed = 4.405
        self.sprint_speed = 5.726
        self.crouch_speed = 1.321
        self.terminal_velocity = -80.0
        
        # Oyuncu Durumu
        self.on_ground = False
        self.is_crouching = False
        self.is_sprinting = False
        self.is_flying = False
        self.in_water = False
        self._y_tick_accum = 0.0  # Unified tick accumulator for Y physics
        
        # AABB Çarpışma Kutusu (Player Boyutları)
        # Genişlik: 0.6 blok (yarıçap 0.3)
        # Yükseklik: 1.8 blok
        self.radius = 0.3
        self.height = 1.8
        
    def _get_player_aabb(self, x, y, z):
        """Oyuncunun şu anki sınır kutusunu (MinX, MinY, MinZ, MaxX, MaxY, MaxZ) döndürür."""
        h = 1.5 if self.is_crouching else self.height
        return (
            x - self.radius, y, z - self.radius,
            x + self.radius, y + h, z + self.radius
        )

    def _check_collision(self, get_block, aabb):
        """Verilen AABB'nin dünyadaki herhangi bir katı blokla çarpışıp çarpışmadığını kontrol eder."""
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        
        # Kutuya değen tüm tamsayı blok koordinatlarını tara
        # int(math.floor()) kullanmak negatif koordinatlarda doğru sonuç verir
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        for x in range(ix0, ix1 + 1):
            for y in range(iy0, iy1 + 1):
                for z in range(iz0, iz1 + 1):
                    block_id = get_block(x, y, z)
                    if block_id > 0 and block_id != 4:  # 0: Air, 4: Water
                        return True
        return False

    def _check_in_water(self, get_block, aabb):
        """AABB içinde su (4) bloğu olup olmadığını kontrol eder."""
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        for x in range(ix0, ix1 + 1):
            for y in range(iy0, iy1 + 1):
                for z in range(iz0, iz1 + 1):
                    block_id = get_block(x, y, z)
                    if block_id == 4:
                        return True
        return False

    def update(self, dt, dx, dz, jump, crouch, sprint, get_block):
        """
        Her karede oyuncunun fiziğini günceller.
        dx, dz: Klavyeden gelen yön vektörü (Normalize edilmiş)
        get_block: Dünyadan blok okuma fonksiyonu callback'i
        """
        self.is_crouching = crouch
        self.is_sprinting = sprint
        
        # Su durumu kontrolü
        # C++: bb->grow(0, -0.4f, 0) — AABB'yi Y'de 0.4 küçültür
        # Bu sayede oyuncu yüzeye çıkınca isInWater=false olur → impulse kesilir → geri batar → bobbing/çırpınma!
        water_aabb = self._get_player_aabb(self.x, self.y, self.z)
        shrunk_aabb = (water_aabb[0], water_aabb[1] + 0.4, water_aabb[2],
                       water_aabb[3], water_aabb[4] - 0.4, water_aabb[5])
        self.in_water = self._check_in_water(get_block, shrunk_aabb)
        
        if self.in_water:
            speed = 2.0
            control = 4.46
        else:
            speed = self.walk_speed
            if self.is_crouching:
                speed = self.crouch_speed
            elif self.is_sprinting:
                speed = self.sprint_speed
            control = 12.1 if self.on_ground else 1.89

        # Y Ekseni: Tamamen Tick-Bazlı C++ Simülasyonu (20 TPS)
        # Su ve hava geçişlerinde ivme birikmemesi için ortak sayaç kullanılır.
        self._y_tick_accum += dt
        while self._y_tick_accum >= 0.05:
            self._y_tick_accum -= 0.05
            vy_tick = self.vy * 0.05  # b/s → b/tick
            
            if self.in_water:
                # Su fiziği
                if jump:
                    vy_tick += 0.04        # C++ aiStep(): yd += 0.04
                vy_tick *= 0.8             # C++ travel(): yd *= 0.8
                vy_tick -= 0.02            # C++ travel(): yd -= 0.02
            else:
                if self.is_flying:
                    # Uçma fiziği
                    if jump:
                        vy_tick = speed * 0.05
                    elif crouch:
                        vy_tick = -speed * 0.05
                    else:
                        vy_tick = 0.0
                else:
                    # Hava fiziği
                    if jump and self.on_ground:
                        vy_tick = 0.42     # C++ jumpFromGround()
                        self.on_ground = False
                    
                    vy_tick -= 0.08        # C++ travel() yerçekimi
                    vy_tick *= 0.98        # C++ travel() drag
            
            self.vy = vy_tick * 20.0       # b/tick → b/s
            
            # Terminal velocity koruması
            if self.vy < self.terminal_velocity:
                self.vy = self.terminal_velocity
        
        target_vx = dx * speed
        target_vz = dz * speed
        
        self.vx += (target_vx - self.vx) * control * dt
        self.vz += (target_vz - self.vz) * control * dt
            
        # Çarpışma Testi ve Hareket Uygulama (Sweep Test)
        # Her ekseni ayrı ayrı test ediyoruz (Slide effect için)
        
        # X Ekseninde Hareket
        new_x = self.x + self.vx * dt
        if not self._check_collision(get_block, self._get_player_aabb(new_x, self.y, self.z)):
            self.x = new_x
        else:
            self.vx = 0.0
            
        # Z Ekseninde Hareket
        new_z = self.z + self.vz * dt
        if not self._check_collision(get_block, self._get_player_aabb(self.x, self.y, new_z)):
            self.z = new_z
        else:
            self.vz = 0.0
            
        # Y Ekseninde Hareket (Düşme / Zıplama / Uçma)
        new_y = self.y + self.vy * dt
        if not self._check_collision(get_block, self._get_player_aabb(self.x, new_y, self.z)) or self.is_flying:
            self.y = new_y
            if not self.is_flying:
                self.on_ground = False
        else:
            # Yere değdik veya kafayı tavana vurduk
            if self.vy < 0:
                self.on_ground = True
                # Yere tam oturt
                self.y = float(math.floor(self.y))
            self.vy = 0.0
            
    def get_eye_position(self):
        """Kameranın olması gereken yer (göz hizası)."""
        eye_height = 1.3 if self.is_crouching else 1.6
        return [self.x, self.y + eye_height, self.z]
