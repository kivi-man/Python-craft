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
        
        # Health and Hunger
        self.health = 20.0
        self.hunger = 20.0
        self.highest_y = y
        self.fall_distance = 0.0
        self.damage_cooldown = 0.0
        self.hunger_timer = 0.0
        self.heal_timer = 0.0
        self.air_supply = 300.0
        self.is_head_in_water = False
        self.tick_count = 0
        
        # Animation attributes for rendering
        self.walk_anim_pos = 0.0
        self.walk_anim_pos_o = 0.0
        self.walk_anim_speed = 0.0
        self.walk_anim_speed_o = 0.0
        
    def take_damage(self, amount):
        if self.damage_cooldown <= 0.0:
            self.health -= amount
            if self.health < 0:
                self.health = 0
            self.damage_cooldown = 0.5 # half a second invulnerability

    def _check_block_intersection(self, get_block_info, aabb, target_block_id):
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        for x in range(ix0, ix1 + 1):
            for y in range(iy0, iy1 + 1):
                for z in range(iz0, iz1 + 1):
                    bid, _ = get_block_info(x, y, z)
                    if bid == target_block_id:
                        return True
        return False
        
    def _get_player_aabb(self, x, y, z):
        """Oyuncunun şu anki sınır kutusunu (MinX, MinY, MinZ, MaxX, MaxY, MaxZ) döndürür."""
        h = 1.5 if self.is_crouching else self.height
        return (
            x - self.radius, y, z - self.radius,
            x + self.radius, y + h, z + self.radius
        )

    def _aabb_intersect(self, a, b):
        return (a[0] < b[3] and a[3] > b[0] and
                a[1] < b[4] and a[4] > b[1] and
                a[2] < b[5] and a[5] > b[2])

    def _check_collision(self, get_block_info, aabb):
        """Verilen AABB'nin dünyadaki herhangi bir katı blokla çarpışıp çarpışmadığını kontrol eder."""
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        
        # Kutuya değen tüm tamsayı blok koordinatlarını tara
        # int(math.floor()) kullanmak negatif koordinatlarda doğru sonuç verir
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        from core.special_blocks import is_stairs, is_slab, get_stair_aabbs, get_slab_aabbs
        
        for x in range(ix0, ix1 + 1):
            for y in range(iy0, iy1 + 1):
                for z in range(iz0, iz1 + 1):
                    block_id, data = get_block_info(x, y, z)
                    if block_id == -1 or (block_id > 0 and block_id != 4):  # 0: Air, 4: Water
                        if is_stairs(block_id):
                            f_id, f_data = get_block_info(x - 1, y, z)
                            b_id, b_data = get_block_info(x + 1, y, z)
                            l_id, l_data = get_block_info(x, y, z - 1)
                            r_id, r_data = get_block_info(x, y, z + 1)
                            aabbs = get_stair_aabbs(x, y, z, data, f_id, f_data, b_id, b_data, l_id, l_data, r_id, r_data)
                            for i in range(3):
                                ba = aabbs[i]
                                if ba[3] > ba[0]: # Not empty
                                    if self._aabb_intersect(aabb, ba): return True
                        elif is_slab(block_id):
                            ba = get_slab_aabbs(x, y, z, data)[0]
                            if self._aabb_intersect(aabb, ba): return True
                        else:
                            return True
        return False

    def _check_in_water(self, get_block_info, aabb):
        """AABB içinde su (4) bloğu olup olmadığını kontrol eder."""
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        for x in range(ix0, ix1 + 1):
            for y in range(iy0, iy1 + 1):
                for z in range(iz0, iz1 + 1):
                    block_id, _ = get_block_info(x, y, z)
                    if block_id == 4:
                        return True
        return False
    def _push_out_of_blocks(self, get_block_info):
        """Eğer oyuncu bir bloğun içine sıkışmışsa, onu en yakın boşluğa doğru iter."""
        # Oyuncunun AABB'sini al
        aabb = self._get_player_aabb(self.x, self.y, self.z)
        min_x, min_y, min_z, max_x, max_y, max_z = aabb
        
        # Sıkışma kontrolü: AABB'nin altını hafifçe yukarı çekiyoruz (yerle teması sıkışma saymasın diye)
        ix0, ix1 = int(math.floor(min_x)), int(math.floor(max_x))
        iy0, iy1 = int(math.floor(min_y + 0.01)), int(math.floor(max_y))
        iz0, iz1 = int(math.floor(min_z)), int(math.floor(max_z))
        
        from core.special_blocks import is_stairs, is_slab, get_stair_aabbs, get_slab_aabbs
        
        stuck_blocks = []
        for bx in range(ix0, ix1 + 1):
            for by in range(iy0, iy1 + 1):
                for bz in range(iz0, iz1 + 1):
                    block_id, data = get_block_info(bx, by, bz)
                    if block_id == -1 or (block_id > 0 and block_id != 4):
                        # Special check
                        if is_stairs(block_id) or is_slab(block_id):
                            pass # Push out doesn't easily handle complex AABBs correctly, we just push out from full blocks for simplicity
                        else:
                            stuck_blocks.append((bx, by, bz))
        
        if not stuck_blocks:
            return
            
        # Sıkışılan her bir blok için oyuncuyu en yakın kenara doğru it
        for bx, by, bz in stuck_blocks:
            b_min_x, b_max_x = bx, bx + 1.0
            b_min_y, b_max_y = by, by + 1.0
            b_min_z, b_max_z = bz, bz + 1.0
            
            dist_left = self.x - b_min_x
            dist_right = b_max_x - self.x
            dist_back = self.z - b_min_z
            dist_front = b_max_z - self.z
            dist_up = b_max_y - self.y
            
            min_dist = min(dist_left, dist_right, dist_back, dist_front, dist_up)
            
            push_speed = 0.1
            if min_dist == dist_up:
                if get_block_info(bx, by + 1, bz)[0] in (0, 4):
                    self.y += push_speed
            elif min_dist == dist_left:
                if get_block_info(bx - 1, by, bz)[0] in (0, 4):
                    self.x -= push_speed
                else:
                    self.y += push_speed
            elif min_dist == dist_right:
                if get_block_info(bx + 1, by, bz)[0] in (0, 4):
                    self.x += push_speed
                else:
                    self.y += push_speed
            elif min_dist == dist_back:
                if get_block_info(bx, by, bz - 1)[0] in (0, 4):
                    self.z -= push_speed
                else:
                    self.y += push_speed
            else:
                if get_block_info(bx, by, bz + 1)[0] in (0, 4):
                    self.z += push_speed
                else:
                    self.y += push_speed

    def update(self, dt, dx, dz, jump, crouch, sprint, get_block_info):
        """
        Her karede oyuncunun fiziğini günceller.
        dx, dz: Klavyeden gelen yön vektörü (Normalize edilmiş)
        get_block_info: Dünyadan blok okuma fonksiyonu callback'i (block_id, data)
        """
        if self.damage_cooldown > 0:
            self.damage_cooldown -= dt
            
        # Hunger system
        self.hunger_timer += dt
        
        self.tick_count += 1
        
        start_x = self.x
        start_z = self.z
        if self.hunger_timer >= 15.0:  # Lose 1 hunger every 15 seconds
            self.hunger_timer = 0.0
            if self.hunger > 0:
                self.hunger -= 1.0
            else:
                self.take_damage(1.0) # Starving damage
                
        # Natural Regeneration
        if self.hunger >= 18.0 and self.health < 20.0:
            self.heal_timer += dt
            if self.heal_timer >= 4.0: # Heal 1 hp every 4 seconds
                self.health += 1.0
                if self.health > 20.0: self.health = 20.0
                self.heal_timer = 0.0
        else:
            self.heal_timer = 0.0
                
        if self.hunger < 7.0: # 3.5 bars
            sprint = False

        # Head in water / Drowning logic
        head_y = self.y + 1.62
        head_block, _ = get_block_info(int(math.floor(self.x)), int(math.floor(head_y)), int(math.floor(self.z)))
        self.is_head_in_water = (head_block == 4)
        
        if self.is_head_in_water:
            self.air_supply -= dt * 20.0
            if self.air_supply <= -20.0:
                self.take_damage(2.0)  # Drown damage (1 heart)
                self.air_supply = 0.0
        else:
            self.air_supply += dt * 20.0 * 2  # Recover quickly
            if self.air_supply > 300.0:
                self.air_supply = 300.0

        # Sıkışma kurtarma mekanizmasını çalıştır
        self._push_out_of_blocks(get_block_info)
        
        self.is_crouching = crouch
        self.is_sprinting = sprint
        
        # Buffer jump input to prevent missed short keypresses between physics ticks
        self.jump_buffer = getattr(self, 'jump_buffer', False) or jump
        
        # Su durumu kontrolü
        # C++: bb->grow(0, -0.4f, 0) — AABB'yi Y'de 0.4 küçültür
        # Bu sayede oyuncu yüzeye çıkınca isInWater=false olur → impulse kesilir → geri batar → bobbing/çırpınma!
        water_aabb = self._get_player_aabb(self.x, self.y, self.z)
        shrunk_aabb = (water_aabb[0], water_aabb[1], water_aabb[2],
                       water_aabb[3], water_aabb[4] - 0.4, water_aabb[5])
        self.in_water = self._check_in_water(get_block_info, shrunk_aabb)
        
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
            
            # Consume buffered jump for this physics tick
            active_jump = self.jump_buffer
            self.jump_buffer = jump # Keep it true if key is still held, otherwise clear it
            
            if self.in_water:
                # Su fiziği
                if active_jump:
                    vy_tick += 0.04        # PlayerMovementCodes: yd += 0.04f
                vy_tick *= 0.8             # C++ travel(): yd *= 0.8
                vy_tick -= 0.02            # C++ travel(): yd -= 0.02
            else:
                if self.is_flying:
                    # Uçma fiziği
                    if active_jump:
                        vy_tick = speed * 0.05
                    elif crouch:
                        vy_tick = -speed * 0.05
                    else:
                        vy_tick = 0.0
                else:
                    # Hava fiziği
                    if active_jump and self.on_ground:
                        # 0.51 değeri, PythonCraft'ta yerçekimi hareketi uygulanmadan önce 
                        # hesaplandığı için C++'taki ilk kare 0.42 zıplama hızına denktir. 
                        # ((0.51 - 0.08) * 0.98 = ~0.42)
                        vy_tick = 0.51
                        self.on_ground = False
                        
                        # PlayerMovementCodes: isSprinting() boost
                        if self.is_sprinting:
                            # 0.2 blocks per tick = 4.0 blocks per second
                            self.vx += dx * 4.0
                            self.vz += dz * 4.0
                    
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
        dx = self.vx * dt
        if self.is_crouching and self.on_ground and not self.is_flying:
            # Shift'teyken kenardan düşmemek için hareketi sınırla
            while dx != 0.0:
                support_aabb = (
                    self.x + dx - self.radius, self.y - 0.5, self.z - self.radius,
                    self.x + dx + self.radius, self.y - 0.01, self.z + self.radius
                )
                if self._check_collision(get_block_info, support_aabb):
                    break
                if abs(dx) <= 0.002:
                    dx = 0.0
                elif dx > 0.0:
                    dx = max(0.0, dx - 0.002)
                else:
                    dx = min(0.0, dx + 0.002)
            if dx == 0.0:
                self.vx = 0.0
                
        new_x = self.x + dx
        check_x = False
        if not self._check_collision(get_block_info, self._get_player_aabb(new_x, self.y, self.z)):
            self.x = new_x
            check_x = True
        else:
            self.vx = 0.0
            
        # Z Ekseninde Hareket
        dz = self.vz * dt
        if self.is_crouching and self.on_ground and not self.is_flying:
            # Shift'teyken kenardan düşmemek için hareketi sınırla
            while dz != 0.0:
                support_aabb = (
                    self.x - self.radius, self.y - 0.5, self.z + dz - self.radius,
                    self.x + self.radius, self.y - 0.01, self.z + dz + self.radius
                )
                if self._check_collision(get_block_info, support_aabb):
                    break
                if abs(dz) <= 0.002:
                    dz = 0.0
                elif dz > 0.0:
                    dz = max(0.0, dz - 0.002)
                else:
                    dz = min(0.0, dz + 0.002)
            if dz == 0.0:
                self.vz = 0.0
                
        new_z = self.z + dz
        check_z = False
        if not self._check_collision(get_block_info, self._get_player_aabb(self.x, self.y, new_z)):
            self.z = new_z
            check_z = True
        else:
            self.vz = 0.0
            
        # Check for stepping up half blocks (Stairs/Slabs)
        step_height = 0.6
        if (not check_x and dx != 0) or (not check_z and dz != 0):
            # We hit something horizontally. Let's see if we can step up.
            if self.on_ground and not self.is_flying:
                # Try stepping up by moving Y up by step_height, then X/Z, then check collision
                step_aabb_y = self._get_player_aabb(self.x, self.y + step_height, self.z)
                if not self._check_collision(get_block_info, step_aabb_y):
                    # We can move up. Now try moving X/Z.
                    can_step_x = False
                    can_step_z = False
                    if not check_x and dx != 0:
                        step_aabb_x = self._get_player_aabb(self.x + dx, self.y + step_height, self.z)
                        if not self._check_collision(get_block_info, step_aabb_x):
                            can_step_x = True
                    if not check_z and dz != 0:
                        step_aabb_z = self._get_player_aabb(self.x, self.y + step_height, self.z + dz)
                        if not self._check_collision(get_block_info, step_aabb_z):
                            can_step_z = True
                            
                    if can_step_x or can_step_z:
                        if can_step_x:
                            self.x += dx
                        if can_step_z:
                            self.z += dz
                        self.y += step_height
                        # Adjust velocity so we don't lose momentum
                        if can_step_x: self.vx = dx / dt
                        if can_step_z: self.vz = dz / dt
            
        # Y Ekseninde Hareket (Düşme / Zıplama / Uçma)
        new_y = self.y + self.vy * dt
        if not self._check_collision(get_block_info, self._get_player_aabb(self.x, new_y, self.z)) or self.is_flying:
            self.y = new_y
            if not self.is_flying:
                if not self._check_collision(get_block_info, self._get_player_aabb(self.x, self.y - 0.01, self.z)):
                    self.on_ground = False
        else:
            # Yere değdik veya kafayı tavana vurduk
            if self.vy < 0:
                # Binary search exactly where the ground is to prevent floating/sliding bugs
                low = new_y
                high = self.y
                for _ in range(12):
                    mid = (low + high) / 2.0
                    if self._check_collision(get_block_info, self._get_player_aabb(self.x, mid, self.z)):
                        low = mid
                    else:
                        high = mid
                self.y = high
                self.on_ground = True
                
                # Calculate fall damage
                fall_dist = self.highest_y - self.y
                if fall_dist > 3.0:
                    damage = math.floor(fall_dist - 3.0)
                    if damage > 0:
                        self.take_damage(damage)
            else:
                # Kafayı tavana vurduk, binary search ile tavana daya
                low = self.y
                high = new_y
                for _ in range(12):
                    mid = (low + high) / 2.0
                    if self._check_collision(get_block_info, self._get_player_aabb(self.x, mid, self.z)):
                        high = mid
                    else:
                        low = mid
                self.y = low
                
            self.vy = 0.0
            
        # Update highest_y for fall damage tracking
        if self.on_ground or self.in_water:
            self.highest_y = self.y
        else:
            if self.y > self.highest_y:
                self.highest_y = self.y

        # Check cactus collision
        if self._check_block_intersection(get_block_info, self._get_player_aabb(self.x, self.y, self.z), 13): # 13 is CACTUS
            self.take_damage(1.0)
            
        # Update walk animation speeds
        self.walk_anim_pos_o = self.walk_anim_pos
        self.walk_anim_speed_o = self.walk_anim_speed
        
        actual_dx = self.x - start_x
        actual_dz = self.z - start_z
        
        dist_moved = math.sqrt(actual_dx * actual_dx + actual_dz * actual_dz)
        # target_speed determines how far the legs swing
        # Make sprint animation much wider (amplitude)
        speed_mult = 1.8 if getattr(self, 'is_sprinting', False) else 1.0
        target_speed = min(dist_moved * 4.0 * speed_mult, 1.0)
        self.walk_anim_speed += (target_speed - self.walk_anim_speed) * 0.4
        
        # position accumulates based on actual distance moved, so legs don't spin wildly
        # Make sprint animation cycle slightly faster 
        pos_mult = 3.5 if getattr(self, 'is_sprinting', False) else 2.5
        self.walk_anim_pos += dist_moved * pos_mult
            
    def get_eye_position(self):
        """Kameranın olması gereken yer (göz hizası)."""
        eye_height = 1.3 if self.is_crouching else 1.6
        return [self.x, self.y + eye_height, self.z]
