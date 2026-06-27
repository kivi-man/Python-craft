from world.terrain import CACTUS
import math

class AABB:
    def __init__(self, min_x, min_y, min_z, max_x, max_y, max_z):
        self.min_x = min_x
        self.min_y = min_y
        self.min_z = min_z
        self.max_x = max_x
        self.max_y = max_y
        self.max_z = max_z

    def intersects(self, other):
        return (self.min_x < other.max_x and self.max_x > other.min_x and
                self.min_y < other.max_y and self.max_y > other.min_y and
                self.min_z < other.max_z and self.max_z > other.min_z)

    def clip_x_collide(self, c, xa):
        epsilon = 1e-5
        if c.max_y <= self.min_y + epsilon or c.min_y >= self.max_y - epsilon: return xa
        if c.max_z <= self.min_z + epsilon or c.min_z >= self.max_z - epsilon: return xa
        if xa > 0.0 and c.min_x >= self.max_x - epsilon:
            max_xa = c.min_x - self.max_x
            if max_xa < xa: xa = max_xa
        if xa < 0.0 and c.max_x <= self.min_x + epsilon:
            max_xa = c.max_x - self.min_x
            if max_xa > xa: xa = max_xa
        return xa

    def clip_y_collide(self, c, ya):
        epsilon = 1e-5
        if c.max_x <= self.min_x + epsilon or c.min_x >= self.max_x - epsilon: return ya
        if c.max_z <= self.min_z + epsilon or c.min_z >= self.max_z - epsilon: return ya
        if ya > 0.0 and c.min_y >= self.max_y - epsilon:
            max_ya = c.min_y - self.max_y
            if max_ya < ya: ya = max_ya
        if ya < 0.0 and c.max_y <= self.min_y + epsilon:
            max_ya = c.max_y - self.min_y
            if max_ya > ya: ya = max_ya
        return ya

    def clip_z_collide(self, c, za):
        epsilon = 1e-5
        if c.max_x <= self.min_x + epsilon or c.min_x >= self.max_x - epsilon: return za
        if c.max_y <= self.min_y + epsilon or c.min_y >= self.max_y - epsilon: return za
        if za > 0.0 and c.min_z >= self.max_z - epsilon:
            max_za = c.min_z - self.max_z
            if max_za < za: za = max_za
        if za < 0.0 and c.max_z <= self.min_z + epsilon:
            max_za = c.max_z - self.min_z
            if max_za > za: za = max_za
        return za

    def move(self, x, y, z):
        self.min_x += x
        self.min_y += y
        self.min_z += z
        self.max_x += x
        self.max_y += y
        self.max_z += z

class Entity:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = x
        self.y = y
        self.z = z
        self.xo = x
        self.yo = y
        self.zo = z
        
        self.dx = 0.0
        self.dy = 0.0
        self.dz = 0.0
        
        self.yRot = 0.0
        self.xRot = 0.0
        self.yRotO = 0.0
        self.xRotO = 0.0
        
        self.width = 0.6
        self.height = 1.8
        
        self.on_ground = False
        self.removed = False
        self.in_water = False
        
        self.fall_distance = 0.0
        self.tick_count = 0
        
    def set_size(self, width, height):
        self.width = width
        self.height = height
        
    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def remove(self):
        self.removed = True
        
    def hurt(self, damage):
        pass # Overridden in LivingEntity
        
    def is_in_water(self, get_block_func):
        b = get_block_func(int(math.floor(self.x)), int(math.floor(self.y)), int(math.floor(self.z)))
        if b == 9 or b == 8: return True
        b = get_block_func(int(math.floor(self.x)), int(math.floor(self.y + 0.4)), int(math.floor(self.z)))
        return b == 9 or b == 8
        
    def tick(self, get_block_func):
        self.xo = self.x
        self.yo = self.y
        self.zo = self.z
        self.yRotO = self.yRot
        self.xRotO = self.xRot
        self.tick_count += 1
        
        # Apply gravity and physics
        self.in_water = self.is_in_water(get_block_func)
        if self.in_water:
            self.dy -= 0.02
            self.dy *= 0.8 # Water vertical friction
            self.fall_distance = 0.0
        else:
            self.dy -= 0.08
            self.dy *= 0.98
            
        if self.dy < 0:
            self.fall_distance -= self.dy
            
        if self.y < -64:
            self.hurt(4)
        
        # Move entity with collision
        self.move(self.dx, self.dy, self.dz, get_block_func)
        
        # Friction
        if self.in_water:
            self.dx *= 0.8 # Water horizontal friction
            self.dz *= 0.8
        elif self.on_ground:
            self.dx *= 0.6
            self.dz *= 0.6
        else:
            self.dx *= 0.91
            self.dz *= 0.91

    def move(self, dx, dy, dz, get_block_func):
        old_dx, old_dy, old_dz = dx, dy, dz
        
        hw = self.width / 2.0
        bounds = AABB(self.x - hw, self.y, self.z - hw, self.x + hw, self.y + self.height, self.z + hw)
        
        # Simple AABB collision checking - optimized bounds to avoid redundant lookups
        min_x = int(math.floor(bounds.min_x + min(0, dx)))
        max_x = int(math.floor(bounds.max_x + max(0, dx)))
        min_y = int(math.floor(bounds.min_y + min(0, dy)))
        max_y = int(math.floor(bounds.max_y + max(0, dy)))
        min_z = int(math.floor(bounds.min_z + min(0, dz)))
        max_z = int(math.floor(bounds.max_z + max(0, dz)))

        blocks_bboxes = []
        for bx in range(min_x, max_x + 1):
            for by in range(min_y, max_y + 1):
                for bz in range(min_z, max_z + 1):
                    block = get_block_func(bx, by, bz)
                    if block == CACTUS: # CACTUS
                        if bounds.intersects(AABB(bx, by, bz, bx + 1, by + 1, bz + 1)):
                            self.hurt(1)
                    if block == -1 or (block > 0 and block not in (32, 34, 37, 38, 39, 40, 83, 106, 175, 176, 8, 9)): # Solid blocks
                        blocks_bboxes.append(AABB(bx, by, bz, bx + 1, by + 1, bz + 1))
        
        # Resolve Y
        for bb in blocks_bboxes:
            dy = bounds.clip_y_collide(bb, dy)
        bounds.move(0, dy, 0)
        
        # Resolve X
        for bb in blocks_bboxes:
            dx = bounds.clip_x_collide(bb, dx)
        bounds.move(dx, 0, 0)
        
        # Resolve Z
        for bb in blocks_bboxes:
            dz = bounds.clip_z_collide(bb, dz)
        bounds.move(0, 0, dz)

        self.on_ground = (old_dy != dy and old_dy < 0)
        
        if self.on_ground:
            if self.fall_distance > 3.0:
                self.hurt(math.ceil(self.fall_distance - 3.0))
            self.fall_distance = 0.0
            
        if old_dx != dx: self.dx = 0
        if old_dy != dy: self.dy = 0
        if old_dz != dz: self.dz = 0
        
        self.x = round((bounds.min_x + bounds.max_x) / 2.0, 5)
        self.y = round(bounds.min_y, 5)
        self.z = round((bounds.min_z + bounds.max_z) / 2.0, 5)
