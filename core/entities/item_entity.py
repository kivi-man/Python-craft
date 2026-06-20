import math
import random
from core.entities.entity import Entity

class ItemEntity(Entity):
    def __init__(self, block_id, x, y, z, xd=None, yd=None, zd=None):
        super().__init__(x, y, z)
        self.block_id = block_id
        self.count = 1
        
        self.set_size(0.25, 0.25)
        
        # Initial velocities identical to C++ example
        self.yRot = random.random() * 360.0
        self.xd = random.random() * 0.2 - 0.1 if xd is None else xd
        self.yd = 0.2 if yd is None else yd
        self.zd = random.random() * 0.2 - 0.1 if zd is None else zd
        
        self.age = 0
        self.pickup_delay = 40 # ticks (2 seconds delay before picking up, typical for dropped blocks)
        self.bob_off = random.random() * math.pi * 2.0
        
    def tick(self, get_block_func):
        self.xo = self.x
        self.yo = self.y
        self.zo = self.z
        self.yRotO = self.yRot
        
        if self.pickup_delay > 0:
            self.pickup_delay -= 1
            
        self.yd -= 0.04 # Gravity
        
        # Move entity
        self.move(self.xd, self.yd, self.zd, get_block_func)
        
        # Friction
        friction = 0.98
        if self.on_ground:
            friction = 0.6 * 0.98
            
        self.xd *= friction
        self.yd *= 0.98
        self.zd *= friction
        
        if self.on_ground:
            self.yd *= -0.5 # Bouncing
            
        self.age += 1
        if self.age >= 6000: # 5 minutes (20 TPS * 60 * 5)
            self.remove()

    def merge_with(self, other):
        if other == self or other.removed or self.removed: return False
        if self.block_id != other.block_id: return False
        if self.count + other.count > 64: return False
        
        self.count += other.count
        self.age = min(self.age, other.age)
        other.remove()
        return True
