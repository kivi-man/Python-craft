import math
import random
from core.entities.entity import Entity

class ParticleEntity(Entity):
    def __init__(self, block_id, x, y, z, xa, ya, za):
        super().__init__(x, y, z)
        self.block_id = block_id
        
        self.set_size(0.2, 0.2)
        
        self.uo = random.random() * 3.0
        self.vo = random.random() * 3.0
        
        # size calculation exactly like C++
        base_size = (random.random() * 0.5 + 0.5) * 2.0
        self.size = base_size / 2.0 # TerrainParticle scales size by half
        
        self.lifetime = int(4.0 / (random.random() * 0.9 + 0.1))
        self.age = 0
        self.gravity = 1.0
        
        xd = xa + (random.random() * 2.0 - 1.0) * 0.4
        yd = ya + (random.random() * 2.0 - 1.0) * 0.4
        zd = za + (random.random() * 2.0 - 1.0) * 0.4
        
        speed = (random.random() + random.random() + 1.0) * 0.15
        dd = math.sqrt(xd*xd + yd*yd + zd*zd) + 1e-6
        
        self.xd = xd / dd * speed * 0.4
        self.yd = yd / dd * speed * 0.4 + 0.1
        self.zd = zd / dd * speed * 0.4

    def set_power(self, power):
        self.xd *= power
        self.yd = (self.yd - 0.1) * power + 0.1
        self.zd *= power
        return self
        
    def set_scale(self, scale):
        self.set_size(0.2 * scale, 0.2 * scale)
        self.size *= scale
        return self

    def tick(self, get_block_func):
        self.xo = self.x
        self.yo = self.y
        self.zo = self.z
        
        self.age += 1
        if self.age >= self.lifetime:
            self.remove()
            return
            
        self.yd -= 0.04 * self.gravity
        self.move(self.xd, self.yd, self.zd, get_block_func)
        
        self.xd *= 0.98
        self.yd *= 0.98
        self.zd *= 0.98
        
        if self.on_ground:
            self.xd *= 0.7
            self.zd *= 0.7
