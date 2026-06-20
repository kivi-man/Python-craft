import random
import math
from core.ai.goal import Goal

class RandomStrollGoal(Goal):
    def __init__(self, mob, speed):
        self.mob = mob
        self.speed = speed
        self.target_x = 0
        self.target_z = 0
        self.active = False
        
    def can_use(self):
        if self.mob.nav_active: return False
        if random.random() < 0.01: # 1% chance per tick to start strolling
            self.target_x = self.mob.x + (random.random() * 10 - 5)
            self.target_z = self.mob.z + (random.random() * 10 - 5)
            return True
        return False
        
    def can_continue_to_use(self):
        return self.active and not self.mob.navigation_done()
        
    def start(self):
        self.mob.set_target_pos(self.target_x, self.target_z, self.speed)
        self.active = True
        self.timeout = random.randint(40, 100) # 2 to 5 seconds max stroll time
        
    def tick(self):
        self.timeout -= 1
        if self.timeout <= 0:
            self.mob.stop_navigation()
            
    def stop(self):
        self.active = False
        self.mob.stop_navigation()

class LookAtPlayerGoal(Goal):
    def __init__(self, mob, radius):
        self.mob = mob
        self.radius = radius
        self.target_player = None
        
    def can_use(self):
        if self.mob.nav_active: return False
        if random.random() < 0.02:
            self.target_player = self.mob.level.get_nearest_player(self.mob.x, self.mob.y, self.mob.z, self.radius)
            return self.target_player is not None
        return False
        
    def can_continue_to_use(self):
        if not self.target_player: return False
        dx = self.target_player.x - self.mob.x
        dy = self.target_player.y - self.mob.y
        dz = self.target_player.z - self.mob.z
        return (dx*dx + dy*dy + dz*dz) <= self.radius*self.radius

    def tick(self):
        self.mob.look_at_head(self.target_player.x, self.target_player.y + 1.5, self.target_player.z)

class RandomLookAroundGoal(Goal):
    def __init__(self, mob):
        self.mob = mob
        self.look_time = 0
        self.relX = 0.0
        self.relZ = 0.0
        
    def can_use(self):
        return random.random() < 0.02
        
    def can_continue_to_use(self):
        return self.look_time >= 0
        
    def start(self):
        rnd = 2 * math.pi * random.random()
        self.relX = math.cos(rnd)
        self.relZ = math.sin(rnd)
        self.look_time = 20 + random.randint(0, 20)
        
    def tick(self):
        self.look_time -= 1
        self.mob.look_at_head(self.mob.x + self.relX, self.mob.y + self.mob.get_head_height(), self.mob.z + self.relZ)

class FloatGoal(Goal):
    def __init__(self, mob):
        self.mob = mob
        
    def can_use(self):
        return self.mob.in_water
        
    def tick(self):
        if random.random() < 0.8:
            self.mob.jump_ticks = 1 # Swim up

class PanicGoal(Goal):
    def __init__(self, mob, speed):
        self.mob = mob
        self.speed = speed
        
    def can_use(self):
        return self.mob.health < self.mob.max_health # Simple panic condition
        
    def can_continue_to_use(self):
        return not self.mob.navigation_done()
        
    def start(self):
        self.mob.set_target_pos(self.mob.x + (random.random() * 20 - 10), 
                                self.mob.z + (random.random() * 20 - 10), self.speed)
