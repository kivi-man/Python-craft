import math
from core.entities.entity import Entity

class LivingEntity(Entity):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        self.health = 20
        self.max_health = 20
        self.move_forward = 0.0
        self.move_strafe = 0.0
        self.speed = 0.2
        self.jump_ticks = 0
        
        self.hurt_time = 0
        self.hurt_duration = 10
        self.death_time = 0
        self.dead = False
        self.air_supply = 300 # 15 seconds of air (20 ticks/sec)
        
        # Animation states
        self.yBodyRot = 0.0
        self.yBodyRotO = 0.0
        self.yHeadRot = 0.0
        self.yHeadRotO = 0.0
        self.walk_anim_pos = 0.0
        self.walk_anim_pos_o = 0.0
        self.walk_anim_speed = 0.0
        self.walk_anim_speed_o = 0.0
        self.has_look_target = False

    def to_dict(self):
        return {
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "health": self.health,
            "yRot": getattr(self, 'yRot', 0.0),
            "xRot": getattr(self, 'xRot', 0.0),
            "yBodyRot": self.yBodyRot,
            "yHeadRot": self.yHeadRot,
            "dead": self.dead,
            "death_time": self.death_time,
            "hurt_time": self.hurt_time
        }

    def from_dict(self, data):
        self.x = data.get("x", self.x)
        self.y = data.get("y", self.y)
        self.z = data.get("z", self.z)
        self.dx = data.get("dx", 0.0)
        self.dy = data.get("dy", 0.0)
        self.dz = data.get("dz", 0.0)
        self.health = data.get("health", self.max_health)
        self.yRot = data.get("yRot", 0.0)
        self.xRot = data.get("xRot", 0.0)
        self.yBodyRot = data.get("yBodyRot", 0.0)
        self.yHeadRot = data.get("yHeadRot", 0.0)
        self.dead = data.get("dead", False)
        self.death_time = data.get("death_time", 0)
        self.hurt_time = data.get("hurt_time", 0)
        self.xo, self.yo, self.zo = self.x, self.y, self.z

    def hurt(self, damage):
        if self.dead or self.hurt_time > 0:
            return
        self.health -= damage
        self.hurt_time = self.hurt_duration
        if self.health <= 0:
            self.dead = True

    def tick(self, get_block_func):
        if self.hurt_time > 0:
            self.hurt_time -= 1
            
        if self.dead:
            self.death_time += 1
            if self.death_time >= 20:
                self.remove()
            # Still apply gravity when dead
            super().tick(get_block_func)
            return

        if getattr(self, 'in_water', False):
            self.air_supply -= 1
            if self.air_supply <= -20: # Take damage every 1 sec when out of air
                self.air_supply = 0
                self.hurt(2)
        else:
            self.air_supply = 300

        self.ai_step()
        
        # Jump logic
        if self.jump_ticks > 0:
            if getattr(self, 'in_water', False):
                self.dy += 0.04
            elif self.on_ground:
                self.dy = 0.42
            self.jump_ticks -= 1
            
        self.move_relative(self.move_strafe, self.move_forward, self.speed if self.on_ground else self.speed * 0.2)
        
        super().tick(get_block_func)
        
        self.update_animations()

    def ai_step(self):
        # Override in subclasses
        self.move_forward = 0.0
        self.move_strafe = 0.0
        
    def move_relative(self, strafe, forward, friction):
        dist = strafe * strafe + forward * forward
        if dist >= 0.0001:
            dist = math.sqrt(dist)
            if dist < 1.0:
                dist = 1.0
            dist = friction / dist
            strafe *= dist
            forward *= dist
            
            sin_yaw = math.sin(math.radians(self.yRot))
            cos_yaw = math.cos(math.radians(self.yRot))
            
            self.dx += strafe * cos_yaw - forward * sin_yaw
            self.dz += forward * cos_yaw + strafe * sin_yaw

    def update_animations(self):
        self.yBodyRotO = self.yBodyRot
        self.yHeadRotO = self.yHeadRot
        self.walk_anim_pos_o = self.walk_anim_pos
        self.walk_anim_speed_o = self.walk_anim_speed
        
        dx = self.x - self.xo
        dz = self.z - self.zo
        dist = math.sqrt(dx*dx + dz*dz)
        
        if dist > 0.05:
            self.yBodyRot = self.yRot
        else:
            diff = self.yHeadRot - self.yBodyRot
            while diff > 180: diff -= 360
            while diff < -180: diff += 360
            if abs(diff) > 50:
                self.yBodyRot += (diff - 50 * (1 if diff > 0 else -1)) * 0.2
                
        # Realign head to body slowly (can be overridden by AI goals)
        if not self.has_look_target:
            diff_head = self.yBodyRot - self.yHeadRot
            while diff_head > 180: diff_head -= 360
            while diff_head < -180: diff_head += 360
            self.yHeadRot += diff_head * 0.05
        
        target_speed = min(dist * 4.0, 1.0)
        self.walk_anim_speed += (target_speed - self.walk_anim_speed) * 0.4
        self.walk_anim_pos += self.walk_anim_speed
