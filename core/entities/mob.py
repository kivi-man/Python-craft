import math
from core.entities.living_entity import LivingEntity
from core.ai.goal import GoalSelector

class Mob(LivingEntity):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        self.goal_selector = GoalSelector()
        self.target_x = 0
        self.target_z = 0
        self.nav_active = False
        self.nav_speed = 0
        self.level = None # Will be set by engine
        self.look_target_yaw = 0.0
        self.look_target_pitch = 0.0

    def set_level(self, level):
        self.level = level

    def ai_step(self):
        self.has_look_target = False
        super().ai_step()
        self.goal_selector.tick()
        
        # Look control logic
        if self.has_look_target:
            # Smooth yaw
            diff = self.look_target_yaw - self.yHeadRot
            while diff < -180.0: diff += 360.0
            while diff >= 180.0: diff -= 360.0
            if diff > 10.0: diff = 10.0
            if diff < -10.0: diff = -10.0
            self.yHeadRot += diff
            
            # Smooth pitch
            diff_p = self.look_target_pitch - self.xRot
            while diff_p < -180.0: diff_p += 360.0
            while diff_p >= 180.0: diff_p -= 360.0
            if diff_p > 10.0: diff_p = 10.0
            if diff_p < -10.0: diff_p = -10.0
            self.xRot += diff_p
            
        # Clamp head to body if moving
        head_diff_body = self.yHeadRot - self.yBodyRot
        while head_diff_body < -180.0: head_diff_body += 360.0
        while head_diff_body >= 180.0: head_diff_body -= 360.0
        
        if self.nav_active:
            if head_diff_body < -75.0: self.yHeadRot = self.yBodyRot - 75.0
            if head_diff_body > 75.0: self.yHeadRot = self.yBodyRot + 75.0

        
        # Simple pathfinding execution (move towards target)
        if self.nav_active:
            dx = self.target_x - self.x
            dz = self.target_z - self.z
            dist = math.sqrt(dx*dx + dz*dz)
            if dist < 0.5:
                self.nav_active = False
                self.move_forward = 0.0
            else:
                self.move_forward = self.nav_speed
                # Rotate towards target
                target_yaw = math.degrees(math.atan2(-dx, dz))
                
                # Smooth rotation
                diff = target_yaw - self.yRot
                while diff > 180: diff -= 360
                while diff < -180: diff += 360
                self.yRot += diff * 0.2

    def set_target_pos(self, x, z, speed):
        self.target_x = x
        self.target_z = z
        self.nav_speed = speed
        self.nav_active = True

    def stop_navigation(self):
        self.nav_active = False
        self.move_forward = 0.0

    def navigation_done(self):
        return not self.nav_active

    def look_at(self, x, y, z):
        dx = x - self.x
        dz = z - self.z
        self.yRot = math.degrees(math.atan2(-dx, dz))

    def get_head_height(self):
        return 0.8

    def look_at_head(self, x, y, z):
        dx = x - self.x
        dz = z - self.z
        dy = y - (self.y + self.get_head_height())
        dist = math.sqrt(dx*dx + dz*dz)
        self.look_target_yaw = math.degrees(math.atan2(-dx, dz))
        self.look_target_pitch = math.degrees(math.atan2(-dy, dist))
        self.has_look_target = True

    def in_water(self):
        return False
