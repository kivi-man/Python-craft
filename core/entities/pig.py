from core.entities.animal import Animal
from core.ai.goals import FloatGoal, PanicGoal, RandomStrollGoal, LookAtPlayerGoal, RandomLookAroundGoal

class Pig(Animal):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        self.set_size(0.9, 0.9)
        self.max_health = 10
        self.health = 10
        self.speed = 0.08
        
        # Matches Pig.cpp goals structure
        self.goal_selector.add_goal(0, FloatGoal(self))
        self.goal_selector.add_goal(1, PanicGoal(self, 1.25))
        self.goal_selector.add_goal(6, RandomStrollGoal(self, 1.0))
        self.goal_selector.add_goal(7, LookAtPlayerGoal(self, 6))
        self.goal_selector.add_goal(8, RandomLookAroundGoal(self))
        
    def to_dict(self):
        data = super().to_dict()
        data['type'] = 'Pig'
        return data
        
    def ai_step(self):
        super().ai_step()
        
        # Step sound
        if not self.dead and (abs(self.dx) > 0.01 or abs(self.dz) > 0.01) and getattr(self, 'on_ground', True):
            if not hasattr(self, 'distance_walked'): self.distance_walked = 0.0
            import math
            # self.dx/self.dz are blocks per tick. Accumulate total blocks moved.
            self.distance_walked += math.sqrt(self.dx**2 + self.dz**2)
            if self.distance_walked > 1.2:
                self.distance_walked = 0.0
                if hasattr(self, 'level') and self.level is not None and hasattr(self.level, 'sound_system'):
                    self.level.sound_system.play("eSoundType_MOB_PIG_STEP", x=self.x, y=self.y, z=self.z, volume=0.2)

    def play_ambient_sound(self):
        if hasattr(self, 'level') and self.level is not None and hasattr(self.level, 'sound_system'):
            self.level.sound_system.play("eSoundType_MOB_PIG_AMBIENT", x=self.x, y=self.y, z=self.z, volume=0.5)

    def hurt(self, damage):
        was_dead = self.dead
        super().hurt(damage)
        if self.dead and not was_dead:
            if hasattr(self, 'level') and self.level is not None:
                self.level.spawn_item_entity(1000, self.x, self.y, self.z)
