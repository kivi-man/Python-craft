from core.entities.animal import Animal
from core.ai.goals import FloatGoal, PanicGoal, RandomStrollGoal, LookAtPlayerGoal, RandomLookAroundGoal

class Pig(Animal):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
        self.set_size(0.9, 0.9)
        self.max_health = 10
        self.health = 10
        self.speed = 0.08
        
    def to_dict(self):
        data = super().to_dict()
        data['type'] = 'Pig'
        return data
        
    def ai_step(self):
        # Matches Pig.cpp goals structure
        self.goal_selector.add_goal(0, FloatGoal(self))
        self.goal_selector.add_goal(1, PanicGoal(self, 1.25))
        self.goal_selector.add_goal(6, RandomStrollGoal(self, 1.0))
        self.goal_selector.add_goal(7, LookAtPlayerGoal(self, 6))
        self.goal_selector.add_goal(8, RandomLookAroundGoal(self))
