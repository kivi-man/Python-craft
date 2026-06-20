from core.entities.mob import Mob

class Animal(Mob):
    def __init__(self, x=0.0, y=0.0, z=0.0):
        super().__init__(x, y, z)
