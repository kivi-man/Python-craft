from renderer.models.quadruped_model import QuadrupedModel

class PigModel(QuadrupedModel):
    def __init__(self, grow=0.0):
        super().__init__(6, grow)
        
        # Add the pig snout to the existing head
        self.head.tex_u = 16
        self.head.tex_v = 16
        self.head.add_box(-2.0, 0.0, -9.0, 4, 3, 1, grow)
