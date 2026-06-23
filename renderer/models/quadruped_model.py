import math
from renderer.models.model import Model
from renderer.models.model_part import ModelPart

class QuadrupedModel(Model):
    def __init__(self, y_offset, grow):
        super().__init__()
        self.head = ModelPart(0, 0).add_box(-4.0, -4.0, -8.0, 8, 8, 8, grow)
        self.head.set_pos(0.0, 18.0 - y_offset, -6.0)
        
        self.body = ModelPart(28, 8).add_box(-5.0, -10.0, -7.0, 10, 16, 8, grow)
        self.body.set_pos(0.0, 17.0 - y_offset, 2.0)
        self.body.xRot = math.pi / 2.0
        
        self.leg0 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, y_offset, 4, grow)
        self.leg0.set_pos(-3.0, 24.0 - y_offset, 7.0)
        
        self.leg1 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, y_offset, 4, grow)
        self.leg1.set_pos(3.0, 24.0 - y_offset, 7.0)
        
        self.leg2 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, y_offset, 4, grow)
        self.leg2.set_pos(-3.0, 24.0 - y_offset, -5.0)
        
        self.leg3 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, y_offset, 4, grow)
        self.leg3.set_pos(3.0, 24.0 - y_offset, -5.0)

    def render(self, entity, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, parent_matrix, u_view_loc):
        super().render(entity, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, parent_matrix, u_view_loc)
        if not self.head.vao:
            self.head.compile(scale)
            self.body.compile(scale, swap_top_bottom=True)
            self.leg0.compile(scale, swap_top_bottom=True)
            self.leg1.compile(scale, swap_top_bottom=True)
            self.leg2.compile(scale, swap_top_bottom=True)
            self.leg3.compile(scale, swap_top_bottom=True)
            
        parts = [self.head, self.body, self.leg0, self.leg1, self.leg2, self.leg3]
        self.render_batched(parts, parent_matrix, u_view_loc, scale)
        
    def setup_anim(self, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale):
        self.head.xRot = head_pitch / 57.2957795
        self.head.yRot = head_yaw / 57.2957795
        
        self.leg0.xRot = math.cos(walk_pos * 0.6662) * 1.4 * walk_speed
        self.leg1.xRot = math.cos(walk_pos * 0.6662 + math.pi) * 1.4 * walk_speed
        self.leg2.xRot = math.cos(walk_pos * 0.6662 + math.pi) * 1.4 * walk_speed
        self.leg3.xRot = math.cos(walk_pos * 0.6662) * 1.4 * walk_speed
