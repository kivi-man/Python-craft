import math
from renderer.models.model import Model
from renderer.models.model_part import ModelPart

class HumanoidModel(Model):
    def __init__(self, grow=0.0, y_offset=0.0):
        super().__init__()
        
        self.head = ModelPart(0, 0).add_box(-4.0, -8.0, -4.0, 8, 8, 8, grow)
        self.head.set_pos(0.0, 0.0 + y_offset, 0.0)
        
        self.body = ModelPart(16, 16).add_box(-4.0, 0.0, -2.0, 8, 12, 4, grow)
        self.body.set_pos(0.0, 0.0 + y_offset, 0.0)
        
        self.arm0 = ModelPart(40, 16).add_box(-3.0, -2.0, -2.0, 4, 12, 4, grow)
        self.arm0.set_pos(-5.0, 2.0 + y_offset, 0.0)
        
        self.arm1 = ModelPart(40, 16).add_box(-1.0, -2.0, -2.0, 4, 12, 4, grow)
        self.arm1.set_pos(5.0, 2.0 + y_offset, 0.0)
        
        self.leg0 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, 12, 4, grow)
        self.leg0.set_pos(-1.9, 12.0 + y_offset, 0.0)
        
        self.leg1 = ModelPart(0, 16).add_box(-2.0, 0.0, -2.0, 4, 12, 4, grow)
        self.leg1.set_pos(1.9, 12.0 + y_offset, 0.0)

    def render(self, entity, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, parent_matrix, u_view_loc):
        self.setup_anim(walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, 
                        sneaking=getattr(entity, 'is_crouching', False), 
                        swinging=getattr(entity, 'swing_progress', 0.0))
        
        if not self.head.vao:
            tw = getattr(self, 'tex_w', 64.0)
            th = getattr(self, 'tex_h', 32.0)
            self.head.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            self.body.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            self.arm0.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            self.arm1.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            self.leg0.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            self.leg1.compile(scale, swap_top_bottom=True, tex_w=tw, tex_h=th)
            
        parts = [self.head, self.body, self.arm0, self.arm1, self.leg0, self.leg1]
        self.render_batched(parts, parent_matrix, u_view_loc, scale)

    def setup_anim(self, walk_pos, walk_speed, alive_ticks, head_yaw, head_pitch, scale, sneaking=False, swinging=0.0):
        self.head.xRot = head_pitch / 57.2957795
        self.head.yRot = head_yaw / 57.2957795
        
        self.arm0.xRot = math.cos(walk_pos * 0.6662 + math.pi) * 2.0 * walk_speed * 0.5
        self.arm1.xRot = math.cos(walk_pos * 0.6662) * 2.0 * walk_speed * 0.5
        self.arm0.zRot = 0.0
        self.arm1.zRot = 0.0
        self.arm0.yRot = 0.0
        self.arm1.yRot = 0.0
        
        self.leg0.xRot = math.cos(walk_pos * 0.6662) * 1.4 * walk_speed
        self.leg1.xRot = math.cos(walk_pos * 0.6662 + math.pi) * 1.4 * walk_speed
        self.leg0.yRot = 0.0
        self.leg1.yRot = 0.0
        
        if sneaking:
            self.body.xRot = 0.5
            self.arm0.xRot += 0.4
            self.arm1.xRot += 0.4
            self.leg0.z = 4.0
            self.leg1.z = 4.0
            self.leg0.y = 12.0
            self.leg1.y = 12.0
            # Lower the entire upper body by 3.2 units to simulate crouching
            # without detaching the head from the body.
            self.head.y = 3.2
            self.body.y = 3.2
            self.arm0.y = 5.2
            self.arm1.y = 5.2
            self.arm0.x = -5.0
            self.arm1.x = 5.0
            self.arm0.z = 0.0
            self.arm1.z = 0.0
        else:
            self.body.xRot = 0.0
            self.leg0.z = 0.1
            self.leg1.z = 0.1
            self.leg0.y = 12.0
            self.leg1.y = 12.0
            self.head.y = 0.0
            self.body.y = 0.0
            self.arm0.y = 2.0
            self.arm1.y = 2.0
            self.arm0.x = -5.0
            self.arm1.x = 5.0
            self.arm0.z = 0.0
            self.arm1.z = 0.0
            
        if swinging > 0.0:
            swing = swinging
            self.body.yRot = math.sin(math.sqrt(swing) * math.pi * 2.0) * 0.2
            self.arm0.z = math.sin(self.body.yRot) * 5.0
            self.arm0.x = -math.cos(self.body.yRot) * 5.0
            self.arm1.z = -math.sin(self.body.yRot) * 5.0
            self.arm1.x = math.cos(self.body.yRot) * 5.0
            self.arm0.yRot += self.body.yRot
            self.arm1.yRot += self.body.yRot
            self.arm1.xRot += self.body.yRot
            
            swing = 1.0 - swinging
            swing *= swing
            swing *= swing
            swing = 1.0 - swing
            aa = math.sin(swing * math.pi)
            bb = math.sin(swinging * math.pi) * -(self.head.xRot - 0.7) * 0.75
            self.arm0.xRot -= (aa * 1.2 + bb)
            self.arm0.yRot += self.body.yRot * 2.0
            self.arm0.zRot -= math.sin(swinging * math.pi) * -0.4
