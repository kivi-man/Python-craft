from pyglet.gl import *
import numpy as np
import math

class EntityRenderer:
    def __init__(self, model):
        self.model = model
        
    def render(self, entity, x, y, z, yaw, pitch, delta_time, camera_view, u_view_loc, u_tint_color_loc=-1):
        walk_speed = entity.walk_anim_speed_o + (entity.walk_anim_speed - entity.walk_anim_speed_o) * delta_time
        walk_pos = entity.walk_anim_pos_o + (entity.walk_anim_pos - entity.walk_anim_pos_o) * delta_time
        # Interpolate body rotation with wrapping
        body_diff = entity.yBodyRot - entity.yBodyRotO
        while body_diff < -180.0: body_diff += 360.0
        while body_diff >= 180.0: body_diff -= 360.0
        y_rot = entity.yBodyRotO + body_diff * delta_time
        
        # Interpolate head rotation with wrapping
        head_diff = entity.yHeadRot - entity.yHeadRotO
        while head_diff < -180.0: head_diff += 360.0
        while head_diff >= 180.0: head_diff -= 360.0
        head_yaw_world = entity.yHeadRotO + head_diff * delta_time
        head_yaw = head_yaw_world - y_rot
        
        # 1. Translate to entity position
        # Add 0.0078125 (0.125/16.0) to prevent z-fighting with the ground block
        t_mat = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y + 1.5078125], 
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 2. Rotate Y
        ry_rad = math.radians(180.0 - y_rot)
        cy, sy = math.cos(ry_rad), math.sin(ry_rad)
        ry_mat = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 3. Scale by -1, -1, 1 (upside down fix)
        s_mat = np.array([
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # 4. Death Rotation
        death_mat = np.eye(4, dtype=np.float32)
        if hasattr(entity, 'death_time') and entity.death_time > 0:
            death_rot = min(90.0, (entity.death_time + delta_time) / 20.0 * 90.0)
            dr_rad = math.radians(death_rot)
            cdr, sdr = math.cos(dr_rad), math.sin(dr_rad)
            # Rotate around Z axis (falling over sideways)
            death_mat = np.array([
                [cdr, -sdr, 0, 0],
                [sdr, cdr,  0, 0],
                [0,   0,    1, 0],
                [0,   0,    0, 1]
            ], dtype=np.float32)
        
        entity_matrix = t_mat @ ry_mat @ death_mat @ s_mat
        cv = np.array(list(camera_view), dtype=np.float32).reshape(4, 4).T
        parent_matrix = cv @ entity_matrix
        
        self.bind_texture(entity)
        
        if u_tint_color_loc != -1:
            if hasattr(entity, 'hurt_time') and entity.hurt_time > 0:
                glUniform4f(u_tint_color_loc, 1.0, 0.3, 0.3, 1.0) # Red flash
            else:
                glUniform4f(u_tint_color_loc, 1.0, 1.0, 1.0, 1.0) # Normal
        
        self.model.render(entity, walk_pos, walk_speed, entity.tick_count, head_yaw, pitch, 1.0/16.0, parent_matrix, u_view_loc)
        
        if u_tint_color_loc != -1:
            glUniform4f(u_tint_color_loc, 1.0, 1.0, 1.0, 1.0) # Reset back to normal for blocks

    def bind_texture(self, entity):
        pass
