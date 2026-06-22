import math
from core.math_utils import normalize_vec, look_at_matrix
from core.raycast import raycast

class Camera:
    def __init__(self):
        self.x, self.y, self.z = 32.0, 45.0, 32.0
        self.yaw = -90.0   # Horizontal look angle (yaw)
        self.pitch = -25.0  # Vertical look angle (pitch)
        self.speed = 30.0
        self.sensitivity = 0.15
        
        # Third Person System
        self.third_person_mode = 0  # 0: First, 1: Third Back, 2: Third Front
        self.tp_distance = 4.0      # Target distance for 3rd person
        self.actual_distance = 4.0  # Actual distance after collision
        self.eye_pos = (0, 0, 0)
        
    def get_front(self):
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        if self.third_person_mode == 2:
            # Front view looks opposite
            rp = -rp
            ry += math.pi
        fx = math.cos(rp) * math.cos(ry)
        fy = math.sin(rp)
        fz = math.cos(rp) * math.sin(ry)
        return normalize_vec([fx, fy, fz])
        
    def update_third_person(self, eye_x, eye_y, eye_z, get_block_info):
        self.eye_pos = (eye_x, eye_y, eye_z)
        if self.third_person_mode == 0:
            self.x, self.y, self.z = eye_x, eye_y, eye_z
            return
            
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        
        dir_x = math.cos(rp) * math.cos(ry)
        dir_y = math.sin(rp)
        dir_z = math.cos(rp) * math.sin(ry)
        
        # In Third Person Back (1), we move backward from look direction
        # In Third Person Front (2), we move forward from look direction (look at face)
        mult = -1.0 if self.third_person_mode == 1 else 1.0
        ray_dx = dir_x * mult
        ray_dy = dir_y * mult
        ray_dz = dir_z * mult
        
        # Collision raycast
        target_dist = self.tp_distance
        
        # Simple raycast logic for camera
        curr_dist = 0.0
        step = 0.1
        while curr_dist < self.tp_distance:
            cx = eye_x + ray_dx * curr_dist
            cy = eye_y + ray_dy * curr_dist
            cz = eye_z + ray_dz * curr_dist
            
            b_id, _ = get_block_info(int(math.floor(cx)), int(math.floor(cy)), int(math.floor(cz)))
            if b_id > 0 and b_id != 4:  # Not air, not water
                # Hit something, stop here
                target_dist = max(0.2, curr_dist - 0.2)
                break
            curr_dist += step
            
        # Smooth interpolation to prevent jitter
        if not hasattr(self, 'actual_distance'):
            self.actual_distance = target_dist
        else:
            self.actual_distance += (target_dist - self.actual_distance) * 0.2
            
        self.x = eye_x + ray_dx * self.actual_distance
        self.y = eye_y + ray_dy * self.actual_distance
        self.z = eye_z + ray_dz * self.actual_distance

    def get_view_matrix(self):
        f = self.get_front()
        # Front mode already inverted pitch/yaw in get_front, but view center must be computed accordingly
        center = [self.x + f[0], self.y + f[1], self.z + f[2]]
        return look_at_matrix([self.x, self.y, self.z], center, [0, 1, 0])

