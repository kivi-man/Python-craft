import math
from core.math_utils import normalize_vec, look_at_matrix

class Camera:
    def __init__(self):
        self.x, self.y, self.z = 32.0, 45.0, 32.0
        self.yaw = -90.0   # Horizontal look angle (yaw)
        self.pitch = -25.0  # Vertical look angle (pitch)
        self.speed = 30.0
        self.sensitivity = 0.15
        
    def get_front(self):
        ry = math.radians(self.yaw)
        rp = math.radians(self.pitch)
        fx = math.cos(rp) * math.cos(ry)
        fy = math.sin(rp)
        fz = math.cos(rp) * math.sin(ry)
        return normalize_vec([fx, fy, fz])
    
    def get_view_matrix(self):
        f = self.get_front()
        center = [self.x + f[0], self.y + f[1], self.z + f[2]]
        return look_at_matrix([self.x, self.y, self.z], center, [0, 1, 0])
