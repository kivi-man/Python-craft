import math
from pyglet.gl import GLfloat

def perspective_matrix(fov, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov) / 2.0)
    return (GLfloat * 16)(
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) / (near - far), -1,
        0, 0, (2 * far * near) / (near - far), 0
    )

def look_at_matrix(eye, center, up):
    f = normalize_vec(sub_vec(center, eye))
    s = normalize_vec(cross_vec(f, up))
    u = cross_vec(s, f)
    
    return (GLfloat * 16)(
        s[0], u[0], -f[0], 0,
        s[1], u[1], -f[1], 0,
        s[2], u[2], -f[2], 0,
        -dot_vec(s, eye), -dot_vec(u, eye), dot_vec(f, eye), 1
    )

def normalize_vec(v):
    l = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if l < 1e-8: return [0, 0, 0]
    return [v[0]/l, v[1]/l, v[2]/l]

def sub_vec(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def cross_vec(a, b):
    return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]

def dot_vec(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

import numpy as np
def get_hand_cube_vertices(block_id, block_layers):
    vertices = []
    faces = [
        ([0, 1, 0], [
            [-0.5, 0.5, -0.5, 0, 0],
            [-0.5, 0.5,  0.5, 0, 1],
            [ 0.5, 0.5,  0.5, 1, 1],
            [-0.5, 0.5, -0.5, 0, 0],
            [ 0.5, 0.5,  0.5, 1, 1],
            [ 0.5, 0.5, -0.5, 1, 0]
        ], 0),
        ([0, -1, 0], [
            [-0.5, -0.5, -0.5, 0, 0],
            [ 0.5, -0.5, -0.5, 1, 0],
            [ 0.5, -0.5,  0.5, 1, 1],
            [-0.5, -0.5, -0.5, 0, 0],
            [ 0.5, -0.5,  0.5, 1, 1],
            [-0.5, -0.5,  0.5, 0, 1]
        ], 1),
        ([1, 0, 0], [
            [0.5, -0.5, -0.5, 0, 0],
            [0.5,  0.5, -0.5, 0, 1],
            [0.5,  0.5,  0.5, 1, 1],
            [0.5, -0.5, -0.5, 0, 0],
            [0.5,  0.5,  0.5, 1, 1],
            [0.5, -0.5,  0.5, 1, 0]
        ], 2),
        ([-1, 0, 0], [
            [-0.5, -0.5, -0.5, 0, 0],
            [-0.5, -0.5,  0.5, 1, 0],
            [-0.5,  0.5,  0.5, 1, 1],
            [-0.5, -0.5, -0.5, 0, 0],
            [-0.5,  0.5,  0.5, 1, 1],
            [-0.5,  0.5, -0.5, 0, 1]
        ], 3),
        ([0, 0, 1], [
            [-0.5, -0.5, 0.5, 0, 0],
            [ 0.5, -0.5, 0.5, 1, 0],
            [ 0.5,  0.5, 0.5, 1, 1],
            [-0.5, -0.5, 0.5, 0, 0],
            [ 0.5,  0.5, 0.5, 1, 1],
            [-0.5,  0.5, 0.5, 0, 1]
        ], 4),
        ([0, 0, -1], [
            [-0.5, -0.5, -0.5, 0, 0],
            [-0.5,  0.5, -0.5, 0, 1],
            [ 0.5,  0.5, -0.5, 1, 1],
            [-0.5, -0.5, -0.5, 0, 0],
            [ 0.5,  0.5, -0.5, 1, 1],
            [ 0.5, -0.5, -0.5, 1, 0]
        ], 5)
    ]
    
    for normal, quad_verts, face_idx in faces:
        layer_idx = block_layers[block_id, face_idx]
        shade = 1.0
        if face_idx == 1: shade = 0.5
        elif face_idx in (2, 3): shade = 0.8
        elif face_idx in (4, 5): shade = 0.7
        
        for v in quad_verts:
            px, py, pz, u, w = v
            vertices.extend([
                px, py, pz,
                normal[0], normal[1], normal[2],
                shade, shade, shade,
                u, w, float(layer_idx),
                3.0, 15.0, 0.0
            ])
            
    return np.array(vertices, dtype=np.float32)
