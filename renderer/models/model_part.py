import math
import numpy as np
from pyglet.gl import *
import ctypes

class ModelPart:
    def __init__(self, tex_u, tex_v):
        self.tex_u = tex_u
        self.tex_v = tex_v
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.xRot = 0.0
        self.yRot = 0.0
        self.zRot = 0.0
        self.boxes = []
        self.vao = None
        self.vbo = None
        self.vertex_count = 0
        # Batch rendering için ham vertex verileri (numpy array, Nx15)
        self._raw_vertices = None
        
    def set_pos(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        
    def add_box(self, x_off, y_off, z_off, width, height, depth, scale=0.0):
        self.boxes.append({
            'off': (x_off, y_off, z_off),
            'dim': (width, height, depth),
            'scale': scale,
            'uv': (self.tex_u, self.tex_v)
        })
        return self
        
    def compile(self, scale=0.0, swap_top_bottom=False, tex_w=64.0, tex_h=32.0):
        verts = []
        
        for box in self.boxes:
            x0 = (box['off'][0] - box['scale']) * scale
            y0 = (box['off'][1] - box['scale']) * scale
            z0 = (box['off'][2] - box['scale']) * scale
            x1 = (box['off'][0] + box['dim'][0] + box['scale']) * scale
            y1 = (box['off'][1] + box['dim'][1] + box['scale']) * scale
            z1 = (box['off'][2] + box['dim'][2] + box['scale']) * scale
            
            u = box['uv'][0]
            v = box['uv'][1]
            w = box['dim'][0]
            h = box['dim'][1]
            d = box['dim'][2]
            
            def add_face(p0, p1, p2, p3, norm, uv_coords):
                pts = [p0, p1, p2, p0, p2, p3]
                uvs = [uv_coords[0], uv_coords[1], uv_coords[2], uv_coords[0], uv_coords[2], uv_coords[3]]
                for i in range(6):
                    px, py, pz = pts[i]
                    tx, ty = uvs[i]
                    verts.extend([px, py, pz, norm[0], norm[1], norm[2], 1.0, 1.0, 1.0, tx, 1.0 - ty, 0.0, 3.0, 15.0, 0.0])

            # Calculate UVs for Top and Bottom
            top_u0, top_v0 = (u + d) / tex_w, (v) / tex_h
            top_u1, top_v1 = (u + d + w) / tex_w, (v + d) / tex_h
            
            bot_u0, bot_v0 = (u + d + w) / tex_w, (v) / tex_h
            bot_u1, bot_v1 = (u + d + w + w) / tex_w, (v + d) / tex_h
            
            if swap_top_bottom:
                top_u0, top_v0, bot_u0, bot_v0 = bot_u0, bot_v0, top_u0, top_v0
                top_u1, top_v1, bot_u1, bot_v1 = bot_u1, bot_v1, top_u1, top_v1

            # Top Face
            add_face((x1,y1,z1), (x1,y1,z0), (x0,y1,z0), (x0,y1,z1), (0,1,0), [(top_u0, top_v0), (top_u0, top_v1), (top_u1, top_v1), (top_u1, top_v0)])
            
            # Bottom Face
            add_face((x0,y0,z1), (x0,y0,z0), (x1,y0,z0), (x1,y0,z1), (0,-1,0), [(bot_u1, bot_v0), (bot_u1, bot_v1), (bot_u0, bot_v1), (bot_u0, bot_v0)])
            
            # Front Face
            u0, v0 = (u + d) / tex_w, (v + d) / tex_h
            u1, v1 = (u + d + w) / tex_w, (v + d + h) / tex_h
            add_face((x1,y1,z0), (x1,y0,z0), (x0,y0,z0), (x0,y1,z0), (0,0,-1), [(u0, v1), (u0, v0), (u1, v0), (u1, v1)])
            
            # Back Face
            u0, v0 = (u + d + w + d) / tex_w, (v + d) / tex_h
            u1, v1 = (u + d + w + d + w) / tex_w, (v + d + h) / tex_h
            add_face((x0,y1,z1), (x0,y0,z1), (x1,y0,z1), (x1,y1,z1), (0,0,1), [(u1, v1), (u1, v0), (u0, v0), (u0, v1)])
            
            # Left Face
            u0, v0 = (u) / tex_w, (v + d) / tex_h
            u1, v1 = (u + d) / tex_w, (v + d + h) / tex_h
            add_face((x0,y1,z0), (x0,y0,z0), (x0,y0,z1), (x0,y1,z1), (-1,0,0), [(u1, v1), (u1, v0), (u0, v0), (u0, v1)])
            
            # Right Face
            u0, v0 = (u + d + w) / tex_w, (v + d) / tex_h
            u1, v1 = (u + d + w + d) / tex_w, (v + d + h) / tex_h
            add_face((x1,y1,z1), (x1,y0,z1), (x1,y0,z0), (x1,y1,z0), (1,0,0), [(u0, v1), (u0, v0), (u1, v0), (u1, v1)])

        self.vertex_count = len(verts) // 15
        if self.vertex_count == 0: return

        self.verts_array = np.array(verts, dtype=np.float32).reshape(-1, 15)

        v_array = (GLfloat * len(verts))(*verts)
        self.vao = GLuint()
        glGenVertexArrays(1, ctypes.byref(self.vao))
        glBindVertexArray(self.vao)
        
        self.vbo = GLuint()
        glGenBuffers(1, ctypes.byref(self.vbo))
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(v_array), v_array, GL_STATIC_DRAW)
        
        stride = 15 * 4
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(24))
        glEnableVertexAttribArray(2)
        glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(36))
        glEnableVertexAttribArray(3)
        glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(48))
        glEnableVertexAttribArray(4)
        glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(52))
        glEnableVertexAttribArray(5)
        glVertexAttribPointer(6, 1, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(56))
        glEnableVertexAttribArray(6)
        
        glBindVertexArray(0)

    def get_transformed_vertices(self, parent_matrix, scale):
        """CPU tarafında dönüştürülmüş vertex verilerini döndürür (batch rendering için).
        
        Pozisyonlar (x,y,z) final_matrix ile çarpılır (w=1),
        normaller (nx,ny,nz) final_matrix ile çarpılır (w=0, çeviri yok).
        Diğer attributelar (renk, uv, layer, ao, light, overlay) olduğu gibi kalır.
        """
        if self._raw_vertices is None:
            return None
        
        local_matrix = self.get_matrix(scale)
        final_matrix = parent_matrix @ local_matrix
        # Geriye dönük uyumluluk için last_final_matrix'i güncelle
        self.last_final_matrix = final_matrix
        
        result = self._raw_vertices.copy()
        n = result.shape[0]
        
        # Pozisyonları dönüştür: (x, y, z) -> final_matrix @ (x, y, z, 1)
        positions = result[:, 0:3]  # Nx3
        ones = np.ones((n, 1), dtype=np.float32)
        pos_h = np.hstack((positions, ones))  # Nx4
        transformed_pos = (final_matrix @ pos_h.T).T  # Nx4
        result[:, 0:3] = transformed_pos[:, 0:3]
        
        # Normalleri dönüştür: (nx, ny, nz) -> final_matrix @ (nx, ny, nz, 0)
        normals = result[:, 3:6]  # Nx3
        zeros = np.zeros((n, 1), dtype=np.float32)
        norm_h = np.hstack((normals, zeros))  # Nx4
        transformed_norm = (final_matrix @ norm_h.T).T  # Nx4
        result[:, 3:6] = transformed_norm[:, 0:3]
        
        return result

    def get_matrix(self, scale):
        tx, ty, tz = self.x * scale, self.y * scale, self.z * scale
        t_mat = np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        rx_rad = self.xRot
        cx, sx = math.cos(rx_rad), math.sin(rx_rad)
        rx_mat = np.array([
            [1, 0, 0, 0],
            [0, cx, -sx, 0],
            [0, sx, cx, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        ry_rad = self.yRot
        cy, sy = math.cos(ry_rad), math.sin(ry_rad)
        ry_mat = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        rz_rad = self.zRot
        cz, sz = math.cos(rz_rad), math.sin(rz_rad)
        rz_mat = np.array([
            [cz, -sz, 0, 0],
            [sz, cz, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)

        return t_mat @ ry_mat @ rx_mat @ rz_mat

    def render(self, parent_matrix, u_view_loc, scale):
        if not self.vao: return
        
        local_matrix = self.get_matrix(scale)
        final_matrix = parent_matrix @ local_matrix
        self.last_final_matrix = final_matrix
        
        flat_matrix = final_matrix.T.flatten()
        view_arr = (GLfloat * 16)(*flat_matrix)
        glUniformMatrix4fv(u_view_loc, 1, GL_FALSE, view_arr)
        
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, self.vertex_count)
        glBindVertexArray(0)

    def get_transformed_vertices(self, parent_matrix, scale):
        if not hasattr(self, 'verts_array') or len(self.verts_array) == 0:
            return None
            
        local_matrix = self.get_matrix(scale)
        final_matrix = parent_matrix @ local_matrix
        self.last_final_matrix = final_matrix
        
        # We need to transform the positions (first 3 floats) and normals (next 3 floats).
        pts = np.ones((len(self.verts_array), 4), dtype=np.float32)
        pts[:, :3] = self.verts_array[:, :3]
        
        norms = np.zeros((len(self.verts_array), 4), dtype=np.float32)
        norms[:, :3] = self.verts_array[:, 3:6]
        
        # P @ M.T transforms row vectors
        t_pts = pts @ final_matrix.T
        t_norms = norms @ final_matrix.T
        
        out_verts = self.verts_array.copy()
        out_verts[:, :3] = t_pts[:, :3]
        out_verts[:, 3:6] = t_norms[:, :3]
        
        return out_verts.flatten()
