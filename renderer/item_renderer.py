from pyglet.gl import *
import ctypes
import numpy as np
import math

class ItemRenderer:
    def render(self, entity, x, y, z, yRot, xRot, partial_tick, camera_view, u_view_loc, u_tint_color_loc, engine):
        # We need the VAO for the block
        if not hasattr(engine, 'hand_block_vaos'): return
        if entity.block_id not in engine.hand_block_vaos: return
        
        vao, vbo, num_verts = engine.hand_block_vaos[entity.block_id]
        
        # Calculate matrix
        scale = 0.5 # Small item
        
        # Bobbing
        bob = math.sin((entity.age + partial_tick) * 0.1 + entity.bob_off) * 0.1 + 0.1
        ty = y + bob
        
        # Spinning
        rot = (entity.age + partial_tick) * 2.0
        
        cy, sy = math.cos(math.radians(rot)), math.sin(math.radians(rot))
        ry = np.array([
            [cy, 0, sy, 0],
            [0, 1, 0, 0],
            [-sy, 0, cy, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        t_mat = np.array([
            [1, 0, 0, x],
            [0, 1, 0, ty],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        s_mat = np.array([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        model_matrix = t_mat @ ry @ s_mat
        
        # Multiply view matrix by model matrix
        # camera_view is a 16-element list or array
        cv_mat = np.array(camera_view, dtype=np.float32).reshape(4, 4).T
        final_mat = (cv_mat @ model_matrix).T.flatten()
        
        glUniformMatrix4fv(u_view_loc, 1, GL_FALSE, (GLfloat * 16)(*final_mat))
        
        # Re-enable face culling specifically for items if disabled by entities
        glEnable(GL_CULL_FACE)
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, engine.texture_id)
        
        glBindVertexArray(vao)
        glDrawArrays(GL_TRIANGLES, 0, num_verts)
        glBindVertexArray(0)
        
        glDisable(GL_CULL_FACE)
