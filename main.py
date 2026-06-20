"""
PythonCraft Engine
Main entry point - Window, camera, world generation, and render loop.

WASD: Movement
Mouse: Look direction
Space: Fly Up / Jump
Shift: Fly Down / Crouch
ESC: Exit
"""
import math
import os
import sys
import numpy as np
import pyglet
from pyglet.gl import *
from pyglet.window import key, mouse
import concurrent.futures
from renderer.mesh_builder import build_chunk_mesh_bg
import ctypes
import numpy as np
import math
import time
import os

from world.mc_terrain import generate_chunk, recalculate_chunk_light, CHUNK_SIZE, CHUNK_HEIGHT, AIR, WATER
from world.terrain import CACTUS, SAND
from renderer.mesh_builder import build_chunk_mesh
from core.player import Player
from core.raycast import raycast
from core.math_utils import perspective_matrix, look_at_matrix, normalize_vec, sub_vec, cross_vec, dot_vec
from renderer.shader import create_shader_program
from renderer.camera import Camera





# ────────────────────────── ANA MOTOR ─────────────────────────────────


from core.frustum import get_visible_chunk_indices
from world.mc_terrain import load_or_generate_chunk
from core.world_db import save_chunk


from renderer.gui_mixin import GUIMixin
from world.chunk_mixin import ChunkMixin
from core.input_mixin import InputMixin
from core.entity_mixin import EntityMixin

class PythonCraftEngine(pyglet.window.Window, InputMixin, ChunkMixin, GUIMixin, EntityMixin):
    def __init__(self, render_distance=4, fast_leaves=False, debug_mode=False, flat_mode=False):
        super().__init__(width=1280, height=720, caption="PythonCraft Engine",
                         resizable=True, vsync=False)
        self.debug_mode = debug_mode
        self.flat_mode = flat_mode
        
        self.set_exclusive_mouse(True)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        
        glClearColor(0.47, 0.65, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        shader_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shaders')
        self.program = create_shader_program(
            os.path.join(shader_dir, 'vertex.glsl'),
            os.path.join(shader_dir, 'fragment.glsl')
        )
        self.water_overlay_program = create_shader_program(
            os.path.join(shader_dir, 'water_overlay_vertex.glsl'),
            os.path.join(shader_dir, 'water_overlay_fragment.glsl')
        )
        self.line_program = create_shader_program(
            os.path.join(shader_dir, 'line_vertex.glsl'),
            os.path.join(shader_dir, 'line_fragment.glsl')
        )
        self.break_program = create_shader_program(
            os.path.join(shader_dir, 'break_vertex.glsl'),
            os.path.join(shader_dir, 'break_fragment.glsl')
        )
        print("[GPU] Shader programs compiled & linked.")
        
        self.dummy_vao = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(self.dummy_vao))
        
        self.u_projection = glGetUniformLocation(self.program, b"projection")
        self.u_view = glGetUniformLocation(self.program, b"view")
        self.u_line_proj = glGetUniformLocation(self.line_program, b"u_projection")
        self.u_line_view = glGetUniformLocation(self.line_program, b"u_view")
        self.u_line_model = glGetUniformLocation(self.line_program, b"u_model")
        
        self.u_break_proj = glGetUniformLocation(self.break_program, b"u_projection")
        self.u_break_view = glGetUniformLocation(self.break_program, b"u_view")
        self.u_break_model = glGetUniformLocation(self.break_program, b"u_model")
        self.u_break_texture = glGetUniformLocation(self.break_program, b"u_texture")
        self.u_break_layer = glGetUniformLocation(self.break_program, b"u_layer")
        
        # Create wireframe cube VAO for block highlighting
        cube_lines = [
            0,0,0, 1,0,0,  1,0,0, 1,0,1,  1,0,1, 0,0,1,  0,0,1, 0,0,0, # bottom
            0,1,0, 1,1,0,  1,1,0, 1,1,1,  1,1,1, 0,1,1,  0,1,1, 0,1,0, # top
            0,0,0, 0,1,0,  1,0,0, 1,1,0,  1,0,1, 1,1,1,  0,0,1, 0,1,1  # sides
        ]
        line_data = (GLfloat * len(cube_lines))(*cube_lines)
        self.line_vao = GLuint(0)
        self.line_vbo = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(self.line_vao))
        glGenBuffers(1, ctypes.byref(self.line_vbo))
        glBindVertexArray(self.line_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.line_vbo)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(line_data), line_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glBindVertexArray(0)

        # Create break block VAO
        cube_break = [
            0,0,1, 0,0,  1,0,1, 1,0,  1,1,1, 1,1,  1,1,1, 1,1,  0,1,1, 0,1,  0,0,1, 0,0, # front
            1,0,0, 0,0,  0,0,0, 1,0,  0,1,0, 1,1,  0,1,0, 1,1,  1,1,0, 0,1,  1,0,0, 0,0, # back
            0,0,0, 0,0,  0,0,1, 1,0,  0,1,1, 1,1,  0,1,1, 1,1,  0,1,0, 0,1,  0,0,0, 0,0, # left
            1,0,1, 0,0,  1,0,0, 1,0,  1,1,0, 1,1,  1,1,0, 1,1,  1,1,1, 0,1,  1,0,1, 0,0, # right
            0,1,1, 0,0,  1,1,1, 1,0,  1,1,0, 1,1,  1,1,0, 1,1,  0,1,0, 0,1,  0,1,1, 0,0, # top
            0,0,0, 0,0,  1,0,0, 1,0,  1,0,1, 1,1,  1,0,1, 1,1,  0,0,1, 0,1,  0,0,0, 0,0  # bottom
        ]
        break_data = (GLfloat * len(cube_break))(*cube_break)
        self.break_vao = GLuint(0)
        self.break_vbo = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(self.break_vao))
        glGenBuffers(1, ctypes.byref(self.break_vbo))
        glBindVertexArray(self.break_vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.break_vbo)
        glBufferData(GL_ARRAY_BUFFER, ctypes.sizeof(break_data), break_data, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(GLfloat), ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * ctypes.sizeof(GLfloat), ctypes.c_void_p(3 * ctypes.sizeof(GLfloat)))
        glEnableVertexAttribArray(1)
        glBindVertexArray(0)

        self.u_texture = glGetUniformLocation(self.program, b"u_texture")
        self.u_tint_color = glGetUniformLocation(self.program, b"u_tint_color")
        
        self.u_inv_proj_view_overlay = glGetUniformLocation(self.water_overlay_program, b"u_inv_proj_view")
        self.u_water_surface_y_overlay = glGetUniformLocation(self.water_overlay_program, b"u_water_surface_y")
        
        glUseProgram(self.program)
        glUniform1i(self.u_texture, 0)
        glUseProgram(0)
        
        from core.texture_manager import TextureManager
        texture_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'textures')
        self.texture_manager = TextureManager(texture_dir, fast_leaves=fast_leaves)
        self.texture_manager.load_textures()
        self.block_layers = self.texture_manager.get_uvs_for_blocks()
        self.block_overlays = self.texture_manager.get_overlays_for_blocks()
        
        self.destroy_stages = []
        for i in range(10):
            tex_name = f"destroy_stage_{i}.png"
            if tex_name in self.texture_manager.tex_names_to_layer:
                self.destroy_stages.append(self.texture_manager.tex_names_to_layer[tex_name])
            else:
                self.destroy_stages.append(0)
        
        tex_data, num_layers = self.texture_manager.get_texture_array_data()
        
        self.texture_id = GLuint(0)
        glGenTextures(1, ctypes.byref(self.texture_id))
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_REPEAT)
        
        from pyglet.gl import glTexImage3D
        tex_buffer = (ctypes.c_ubyte * len(tex_data)).from_buffer_copy(tex_data)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 16, 16, num_layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_buffer)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        
        self.camera = Camera()
        self.player = Player(32.0, 100.0, 32.0)
        self.hotbar_blocks = [0] * 9
        self.hotbar_counts = [0] * 9
        self.selected_slot = 0
        self.selected_block_id = self.hotbar_blocks[self.selected_slot]
        
        # Mouse hold state and action cooldown tracking
        self.mouse_held = {mouse.LEFT: False, mouse.RIGHT: False}
        self.mouse_action_cooldown = 0.0
        self.breaking_pos = None
        self.breaking_progress = 0.0
        
        self.world_chunks = {}
        self.world_light_maps = {}
        self.world_biomes = {}
        self.pending_decorations = {}
        self.total_verts = 0
        self.rendered_chunks = 0
        
        self._init_world_system(render_distance)
        self._init_gui()
        self.bob_time = 0.0
        self.swing_time = 0.0
        self._init_hand_blocks()
        self._init_entities()
        
        self._frame_count = 0
        pyglet.clock.schedule_interval(self._update_title, 0.5)
        pyglet.clock.schedule_interval(self.update, 1.0 / 120.0)
        
        print("=============================================")
        print("   ENGINE READY. ENTERING MAIN GAME LOOP.    ")
        print("   WASD: Movement | Mouse: Look | ESC: Exit ")
        print("=============================================")

    def update(self, dt):
        import time
        t_start = time.perf_counter()
        dt = min(dt, 0.05)
        
        # Update hand swing and bobbing animations
        if self.swing_time > 0.0:
            self.swing_time += dt * 5.0
            if self.swing_time >= 1.0:
                self.swing_time = 0.0
                
        # Bobbing speed depends on walking state
        is_moving = False
        dx, dz = 0.0, 0.0
        front = self.camera.get_front()
        flat_front = normalize_vec([front[0], 0, front[2]])
        right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
        if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
        if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
        if self.keys[key.A]: dx -= right[0]; dz -= right[2]
        if self.keys[key.D]: dx += right[0]; dz += right[2]
        if math.sqrt(dx*dx + dz*dz) > 0.001:
            is_moving = True
            
        if is_moving and not self.player.is_flying:
            self.bob_time += dt * 10.0
        else:
            self.bob_time += dt * 1.5 # Slow breathing bobbing animation
            
        # Mouse hold actions (continuous block break/place)
        if self.mouse_action_cooldown > 0.0:
            self.mouse_action_cooldown -= dt
            
        if self.mouse_action_cooldown <= 0.005:
            if self.mouse_held[mouse.LEFT]:
                self.swing_time = 0.01
                self._handle_mouse_action(mouse.LEFT)
                self.mouse_action_cooldown = 0.20
            elif self.mouse_held[mouse.RIGHT]:
                from world.terrain import PORKCHOP_RAW
                if self.selected_block_id == PORKCHOP_RAW:
                    pass # Handled below for continuous eating
                else:
                    self.swing_time = 0.01
                    self._handle_mouse_action(mouse.RIGHT)
                    self.mouse_action_cooldown = 0.20
                
        # Continuous Eating Progress
        if self.mouse_held[mouse.RIGHT]:
            from world.terrain import PORKCHOP_RAW
            if self.selected_block_id == PORKCHOP_RAW and self.player.hunger < 20.0:
                if not hasattr(self, 'eating_progress'):
                    self.eating_progress = 0.0
                self.eating_progress += dt
                # Bob hand slightly while eating
                self.swing_time += dt * 3.0
                
                if self.eating_progress >= 1.6: # Takes 1.6 seconds to eat
                    self.eating_progress = 0.0
                    self.player.hunger = min(20.0, getattr(self.player, 'hunger', 20.0) + 4.0) # Restores 4 hunger points (2 shanks)
                    self.hotbar_counts[self.selected_slot] -= 1
                    if self.hotbar_counts[self.selected_slot] <= 0:
                        self.hotbar_blocks[self.selected_slot] = 0
                        self.selected_block_id = 0
            else:
                self.eating_progress = 0.0
        else:
            if hasattr(self, 'eating_progress'):
                self.eating_progress = 0.0
                
        # Always update targeted block for highlighting
        eye_pos = self.player.get_eye_position()
        direction = self.camera.get_front()
        from core.raycast import raycast
        hx, hy, hz, px, py, pz = raycast(eye_pos, direction, self.get_block)
        if hx is not None:
            self.targeted_block = (hx, hy, hz)
        else:
            self.targeted_block = None

        # Continuous Block Breaking Progress
        if self.mouse_held[mouse.LEFT]:
            if self.targeted_block is not None:
                hx, hy, hz = self.targeted_block
                current_target = (hx, hy, hz)
                if getattr(self, 'breaking_pos', None) == current_target:
                    broken_id = self.get_block(hx, hy, hz)
                    if broken_id > 0:
                        from world.terrain import BLOCK_HARDNESS_ARRAY
                        hardness = BLOCK_HARDNESS_ARRAY[broken_id]
                        if hardness >= 0:
                            self.breaking_progress += dt
                            import random
                            if random.random() < 0.25:
                                self.spawn_crack_particles(hx, hy, hz, broken_id, 1)
                                
                            required_time = hardness * 5.0
                            if self.breaking_progress >= required_time:
                                self.spawn_destruction_particles(hx, hy, hz, broken_id)
                                self.spawn_item_entity(broken_id, hx + 0.5, hy + 0.5, hz + 0.5)
                                self.set_block(hx, hy, hz, 0)
                                self.breaking_pos = None
                                self.breaking_progress = 0.0
                                self.mouse_action_cooldown = 0.20
                else:
                    self.breaking_pos = current_target
                    self.breaking_progress = 0.0
            else:
                self.breaking_pos = None
                self.breaking_progress = 0.0
        
        t_load_start = time.perf_counter()
        self._update_chunk_loading()
        t_load_end = time.perf_counter()
        
        t_queue_start = time.perf_counter()
        self._process_chunk_queues()
        t_queue_end = time.perf_counter()
        
        front = self.camera.get_front()
        flat_front = normalize_vec([front[0], 0, front[2]])
        right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
        
        dx, dz = 0.0, 0.0
        if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
        if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
        if self.keys[key.A]: dx -= right[0]; dz -= right[2]
        if self.keys[key.D]: dx += right[0]; dz += right[2]
            
        length = math.sqrt(dx*dx + dz*dz)
        if length > 0.001:
            dx /= length
            dz /= length
            
        jump = self.keys[key.SPACE]
        crouch = self.keys[key.LSHIFT]
        sprint = self.keys[key.LCTRL]
        
        # Prevent player from falling if inside an unloaded chunk
        pcx = int(math.floor(self.player.x / CHUNK_SIZE))
        pcz = int(math.floor(self.player.z / CHUNK_SIZE))
        
        t_player_start = time.perf_counter()
        if (pcx, pcz) in self.world_chunks:
            self.player.update(dt, dx, dz, jump, crouch, sprint, self.get_block)
        else:
            self.player.vy = 0.0 # Reset gravity accumulation
        t_player_end = time.perf_counter()
        
        self._update_entities(dt)
            
        eye_pos = self.player.get_eye_position()
        self.camera.x, self.camera.y, self.camera.z = eye_pos[0], eye_pos[1], eye_pos[2]
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 2.0:
            self.log(f"[SLOW UPDATE] Total: {dur:.2f}ms | Load: {(t_load_end-t_load_start)*1000.0:.2f}ms | Queues: {(t_queue_end-t_queue_start)*1000.0:.2f}ms | Player: {(t_player_end-t_player_start)*1000.0:.2f}ms")
    
    def on_draw(self):
        import time
        t_start = time.perf_counter()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        
        aspect = self.width / max(self.height, 1)
        proj = perspective_matrix(70.0, aspect, 0.1, 1000.0)
        view = self.camera.get_view_matrix()
        
        glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, view)
        
        t_cull_start = time.perf_counter()
        # Proj * View matrix multiplication (Manual)
        clip = [0.0] * 16
        for i in range(4):
            for j in range(4):
                clip[i*4+j] = proj[0*4+j]*view[i*4+0] + proj[1*4+j]*view[i*4+1] + proj[2*4+j]*view[i*4+2] + proj[3*4+j]*view[i*4+3]
        proj_view = np.array(clip, dtype=np.float32)
        
        # Light-speed Frustum Culling using Numba
        visible_count = get_visible_chunk_indices(proj_view, self.chunk_bounds, self.chunk_active, self.visible_indices)
        self.rendered_chunks = visible_count
        t_cull_end = time.perf_counter()
        
        # Trigger draw calls only for visible chunks
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        t_opaque_start = time.perf_counter()
        # OPAQUE PASS
        for i in range(visible_count):
            chunk_idx = self.visible_indices[i]
            o_vao, _, o_count, _, _, _ = self.chunk_vaos_array[chunk_idx]
            if o_count > 0:
                glBindVertexArray(o_vao)
                glDrawArrays(GL_TRIANGLES, 0, o_count)
        t_opaque_end = time.perf_counter()
        
        # ENTITY PASS
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE) # Disable culling for entities
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        self._render_entities(view, self.u_view, getattr(self, 'u_tint_color', -1))
        
        # Restore the original camera view after entity rendering
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, view)
        glEnable(GL_CULL_FACE) # Re-enable for chunks
        
        # Re-bind main chunk texture atlas
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)

        t_trans_start = time.perf_counter()
        # TRANSPARENT PASS
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        for i in range(visible_count):
            chunk_idx = self.visible_indices[i]
            _, _, _, t_vao, _, t_count = self.chunk_vaos_array[chunk_idx]
            if t_count > 0:
                glBindVertexArray(t_vao)
                glDrawArrays(GL_TRIANGLES, 0, t_count)
                
        glDisable(GL_BLEND)
        glBindVertexArray(0)
        glUseProgram(0)

        # RENDER BLOCK HIGHLIGHT
        if getattr(self, 'targeted_block', None) is not None:
            bx, by, bz = self.targeted_block
            glUseProgram(self.line_program)
            glUniformMatrix4fv(self.u_line_proj, 1, GL_FALSE, proj)
            glUniformMatrix4fv(self.u_line_view, 1, GL_FALSE, view)
            
            s = 1.005
            o = (1.0 - s) / 2.0
            model_mat = np.array([
                [s, 0, 0, 0],
                [0, s, 0, 0],
                [0, 0, s, 0],
                [bx + o, by + o, bz + o, 1]
            ], dtype=np.float32)
            
            glUniformMatrix4fv(self.u_line_model, 1, GL_FALSE, (GLfloat * 16)(*model_mat.flatten()))
            
            glLineWidth(2.0)
            glBindVertexArray(self.line_vao)
            glDrawArrays(GL_LINES, 0, 24)
            glBindVertexArray(0)
            glLineWidth(1.0)
            glUseProgram(0)
            
            # Break animation overlay
            if getattr(self, 'breaking_pos', None) == (bx, by, bz) and getattr(self, 'breaking_progress', 0.0) > 0.0:
                block_id = self.get_block(bx, by, bz)
                if block_id > 0:
                    from world.terrain import BLOCK_HARDNESS_ARRAY
                    hardness = BLOCK_HARDNESS_ARRAY[block_id]
                    if hardness > 0:
                        req_time = hardness * 5.0
                        stage = int((self.breaking_progress / req_time) * 10)
                        if stage > 9: stage = 9
                        layer_idx = self.destroy_stages[stage]
                        if layer_idx > 0:
                            glUseProgram(self.break_program)
                            glUniformMatrix4fv(self.u_break_proj, 1, GL_FALSE, proj)
                            glUniformMatrix4fv(self.u_break_view, 1, GL_FALSE, view)
                            glUniformMatrix4fv(self.u_break_model, 1, GL_FALSE, (GLfloat * 16)(*model_mat.flatten()))
                            glUniform1f(self.u_break_layer, float(layer_idx))
                            
                            glEnable(GL_BLEND)
                            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
                            glActiveTexture(GL_TEXTURE0)
                            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
                            glUniform1i(self.u_break_texture, 0)
                            
                            glBindVertexArray(self.break_vao)
                            glDrawArrays(GL_TRIANGLES, 0, 36)
                            glBindVertexArray(0)
                            glDisable(GL_BLEND)
                            glUseProgram(0)
        
        # WATER OVERLAY PASS
        cx, cy, cz = self.camera.x, self.camera.y, self.camera.z
        cy_int = int(math.floor(cy))
        
        # Find closest water surface
        u_water_surface_y = -1000.0
        if self.get_block(cx, cy_int + 1, cz) == 4:
            u_water_surface_y = cy_int + 2.0
        elif self.get_block(cx, cy_int, cz) == 4:
            u_water_surface_y = cy_int + 1.0
        elif self.get_block(cx, cy_int - 1, cz) == 4:
            u_water_surface_y = float(cy_int)
            
        if u_water_surface_y > -999.0:
            glUseProgram(self.water_overlay_program)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST) # Overlay covers screen
            
            # Inverse proj*view for world reconstruction
            inv_pv = np.linalg.inv(proj_view.reshape(4,4)).flatten().astype(np.float32)
            glUniformMatrix4fv(self.u_inv_proj_view_overlay, 1, GL_FALSE, (ctypes.c_float * 16)(*inv_pv))
            glUniform1f(self.u_water_surface_y_overlay, u_water_surface_y)
            
            glBindVertexArray(self.dummy_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glBindVertexArray(0)
            
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glUseProgram(0)

        # 3D Held Block Viewmodel Rendering
        if hasattr(self, 'hand_block_vaos') and self.selected_block_id in self.hand_block_vaos:
            glClear(GL_DEPTH_BUFFER_BIT)
            glEnable(GL_DEPTH_TEST)
            glEnable(GL_CULL_FACE)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glUseProgram(self.program)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
            
            # Bobbing and swing animations
            bob_x = math.sin(self.bob_time) * 0.02
            bob_y = math.cos(self.bob_time * 2.0) * 0.015 - 0.01
            
            # Movement checks
            dx, dz = 0.0, 0.0
            front = self.camera.get_front()
            flat_front = normalize_vec([front[0], 0, front[2]])
            right = normalize_vec(cross_vec(flat_front, [0, 1, 0]))
            if self.keys[key.W]: dx += flat_front[0]; dz += flat_front[2]
            if self.keys[key.S]: dx -= flat_front[0]; dz -= flat_front[2]
            if self.keys[key.A]: dx -= right[0]; dz -= right[2]
            if self.keys[key.D]: dx += right[0]; dz += right[2]
            is_moving = math.sqrt(dx*dx + dz*dz) > 0.001
            
            if is_moving and not self.player.is_flying:
                bob_x = math.sin(self.bob_time) * 0.04
                bob_y = math.cos(self.bob_time * 2.0) * 0.03 - 0.02
                
            swing_tx = 0.0
            swing_ty = 0.0
            swing_tz = 0.0
            swing_rx = 0.0
            swing_ry = 0.0
            swing_rz = 0.0
            
            swing1 = 0.0
            swing2 = 0.0
            swing3 = 0.0
            
            if self.swing_time > 0.0:
                swing = self.swing_time
                swing1 = math.sin(swing * math.pi)
                swing2 = math.sin(math.sqrt(swing) * math.pi)
                swing3 = math.sin(swing * swing * math.pi)
                
                swing_tx = -swing2 * 0.4
                swing_ty = math.sin(math.sqrt(swing) * math.pi * 2.0) * 0.2
                swing_tz = -swing1 * 0.2
                
                swing_rx = -swing2 * 80.0
                swing_ry = -swing3 * 20.0
                swing_rz = -swing2 * 20.0
                
            tx = 0.48 + bob_x + swing_tx
            ty = -0.45 + bob_y + swing_ty
            tz = -0.75 + swing_tz
            
            scale = 0.36
            
            from world.terrain import PORKCHOP_RAW
            if self.selected_block_id == PORKCHOP_RAW:
                # Use Player gui ItemInHandRenderer transform
                # glTranslatef(0.7f * d, -0.65f * d - (1 - h) * 0.6f, -0.9f * d);
                # glRotatef(45, 0, 1, 0);
                # glRotatef(-swing3 * 20, 0, 1, 0);
                # glRotatef(-swing2 * 20, 0, 0, 1);
                # glRotatef(-swing2 * 80, 1, 0, 0);
                # glScalef(0.4, 0.4, 0.4)
                tx = 0.7 * 0.8 - swing2 * 0.4 * 0.8
                ty = -0.65 * 0.8 + math.sin(math.sqrt(self.swing_time)*math.pi*2)*0.2*0.8 + bob_y
                tz = -0.9 * 0.8 - swing1 * 0.2 * 0.8
                
                scale = 0.4 * 1.5
                
                rad_y = math.radians(45 - swing3 * 20)
                cy, sy = math.cos(rad_y), math.sin(rad_y)
                ry = np.array([
                    [cy, 0, sy, 0],
                    [0, 1, 0, 0],
                    [-sy, 0, cy, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_x = math.radians(-swing2 * 80)
                cx_val, sx_val = math.cos(rad_x), math.sin(rad_x)
                rx = np.array([
                    [1, 0, 0, 0],
                    [0, cx_val, -sx_val, 0],
                    [0, sx_val, cx_val, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z = math.radians(-swing2 * 20)
                cz, sz = math.cos(rad_z), math.sin(rad_z)
                rz = np.array([
                    [cz, -sz, 0, 0],
                    [sz, cz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                t_mat = np.array([
                    [1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                s_mat = np.array([
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                # Extra item rotations to tilt it forward and right
                rad_y_item = math.radians(50)
                cy_item, sy_item = math.cos(rad_y_item), math.sin(rad_y_item)
                ry_item = np.array([
                    [cy_item, 0, sy_item, 0],
                    [0, 1, 0, 0],
                    [-sy_item, 0, cy_item, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z_item = math.radians(335)
                cz_item, sz_item = math.cos(rad_z_item), math.sin(rad_z_item)
                rz_item = np.array([
                    [cz_item, -sz_item, 0, 0],
                    [sz_item, cz_item, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                hand_matrix = t_mat @ ry @ rx @ rz @ s_mat @ ry_item @ rz_item
            else:
                rad_y = math.radians(15 + swing_ry)
                cy, sy = math.cos(rad_y), math.sin(rad_y)
                ry = np.array([
                    [cy, 0, sy, 0],
                    [0, 1, 0, 0],
                    [-sy, 0, cy, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_x = math.radians(-15 + swing_rx)
                cx_val, sx_val = math.cos(rad_x), math.sin(rad_x)
                rx = np.array([
                    [1, 0, 0, 0],
                    [0, cx_val, -sx_val, 0],
                    [0, sx_val, cx_val, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                rad_z = math.radians(8 + swing_rz)
                cz, sz = math.cos(rad_z), math.sin(rad_z)
                rz = np.array([
                    [cz, -sz, 0, 0],
                    [sz, cz, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                t_mat = np.array([
                    [1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                s_mat = np.array([
                    [scale, 0, 0, 0],
                    [0, scale, 0, 0],
                    [0, 0, scale, 0],
                    [0, 0, 0, 1]
                ], dtype=np.float32)
                
                hand_matrix = t_mat @ ry @ rx @ rz @ s_mat
                
            flat_hand_matrix = hand_matrix.T.flatten()
            
            glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
            hand_view = (GLfloat * 16)(*flat_hand_matrix)
            glUniformMatrix4fv(self.u_view, 1, GL_FALSE, hand_view)
            
            h_vao, _, num_verts = self.hand_block_vaos[self.selected_block_id]
            glBindVertexArray(h_vao)
            glDrawArrays(GL_TRIANGLES, 0, num_verts)
            
            glBindVertexArray(0)
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
            glDisable(GL_BLEND)
            glUseProgram(0)
            
        # 2D GUI Elements (Crosshair and Hotbar)
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        
        # 1. Crosshair (using color inversion blending)
        glBlendFunc(GL_ONE_MINUS_DST_COLOR, GL_ONE_MINUS_SRC_COLOR)
        if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
            self.crosshair_sprite.draw()
            
        # 2. Hotbar ve Blok Simgeleri (Standart Alpha Blending ile)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
            self.hotbar_bg_sprite.draw()
            
            # Render block icons (scaled from 64x64 to 13*scale and centered in slots)
            bg_scale = self.hotbar_bg_sprite.scale
            if not hasattr(self, 'count_labels'):
                import pyglet.text
                import pyglet.font
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                try:
                    pyglet.font.add_file(font_path)
                    target_font = 'Minecraftia'
                except Exception:
                    target_font = 'Arial'
                self.count_labels = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(9)]
                
            # Render active slot selection frame FIRST (behind items and text)
            if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
                scale = self.hotbar_sel_sprite.scale
                self.hotbar_sel_sprite.x = int(self.hotbar_bg_sprite.x - 1 * scale + self.selected_slot * 20 * scale)
                self.hotbar_sel_sprite.y = int(self.hotbar_bg_sprite.y - 1 * scale)
                self.hotbar_sel_sprite.draw()

            for slot_idx, b_id in enumerate(self.hotbar_blocks):
                if b_id > 0 and b_id in self.block_icon_sprites:
                    sprite = self.block_icon_sprites[b_id]
                    sprite_size = getattr(sprite, 'original_width', 64.0)
                    sprite.scale = (13.0 * bg_scale) / sprite_size
                    sprite.x = int(self.hotbar_bg_sprite.x + (3 + slot_idx * 20) * bg_scale + 1.5 * bg_scale)
                    sprite.y = int(self.hotbar_bg_sprite.y + 3 * bg_scale + 1.5 * bg_scale)
                    sprite.draw()
                    
                    count = self.hotbar_counts[slot_idx]
                    if count > 0:
                        lbl = self.count_labels[slot_idx]
                        
                        target_size = max(8, int(8 * bg_scale))
                        if lbl.font_size != target_size:
                            lbl.font_size = target_size
                            
                        if lbl.text != str(count):
                            lbl.text = str(count)
                        
                        # Position slightly outside the item bounds (x+20, y+2 relative to item top-left)
                        base_x = self.hotbar_bg_sprite.x + (20 + slot_idx * 20) * bg_scale
                        base_y = self.hotbar_bg_sprite.y + 2 * bg_scale
                        offset = max(1, int(1 * bg_scale))
                        
                        # Draw Shadow
                        lbl.x = base_x + offset
                        lbl.y = base_y - offset
                        lbl.color = (63, 63, 63, 255)
                        lbl.draw()
                        
                        # Draw Main Text
                        lbl.x = base_x
                        lbl.y = base_y
                        lbl.color = (255, 255, 255, 255)
                        lbl.draw()

            # Render XP Bar (just for visual for now, 30% full)
            if hasattr(self, 'spr_xp_empty') and hasattr(self, 'spr_xp_full'):
                xp_x = self.hotbar_bg_sprite.x
                xp_y = int(self.hotbar_bg_sprite.y + 24 * self.hotbar_bg_sprite.scale)
                
                self.spr_xp_empty.scale = self.hotbar_bg_sprite.scale
                self.spr_xp_empty.x = xp_x
                self.spr_xp_empty.y = xp_y
                self.spr_xp_empty.draw()
                
                # XP bar is completely empty for now
                # self.spr_xp_full.draw()

            # Render Health and Hunger Bars
            if hasattr(self, 'spr_heart') and hasattr(self, 'spr_hunger'):
                # Draw Health (10 hearts)
                heart_start_x = self.hotbar_bg_sprite.x
                bar_y = int(self.hotbar_bg_sprite.y + 32 * self.hotbar_bg_sprite.scale)
                
                player_health = getattr(self.player, 'health', 20.0)
                player_hunger = getattr(self.player, 'hunger', 20.0)
                tick_count = int(time.perf_counter() * 20)
                
                heart_offset_index = -1
                if player_health < 20.0 and player_hunger >= 18.0:
                    heart_offset_index = tick_count % 25
                    
                import random
                for i in range(10):
                    yo = bar_y
                    if player_health <= 4.0:
                        yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                    if i == heart_offset_index:
                        yo += 2 * int(self.hotbar_bg_sprite.scale)
                        
                    self.spr_heart.x = heart_start_x + i * 8 * self.spr_heart.scale
                    self.spr_heart.y = yo
                    
                    # Always draw the empty heart (black frame) first
                    self.spr_heart.image = self.tex_heart_empty
                    self.spr_heart.draw()
                    
                    # Draw the red inner heart (full or half) on top if needed
                    h_val = player_health - (i * 2)
                    if h_val >= 2:
                        self.spr_heart.image = self.tex_heart_full
                        self.spr_heart.draw()
                    elif h_val >= 1:
                        self.spr_heart.image = self.tex_heart_half
                        self.spr_heart.draw()
                    
                # Draw Hunger (10 icons, right-aligned)
                hunger_start_x = self.hotbar_bg_sprite.x + self.hotbar_bg_sprite.width - 9 * self.spr_hunger.scale
                for i in range(10):
                    yo = bar_y
                    if player_hunger <= 0.0 and tick_count % 20 < 10: # shake slightly when starving
                         yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                         
                    self.spr_hunger.x = hunger_start_x - i * 8 * self.spr_hunger.scale
                    self.spr_hunger.y = yo
                    
                    # Always draw the empty hunger background first
                    self.spr_hunger.image = self.tex_hunger_empty
                    self.spr_hunger.draw()
                    
                    # Draw the inner food (full or half) on top if needed
                    f_val = player_hunger - (i * 2)
                    if f_val >= 2:
                        self.spr_hunger.image = self.tex_hunger_full
                        self.spr_hunger.draw()
                    elif f_val >= 1:
                        self.spr_hunger.image = self.tex_hunger_half
                        self.spr_hunger.draw()

                # Render Bubbles if underwater
                if getattr(self.player, 'is_head_in_water', False) and hasattr(self, 'spr_bubble'):
                    bubble_y = bar_y + 10 * scale
                    air_supply = getattr(self.player, 'air_supply', 300.0)
                    air_scale = 10.0 / 300.0
                    air_scaled = air_supply * air_scale
                    count = int(math.ceil((air_supply - 2) * air_scale))
                    extra = int(math.ceil(air_scaled)) - count
                    
                    for i in range(count + extra):
                        bx = hunger_start_x - (i * 8 * scale)
                        self.spr_bubble.x = bx
                        self.spr_bubble.y = bubble_y
                        if i < count:
                            self.spr_bubble.image = self.tex_bubble_full
                        else:
                            self.spr_bubble.image = self.tex_bubble_popped
                        self.spr_bubble.draw()

        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        t_trans_end = time.perf_counter()
        
        self._frame_count += 1
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 4.0:
            self.log(f"[SLOW DRAW] Total: {dur:.2f}ms | Cull: {(t_cull_end-t_cull_start)*1000.0:.2f}ms | Opaque: {(t_opaque_end-t_opaque_start)*1000.0:.2f}ms | Trans: {(t_trans_end-t_trans_start)*1000.0:.2f}ms")

    def on_resize(self, width, height):
        super().on_resize(width, height)
        
        # Minecraft-style GUI scale (scales up in integer steps based on resolution)
        scale = max(2, int(width // 400))
        
        if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
            self.hotbar_bg_sprite.scale = scale
        if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
            self.crosshair_sprite.scale = scale
        if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
            self.hotbar_sel_sprite.scale = scale
        if hasattr(self, 'spr_heart'):
            self.spr_heart.scale = scale
        if hasattr(self, 'spr_hunger'):
            self.spr_hunger.scale = scale
        if hasattr(self, 'spr_bubble'):
            self.spr_bubble.scale = scale
        if hasattr(self, 'spr_xp_empty'):
            self.spr_xp_empty.scale = scale
        if hasattr(self, 'spr_xp_full'):
            self.spr_xp_full.scale = scale
            
        if hasattr(self, '_update_gui_positions'):
            self._update_gui_positions(width, height)

    def on_key_press(self, symbol, modifiers):
        super().on_key_press(symbol, modifiers)
        InputMixin.on_key_press(self, symbol, modifiers)
        if symbol == key.P:
            self.spawn_pig(self.player.x, self.player.y + 2.5, self.player.z)
    
    def _update_title(self, dt):
        fps = self._frame_count / max(dt, 0.001)
        self._frame_count = 0
        cam = self.camera
        queued = len(self.chunk_load_queue) + len(self.chunk_mesh_queue)
        
        self.set_caption(
            f"Pythoncraft | FPS: {fps:.0f} | "
            f"Pos: ({cam.x:.0f}, {cam.y:.0f}, {cam.z:.0f}) | "
            f"Chunks: {self.rendered_chunks}/{len(self.world_chunks)} (Q:{queued}) | "
            f"Verts: {self.total_verts:,}"
        )

def main():
    import sys
    
    print("=============================================")
    print("      PYTHONCRAFT ENGINE INITIATING...       ")
    print("=============================================")
    
    distance = 4
    fast_leaves = False
    debug_mode = False
    flat_mode = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == "-fast":
                fast_leaves = True
                print("Fast mode enabled: Leaves are now opaque.")
            elif arg == "-debug":
                debug_mode = True
                print("Debug mode enabled: Performance metrics will be printed to console.")
            elif arg == "-flat":
                flat_mode = True
                print("Flat mode enabled: World will be generated as flat.")
            else:
                try:
                    distance = int(arg)
                    print(f"User requested Render Distance: {distance} ({distance*2}x{distance*2} = {distance*distance*4} chunks)")
                except ValueError:
                    pass
                    
    if fast_leaves:
        from world.terrain import BLOCK_OPAQUE_ARRAY
        BLOCK_OPAQUE_ARRAY[12] = True
        BLOCK_OPAQUE_ARRAY[16] = True
        BLOCK_OPAQUE_ARRAY[17] = True
    
    engine = PythonCraftEngine(render_distance=distance, fast_leaves=fast_leaves, debug_mode=debug_mode, flat_mode=flat_mode)
    pyglet.app.run()

if __name__ == '__main__':
    main()
