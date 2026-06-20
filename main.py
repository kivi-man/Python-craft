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
        print("[GPU] Shader programs compiled & linked.")
        
        self.dummy_vao = GLuint(0)
        glGenVertexArrays(1, ctypes.byref(self.dummy_vao))
        
        self.u_projection = glGetUniformLocation(self.program, b"projection")
        self.u_view = glGetUniformLocation(self.program, b"view")
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
        self.hotbar_blocks = [1, 3, 20, 12, 4, CACTUS, 0, 0, 0]
        self.selected_slot = 1 # Start with GRASS selected (slot 1)
        self.selected_block_id = self.hotbar_blocks[self.selected_slot]
        
        # Mouse hold state and action cooldown tracking
        self.mouse_held = {mouse.LEFT: False, mouse.RIGHT: False}
        self.mouse_action_cooldown = 0.0
        
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
                self.swing_time = 0.01
                self._handle_mouse_action(mouse.RIGHT)
                self.mouse_action_cooldown = 0.20
        
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
            
            h_vao, _ = self.hand_block_vaos[self.selected_block_id]
            glBindVertexArray(h_vao)
            glDrawArrays(GL_TRIANGLES, 0, 36)
            
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
            for slot_idx, b_id in enumerate(self.hotbar_blocks):
                if b_id in self.block_icon_sprites:
                    sprite = self.block_icon_sprites[b_id]
                    sprite.scale = (13.0 * bg_scale) / 64.0
                    sprite.x = int(self.hotbar_bg_sprite.x + (3 + slot_idx * 20) * bg_scale + 1.5 * bg_scale)
                    sprite.y = int(self.hotbar_bg_sprite.y + 3 * bg_scale + 1.5 * bg_scale)
                    sprite.draw()
                    
            # Render active slot selection frame
            if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
                scale = self.hotbar_sel_sprite.scale
                self.hotbar_sel_sprite.x = int(self.hotbar_bg_sprite.x - 1 * scale + self.selected_slot * 20 * scale)
                self.hotbar_sel_sprite.y = int(self.hotbar_bg_sprite.y - 1 * scale)
                self.hotbar_sel_sprite.draw()
            
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        t_trans_end = time.perf_counter()
        
        self._frame_count += 1
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 4.0:
            self.log(f"[SLOW DRAW] Total: {dur:.2f}ms | Cull: {(t_cull_end-t_cull_start)*1000.0:.2f}ms | Opaque: {(t_opaque_end-t_opaque_start)*1000.0:.2f}ms | Trans: {(t_trans_end-t_trans_start)*1000.0:.2f}ms")

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
