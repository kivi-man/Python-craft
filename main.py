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
import time
import ctypes
import random
import concurrent.futures

import numpy as np
import pyglet
import pyglet.font
import pyglet.shapes
import pyglet.text
from pyglet.gl import *
from pyglet.window import key, mouse

from core.player import Player
from core.world_db import load_level_dat, save_level_dat
from core.raycast import raycast
from core.math_utils import perspective_matrix, look_at_matrix, normalize_vec, sub_vec, cross_vec, dot_vec
from core.recipes import RecipeManager
from core.texture_manager import TextureManager

from renderer.mesh_builder import build_chunk_mesh, build_chunk_mesh_bg
from renderer.shader import create_shader_program
from renderer.camera import Camera

from world.mc_terrain import generate_chunk, recalculate_chunk_light, CHUNK_SIZE, CHUNK_HEIGHT, AIR, WATER
from world.terrain import CACTUS, SAND, BLOCK_MAX_STACK_ARRAY, BLOCK_HARDNESS_ARRAY, BLOCK_OPAQUE_ARRAY, PORKCHOP_RAW, get_break_time
from core.inventory_mixin import InventoryMixin
# ────────────────────────── ANA MOTOR ─────────────────────────────────


from core.frustum import get_visible_chunk_indices
from world.mc_terrain import load_or_generate_chunk
from core.world_db import save_chunk
from core.sound_system import SoundSystem


def check_gl_version():
    """Checks for GL 4.3 support."""
    version_str = pyglet.gl.gl_info.get_version()
    try:
        major, minor = map(int, version_str.split('.')[:2])
        if major < 4 or (major == 4 and minor < 3):
            print(f"Warning: GL version {version_str} is below 4.3. Some features may not work.")
    except:
        pass

from renderer.gui_mixin import GUIMixin
from world.chunk_mixin import ChunkMixin
from core.input_mixin import InputMixin
from core.entity_mixin import EntityMixin
from core.inventory_mixin import InventoryMixin
from core.engine_setup_mixin import EngineSetupMixin
from core.player_update_mixin import PlayerUpdateMixin

class PythonCraftEngine(pyglet.window.Window, InputMixin, ChunkMixin, GUIMixin, EntityMixin, InventoryMixin, EngineSetupMixin, PlayerUpdateMixin):
    def __init__(self, render_distance=4, simulation_distance=4, fast_leaves=False, debug_mode=False, flat_mode=False, console_mode=False, config=None, gpu_mode=False):
        super().__init__(width=1280, height=720, caption="PythonCraft Engine", resizable=True, vsync=False, config=config)
        self.gpu_mode = gpu_mode
        self.render_distance = render_distance
        self.simulation_distance = simulation_distance
        self.debug_mode = debug_mode
        self.flat_mode = flat_mode
        self.console_mode = console_mode
        self.set_exclusive_mouse(True)
        self.keys = key.KeyStateHandler()
        self.push_handlers(self.keys)
        self._init_core_variables()
        self._setup_opengl_and_shaders(fast_leaves)
        self._setup_textures_and_materials(fast_leaves)
        self._init_world_system(render_distance)
        self._init_gui()
        self.bob_time = 0.0
        self.swing_time = 0.0
        
        if self.console_mode:
            import socket
            import subprocess
            self.console_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.console_sock.bind(('127.0.0.1', 25565))
            self.console_sock.setblocking(False)
            pyglet.clock.schedule_interval(self.poll_console, 0.1)
            try:
                # Open a new terminal window on Windows
                subprocess.Popen([sys.executable, "console_client.py"], creationflags=subprocess.CREATE_NEW_CONSOLE)
                print("[Console] Server started on port 25565, console window spawned.")
            except Exception as e:
                print(f"[Console] Failed to launch console window: {e}")
                
        self._init_hand_blocks()
        self._init_entities()
        
        # Pre-initialized update() state (avoids hasattr checks in hot loop)
        self.eating_progress = 0.0
        self.last_eat_sound = 0.0
        self.distance_walked = 0.0
        self.last_dig_sound_time = 0.0
        self.last_raycast_eye_pos = None
        self.last_raycast_dir = None
        self._camera_matrix_dirty = True
        self._cached_inv_pv = None
        
        self._frame_count = 0
        pyglet.clock.schedule_interval(self._update_title, 0.5)
        pyglet.clock.schedule_interval(self.update, 1.0 / 120.0)
        
        print("=============================================")
        print("   ENGINE READY. ENTERING MAIN GAME LOOP.    ")
        print("   WASD: Movement | Mouse: Look | ESC: Exit ")
        print("=============================================")

    def poll_console(self, dt):
        try:
            while True:
                data, addr = self.console_sock.recvfrom(1024)
                cmd = data.decode('utf-8').strip()
                self._handle_console_command(cmd)
        except BlockingIOError:
            pass
        except Exception as e:
            print(f"[Console] Error receiving: {e}")

    def _handle_console_command(self, cmd):
        parts = cmd.split()
        if not parts:
            return
        if parts[0].lower() == 'tp' and len(parts) >= 4:
            try:
                x = float(parts[1])
                z = float(parts[2])
                y = float(parts[3])
                self.player.x = x
                self.player.y = y
                self.player.z = z
                self.camera.position = (x, y, z)
                print(f"[Console] Teleported to X:{x}, Z:{z}, Y:{y}")
            except ValueError:
                print("[Console] Invalid tp coordinates. Use: tp x z y")

    def on_close(self):
        self.is_closing = True
        print("[MAIN] Saving all chunks before exit. Please wait...")
        # Save all chunks
        for cx, cz in list(getattr(self, 'world_chunks', {}).keys()):
            if hasattr(self, '_unload_chunk'):
                self._unload_chunk(cx, cz)
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            
        print("[MAIN] Saving player data...")
        save_level_dat(
            [self.player.x, self.player.y, self.player.z],
            [self.camera.yaw, self.camera.pitch],
            self.inventory_blocks,
            self.inventory_counts
        )
            
        super().on_close()

    def update(self, dt):
        if getattr(self, 'is_closing', False): return
        if hasattr(self, 'sound_system'):
            self.sound_system.update_music(dt, dimension="OVERWORLD")
        t_start = time.perf_counter()
        dt = min(dt, 0.05)
        
        self._update_player_actions(dt)
        
        t_load_start = time.perf_counter()
        self._update_chunk_loading()
        t_load_end = time.perf_counter()
        
        t_queue_start = time.perf_counter()
        self._process_chunk_queues()
        t_queue_end = time.perf_counter()
        
        t_player_start = time.perf_counter()
        self._update_player_movement_and_physics(dt)
        t_player_end = time.perf_counter()
        
        self._update_entities(dt)
        self._update_audio_listener()
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 2.0:
            self.log(f"[SLOW UPDATE] Total: {dur:.2f}ms | Load: {(t_load_end-t_load_start)*1000.0:.2f}ms | Queues: {(t_queue_end-t_queue_start)*1000.0:.2f}ms | Player: {(t_player_end-t_player_start)*1000.0:.2f}ms")
    def on_draw(self):
        t_start = time.perf_counter()
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(self.program)
        
        aspect = self.width / max(self.height, 1)
        proj = perspective_matrix(70.0, aspect, 0.1, 1000.0)
        
        # Update camera position right before rendering to sync with mouse motion
        eye_pos = self.player.get_eye_position()
        self.camera.update_third_person(eye_pos[0], eye_pos[1], eye_pos[2], self.get_block_info)
        view = self.camera.get_view_matrix()
        
        glUniformMatrix4fv(self.u_projection, 1, GL_FALSE, proj)
        glUniformMatrix4fv(self.u_view, 1, GL_FALSE, view)
        
        t_cull_start = time.perf_counter()
        # Proj * View matrix multiplication (NumPy optimized)
        p = np.array(proj, dtype=np.float32).reshape(4, 4)
        v = np.array(view, dtype=np.float32).reshape(4, 4)
        proj_view = (v @ p).flatten()
        
        # Light-speed Frustum Culling using Numba
        self.chunk_renderer.update_frustum(proj_view)
        t_cull_end = time.perf_counter()
        
        # Re-bind the main rendering program (since compute shader might have changed it)
        glUseProgram(self.program)
        
        # Trigger draw calls only for visible chunks
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.texture_id)
        
        t_opaque_start = time.perf_counter()
        # OPAQUE PASS
        self.chunk_renderer.render_opaque()
        t_opaque_end = time.perf_counter()
        
        # ENTITY PASS
        t_entity_start = time.perf_counter()
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
        t_entity_end = time.perf_counter()

        t_trans_start = time.perf_counter()
        # TRANSPARENT PASS
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        self.chunk_renderer.render_transparent()
                
        glDisable(GL_BLEND)
        glBindVertexArray(0)
        glUseProgram(0)

        self._draw_block_highlight(proj, view)
        self._draw_water_overlay(proj_view)
        self._draw_held_block(proj, view)
        self._draw_hud()
        glDisable(GL_BLEND)
        glEnable(GL_DEPTH_TEST)
        
        t_trans_end = time.perf_counter()
        
        self._frame_count += 1
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 4.0:
            self.log(f"[SLOW DRAW] Total: {dur:.2f}ms | Cull: {(t_cull_end-t_cull_start)*1000.0:.2f}ms | Opaque: {(t_opaque_end-t_opaque_start)*1000.0:.2f}ms | Entity: {(t_entity_end-t_entity_start)*1000.0:.2f}ms | Trans: {(t_trans_end-t_trans_start)*1000.0:.2f}ms")

    def on_resize(self, width, height):
        super().on_resize(width, height)
        
        # Minecraft-style GUI scale (scales up in integer steps based on resolution)
        scale = max(2, int(width // 400))
        
        if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
            self.hotbar_bg_sprite.scale = scale
        if hasattr(self, 'inventory_bg_sprite') and self.inventory_bg_sprite is not None:
            self.inventory_bg_sprite.scale = scale
        if hasattr(self, 'crafting_bg_sprite') and self.crafting_bg_sprite is not None:
            self.crafting_bg_sprite.scale = scale
        if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
            self.crosshair_sprite.scale = scale
        if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
            self.hotbar_sel_sprite.scale = scale
        if hasattr(self, 'heart_bg_sprites'):
            for spr in self.heart_bg_sprites + self.heart_fg_sprites + self.hunger_bg_sprites + self.hunger_fg_sprites + self.bubble_sprites:
                spr.scale = scale
                
        if hasattr(self, 'spr_xp_empty'):
            self.spr_xp_empty.scale = scale
        if hasattr(self, 'spr_xp_full'):
            self.spr_xp_full.scale = scale
            
        if hasattr(self, '_update_gui_positions'):
            self._update_gui_positions(width, height)



    def on_key_press(self, symbol, modifiers):
        if symbol == key.W:
            current_time = time.time()
            if current_time - getattr(self, 'last_w_press_time', 0) < 0.3:
                self.is_sprinting_w = True
            self.last_w_press_time = current_time

        if symbol == key.ESCAPE:
            if getattr(self, 'inventory_open', False):
                self.inventory_open = False
                self.set_exclusive_mouse(True)
                self.mouse_locked = True
                self._drop_inventory_excess()
                return pyglet.event.EVENT_HANDLED
            elif getattr(self, 'crafting_open', False):
                self.crafting_open = False
                self.set_exclusive_mouse(True)
                self.mouse_locked = True
                self._drop_inventory_excess()
                return pyglet.event.EVENT_HANDLED
        
        super().on_key_press(symbol, modifiers)
        
        if symbol == key.E:
            if getattr(self, 'crafting_open', False):
                self.crafting_open = False
                self.set_exclusive_mouse(True)
                self.mouse_locked = True
                self._drop_inventory_excess()
                self._evaluate_crafting()
            else:
                self.inventory_open = not getattr(self, 'inventory_open', False)
                self.set_exclusive_mouse(not self.inventory_open)
                self.mouse_locked = not self.inventory_open
                if not self.inventory_open:
                    self._drop_inventory_excess()
                self._evaluate_crafting()
        elif symbol == key.P:
            self.spawn_pig(self.player.x, self.player.y + 2.5, self.player.z)
    

    def on_key_release(self, symbol, modifiers):
        if symbol == key.W:
            self.is_sprinting_w = False
        # super().on_key_release(symbol, modifiers)
        if hasattr(InputMixin, 'on_key_release'):
            InputMixin.on_key_release(self, symbol, modifiers)
            
    def on_deactivate(self):
        self.set_exclusive_mouse(False)
        self.mouse_locked = False
        if hasattr(super(), 'on_deactivate'):
            super().on_deactivate()
            
    def _update_title(self, dt):
        fps = self._frame_count / max(dt, 0.001)
        self._frame_count = 0
        cam = self.camera
        queued = len(self.chunk_load_queue) + len(self.chunk_mesh_queue)
        
        self.set_caption(
            f"Pythoncraft | FPS: {fps:.0f} | "
            f"Pos: ({cam.x:.0f}, {cam.y:.0f}, {cam.z:.0f}) | "
            f"Chunks: {self.chunk_renderer.visible_count}/{len(self.world_chunks)} (Q:{queued}) | "
            f"Sim: {getattr(self, 'simulation_distance', 4)} | "
            f"Verts: {self.chunk_renderer.total_verts:,}"
        )

def main():
    print("=============================================")
    print("      PYTHONCRAFT ENGINE INITIATING...       ")
    print("=============================================")
    
    distance = 4
    sim_distance = 4
    fast_leaves = False
    debug_mode = False
    flat_mode = False
    console_mode = False
    force_legacy = False
    
    import sys
    if len(sys.argv) > 1:
        args = sys.argv[1:]
        i = 0
        while i < len(args):
            arg = args[i]
            if arg == "-fast":
                fast_leaves = True
                print("Fast mode enabled: Leaves are now opaque.")
            elif arg == "-debug":
                debug_mode = True
                print("Debug mode enabled: Performance metrics will be printed to console.")
            elif arg == "-flat":
                flat_mode = True
                print("Flat mode enabled: World will be generated as flat.")
            elif arg == "-sim" and i + 1 < len(args):
                try:
                    sim_distance = int(args[i+1])
                    print(f"Simulation Distance set to: {sim_distance}")
                except ValueError:
                    pass
                i += 1
            elif arg == "-console":
                console_mode = True
                print("Console mode enabled.")
            elif arg == "-legacy":
                force_legacy = True
                print("Legacy mode forced via command line.")
            else:
                try:
                    distance = int(arg)
                    print(f"User requested Render Distance: {distance} ({distance*2}x{distance*2} = {distance*distance*4} chunks)")
                except ValueError:
                    pass
            i += 1
                    
    from world.terrain import BLOCK_OPAQUE_ARRAY
    if fast_leaves:
        BLOCK_OPAQUE_ARRAY[12] = True
        BLOCK_OPAQUE_ARRAY[16] = True
        BLOCK_OPAQUE_ARRAY[17] = True
        
    gpu_mode = False
    config = None
    import pyglet
    if force_legacy:
        gpu_mode = False
        config = pyglet.gl.Config(depth_size=24, double_buffer=True)
        print("GPU_MODE [PASIF]: Force Legacy Mode aktif. OpenGL 3.3 kullanilacak.")
    else:
        try:
            pyglet.options['search_local_libs'] = True
            config_43 = pyglet.gl.Config(major_version=4, minor_version=3, forward_compatible=True, depth_size=24, double_buffer=True)
            test_win = pyglet.window.Window(width=1, height=1, config=config_43, visible=False)
            test_win.close()
            gpu_mode = True
            config = config_43
            print("GPU_MODE [AKTIF]: OpenGL 4.3 destekleniyor. Compute Shaders kullanilacak.")
        except Exception as e:
            gpu_mode = False
            config = pyglet.gl.Config(depth_size=24, double_buffer=True)
            print(f"GPU_MODE [PASIF]: OpenGL 4.3 desteklenmiyor. Legacy Mode kullanilacak. ({e})")
        
    engine = PythonCraftEngine(render_distance=distance, simulation_distance=sim_distance, fast_leaves=fast_leaves, debug_mode=debug_mode, flat_mode=flat_mode, console_mode=console_mode, config=config, gpu_mode=gpu_mode)
    pyglet.app.run()

if __name__ == '__main__':
    main()
