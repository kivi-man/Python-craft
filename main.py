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
from world.terrain import CACTUS, SAND, BLOCK_MAX_STACK_ARRAY
from renderer.mesh_builder import build_chunk_mesh
from core.player import Player
from core.world_db import load_level_dat, save_level_dat
from core.raycast import raycast
from core.math_utils import perspective_matrix, look_at_matrix, normalize_vec, sub_vec, cross_vec, dot_vec
from renderer.shader import create_shader_program
from renderer.camera import Camera

# --- MOVED IMPORTS ---
from core.raycast import raycast
from core.recipes import RecipeManager
from core.texture_manager import TextureManager
from pyglet.gl import glTexImage3D
from pyglet.window import mouse
from world.terrain import BLOCK_HARDNESS_ARRAY
from world.terrain import BLOCK_OPAQUE_ARRAY
from world.terrain import PORKCHOP_RAW
from world.terrain import get_break_time
import os
import pyglet
import pyglet.font
import pyglet.shapes
import pyglet.text
import random
import sys
import time
# ---------------------






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

class PythonCraftEngine(pyglet.window.Window, InputMixin, ChunkMixin, GUIMixin, EntityMixin):
    def __init__(self, render_distance=4, simulation_distance=4, fast_leaves=False, debug_mode=False, flat_mode=False, console_mode=False, config=None, gpu_mode=False):
        super().__init__(width=1280, height=720, caption="PythonCraft Engine",
                         resizable=True, vsync=False, config=config)
        self.gpu_mode = gpu_mode
        self.render_distance = render_distance
        self.simulation_distance = simulation_distance
        self.debug_mode = debug_mode
        self.flat_mode = flat_mode
        self.console_mode = console_mode
        
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
        
        tex_buffer = (ctypes.c_ubyte * len(tex_data)).from_buffer_copy(tex_data)
        glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA8, 16, 16, num_layers, 0, GL_RGBA, GL_UNSIGNED_BYTE, tex_buffer)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)
        
        self.camera = Camera()
        self.player = Player(44.0, 100.0, -4.0)
        self.sound_system = SoundSystem(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'SFX', 'Pythoncraft'))
        self.inventory_blocks = [0] * 55
        self.inventory_counts = [0] * 55
        
        player_data = load_level_dat()
        self.void_mode = False
        if player_data:
            if "pos" in player_data:
                px, py, pz = player_data["pos"]
                self.player.x, self.player.y, self.player.z = px, py, pz
            if "rot" in player_data:
                ryaw, rpitch = player_data["rot"]
                self.player.rotation = [ryaw, rpitch]
            if "inventory_blocks" in player_data:
                self.inventory_blocks = player_data["inventory_blocks"]
                self.inventory_counts = player_data["inventory_counts"]
            
            # Autodetect if this map is a flat/void map to prevent procedural generation over it
            gen_name = player_data.get("generatorName", "").lower()
            gen_opts = player_data.get("generatorOptions", "")
            if gen_name == "flat" or gen_name == "void":
                # If there are no options, or it specifies air, treat it as a void map
                # This prevents generating huge mountains around a parkour map
                if "minecraft:air" in str(gen_opts) or not gen_opts:
                    self.void_mode = True
                else:
                    self.flat_mode = True

        self.selected_slot = 0
        self.selected_block_id = self.inventory_blocks[self.selected_slot]
        
        self.recipe_manager = RecipeManager(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'recipes.json'))
        
        self.inventory_open = False
        self.cursor_item_id = 0
        self.cursor_item_count = 0
        
        # Mouse hold state and action cooldown tracking
        self.mouse_held = {mouse.LEFT: False, mouse.RIGHT: False}
        self.mouse_action_cooldown = 0.0
        self.breaking_pos = None
        self.breaking_progress = 0.0
        self.world_chunks = {}
        self.world_data_maps = {}
        self.world_light_maps = {}
        self.world_biomes = {}
        self.pending_decorations = {}
        self.total_verts = 0
        self.rendered_chunks = 0
        
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
            self.player.rotation,
            self.inventory_blocks,
            self.inventory_counts
        )
            
        super().on_close()

    def update(self, dt):
        # Update Sound System (Music and ambiance)
        if hasattr(self, 'sound_system'):
            self.sound_system.update_music(dt, dimension="OVERWORLD")

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
            
        if not getattr(self, 'inventory_open', False):
            # Mouse hold actions (continuous block break/place)
            if self.mouse_action_cooldown > 0.0:
                self.mouse_action_cooldown -= dt
                
            if self.mouse_action_cooldown <= 0.005:
                if self.mouse_held[mouse.LEFT]:
                    self.swing_time = 0.01
                    self._handle_mouse_action(mouse.LEFT)
                    self.mouse_action_cooldown = 0.20
                elif self.mouse_held[mouse.RIGHT]:
                    if self.selected_block_id == PORKCHOP_RAW:
                        pass # Handled below for continuous eating
                    else:
                        self.swing_time = 0.01
                        self._handle_mouse_action(mouse.RIGHT)
                        self.mouse_action_cooldown = 0.20
                    
            # Continuous Eating Progress
            if self.mouse_held[mouse.RIGHT]:
                if self.selected_block_id == PORKCHOP_RAW and self.player.hunger < 20.0:
                    self.eating_progress += dt
                    
                    # Play random eating sound every 0.3 seconds
                    if self.eating_progress - self.last_eat_sound > 0.3:
                        self.last_eat_sound = self.eating_progress
                        self.sound_system.play("eSoundType_EATING", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5, pitch=random.uniform(0.9, 1.1))
                        
                    # Bob hand slightly while eating
                    self.swing_time += dt * 3.0
                    
                    if self.eating_progress >= 1.6: # Takes 1.6 seconds to eat
                        self.eating_progress = 0.0
                        self.player.hunger = min(20.0, getattr(self.player, 'hunger', 20.0) + 4.0) # Restores 4 hunger points (2 shanks)
                        self.inventory_counts[self.selected_slot] -= 1
                        if self.inventory_counts[self.selected_slot] <= 0:
                            self.inventory_blocks[self.selected_slot] = 0
                            self.selected_block_id = 0
                else:
                    self.eating_progress = 0.0
            else:
                self.eating_progress = 0.0
                    
            # Always update targeted block for highlighting
            eye_pos = self.player.get_eye_position()
            direction = self.camera.get_front()
            if eye_pos != self.last_raycast_eye_pos or direction != self.last_raycast_dir:
                self.last_raycast_eye_pos = eye_pos
                self.last_raycast_dir = direction
                self._camera_matrix_dirty = True
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
                            hardness = BLOCK_HARDNESS_ARRAY[broken_id]
                            if hardness >= 0:
                                self.breaking_progress += dt
                                if random.random() < 0.25:
                                    self.spawn_crack_particles(hx, hy, hz, broken_id, 1)
                                    # Play continuous digging sound
                                    if time.perf_counter() - self.last_dig_sound_time > 0.2:
                                        self.last_dig_sound_time = time.perf_counter()
                                        sound_enum = self.sound_system.get_dig_sound(broken_id)
                                        self.sound_system.play(sound_enum, x=hx+0.5, y=hy+0.5, z=hz+0.5, volume=0.3, pitch=random.uniform(0.9, 1.1))
                                    
                                required_time = get_break_time(broken_id, self.selected_block_id)
                                if self.breaking_progress >= required_time:
                                    broken_id, broken_data = self.get_block_info(hx, hy, hz)
                                    self.spawn_destruction_particles(hx, hy, hz, broken_id)
                                    self.spawn_item_entity(broken_id, hx + 0.5, hy + 0.5, hz + 0.5)
                                    self.set_block(hx, hy, hz, 0)
                                    
                                    from core.special_blocks import is_door
                                    if is_door(broken_id):
                                        if (broken_data & 8) != 0:
                                            lower_id, _ = self.get_block_info(hx, hy - 1, hz)
                                            if lower_id == broken_id:
                                                self.set_block(hx, hy - 1, hz, 0)
                                        else:
                                            upper_id, _ = self.get_block_info(hx, hy + 1, hz)
                                            if upper_id == broken_id:
                                                self.set_block(hx, hy + 1, hz, 0)
                                    
                                    upper_id, upper_data = self.get_block_info(hx, hy + 1, hz)
                                    if is_door(upper_id) and (upper_data & 8) == 0:
                                        self.set_block(hx, hy + 1, hz, 0)
                                        self.spawn_item_entity(upper_id, hx + 0.5, hy + 1.5, hz + 0.5)
                                        uu_id, _ = self.get_block_info(hx, hy + 2, hz)
                                        if uu_id == upper_id:
                                            self.set_block(hx, hy + 2, hz, 0)
                                    
                                    # Play break sound
                                    sound_enum = self.sound_system.get_dig_sound(broken_id)
                                    self.sound_system.play(sound_enum, x=hx+0.5, y=hy+0.5, z=hz+0.5, volume=0.5)
                                    
                                    self.breaking_pos = None
                                    self.breaking_progress = 0.0
                                    self.mouse_action_cooldown = 0.20
                    else:
                        self.breaking_pos = current_target
                        self.breaking_progress = 0.0
                else:
                    self.breaking_pos = None
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
        sprint = self.keys[key.LCTRL] or getattr(self, 'is_sprinting_w', False)
        
        # Prevent player from falling if inside an unloaded chunk
        pcx = int(math.floor(self.player.x / CHUNK_SIZE))
        pcz = int(math.floor(self.player.z / CHUNK_SIZE))
        
        t_player_start = time.perf_counter()
        if (pcx, pcz) in self.world_chunks:
            old_x, old_z = self.player.x, self.player.z
            old_vy = self.player.vy
            was_on_ground = getattr(self.player, 'on_ground', False)
            was_in_water = getattr(self.player, 'in_water', False)
            
            self.player.update(dt, dx, dz, jump, crouch, sprint, self.get_block_info)
            
            is_on_ground = getattr(self.player, 'on_ground', False)
            is_in_water = getattr(self.player, 'in_water', False)
            
            # Fall Damage Sound
            if is_on_ground and not was_on_ground and not is_in_water:
                if old_vy < -25.0:
                    self.sound_system.play("eSoundType_DAMAGE_FALL_BIG", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.8)
                    self.sound_system.play("eSoundType_DAMAGE_HURT", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.8)
                elif old_vy < -15.0:
                    self.sound_system.play("eSoundType_DAMAGE_FALL_SMALL", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5)
                    self.sound_system.play("eSoundType_DAMAGE_HURT", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.5)
            
            # Splash Sound
            if is_in_water and not was_in_water and old_vy < -10.0:
                self.sound_system.play("eSoundType_RANDOM_SPLASH", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.6)
                
            # Accumulate walking distance regardless of being on ground (for bunny hopping)
            dx_moved = self.player.x - old_x
            dz_moved = self.player.z - old_z
            dist = math.sqrt(dx_moved*dx_moved + dz_moved*dz_moved)
            self.distance_walked += dist
            
            if is_on_ground or is_in_water:
                step_threshold = 1.0 if sprint else 1.5
                
                # Force a landing step sound if landing from a small jump
                if is_on_ground and not was_on_ground and not is_in_water:
                    if old_vy >= -15.0:
                        self.distance_walked = step_threshold + 0.1
                        
                if self.distance_walked > step_threshold:
                    self.distance_walked = 0.0
                    if is_in_water:
                        self.sound_system.play("eSoundType_LIQUID_SWIM", x=self.player.x, y=self.player.y, z=self.player.z, volume=0.3)
                    else:
                        block_below = self.get_block(self.player.x, self.player.y - 0.1, self.player.z)
                        if block_below > 0:
                            sound_enum = self.sound_system.get_step_sound(block_below)
                            self.sound_system.play(sound_enum, x=self.player.x, y=self.player.y, z=self.player.z, volume=0.3)
        else:
            self.player.vy = 0.0 # Reset gravity accumulation
        t_player_end = time.perf_counter()
        
        self._update_entities(dt)
            
        eye_pos = self.player.get_eye_position()
        
        # 3D Audio Listener Update
        try:
            listener = pyglet.media.get_audio_driver().get_listener()
            listener.position = (self.camera.x, self.camera.y, self.camera.z)
            fx, fy, fz = self.camera.get_front()
            listener.forward_orientation = (fx, fy, fz)
            listener.up_orientation = (0, 1, 0)
        except Exception as e:
            pass # Failsafe if audio driver is missing
        
        dur = (time.perf_counter() - t_start) * 1000.0
        if dur > 2.0:
            self.log(f"[SLOW UPDATE] Total: {dur:.2f}ms | Load: {(t_load_end-t_load_start)*1000.0:.2f}ms | Queues: {(t_queue_end-t_queue_start)*1000.0:.2f}ms | Player: {(t_player_end-t_player_start)*1000.0:.2f}ms")
    

    def _get_slot_rect(self, slot_idx):
        if getattr(self, 'crafting_open', False) and hasattr(self, 'crafting_bg_sprite'):
            bg_x = self.crafting_bg_sprite.x
            bg_y = self.crafting_bg_sprite.y
            scale = self.crafting_bg_sprite.scale
        else:
            if not hasattr(self, 'inventory_bg_sprite') or self.inventory_bg_sprite is None:
                return 0, 0, 0, 0
                
            bg_x = self.inventory_bg_sprite.x
            bg_y = self.inventory_bg_sprite.y
            scale = self.inventory_bg_sprite.scale
        
        if 0 <= slot_idx <= 8:
            px = 8 + slot_idx * 18
            py = 8
        elif 9 <= slot_idx <= 35:
            rel = slot_idx - 9
            col = rel % 9
            row = rel // 9
            px = 8 + col * 18
            py = 30 + (2 - row) * 18
        elif 36 <= slot_idx <= 39:
            px = 8
            py = 88 + (slot_idx - 36) * 18
        elif 40 <= slot_idx <= 43:
            rel = slot_idx - 40
            col = rel % 2
            row = rel // 2
            px = 88 + col * 18
            py = 124 - row * 18
        elif slot_idx == 44:
            px = 144
            py = 114
        elif 45 <= slot_idx <= 53:
            rel = slot_idx - 45
            col = rel % 3
            row = rel // 3
            px = 30 + col * 18
            py = 133 - row * 18
        elif slot_idx == 54:
            px = 124
            py = 115
        else:
            return 0, 0, 0, 0
            
        return bg_x + px * scale, bg_y + py * scale, 16 * scale, 16 * scale

    def _draw_inventory_gui(self):
        if hasattr(self, 'inventory_bg_sprite') and self.inventory_bg_sprite is not None:
            self.inventory_bg_sprite.draw()
            bg_scale = self.inventory_bg_sprite.scale
            
            # Init labels if not present
            if not hasattr(self, 'count_labels'):
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                try:
                    pyglet.font.add_file(font_path)
                    target_font = 'Minecraftia'
                except Exception:
                    target_font = 'Arial'
                self.count_labels = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55)]
                self.crafting_label = pyglet.text.Label("Crafting", font_name=target_font, font_size=8, color=(64, 64, 64, 255), anchor_x="left", anchor_y="bottom")
                
            elif len(self.count_labels) < 55:
                target_font = self.count_labels[0].font_name
                self.count_labels.extend([pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55 - len(self.count_labels))])

            # Update and draw crafting label
            if hasattr(self, 'crafting_label'):
                self.crafting_label.font_size = max(8, int(8 * bg_scale))
                # Pyglet coordinates for Crafting grid start at px=88. The text should be at px=98, py=142 roughly
                self.crafting_label.x = int(self.inventory_bg_sprite.x + 98 * bg_scale)
                self.crafting_label.y = int(self.inventory_bg_sprite.y + 144 * bg_scale)
                self.crafting_label.draw()                
            # Hovered slot
            mouse_x, mouse_y = getattr(self, 'mouse_pos', (0, 0))
            hovered_slot = -1
            
            # Draw items and labels
            for slot_idx, b_id in enumerate(self.inventory_blocks[:45]):
                x, y, w, h = self._get_slot_rect(slot_idx)
                
                # Check hover
                if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                    hovered_slot = slot_idx
                    
                if b_id > 0 and b_id in getattr(self, 'block_icon_sprites', {}):
                    sprite = self.block_icon_sprites[b_id]
                    sprite_size = getattr(sprite, 'original_width', 64.0)
                    sprite.scale = (16.0 * bg_scale) / sprite_size
                    sprite.x = int(x)
                    sprite.y = int(y)
                    sprite.draw()
                    
                    count = self.inventory_counts[slot_idx]
                    if count > 0:
                        lbl = self.count_labels[slot_idx]
                        target_size = max(8, int(8 * bg_scale))
                        if lbl.font_size != target_size:
                            lbl.font_size = target_size
                        if lbl.text != str(count):
                            lbl.text = str(count)
                        lbl.x = int(x + 16 * bg_scale)
                        lbl.y = int(y)
                        lbl.draw()
                        
            # Draw Hover highlight
            if hovered_slot != -1:
                hx, hy, hw, hh = self._get_slot_rect(hovered_slot)

                hover_rect = pyglet.shapes.Rectangle(hx, hy, hw, hh, color=(255, 255, 255, 100))
                hover_rect.draw()
                
            # Draw cursor item
            cursor_id = getattr(self, 'cursor_item_id', 0)
            if cursor_id > 0 and cursor_id in getattr(self, 'block_icon_sprites', {}):
                sprite = self.block_icon_sprites[cursor_id]
                sprite_size = getattr(sprite, 'original_width', 64.0)
                sprite.scale = (16.0 * bg_scale) / sprite_size
                sprite.x = int(mouse_x - 8 * bg_scale)
                sprite.y = int(mouse_y - 8 * bg_scale)
                sprite.draw()
                
                count = getattr(self, 'cursor_item_count', 0)
                if count > 0:
                    lbl = getattr(self, 'cursor_label', None)
                    if not lbl:
                        self.cursor_label = pyglet.text.Label("", font_name='Arial', font_size=8, anchor_x="right", anchor_y="bottom")
                        lbl = self.cursor_label
                    target_size = max(8, int(8 * bg_scale))
                    if lbl.font_size != target_size:
                        lbl.font_size = target_size
                    if lbl.text != str(count):
                        lbl.text = str(count)
                    lbl.x = int(sprite.x + 16 * bg_scale)
                    lbl.y = int(sprite.y)
                    lbl.draw()

    def _draw_crafting_gui(self):
        if hasattr(self, 'crafting_bg_sprite') and self.crafting_bg_sprite is not None:
            self.crafting_bg_sprite.draw()
            bg_scale = self.crafting_bg_sprite.scale
            
            if not hasattr(self, 'count_labels'):
                font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                try:
                    pyglet.font.add_file(font_path)
                    target_font = 'Minecraftia'
                except Exception:
                    target_font = 'Arial'
                self.count_labels = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55)]
                self.crafting_label = pyglet.text.Label("Crafting", font_name=target_font, font_size=8, color=(64, 64, 64, 255), anchor_x="left", anchor_y="bottom")
            elif len(self.count_labels) < 55:
                target_font = self.count_labels[0].font_name
                self.count_labels.extend([pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom") for _ in range(55 - len(self.count_labels))])

            if hasattr(self, 'crafting_label'):
                self.crafting_label.font_size = max(8, int(8 * bg_scale))
                self.crafting_label.x = int(self.crafting_bg_sprite.x + 30 * bg_scale)
                self.crafting_label.y = int(self.crafting_bg_sprite.y + 144 * bg_scale)
                self.crafting_label.draw()
                
            mouse_x, mouse_y = getattr(self, 'mouse_pos', (0, 0))
            hovered_slot = -1
            
            # Draw player inventory (0-35) + Crafting grid & output (45-54)
            slots_to_draw = list(range(36)) + list(range(45, 55))
            
            for slot_idx in slots_to_draw:
                b_id = self.inventory_blocks[slot_idx]
                x, y, w, h = self._get_slot_rect(slot_idx)
                
                if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                    hovered_slot = slot_idx
                    
                if b_id > 0 and b_id in getattr(self, 'block_icon_sprites', {}):
                    sprite = self.block_icon_sprites[b_id]
                    sprite_size = getattr(sprite, 'original_width', 64.0)
                    sprite.scale = (16.0 * bg_scale) / sprite_size
                    sprite.x = int(x)
                    sprite.y = int(y)
                    sprite.draw()
                    
                    count = self.inventory_counts[slot_idx]
                    if count > 0:
                        lbl = self.count_labels[slot_idx]
                        target_size = max(8, int(8 * bg_scale))
                        if lbl.font_size != target_size:
                            lbl.font_size = target_size
                        if lbl.text != str(count):
                            lbl.text = str(count)
                        lbl.x = int(x + 16 * bg_scale)
                        lbl.y = int(y)
                        lbl.draw()
                        
            if hovered_slot != -1:
                hx, hy, hw, hh = self._get_slot_rect(hovered_slot)
                hover_rect = pyglet.shapes.Rectangle(hx, hy, hw, hh, color=(255, 255, 255, 100))
                hover_rect.draw()
                
            cursor_id = getattr(self, 'cursor_item_id', 0)
            if cursor_id > 0 and cursor_id in getattr(self, 'block_icon_sprites', {}):
                sprite = self.block_icon_sprites[cursor_id]
                sprite_size = getattr(sprite, 'original_width', 64.0)
                sprite.scale = (16.0 * bg_scale) / sprite_size
                sprite.x = int(mouse_x - 8 * bg_scale)
                sprite.y = int(mouse_y - 8 * bg_scale)
                sprite.draw()
                
                count = getattr(self, 'cursor_item_count', 0)
                if count > 0:
                    lbl = getattr(self, 'cursor_label', None)
                    if not lbl:
                        self.cursor_label = pyglet.text.Label("", font_name='Arial', font_size=8, anchor_x="right", anchor_y="bottom")
                        lbl = self.cursor_label
                    target_size = max(8, int(8 * bg_scale))
                    if lbl.font_size != target_size:
                        lbl.font_size = target_size
                    if lbl.text != str(count):
                        lbl.text = str(count)
                    lbl.x = int(sprite.x + 16 * bg_scale)
                    lbl.y = int(sprite.y)
                    lbl.draw()

    def _handle_inventory_click(self, x, y, button):
        
        clicked_slot = -1
        for i in range(55):
            sx, sy, sw, sh = self._get_slot_rect(i)
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                if getattr(self, 'crafting_open', False) and 40 <= i <= 44:
                    continue
                if getattr(self, 'inventory_open', False) and 45 <= i <= 54:
                    continue
                clicked_slot = i
                break
                
        if clicked_slot == -1:
            if button == mouse.LEFT and getattr(self, 'cursor_item_id', 0) > 0:
                for _ in range(self.cursor_item_count):
                    self.spawn_item_entity(self.cursor_item_id, self.player.x, self.player.y + 1.5, self.player.z)
                self.cursor_item_id = 0
                self.cursor_item_count = 0
            return
            
        if 36 <= clicked_slot <= 39:
            if getattr(self, 'cursor_item_id', 0) > 0:
                return
                
        if clicked_slot in (44, 54):
            slot_id = self.inventory_blocks[clicked_slot]
            if slot_id == 0: return
            
            cursor_id = getattr(self, 'cursor_item_id', 0)
            cursor_count = getattr(self, 'cursor_item_count', 0)
            slot_count = self.inventory_counts[clicked_slot]
            
            max_stack = BLOCK_MAX_STACK_ARRAY[slot_id] if slot_id > 0 else 64
            
            if cursor_id == 0 or (cursor_id == slot_id and cursor_count + slot_count <= max_stack):
                self.cursor_item_id = slot_id
                self.cursor_item_count = cursor_count + slot_count
                self.inventory_blocks[clicked_slot] = 0
                self.inventory_counts[clicked_slot] = 0
                
                # Consume grid
                grid_slots = range(45, 54) if clicked_slot == 54 else range(40, 44)
                for i in grid_slots:
                    if self.inventory_counts[i] > 0:
                        self.inventory_counts[i] -= 1
                        if self.inventory_counts[i] == 0:
                            self.inventory_blocks[i] = 0
                self._evaluate_crafting()
                
                if hasattr(self, 'sound_system'):
                    self.sound_system.play("eSoundType_RANDOM_CLICK", volume=0.5)
            return
                
        slot_id = self.inventory_blocks[clicked_slot]
        slot_count = self.inventory_counts[clicked_slot]
        
        # Play UI click sound
        if hasattr(self, 'sound_system'):
            self.sound_system.play("eSoundType_RANDOM_CLICK", volume=0.5)
        
        cursor_id = getattr(self, 'cursor_item_id', 0)
        cursor_count = getattr(self, 'cursor_item_count', 0)
        
        if button == mouse.LEFT:
            if cursor_id == 0:
                if slot_id > 0:
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
                    self.inventory_blocks[clicked_slot] = 0
                    self.inventory_counts[clicked_slot] = 0
            else:
                if slot_id == cursor_id:
                    max_stack = BLOCK_MAX_STACK_ARRAY[slot_id]
                    space = max_stack - slot_count
                    if space >= cursor_count:
                        self.inventory_counts[clicked_slot] += cursor_count
                        self.cursor_item_id = 0
                        self.cursor_item_count = 0
                    else:
                        self.inventory_counts[clicked_slot] = max_stack
                        self.cursor_item_count -= space
                else:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = cursor_count
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
        elif button == mouse.RIGHT:
            if cursor_id == 0:
                if slot_id > 0:
                    half = int(slot_count / 2)
                    rem = slot_count - half
                    if half > 0:
                        self.cursor_item_id = slot_id
                        self.cursor_item_count = half
                        self.inventory_counts[clicked_slot] = rem
                    else:
                        self.cursor_item_id = slot_id
                        self.cursor_item_count = 1
                        self.inventory_counts[clicked_slot] = 0
                        self.inventory_blocks[clicked_slot] = 0
            else:
                if slot_id == 0:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = 1
                    self.cursor_item_count -= 1
                    if self.cursor_item_count <= 0:
                        self.cursor_item_id = 0
                elif slot_id == cursor_id:
                    max_stack = BLOCK_MAX_STACK_ARRAY[slot_id]
                    if slot_count < max_stack:
                        self.inventory_counts[clicked_slot] += 1
                        self.cursor_item_count -= 1
                        if self.cursor_item_count <= 0:
                            self.cursor_item_id = 0
                else:
                    self.inventory_blocks[clicked_slot] = cursor_id
                    self.inventory_counts[clicked_slot] = cursor_count
                    self.cursor_item_id = slot_id
                    self.cursor_item_count = slot_count
                    
        # Synchronize selected block if hotbar is changed
        self.selected_block_id = self.inventory_blocks[self.selected_slot]
        self._evaluate_crafting()

    def _evaluate_crafting(self):
        if getattr(self, 'crafting_open', False):
            grid_width, grid_height = 3, 3
            start_slot = 45
            out_slot = 54
        elif getattr(self, 'inventory_open', False):
            grid_width, grid_height = 2, 2
            start_slot = 40
            out_slot = 44
        else:
            return
            
        grid = []
        for y in range(grid_height):
            row = []
            for x in range(grid_width):
                idx = start_slot + y * grid_width + x
                row.append(self.inventory_blocks[idx])
            grid.append(row)
            
        res_id, res_count = self.recipe_manager.match(grid, grid_width, grid_height)
        if res_id > 0:
            self.inventory_blocks[out_slot] = res_id
            self.inventory_counts[out_slot] = res_count
        else:
            self.inventory_blocks[out_slot] = 0
            self.inventory_counts[out_slot] = 0

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

        # RENDER BLOCK HIGHLIGHT
        if getattr(self, 'targeted_block', None) is not None:
            bx, by, bz = self.targeted_block
            glUseProgram(self.line_program)
            glUniformMatrix4fv(self.u_line_proj, 1, GL_FALSE, proj)
            glUniformMatrix4fv(self.u_line_view, 1, GL_FALSE, view)
            
            s_x, s_y, s_z = 1.0, 1.0, 1.0
            t_x, t_y, t_z = 0.0, 0.0, 0.0
            block_id, block_data = self.get_block_info(bx, by, bz)
            from core.special_blocks import is_door, is_slab
            if is_door(block_id):
                is_upper = (block_data & 8) != 0
                lower_data = block_data
                upper_data = block_data
                if is_upper:
                    adj_id, adj_data = self.get_block_info(bx, by - 1, bz)
                    if adj_id == block_id: lower_data = adj_data
                else:
                    adj_id, adj_data = self.get_block_info(bx, by + 1, bz)
                    if adj_id == block_id: upper_data = adj_data
                has_right_hinge = (upper_data & 1) != 0
                dir_val = lower_data & 3
                is_open = (lower_data & 4) != 0
                r = 3.0 / 16.0
                minX, minZ, maxX, maxZ = 0.0, 0.0, 1.0, 1.0
                if dir_val == 0:
                    if is_open:
                        if not has_right_hinge: maxZ = r
                        else: minZ = 1.0 - r
                    else: maxX = r
                elif dir_val == 1:
                    if is_open:
                        if not has_right_hinge: minX = 1.0 - r
                        else: maxX = r
                    else: maxZ = r
                elif dir_val == 2:
                    if is_open:
                        if not has_right_hinge: minZ = 1.0 - r
                        else: maxZ = r
                    else: minX = 1.0 - r
                elif dir_val == 3:
                    if is_open:
                        if not has_right_hinge: maxX = r
                        else: minX = 1.0 - r
                    else: minZ = 1.0 - r
                s_x, s_z = (maxX - minX), (maxZ - minZ)
                t_x, t_z = minX, minZ
            elif is_slab(block_id):
                s_y = 0.5
                if (block_data & 4) != 0: t_y = 0.5
            
            s = 1.005
            model_mat = np.array([
                [s_x * s, 0, 0, 0],
                [0, s_y * s, 0, 0],
                [0, 0, s_z * s, 0],
                [bx + t_x - (s_x * s - s_x) / 2.0, by + t_y - (s_y * s - s_y) / 2.0, bz + t_z - (s_z * s - s_z) / 2.0, 1]
            ], dtype=np.float32)
            
            glUniformMatrix4fv(self.u_line_model, 1, GL_FALSE, (GLfloat * 16)(*model_mat.flatten()))
            
            glBindVertexArray(self.line_vao)
            glDrawArrays(GL_LINES, 0, 24)
            glBindVertexArray(0)
            glUseProgram(0)
            
            # Break animation overlay
            if getattr(self, 'breaking_pos', None) == (bx, by, bz) and getattr(self, 'breaking_progress', 0.0) > 0.0:
                block_id = self.get_block(bx, by, bz)
                if block_id > 0:
                    hardness = BLOCK_HARDNESS_ARRAY[block_id]
                    if hardness > 0:
                        req_time = get_break_time(block_id, self.selected_block_id)
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
        if self.get_block(cx, cy_int + 1, cz) in (8, 9):
            u_water_surface_y = cy_int + 2.0
        elif self.get_block(cx, cy_int, cz) in (8, 9):
            u_water_surface_y = cy_int + 1.0
        elif self.get_block(cx, cy_int - 1, cz) in (8, 9):
            u_water_surface_y = float(cy_int)
            
        if u_water_surface_y > -999.0:
            glUseProgram(self.water_overlay_program)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_DEPTH_TEST) # Overlay covers screen
            
            # Inverse proj*view for world reconstruction
            if self._camera_matrix_dirty:
                self._cached_inv_pv = np.linalg.inv(proj_view.reshape(4,4)).flatten().astype(np.float32)
                self._camera_matrix_dirty = False
            inv_pv = self._cached_inv_pv
            glUniformMatrix4fv(self.u_inv_proj_view_overlay, 1, GL_FALSE, (ctypes.c_float * 16)(*inv_pv))
            glUniform1f(self.u_water_surface_y_overlay, u_water_surface_y)
            
            glBindVertexArray(self.dummy_vao)
            glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
            glBindVertexArray(0)
            
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)
            glUseProgram(0)

        # 3D Held Block Viewmodel Rendering
        if hasattr(self, 'hand_block_vaos') and self.selected_block_id in self.hand_block_vaos and getattr(self.camera, 'third_person_mode', 0) == 0:
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
            
        # 2D GUI Elements
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        # UI/Inventory tick
        if getattr(self, 'crafting_open', False):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            overlay_rect = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(0, 0, 0, 153))
            overlay_rect.draw()
            self._draw_crafting_gui()
        elif getattr(self, 'inventory_open', False):
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            overlay_rect = pyglet.shapes.Rectangle(0, 0, self.width, self.height, color=(0, 0, 0, 153))
            overlay_rect.draw()
            self._draw_inventory_gui()
        else:
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
                if not hasattr(self, 'count_labels_main'):
                    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets', 'fonts', 'Minecraftia-Regular.ttf')
                    try:
                        pyglet.font.add_file(font_path)
                        target_font = 'Minecraftia'
                    except Exception:
                        target_font = 'Arial'
                    self.count_labels_shadow = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom", batch=self.ui_batch, color=(63, 63, 63, 255)) for _ in range(9)]
                    self.count_labels_main = [pyglet.text.Label("", font_name=target_font, font_size=8, anchor_x="right", anchor_y="bottom", batch=self.ui_batch, color=(255, 255, 255, 255)) for _ in range(9)]
                
                # Render active slot selection frame FIRST (behind items and text)
                if hasattr(self, 'hotbar_sel_sprite') and self.hotbar_sel_sprite is not None:
                    scale = self.hotbar_sel_sprite.scale
                    self.hotbar_sel_sprite.x = int(self.hotbar_bg_sprite.x - 1 * scale + self.selected_slot * 20 * scale)
                    self.hotbar_sel_sprite.y = int(self.hotbar_bg_sprite.y - 1 * scale)
                    self.hotbar_sel_sprite.draw()

                # Reset labels visibility
                for i in range(9):
                    self.count_labels_shadow[i].text = ""
                    self.count_labels_main[i].text = ""

                for slot_idx, b_id in enumerate(self.inventory_blocks[:9]):
                    if b_id > 0 and b_id in self.block_icon_sprites:
                        sprite = self.block_icon_sprites[b_id]
                        sprite_size = getattr(sprite, 'original_width', 64.0)
                        sprite.scale = (13.0 * bg_scale) / sprite_size
                        sprite.x = int(self.hotbar_bg_sprite.x + (3 + slot_idx * 20) * bg_scale + 1.5 * bg_scale)
                        sprite.y = int(self.hotbar_bg_sprite.y + 3 * bg_scale + 1.5 * bg_scale)
                        sprite.draw()
                    
                        count = self.inventory_counts[slot_idx]
                        if count > 0:
                            target_size = max(8, int(8 * bg_scale))
                            base_x = self.hotbar_bg_sprite.x + (20 + slot_idx * 20) * bg_scale
                            base_y = self.hotbar_bg_sprite.y + 2 * bg_scale
                            offset = max(1, int(1 * bg_scale))
                            
                            lbl_shadow = self.count_labels_shadow[slot_idx]
                            lbl_main = self.count_labels_main[slot_idx]
                            
                            if lbl_shadow.font_size != target_size:
                                lbl_shadow.font_size = target_size
                                lbl_main.font_size = target_size
                            
                            s_count = str(count)
                            if lbl_shadow.text != s_count:
                                lbl_shadow.text = s_count
                                lbl_main.text = s_count
                        
                            lbl_shadow.x = base_x + offset
                            lbl_shadow.y = base_y - offset
                            lbl_main.x = base_x
                            lbl_main.y = base_y

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

                # Reset batched sprites visibility
                for spr in self.heart_bg_sprites + self.heart_fg_sprites + self.hunger_bg_sprites + self.hunger_fg_sprites + self.bubble_sprites:
                    spr.visible = False

                # Render Health and Hunger Bars
                if hasattr(self, 'heart_bg_sprites'):
                    # Draw Health (10 hearts)
                    heart_start_x = self.hotbar_bg_sprite.x
                    bar_y = int(self.hotbar_bg_sprite.y + 32 * self.hotbar_bg_sprite.scale)
                
                    player_health = getattr(self.player, 'health', 20.0)
                    player_hunger = getattr(self.player, 'hunger', 20.0)
                    tick_count = int(time.perf_counter() * 20)
                
                    heart_offset_index = -1
                    if player_health < 20.0 and player_hunger >= 18.0:
                        heart_offset_index = tick_count % 25
                    
                    for i in range(10):
                        yo = bar_y
                        if player_health <= 4.0:
                            yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                        if i == heart_offset_index:
                            yo += 2 * int(self.hotbar_bg_sprite.scale)
                        
                        hx = heart_start_x + i * 8 * self.heart_bg_sprites[i].scale
                        
                        bg_spr = self.heart_bg_sprites[i]
                        bg_spr.x = hx
                        bg_spr.y = yo
                        bg_spr.visible = True
                    
                        # Inner heart
                        h_val = player_health - (i * 2)
                        if h_val >= 1:
                            fg_spr = self.heart_fg_sprites[i]
                            fg_spr.x = hx
                            fg_spr.y = yo
                            fg_spr.image = self.tex_heart_full if h_val >= 2 else self.tex_heart_half
                            fg_spr.visible = True
                    
                    # Draw Hunger (10 icons, right-aligned)
                    hunger_start_x = self.hotbar_bg_sprite.x + self.hotbar_bg_sprite.width - 9 * self.hunger_bg_sprites[0].scale
                    for i in range(10):
                        yo = bar_y
                        if player_hunger <= 0.0 and tick_count % 20 < 10:
                             yo += random.randint(0, 1) * int(self.hotbar_bg_sprite.scale)
                         
                        hx = hunger_start_x - i * 8 * self.hunger_bg_sprites[i].scale
                        
                        bg_spr = self.hunger_bg_sprites[i]
                        bg_spr.x = hx
                        bg_spr.y = yo
                        bg_spr.visible = True
                    
                        # Inner food
                        f_val = player_hunger - (i * 2)
                        if f_val >= 1:
                            fg_spr = self.hunger_fg_sprites[i]
                            fg_spr.x = hx
                            fg_spr.y = yo
                            fg_spr.image = self.tex_hunger_full if f_val >= 2 else self.tex_hunger_half
                            fg_spr.visible = True

                    # Render Bubbles if underwater
                    if getattr(self.player, 'is_head_in_water', False):
                        bubble_y = bar_y + 10 * scale
                        air_supply = getattr(self.player, 'air_supply', 300.0)
                        air_scale = 10.0 / 300.0
                        air_scaled = air_supply * air_scale
                        count = int(math.ceil((air_supply - 2) * air_scale))
                        extra = int(math.ceil(air_scaled)) - count
                    
                        for i in range(min(10, count + extra)):
                            bx = hunger_start_x - (i * 8 * scale)
                            bspr = self.bubble_sprites[i]
                            bspr.x = bx
                            bspr.y = bubble_y
                            bspr.image = self.tex_bubble_full if i < count else self.tex_bubble_popped
                            bspr.visible = True

                # Draw UI Batch! (One draw call for all labels, hearts, hunger, and bubbles)
                if hasattr(self, 'ui_batch'):
                    self.ui_batch.draw()

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

    def _drop_inventory_excess(self):
        # Drop cursor item
        if getattr(self, 'cursor_item_count', 0) > 0 and getattr(self, 'cursor_item_id', 0) > 0:
            for _ in range(self.cursor_item_count):
                self.spawn_item_entity(self.cursor_item_id, self.player.x, self.player.y + 1.5, self.player.z)
            self.cursor_item_count = 0
            self.cursor_item_id = 0
            
        # Drop crafting grid items (slots 40, 41, 42, 43)
        for i in range(40, 44):
            if self.inventory_counts[i] > 0 and self.inventory_blocks[i] > 0:
                for _ in range(self.inventory_counts[i]):
                    self.spawn_item_entity(self.inventory_blocks[i], self.player.x, self.player.y + 1.5, self.player.z)
                self.inventory_counts[i] = 0
                self.inventory_blocks[i] = 0

        # Drop large crafting grid items (slots 45-53)
        for i in range(45, 54):
            if self.inventory_counts[i] > 0 and self.inventory_blocks[i] > 0:
                for _ in range(self.inventory_counts[i]):
                    self.spawn_item_entity(self.inventory_blocks[i], self.player.x, self.player.y + 1.5, self.player.z)
                self.inventory_counts[i] = 0
                self.inventory_blocks[i] = 0

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
                self._drop_inventory_excess()
                return pyglet.event.EVENT_HANDLED
            elif getattr(self, 'crafting_open', False):
                self.crafting_open = False
                self.set_exclusive_mouse(True)
                self._drop_inventory_excess()
                return pyglet.event.EVENT_HANDLED
        
        super().on_key_press(symbol, modifiers)
        InputMixin.on_key_press(self, symbol, modifiers)
        
        if symbol == key.E:
            if getattr(self, 'crafting_open', False):
                self.crafting_open = False
                self.set_exclusive_mouse(True)
                self._drop_inventory_excess()
                self._evaluate_crafting()
            else:
                self.inventory_open = not getattr(self, 'inventory_open', False)
                self.set_exclusive_mouse(not self.inventory_open)
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
