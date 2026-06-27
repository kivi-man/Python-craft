import os
import ctypes
from pyglet.gl import *
from pyglet.window import mouse
from core.texture_manager import TextureManager
from core.recipes import RecipeManager
from core.sound_system import SoundSystem
from renderer.camera import Camera
from renderer.shader import create_shader_program
from core.player import Player
from core.world_db import load_level_dat

class EngineSetupMixin:
    def _setup_opengl_and_shaders(self, fast_leaves):
        glClearColor(0.47, 0.65, 1.0, 1.0)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        shader_dir = os.path.join(root_dir, 'shaders')
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
        

    def _setup_textures_and_materials(self, fast_leaves):
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        texture_dir = os.path.join(root_dir, 'assets', 'textures')
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
        

    def _init_core_variables(self):
        self.camera = Camera()
        self.player = Player(44.0, 100.0, -4.0)
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.sound_system = SoundSystem(os.path.join(root_dir, 'assets', 'SFX', 'Pythoncraft'))
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
        
        root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.recipe_manager = RecipeManager(os.path.join(root_dir, 'recipes.json'))
        
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

