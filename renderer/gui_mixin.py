import os
import ctypes
import pyglet
from pyglet.gl import *
import numpy as np
from core.math_utils import get_hand_cube_vertices
from PIL import Image

# Import CACTUS which is used in _init_gui
from world.terrain import CACTUS

class GUIMixin:

    def _create_3d_block_sprite(self, top_name, left_name, right_name):
        from PIL import Image
        import pyglet.image
        import pyglet.sprite
        
        blocks_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'blocks')
        
        def load_pillow_img(name):
            p = os.path.join(blocks_dir, name)
            if os.path.exists(p):
                return Image.open(p).convert("RGBA")
            return None
            
        top_img = load_pillow_img(top_name)
        left_img = load_pillow_img(left_name)
        right_img = load_pillow_img(right_name)
        
        if not top_img and not left_img and not right_img:
            # Fallback
            fallback = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
            raw_data = fallback.tobytes()
            p_img = pyglet.image.ImageData(64, 64, 'RGBA', raw_data)
            tex = p_img.get_texture()
            sprite = pyglet.sprite.Sprite(img=tex)
            return sprite
            
        if not top_img: top_img = left_img or right_img
        if not left_img: left_img = top_img or right_img
        if not right_img: right_img = top_img or left_img
        
        dest = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
        dest_pix = dest.load()
        top_pix = top_img.load()
        left_pix = left_img.load()
        right_pix = right_img.load()
        
        for y in range(64):
            for x in range(64):
                # Top Face
                if (x + 2*y >= 32) and (x - 2*y <= 32) and (x - 2*y >= -32) and (x + 2*y <= 96) and (y < 32):
                    x_prime = (x - 32) / 32.0
                    y_prime = (y - 16) / 16.0
                    u = int(8 * (x_prime + y_prime + 1))
                    v = int(8 * (-x_prime + y_prime + 1))
                    u = max(0, min(15, u))
                    v = max(0, min(15, v))
                    dest_pix[x, y] = top_pix[u, v]
                # Left Face (Shaded 0.6)
                elif (x >= 0) and (x < 32) and (y >= 16 + x/2) and (y < 48 + x/2):
                    u = int(x / 2)
                    v = int((y - (16 + x/2.0)) / 2)
                    u = max(0, min(15, u))
                    v = max(0, min(15, v))
                    r, g, b, a = left_pix[u, v]
                    dest_pix[x, y] = (int(r * 0.6), int(g * 0.6), int(b * 0.6), a)
                # Right Face (Shaded 0.8)
                elif (x >= 32) and (x < 64) and (y >= 48 - x/2) and (y < 80 - x/2):
                    u = int((x - 32) / 2)
                    v = int((y - (48 - x/2.0)) / 2)
                    u = max(0, min(15, u))
                    v = max(0, min(15, v))
                    r, g, b, a = right_pix[u, v]
                    dest_pix[x, y] = (int(r * 0.8), int(g * 0.8), int(b * 0.8), a)
                    
        # Flip vertically for Pyglet coordinates compatibility
        dest = dest.transpose(Image.FLIP_TOP_BOTTOM)
        raw_data = dest.tobytes()
        pyglet_img = pyglet.image.ImageData(64, 64, 'RGBA', raw_data)
        tex = pyglet_img.get_texture()
        
        # Apply linear filtering to make cube edges smooth like a high-res 3D model
        glBindTexture(tex.target, tex.id)
        glTexParameteri(tex.target, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(tex.target, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(tex.target, 0)
        
        tex.min_filter = GL_LINEAR
        tex.mag_filter = GL_LINEAR
        
        sprite = pyglet.sprite.Sprite(img=tex)
        return sprite


    def _init_gui(self):
        import pyglet.image
        import pyglet.sprite
        
        try:
            icons_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'icons.png')
            crosshair_img = pyglet.image.load(icons_path)
            # The top-left 16x16 region in icons.png is the crosshair
            # Pyglet coordinates start from bottom-left, so we set y = height - 16
            region = crosshair_img.get_region(0, crosshair_img.height - 16, 16, 16)
            texture = region.get_texture()
            
            # Apply nearest-neighbor filtering to prevent texture blurring
            glBindTexture(texture.target, texture.id)
            glTexParameteri(texture.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(texture.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(texture.target, 0)
            
            texture.min_filter = GL_NEAREST
            texture.mag_filter = GL_NEAREST
            
            self.crosshair_sprite = pyglet.sprite.Sprite(img=texture)
            self.crosshair_sprite.scale = 2
            
            # Load Hotbar Textures
            gui_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'gui.png')
            gui_img = pyglet.image.load(gui_path)
            
            # Hotbar Background (182x22)
            bg_region = gui_img.get_region(0, gui_img.height - 22, 182, 22)
            bg_tex = bg_region.get_texture()
            glBindTexture(bg_tex.target, bg_tex.id)
            glTexParameteri(bg_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(bg_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            self.hotbar_bg_sprite = pyglet.sprite.Sprite(img=bg_tex)
            self.hotbar_bg_sprite.scale = 2
            
            # Active Slot Selection Frame (24x24)
            sel_region = gui_img.get_region(0, gui_img.height - 46, 24, 24)
            sel_tex = sel_region.get_texture()
            glBindTexture(sel_tex.target, sel_tex.id)
            glTexParameteri(sel_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(sel_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            self.hotbar_sel_sprite = pyglet.sprite.Sprite(img=sel_tex)
            self.hotbar_sel_sprite.scale = 2
            
            # Load Block Icons as 3D Isometric Cubes
            self.block_icon_sprites = {}
            block_icons = {
                1: ('stone.png', 'stone.png', 'stone.png'),
                3: ('grass_top.png', 'grass_side.png', 'grass_side.png'),
                20: ('glass.png', 'glass.png', 'glass.png'),
                12: ('leaves_oak.png', 'leaves_oak.png', 'leaves_oak.png'),
                4: ('water.png', 'water.png', 'water.png'),
                CACTUS: ('cactus_top.png', 'cactus_side.png', 'cactus_side.png')
            }
            
            for b_id, faces in block_icons.items():
                sprite = self._create_3d_block_sprite(faces[0], faces[1], faces[2])
                if sprite:
                    self.block_icon_sprites[b_id] = sprite
            
            # Position user interface elements
            self._update_gui_positions(self.width, self.height)
        except Exception as e:
            print(f"[GUI] Failed to load user interface: {e}")
            self.crosshair_sprite = None
            self.hotbar_bg_sprite = None
            self.hotbar_sel_sprite = None


    def _update_gui_positions(self, width, height):
        if hasattr(self, 'crosshair_sprite') and self.crosshair_sprite is not None:
            # Round coordinates to integers to prevent sub-pixel rendering blurriness
            self.crosshair_sprite.x = int((width - self.crosshair_sprite.width) // 2)
            self.crosshair_sprite.y = int((height - self.crosshair_sprite.height) // 2)
            
        if hasattr(self, 'hotbar_bg_sprite') and self.hotbar_bg_sprite is not None:
            # Center hotbar horizontally, keep 10 pixels margin from the bottom
            self.hotbar_bg_sprite.x = int((width - self.hotbar_bg_sprite.width) // 2)
            self.hotbar_bg_sprite.y = 10


    def _init_hand_blocks(self):
        self.hand_block_vaos = {}
        
        def create_hand_vao(mesh):
            vao = GLuint(0)
            glGenVertexArrays(1, ctypes.byref(vao))
            glBindVertexArray(vao)
            vbo = GLuint(0)
            glGenBuffers(1, ctypes.byref(vbo))
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, mesh.nbytes, mesh.ctypes.data, GL_STATIC_DRAW)
            
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
            
            glBindBuffer(GL_ARRAY_BUFFER, 0)
            glBindVertexArray(0)
            return vao, vbo

        for b_id in [1, 3, 20, 12, 4, CACTUS]:
            mesh = get_hand_cube_vertices(b_id, self.block_layers)
            vao, vbo = create_hand_vao(mesh)
            self.hand_block_vaos[b_id] = (vao, vbo)

