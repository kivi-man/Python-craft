import os
import ctypes
import pyglet
from pyglet.gl import *
from pyglet.gl import *
import numpy as np
from core.math_utils import get_hand_cube_vertices, get_item_sprite_vertices
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

    def _create_2d_item_sprite(self, texture_name):
        import os
        from PIL import Image
        import pyglet.image
        import pyglet.sprite
        
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        items_dir = os.path.join(base_dir, 'assets', 'textures', 'items')
        blocks_dir = os.path.join(base_dir, 'assets', 'textures', 'blocks')
        
        p = os.path.join(items_dir, texture_name)
        if not os.path.exists(p):
            p = os.path.join(blocks_dir, texture_name)
            if not os.path.exists(p):
                return None
            
        img = Image.open(p).convert("RGBA")
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        raw_data = img.tobytes()
        pyglet_img = pyglet.image.ImageData(img.width, img.height, 'RGBA', raw_data)
        tex = pyglet_img.get_texture()
        
        glBindTexture(tex.target, tex.id)
        glTexParameteri(tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glBindTexture(tex.target, 0)
        
        sprite = pyglet.sprite.Sprite(img=tex)
        sprite.original_width = float(img.width)
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
            
            # Health and Hunger Icons (9x9)
            heart_y = crosshair_img.height - 9
            hunger_y = crosshair_img.height - 36
            
            def get_icon_tex(x, y):
                reg = crosshair_img.get_region(x, y, 9, 9).get_texture()
                glBindTexture(reg.target, reg.id)
                glTexParameteri(reg.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(reg.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glBindTexture(reg.target, 0)
                return reg
                
            self.tex_heart_empty = get_icon_tex(16, heart_y)
            self.tex_heart_half = get_icon_tex(61, heart_y)
            self.tex_heart_full = get_icon_tex(52, heart_y)
            
            self.tex_hunger_empty = get_icon_tex(16, hunger_y)
            self.tex_hunger_half = get_icon_tex(61, hunger_y)
            self.tex_hunger_full = get_icon_tex(52, hunger_y)
            
            # Bubble Icons (9x9)
            bubble_y = crosshair_img.height - 27
            self.tex_bubble_full = get_icon_tex(16, bubble_y)
            self.tex_bubble_popped = get_icon_tex(25, bubble_y)
            
            self.ui_batch = pyglet.graphics.Batch()
            
            # Create sprite pools for batched drawing
            self.heart_bg_sprites = [pyglet.sprite.Sprite(img=self.tex_heart_empty, batch=self.ui_batch) for _ in range(10)]
            self.heart_fg_sprites = [pyglet.sprite.Sprite(img=self.tex_heart_full, batch=self.ui_batch) for _ in range(10)]
            self.hunger_bg_sprites = [pyglet.sprite.Sprite(img=self.tex_hunger_empty, batch=self.ui_batch) for _ in range(10)]
            self.hunger_fg_sprites = [pyglet.sprite.Sprite(img=self.tex_hunger_full, batch=self.ui_batch) for _ in range(10)]
            self.bubble_sprites = [pyglet.sprite.Sprite(img=self.tex_bubble_full, batch=self.ui_batch) for _ in range(10)]
            
            for spr in self.heart_bg_sprites + self.heart_fg_sprites + self.hunger_bg_sprites + self.hunger_fg_sprites + self.bubble_sprites:
                spr.scale = 2
                spr.visible = False
            
            # XP Bar (182x5)
            xp_empty_y = crosshair_img.height - 69
            xp_full_y = crosshair_img.height - 74
            
            reg_xp_empty = crosshair_img.get_region(0, xp_empty_y, 182, 5).get_texture()
            glBindTexture(reg_xp_empty.target, reg_xp_empty.id)
            glTexParameteri(reg_xp_empty.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(reg_xp_empty.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(reg_xp_empty.target, 0)
            self.tex_xp_empty = reg_xp_empty
            
            reg_xp_full = crosshair_img.get_region(0, xp_full_y, 182, 5).get_texture()
            glBindTexture(reg_xp_full.target, reg_xp_full.id)
            glTexParameteri(reg_xp_full.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(reg_xp_full.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
            glBindTexture(reg_xp_full.target, 0)
            self.tex_xp_full = reg_xp_full
            
            self.spr_xp_empty = pyglet.sprite.Sprite(img=self.tex_xp_empty)
            self.spr_xp_full = pyglet.sprite.Sprite(img=self.tex_xp_full)
            
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

            # Load Full Inventory Background
            inv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'container', 'inventory.png')
            if os.path.exists(inv_path):
                inv_img = pyglet.image.load(inv_path)
                # The active GUI region in inventory.png is 176x166 (top-left aligned, but Pyglet is bottom-left, so y=height-166)
                inv_region = inv_img.get_region(0, inv_img.height - 166, 176, 166)
                inv_tex = inv_region.get_texture()
                glBindTexture(inv_tex.target, inv_tex.id)
                glTexParameteri(inv_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(inv_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                self.inventory_bg_sprite = pyglet.sprite.Sprite(img=inv_tex)
            else:
                self.inventory_bg_sprite = None

            # Load Crafting Table Background
            craft_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'assets', 'textures', 'gui', 'container', 'crafting_table.png')
            if os.path.exists(craft_path):
                craft_img = pyglet.image.load(craft_path)
                craft_region = craft_img.get_region(0, craft_img.height - 166, 176, 166)
                craft_tex = craft_region.get_texture()
                glBindTexture(craft_tex.target, craft_tex.id)
                glTexParameteri(craft_tex.target, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexParameteri(craft_tex.target, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                self.crafting_bg_sprite = pyglet.sprite.Sprite(img=craft_tex)
            else:
                self.crafting_bg_sprite = None

            
            # Load Block Icons as 3D Isometric Cubes
            self.block_icon_sprites = {}
            from world.terrain import BLOCK_REGISTRY
            sprite_blocks = {31, 37, 38, 175, 176, 177, 178}
            
            for name, data in BLOCK_REGISTRY.items():
                b_id = data["id"]
                if b_id == 0: continue
                
                tex = data.get("texture")
                sprite = None
                
                if b_id == 64:
                    sprite = self._create_2d_item_sprite("door_wood.png")
                elif b_id == 71:
                    sprite = self._create_2d_item_sprite("door_iron.png")
                elif b_id >= 1000 or b_id in sprite_blocks:
                    if isinstance(tex, str):
                        sprite = self._create_2d_item_sprite(tex)
                
                if sprite is None:
                    if not tex:
                        sprite = self._create_3d_block_sprite('stone.png', 'stone.png', 'stone.png')
                    elif isinstance(tex, str):
                        sprite = self._create_3d_block_sprite(tex, tex, tex)
                    elif isinstance(tex, dict):
                        top = tex.get('top', 'stone.png')
                        side = tex.get('side', top)
                        sprite = self._create_3d_block_sprite(top, side, side)
                    else:
                        sprite = self._create_3d_block_sprite('stone.png', 'stone.png', 'stone.png')
                        
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

        if hasattr(self, 'inventory_bg_sprite') and self.inventory_bg_sprite is not None:
            self.inventory_bg_sprite.x = int((width - self.inventory_bg_sprite.width) // 2)
            self.inventory_bg_sprite.y = int((height - self.inventory_bg_sprite.height) // 2)

        if hasattr(self, 'crafting_bg_sprite') and self.crafting_bg_sprite is not None:
            self.crafting_bg_sprite.x = int((width - self.crafting_bg_sprite.width) // 2)
            self.crafting_bg_sprite.y = int((height - self.crafting_bg_sprite.height) // 2)


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
            num_verts = len(mesh) // 15
            return vao, vbo, num_verts

        from world.terrain import BLOCK_REGISTRY
        from core.special_blocks import is_slab, is_stairs
        
        # Define block IDs that should be rendered as flat 2D item sprites in hand
        # 31: Tallgrass, 37: Dandelion, 38: Rose, 175-178: Double plants, 1000: Raw Porkchop
        # 1001-1033: Tools and items
        SPRITE_BLOCKS = {31, 37, 38, 162, 163, 175, 176, 177, 178} | set(range(1000, 1034))
        
        for name, data in BLOCK_REGISTRY.items():
            b_id = data["id"]
            if b_id > 0:
                if b_id in SPRITE_BLOCKS:
                    layer_idx = 0
                    if b_id < len(self.block_layers):
                        layer_idx = self.block_layers[b_id, 0]
                    if b_id == 64 and "door_wood.png" in self.texture_manager.tex_names_to_layer:
                        layer_idx = self.texture_manager.tex_names_to_layer["door_wood.png"]
                    elif b_id == 71 and "door_iron.png" in self.texture_manager.tex_names_to_layer:
                        layer_idx = self.texture_manager.tex_names_to_layer["door_iron.png"]
                    elif b_id >= len(self.block_layers) and hasattr(self.texture_manager, 'item_layers') and b_id in self.texture_manager.item_layers:
                         pass
                    
                    mask = self.texture_manager.alpha_masks[int(layer_idx)] if hasattr(self.texture_manager, 'alpha_masks') and int(layer_idx) < len(self.texture_manager.alpha_masks) else None
                    if mask is not None:
                        mesh = get_item_sprite_vertices(layer_idx, mask)
                        if mesh is not None and len(mesh) > 0:
                            vao, vbo, num_verts = create_hand_vao(mesh)
                            self.hand_block_vaos[b_id] = (vao, vbo, num_verts)
                else:
                    aabbs = None
                    if is_slab(b_id):
                        aabbs = [[-0.5, -0.5, -0.5, 0.5, 0.0, 0.5]]
                    elif is_stairs(b_id):
                        # Base + step
                        aabbs = [
                            [-0.5, -0.5, -0.5, 0.5, 0.0, 0.5],
                            [0.0, 0.0, -0.5, 0.5, 0.5, 0.5]
                        ]
                    elif b_id == 81: # Cactus
                        aabbs = [[-0.4375, -0.5, -0.4375, 0.4375, 0.5, 0.4375]]
                        
                    mesh = get_hand_cube_vertices(b_id, self.block_layers, aabbs=aabbs)
                    if mesh is not None and len(mesh) > 0:
                        vao, vbo, num_verts = create_hand_vao(mesh)
                        self.hand_block_vaos[b_id] = (vao, vbo, num_verts)

